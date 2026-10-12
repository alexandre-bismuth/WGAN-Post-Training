"""Compounding-error (exposure-bias) token dump — per-position divergence raw material.

WHY THIS EXISTS: the sealed evals score pooled distributional realism over whole
continuations; nothing measures error growth AS A FUNCTION OF continuation position, which
is the project's premise (adversarial post-training should reduce autoregressive
compounding error). This script produces the raw material for that curve: for X held-out
contexts it saves, index-for-index aligned,
  tokens_real.npz    uint16 [X, n_gen*26]  the TRUE continuation token stream
  tokens_<row>.npz   uint16 [X, n_gen*26]  each model row's generated token stream
using the same CRN protocol as test_eval (all rows in ONE program, one rng_grid, identical
contexts) — discrete sampling amplifies TF32 cross-program noise into divergent token
sequences, so cross-row comparability requires a single program.

The generated tokens are generate()'s out[3] (msgs_tokens [Q, n_gen, 26]), which the
test_eval path computes and discards; the true tokens are _prep_real_batch's
real_cont_tokens (m_seq[:, n_cond*26+1 : n_cond*26+1+n_gen*26] — the +1 START offset), the
exact alignment the GAN critic trains on. Analysis (per-position KL over phase alphabets,
entropy normalization, split-half floor, z-drift) is CPU-side in
compounding_error_analysis.py — this job does generation only.

Row mechanisms are test_eval's: --eggroll proj rows (breadcrumb + generator_proj payload,
anchor decoder head), --full_ckpt rows (complete generator, own head), and the merge_noop
canary whose token stream must be BIT-identical to the anchor row's (exit 3 otherwise).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

from ..config import DEFAULT as CFG
from .test_eval import apply_flat_subtree, load_proj_payload, _tree_get


def flatten_tokens(msgs_tokens):
    """generate()'s out[3] [Q, n_gen, msg_len] -> [Q, n_gen*msg_len] token stream, aligned
    index-for-index with real_cont_tokens (both start on a message boundary)."""
    a = np.asarray(msgs_tokens)
    return a.reshape(a.shape[0], -1)


def to_u16(tok, name):
    """Vocab (2112) fits uint16 with huge headroom; fail loudly on anything unexpected."""
    a = np.asarray(tok)
    lo, hi = int(a.min()), int(a.max())
    if lo < 0 or hi > 65535:
        raise ValueError(f"{name}: token ids outside uint16 range [{lo}, {hi}]")
    return a.astype(np.uint16)


def run(args):
    from ..tests.s3_es_rollout import _prep_real_batch
    from ..es.es_generator import make_generate_es_sharded, tile_dirs_over_Q, grid_rngs

    Q = args.n_eval_ctx
    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, Q,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels, wide_book_dir=args.wide_book_dir)
    kernel0, bias0 = P["kernel"], P["bias"]
    top_n = int(CFG.rollout.sample_top_n) if args.top_n is None else int(args.top_n)
    print(f"[compounding_error] Q={Q} n_cond={args.n_cond} n_gen={args.n_gen} top_n={top_n} "
          f"wide_levels={args.wide_levels} seed={args.seed} data={args.data_dir}", flush=True)

    CH = int(args.gen_chunk or 0)
    if CH > 0:
        assert args.shard == "off", "--gen_chunk is a single-device path (use --shard off)"
        assert Q % CH == 0, f"--gen_chunk {CH} must divide n_eval_ctx {Q}"

    def _sliced(tree, sl):
        return jax.tree_util.tree_map(lambda x: x[sl], tree)

    os.makedirs(args.out_dir, exist_ok=True)
    real_tok = to_u16(P["real_cont_tokens"], "real")
    assert real_tok.shape == (Q, args.n_gen * 26), f"real tokens shape {real_tok.shape}"
    np.savez_compressed(os.path.join(args.out_dir, "tokens_real.npz"), tokens=real_tok)

    # Rows: (name, head_pop, params). One gen build + one rng draw = CRN across all rows.
    anchor_head = {"kernel": kernel0[None], "bias": bias0[None]}
    rows = [("anchor", anchor_head, P["bb_params"])]
    first_payload = None
    for spec in args.eggroll or []:
        name, _, d = spec.partition("=")
        flat, bc = load_proj_payload(d)
        first_payload = first_payload or flat
        merged = apply_flat_subtree(P["bb_params"], {k: jnp.asarray(v) for k, v in flat.items()})
        rows.append((name, anchor_head, merged))
        print(f"[compounding_error] row '{name}': proj ckpt step {bc['step']} "
              f"({len(flat)} leaves) <- {d}", flush=True)
    for spec in args.full_ckpt or []:
        from ..data import checkpoint_utils as ck
        name, _, rest = spec.partition("=")
        d, _, st = rest.rpartition(":")
        fl = ck.load_pretrained_generator(d, int(st), build_loaders=False)
        fp = fl["train_state"].params
        assert (jax.tree_util.tree_structure(fp)
                == jax.tree_util.tree_structure(P["bb_params"])), \
            f"full_ckpt '{name}': param tree structure differs from the anchor"
        for a, b in zip(jax.tree_util.tree_leaves(fp), jax.tree_util.tree_leaves(P["bb_params"])):
            assert np.shape(a) == np.shape(b), \
                f"full_ckpt '{name}': leaf shape {np.shape(a)} != anchor {np.shape(b)}"
        fhead = {"kernel": jnp.asarray(fp["decoder"]["kernel"])[None],
                 "bias": jnp.asarray(fp["decoder"]["bias"])[None]}
        rows.append((name, fhead, fp))
        print(f"[compounding_error] row '{name}': FULL ckpt step {st} <- {d}", flush=True)
    if first_payload is not None:
        anchor_resub = apply_flat_subtree(
            P["bb_params"], {k: _tree_get(P["bb_params"], k) for k in first_payload})
        rows.append(("merge_noop", anchor_head, anchor_resub))
    if args.rows:
        keep = {s.strip() for s in args.rows.split(",") if s.strip()}
        rows = [r for r in rows if r[0] == "anchor"
                or any(r[0] == k or r[0].startswith(k) for k in keep)]
        print(f"[compounding_error] --rows filter -> {[r[0] for r in rows]}", flush=True)

    gen = make_generate_es_sharded(P["model"], P["batchnorm"], P["encoder"], top_n,
                                   P["tick_size"], args.n_gen, P["sim_init"],
                                   P["valid_mask_array"], conditional=True, shard=args.shard)
    grid = dict(m=P["m_seq_inp"], b=P["b_seq_inp"], sim=P["sim_states_init"],
                ih=P["init_hidden_batched"], it=P["init_time_batched"])
    rng_grid = grid_rngs(jax.random.PRNGKey(args.seed + 1), 1, Q)        # G=1; CRN across rows

    def _gen_row(head_pop, ts):
        """gen() over all Q rollouts, keeping ONLY (tokens, num_errors) — out[3] and out[2].
        Same chunked head tiling as test_eval._gen_row: the single-member head is tiled per
        chunk, never over the full Q (a full-Q tile OOMs at Q=4096 before generation)."""
        if CH <= 0 or Q <= CH:
            o = gen(tile_dirs_over_Q(head_pop, Q), ts, grid["m"], grid["b"], grid["sim"],
                    rng_grid, grid["ih"], grid["it"])
            return np.asarray(o[3]), np.asarray(o[2])
        assert all(x.shape[0] == 1 for x in jax.tree_util.tree_leaves(head_pop)), \
            "chunked _gen_row assumes single-member heads"
        toks, nerrs = [], []
        for s in range(0, Q, CH):
            sl = slice(s, s + CH)
            o = gen(tile_dirs_over_Q(head_pop, min(CH, Q - s)), ts, grid["m"][sl], grid["b"][sl],
                    _sliced(grid["sim"], sl), rng_grid[sl], _sliced(grid["ih"], sl),
                    grid["it"][sl])
            toks.append(np.asarray(o[3]))
            nerrs.append(np.asarray(o[2]))
        return np.concatenate(toks, axis=0), np.concatenate(nerrs, axis=0)

    results, anchor_tok, gate_ok = {}, None, True
    for name, head_pop, params in rows:
        ts = P["train_state"].replace(params=params)
        gtok, g_nerr = _gen_row(head_pop, ts)                 # [Q, n_gen, 26], [Q]
        gtok = to_u16(flatten_tokens(gtok), name)
        assert gtok.shape == real_tok.shape, f"row '{name}': {gtok.shape} vs {real_tok.shape}"
        if name == "anchor":
            anchor_tok = gtok
        if name == "merge_noop":
            bitexact = bool(np.array_equal(gtok, anchor_tok))
            gate_ok = bitexact
            results[name] = dict(gate_bitexact=bitexact)
            print(f"[compounding_error] (G1) merge_noop tokens bit-exact vs anchor: {bitexact}",
                  flush=True)
            continue
        np.savez_compressed(os.path.join(args.out_dir, f"tokens_{name}.npz"), tokens=gtok)
        agree = float((gtok == real_tok).mean())              # coarse per-run sanity signal
        results[name] = dict(mean_num_errors=float(np.asarray(g_nerr, np.float32).mean()),
                             token_agreement_vs_real=agree)
        print(f"[compounding_error] row '{name}': token-agreement vs real {agree:.4f} "
              f"num_errors {results[name]['mean_num_errors']:.2f}", flush=True)

    # Context -> trading-day mapping (day-clustered bootstrap needs it). Same block as
    # test_eval: the idx draw is PRNGKey(seed)-deterministic over the staged day files.
    try:
        import re as _re
        ds_, idx_ = P["ds"], np.asarray(P["idx"], np.int64)
        fi = np.searchsorted(np.asarray(ds_._seqs_cumsum, np.int64), idx_, side="right") - 1

        def _day_of(p):
            m = _re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(str(p)))
            return m.group(0) if m else "unknown"
        _days = [_day_of(ds_.message_files[int(f)]) for f in fi]
        with open(os.path.join(args.out_dir, "ctx_days.json"), "w") as f:
            json.dump({"seed": args.seed, "n_eval_ctx": Q, "idx": idx_.tolist(), "day": _days}, f)
        print(f"[compounding_error] ctx_days.json: {len(set(_days))} day(s)", flush=True)
    except Exception as _e:
        print(f"[compounding_error] ctx->day mapping skipped: {_e}", flush=True)

    with open(os.path.join(args.out_dir, "compounding_error_gen.json"), "w") as f:
        json.dump(dict(stage="compounding_error_gen", n_cond=args.n_cond, n_gen=args.n_gen,
                       n_eval_ctx=Q, seed=args.seed, top_n=top_n,
                       wide_levels=args.wide_levels, ckpt_dir=args.ckpt_dir,
                       ckpt_step=args.ckpt_step, rows=results), f, indent=2)
    print(f"[compounding_error] done -> {args.out_dir} "
          f"(G1 merge_noop={'PASS' if gate_ok else 'FAIL'})", flush=True)
    return 0 if gate_ok else 3


def cpu_checks():
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
              flush=True)
        if not cond:
            fails.append(name)

    print("[compounding_error] CPU checks — alignment / dtype / phase alphabets", flush=True)
    # (C1) flatten_tokens is the message-boundary-preserving reshape: token t of the flat
    # stream is message t//26, phase t%26.
    a = np.arange(2 * 3 * 26).reshape(2, 3, 26)
    fl = flatten_tokens(a)
    chk("(C1) flatten_tokens reshape identity",
        fl.shape == (2, 78) and int(fl[1, 27]) == int(a[1, 1, 1]))
    # (C2) uint16 round-trip on the full vocab range; loud failure outside it.
    ok = bool((to_u16(np.array([[0, 2111]]), "t") == np.array([[0, 2111]])).all())
    try:
        to_u16(np.array([[-1]]), "t"); ok = False
    except ValueError:
        pass
    chk("(C2) to_u16 round-trip + loud range failure", ok)
    # (C3) the syntax matrix agrees with the phase-alphabet table the analysis hardcodes:
    # phase 0 = event_type ids {1004..1007}, phase 2 sign-only, NA (2) legal only at >=16.
    try:
        from lob import validation_helpers as valh
        M = np.asarray(valh.syntax_validation_matrix(block_start_tok=False))
        p0 = set(np.where(M[0])[0].tolist())
        chk("(C3) syntax matrix: phase0={1004..1007}, phase2 sign-only, NA only >=16",
            M.shape == (26, 2112) and p0 >= {1004, 1005, 1006, 1007}
            and set(np.where(M[2])[0].tolist()) <= {2108, 2109, 3}
            and not M[0, 2] and bool(M[16, 2]),
            f"|phase0|={len(p0)}")
    except Exception as e:
        print(f"   [SKIP] (C3) syntax matrix unavailable here: {e}", flush=True)
    print("[compounding_error] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"),
          flush=True)
    return fails


def main():
    ap = argparse.ArgumentParser(description="compounding-error token dump (generation only)")
    ap.add_argument("--run", action="store_true", help="run the GPU dump; else CPU checks only")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                                      "compounding_error_out"))
    ap.add_argument("--eggroll", action="append", default=None, metavar="NAME=CKPT_DIR",
                    help="proj-scope checkpoint row (repeatable)")
    ap.add_argument("--full_ckpt", action="append", default=None, metavar="NAME=CKPT_DIR:STEP",
                    help="full-checkpoint row (repeatable), own decoder head")
    ap.add_argument("--n_cond", type=int, default=500)
    ap.add_argument("--n_gen", type=int, default=500)
    ap.add_argument("--n_eval_ctx", type=int, default=256)
    ap.add_argument("--top_n", type=int, default=None)
    ap.add_argument("--rows", default=None, help="comma row-name filter; anchor always kept")
    ap.add_argument("--wide_book_dir", default=None)
    ap.add_argument("--wide_levels", type=int, default=50)
    ap.add_argument("--shard", choices=["auto", "on", "off"], default="off")
    ap.add_argument("--gen_chunk", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fails = cpu_checks()
    if fails:
        print(f"[compounding_error] CPU checks FAILED: {fails}"); sys.exit(1)
    if args.run:
        if not args.data_dir:
            print("[compounding_error] --run requires --data_dir"); sys.exit(2)
        sys.exit(run(args))
    print("\n[compounding_error] (skipped GPU run — pass --run on the GH200; CPU checks PASSED)")


if __name__ == "__main__":
    main()
