"""Directional-accuracy mid-price dump — downstream forecasting skill raw material.

WHY THIS EXISTS: the sealed evals score distributional realism (WS-21/L1) and the
compounding-error curves score per-position token divergence; neither asks the DOWNSTREAM
question "is the post-trained generator a better mid-price direction forecaster?". This
script produces the raw material for that: for X held-out contexts it saves,
index-for-index aligned,
  mids_real.npz    float32 [X, H+1]  TRUE mid-price path: col 0 = the context-boundary mid
                                     (book state prior to continuation message 0), col j =
                                     mid after j true continuation messages, straight from
                                     the raw book day files (rows seq_start+n_cond .. +H).
  mids_<row>.npz   float32 [X, H]    each model row's GENERATED mid path: col j = sim book
                                     mid after j generated messages, from generate()'s
                                     out[1] l2_book_states (level-1 ask/bid cols 0/2).
  mid0_sim.npy     int64   [X]       the sim's own boundary mid (_get_safe_mid_price on the
                                     replayed context states) — the self-consistent m0 for
                                     generated directions; analysis verifies it against
                                     mids_real[:, 0].
Directional accuracy at horizon h is then sign(gen[:, h-1] - m0) vs
sign(real[:, h] - real[:, 0]) — computed CPU-side in directional_accuracy_analysis.py.

HORIZON > n_gen DESIGN: deep wide-book init aligns ONLY at n_cond+n_gen=1000 windows
(500/500 — docs/reference/gotchas.md), so the dataset/window/init machinery is prepped at
the sealed 500/500 exactly as test_eval, and the generator scan is built with
n_msg_todo=horizon_max (its scan length is a static independent of the data windowing).
True mids at horizons past the window end are direct reads of the SAME day's book file
(book row r = state prior to message r; window stride = n_cond+n_gen, offsets 0), NaN-
padded where a window's tail crosses the end of its day file (analysis drops per-horizon).

CRN protocol is test_eval's: all rows in ONE program, one rng_grid, identical contexts;
the merge_noop canary's mid path must be BIT-identical to the anchor row's (exit 3
otherwise). Row mechanisms: --eggroll proj rows, --full_ckpt rows.
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

# JaxLOB empty-side sentinels are <= 0 (get_best_* returns -1) or huge init prices; any
# level-1 price outside (0, PRICE_SENTINEL) means "no real quote" -> mid is undefined.
PRICE_SENTINEL = 999_999_999


def mids_from_l2(l2):
    """generate()'s out[1] [Q, H, 4*L2_STATE_N] -> (mid float32 [Q, H], bad_frac).
    Per level layout [ask_p, ask_v, bid_p, bid_v]; level-1 ask/bid = cols 0/2. Empty or
    sentinel sides -> NaN (analysis treats NaN as no-forecast)."""
    a = np.asarray(l2[..., 0], np.float64)
    b = np.asarray(l2[..., 2], np.float64)
    bad = (a <= 0) | (b <= 0) | (a >= PRICE_SENTINEL) | (b >= PRICE_SENTINEL)
    mid = np.where(bad, np.nan, (a + b) / 2.0).astype(np.float32)
    return mid, float(bad.mean())


def select_rows(names, rows_filter=None, skip_anchor=False):
    """Row-name selection for the two split modes (pure, so cpu_checks can cover it —
    this logic sits behind the GPU path and a smoke run never reaches it).

    rows_filter: comma name filter (prefix match); the anchor is always kept, so the
      anchor+canary job uses ROWS=merge_noop to drop the payload-supplying pick.
    skip_anchor: per-row seed job — drops BOTH the anchor and the merge_noop canary (the
      canary's gate compares against the in-program anchor, so it cannot run here)."""
    out = list(names)
    if rows_filter:
        keep = {s.strip() for s in rows_filter.split(",") if s.strip()}
        out = [n for n in out if n == "anchor" or any(n == k or n.startswith(k) for k in keep)]
    if skip_anchor:
        out = [n for n in out if n not in ("anchor", "merge_noop")]
        assert out, "--skip_anchor left no rows"
    return out


def window_rows(seqs_cumsum, idx, window_len, n_cond):
    """Dataset window index -> (file_idx, boundary book row). Window k of file f spans
    message rows [k*window_len, (k+1)*window_len) (randomize_offset off => offset 0);
    book row r is the state PRIOR to message r, so the context-boundary state is row
    k*window_len + n_cond."""
    cs = np.asarray(seqs_cumsum, np.int64)
    ix = np.asarray(idx, np.int64)
    fi = np.searchsorted(cs, ix, side="right") - 1
    si = ix - cs[fi]
    return fi, si * int(window_len) + int(n_cond)


def true_mid_paths(book_files, load_fn, fi, r0, horizon):
    """float32 [X, horizon+1] true mid path from raw book day files; col j = mid at book
    row r0+j (= state after j true continuation messages). Raw layout: cols 3/5 = best
    ask/bid price (LOBSTER $x1e4). NaN past the end of the day file."""
    out = np.full((len(fi), horizon + 1), np.nan, np.float32)
    cache = {}
    for k in range(len(fi)):
        f = int(fi[k])
        if f not in cache:
            cache[f] = load_fn(book_files[f], mmap_mode="r")
        b = cache[f]
        lo = int(r0[k]); hi = min(lo + horizon + 1, int(b.shape[0]))
        assert lo < int(b.shape[0]), f"window {k}: boundary row {lo} outside file ({b.shape[0]})"
        rows = np.asarray(b[lo:hi, 3:6:2], np.float64)          # cols [3, 5]
        out[k, : hi - lo] = (rows[:, 0] + rows[:, 1]) / 2.0
    return out


def run(args):
    from ..tests.s3_es_rollout import _prep_real_batch
    from ..es.es_generator import make_generate_es_sharded, tile_dirs_over_Q, grid_rngs

    Q = args.n_eval_ctx
    H = args.horizon_max
    # Windowing/init at the sealed 500/500 (wide-book alignment); ONLY the generator scan
    # runs to H. The n_gen=500 real-continuation arrays in P are unused here.
    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, Q,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels, wide_book_dir=args.wide_book_dir)
    inf = P["inf"]
    kernel0, bias0 = P["kernel"], P["bias"]
    top_n = int(CFG.rollout.sample_top_n) if args.top_n is None else int(args.top_n)
    print(f"[directional_accuracy] Q={Q} n_cond={args.n_cond} n_gen={args.n_gen} "
          f"horizon_max={H} top_n={top_n} wide_levels={args.wide_levels} seed={args.seed} "
          f"data={args.data_dir}", flush=True)

    CH = int(args.gen_chunk or 0)
    if CH > 0:
        assert args.shard == "off", "--gen_chunk is a single-device path (use --shard off)"
        assert Q % CH == 0, f"--gen_chunk {CH} must divide n_eval_ctx {Q}"

    def _sliced(tree, sl):
        return jax.tree_util.tree_map(lambda x: x[sl], tree)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- TRUE side: mid paths straight from the staged raw book day files.
    ds = P["ds"]
    off = np.asarray(ds.seq_offsets, np.int64)
    assert (off == 0).all(), "randomize_offset must be off (nonzero seq_offsets)"
    window_len = args.n_cond + args.n_gen
    fi, r0 = window_rows(ds._seqs_cumsum, P["idx"], window_len, args.n_cond)
    try:
        from lob.lobster_dataloader import _np_load_zst as _load
    except ImportError:
        _load = np.load
    real_mids = true_mid_paths(ds.book_files, _load, fi, r0, H)
    n_tail_nan = int(np.isnan(real_mids[:, H]).sum())
    np.savez_compressed(os.path.join(args.out_dir, "mids_real.npz"), mids=real_mids)
    print(f"[directional_accuracy] true mid paths saved; windows lacking +{H} coverage "
          f"(day-file end): {n_tail_nan}/{Q}", flush=True)

    # Sim's own boundary mid (the m0 the generated book evolves from).
    mid0_sim = np.asarray(jax.vmap(
        lambda st: inf._get_safe_mid_price(P["sim_init"], st, P["tick_size"])
    )(P["sim_states_init"]))
    np.save(os.path.join(args.out_dir, "mid0_sim.npy"), mid0_sim)
    d0 = np.abs(mid0_sim.astype(np.float64) - real_mids[:, 0].astype(np.float64))
    print(f"[directional_accuracy] boundary check |mid0_sim - real0|: mean {np.nanmean(d0):.2f} "
          f"max {np.nanmax(d0):.0f} frac>{P['tick_size'] / 2}: "
          f"{float((d0 > P['tick_size'] / 2).mean()):.4f}", flush=True)

    # ---- Rows: (name, head_pop, params). One gen build + one rng draw = CRN across rows.
    anchor_head = {"kernel": kernel0[None], "bias": bias0[None]}
    rows = [("anchor", anchor_head, P["bb_params"])]
    first_payload = None
    for spec in args.eggroll or []:
        name, _, d = spec.partition("=")
        flat, bc = load_proj_payload(d)
        first_payload = first_payload or flat
        merged = apply_flat_subtree(P["bb_params"], {k: jnp.asarray(v) for k, v in flat.items()})
        rows.append((name, anchor_head, merged))
        print(f"[directional_accuracy] row '{name}': proj ckpt step {bc['step']} "
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
        print(f"[directional_accuracy] row '{name}': FULL ckpt step {st} <- {d}", flush=True)
    if first_payload is not None:
        anchor_resub = apply_flat_subtree(
            P["bb_params"], {k: _tree_get(P["bb_params"], k) for k in first_payload})
        rows.append(("merge_noop", anchor_head, anchor_resub))
    if args.rows or args.skip_anchor:
        keep = set(select_rows([r[0] for r in rows], args.rows, args.skip_anchor))
        rows = [r for r in rows if r[0] in keep]
        print(f"[directional_accuracy] row selection (rows={args.rows!r} "
              f"skip_anchor={args.skip_anchor}) -> {[r[0] for r in rows]}", flush=True)

    # Generator scan runs to H, independent of the 500/500 data windowing.
    gen = make_generate_es_sharded(P["model"], P["batchnorm"], P["encoder"], top_n,
                                   P["tick_size"], H, P["sim_init"],
                                   P["valid_mask_array"], conditional=True, shard=args.shard)
    grid = dict(m=P["m_seq_inp"], b=P["b_seq_inp"], sim=P["sim_states_init"],
                ih=P["init_hidden_batched"], it=P["init_time_batched"])
    rng_grid = grid_rngs(jax.random.PRNGKey(args.seed + 1), 1, Q)        # G=1; CRN across rows

    def _gen_row(head_pop, ts):
        """gen() over all Q rollouts, keeping ONLY the mid path from out[1] (l2 states)
        plus out[2] (num_errors). Same chunked head tiling as test_eval._gen_row; l2
        states are reduced to mids per chunk so the [Q, H, 4*N] block never accumulates
        host-side."""
        if CH <= 0 or Q <= CH:
            o = gen(tile_dirs_over_Q(head_pop, Q), ts, grid["m"], grid["b"], grid["sim"],
                    rng_grid, grid["ih"], grid["it"])
            mid, bf = mids_from_l2(np.asarray(o[1]))
            return mid, np.asarray(o[2]), bf
        assert all(x.shape[0] == 1 for x in jax.tree_util.tree_leaves(head_pop)), \
            "chunked _gen_row assumes single-member heads"
        mids, nerrs, bfs = [], [], []
        for s in range(0, Q, CH):
            sl = slice(s, s + CH)
            o = gen(tile_dirs_over_Q(head_pop, min(CH, Q - s)), ts, grid["m"][sl], grid["b"][sl],
                    _sliced(grid["sim"], sl), rng_grid[sl], _sliced(grid["ih"], sl),
                    grid["it"][sl])
            mid, bf = mids_from_l2(np.asarray(o[1]))
            mids.append(mid)
            nerrs.append(np.asarray(o[2]))
            bfs.append(bf)
        return np.concatenate(mids, axis=0), np.concatenate(nerrs, axis=0), float(np.mean(bfs))

    results, anchor_mid, gate_ok = {}, None, True
    for name, head_pop, params in rows:
        ts = P["train_state"].replace(params=params)
        gmid, g_nerr, bad_frac = _gen_row(head_pop, ts)          # [Q, H], [Q], scalar
        assert gmid.shape == (Q, H), f"row '{name}': mid shape {gmid.shape}"
        if name == "anchor":
            anchor_mid = gmid
        if name == "merge_noop":
            assert anchor_mid is not None, "merge_noop gate requires the in-program anchor"
            bitexact = bool(np.array_equal(gmid, anchor_mid, equal_nan=True))
            gate_ok = bitexact
            results[name] = dict(gate_bitexact=bitexact)
            print(f"[directional_accuracy] (G1) merge_noop mids bit-exact vs anchor: {bitexact}",
                  flush=True)
            continue
        np.savez_compressed(os.path.join(args.out_dir, f"mids_{name}.npz"),
                            mids=gmid, num_errors=np.asarray(g_nerr, np.float32))
        results[name] = dict(mean_num_errors=float(np.asarray(g_nerr, np.float32).mean()),
                             bad_l2_frac=bad_frac)
        print(f"[directional_accuracy] row '{name}': num_errors {results[name]['mean_num_errors']:.2f} "
              f"bad_l2_frac {bad_frac:.5f}", flush=True)

    # Context -> trading-day mapping (day-clustered bootstrap needs it); same block as
    # test_eval/compounding_error.
    try:
        import re as _re
        idx_ = np.asarray(P["idx"], np.int64)

        def _day_of(p):
            m = _re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(str(p)))
            return m.group(0) if m else "unknown"
        _days = [_day_of(ds.message_files[int(f)]) for f in fi]
        with open(os.path.join(args.out_dir, "ctx_days.json"), "w") as f:
            json.dump({"seed": args.seed, "n_eval_ctx": Q, "idx": idx_.tolist(), "day": _days}, f)
        print(f"[directional_accuracy] ctx_days.json: {len(set(_days))} day(s)", flush=True)
    except Exception as _e:
        print(f"[directional_accuracy] ctx->day mapping skipped: {_e}", flush=True)

    with open(os.path.join(args.out_dir, "directional_accuracy_gen.json"), "w") as f:
        json.dump(dict(stage="directional_accuracy_gen", n_cond=args.n_cond, n_gen=args.n_gen,
                       horizon_max=H, horizons=[int(h) for h in args.horizons.split(",")],
                       n_eval_ctx=Q, seed=args.seed, top_n=top_n,
                       tick_size=int(P["tick_size"]), wide_levels=args.wide_levels,
                       ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step,
                       n_tail_nan=n_tail_nan, rows=results), f, indent=2)
    print(f"[directional_accuracy] done -> {args.out_dir} "
          f"(G1 merge_noop={'PASS' if gate_ok else 'FAIL'})", flush=True)
    return 0 if gate_ok else 3


def cpu_checks():
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
              flush=True)
        if not cond:
            fails.append(name)

    print("[directional_accuracy] CPU checks — mid extraction / row mapping", flush=True)
    # (C1) mids_from_l2: level-1 cols, sentinel guard -> NaN.
    l2 = np.zeros((2, 3, 8), np.int64)
    l2[..., 0] = 1000100; l2[..., 2] = 1000000                    # ask/bid level 1
    l2[0, 1, 0] = -1                                              # empty ask side
    l2[1, 2, 2] = PRICE_SENTINEL                                  # sentinel bid
    m, bf = mids_from_l2(l2)
    chk("(C1) mids_from_l2 level-1 mid + sentinel NaN",
        m.shape == (2, 3) and abs(float(m[0, 0]) - 1000050.0) < 1e-3
        and np.isnan(m[0, 1]) and np.isnan(m[1, 2]) and abs(bf - 2 / 6) < 1e-9)
    # (C2) window_rows: file/window mapping + boundary-row arithmetic.
    cs = np.array([0, 5, 12])                                     # file0: 5 windows, file1: 7
    fi_, r0_ = window_rows(cs, [0, 4, 5, 11], 1000, 500)
    chk("(C2) window_rows mapping",
        fi_.tolist() == [0, 0, 1, 1] and r0_.tolist() == [500, 4500, 500, 6500])
    # (C3) true_mid_paths: direct-read mids, NaN tail past file end.
    b = np.zeros((7, 8), np.float64)
    b[:, 3] = 100 + np.arange(7); b[:, 5] = 98 + np.arange(7)     # mid = 99+j
    paths = true_mid_paths(["f0"], lambda p, mmap_mode=None: b,
                           np.array([0]), np.array([3]), 5)
    chk("(C3) true_mid_paths reads + NaN pad",
        paths.shape == (1, 6) and abs(float(paths[0, 0]) - 102.0) < 1e-9
        and abs(float(paths[0, 3]) - 105.0) < 1e-9 and np.isnan(paths[0, 4])
        and np.isnan(paths[0, 5]))
    # (C4) split-mode row selection. Regression: the auto-appended merge_noop row used to
    # trip a --skip_anchor assert, killing every seed job in the split program (jobs
    # 5918389-5918415, 2026-08-06); a smoke run never exercises this path.
    NAMES = ["anchor", "h_s3pick", "merge_noop"]
    ok_seed = select_rows(NAMES, None, True) == ["h_s3pick"]
    ok_canary = select_rows(NAMES, "merge_noop", False) == ["anchor", "merge_noop"]
    ok_plain = select_rows(NAMES) == NAMES
    empty_raises = False
    try:
        select_rows(["anchor", "merge_noop"], None, True)
    except AssertionError:
        empty_raises = True
    chk("(C4) split-mode row selection (seed / canary / plain / empty)",
        ok_seed and ok_canary and ok_plain and empty_raises,
        f"seed={select_rows(NAMES, None, True)} canary={select_rows(NAMES, 'merge_noop', False)}")
    print("[directional_accuracy] "
          + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


def main():
    ap = argparse.ArgumentParser(description="directional-accuracy mid dump (generation only)")
    ap.add_argument("--run", action="store_true", help="run the GPU dump; else CPU checks only")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                                      "directional_accuracy_out"))
    ap.add_argument("--eggroll", action="append", default=None, metavar="NAME=CKPT_DIR",
                    help="proj-scope checkpoint row (repeatable)")
    ap.add_argument("--full_ckpt", action="append", default=None, metavar="NAME=CKPT_DIR:STEP",
                    help="full-checkpoint row (repeatable), own decoder head")
    ap.add_argument("--n_cond", type=int, default=500)
    ap.add_argument("--n_gen", type=int, default=500,
                    help="data windowing only (wide-init alignment); generation runs to --horizon_max")
    ap.add_argument("--horizon_max", type=int, default=1000)
    ap.add_argument("--horizons", default="100,250,500,1000", help="recorded in meta for analysis")
    ap.add_argument("--n_eval_ctx", type=int, default=256)
    ap.add_argument("--top_n", type=int, default=None)
    ap.add_argument("--rows", default=None, help="comma row-name filter; anchor always kept")
    ap.add_argument("--skip_anchor", action="store_true",
                    help="per-row split mode: drop the anchor row (it runs in its own job)")
    ap.add_argument("--wide_book_dir", default=None)
    ap.add_argument("--wide_levels", type=int, default=50)
    ap.add_argument("--shard", choices=["auto", "on", "off"], default="off")
    ap.add_argument("--gen_chunk", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fails = cpu_checks()
    if fails:
        print(f"[directional_accuracy] CPU checks FAILED: {fails}"); sys.exit(1)
    if args.run:
        if not args.data_dir:
            print("[directional_accuracy] --run requires --data_dir"); sys.exit(2)
        assert args.horizon_max >= 1
        sys.exit(run(args))
    print("\n[directional_accuracy] (skipped GPU run — pass --run on the GH200; CPU checks PASSED)")


if __name__ == "__main__":
    main()
