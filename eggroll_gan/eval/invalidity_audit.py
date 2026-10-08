"""Measure the TRUE proportion of generated messages that cannot be matched on a LOB.

WHY THIS EXISTS — `num_errors` IS NOT INVALIDITY
  The generator's `num_errors` (inference_no_errcorr) is
      num_errors = (l2_book_states[1:] == l2_book_states[:-1]).all(axis=1).sum()
  i.e. the number of steps whose *top-`L2_STATE_N` snapshot* is byte-identical to the previous
  step. A perfectly VALID limit order / cancel / execution that only touches the book BELOW the
  visible levels leaves the top-N snapshot unchanged and is counted. For a deep name like GOOG even
  REAL replayed data scores a high "unchanged" rate (see archive/eggroll_gan/noop_audit.py). So that
  number conflates (a) genuinely un-matchable messages with (b) valid-but-deep activity. This module
  does NOT use it.

WHAT "CANNOT BE MATCHED" MEANS HERE (read off the JAX-LOB engine's ACTUAL effect, not a re-derived
rule — so the engine's own cancel-by-price init fallback, crossing logic, etc. are captured exactly):
  * new limit (type 1)   : always placeable (rests or crosses) -> VALID, unless malformed.
  * cancel/delete (2/3)  : apply it; `removed` = drop in total resting qty (only this message acts in
                           its step). removed == 0  -> HARD-INVALID (phantom: id not resting AND no
                           init order at that price). 0 < removed < size -> PARTIAL (over-cancel).
                           removed >= size -> VALID.
  * execution (type 4)   : engine treats it as an opposite-side crossing limit; `traded` = matched qty
                           this step (from the engine trades array). traded == 0 -> HARD-INVALID
                           (no counterparty). 0 < traded < size -> PARTIAL (partial fill). else VALID.
  * malformed            : size <= 0, price <= 0, or event-type not in {1,2,3,4} -> HARD-INVALID.

The headline "impossible on a LOB / cannot be matched" rate = HARD-INVALID = phantom + no-counterparty
+ malformed. Partials are reported as a SEPARATE category (the message partly applied), decomposed by
event type.

CORRECTNESS GATE: the IDENTICAL verdict scan is run on the REAL continuation
replayed through the SAME engine from the SAME init state. Real messages are valid by construction, so
real HARD-INVALID must be ~0. If it isn't, the predicate (or order-id matching) is wrong and the GEN
numbers are not to be trusted. A second gate reconstructs the gen L2 from our own replay and checks it
equals `generate_batched`'s returned L2 (so the verdicts are on the exact trajectory generation saw).

CPU SELF-TEST (login-safe, no model, no GPU): `python -m eggroll_gan.eval.invalidity_audit --self_test`
builds a tiny book and crafts messages hitting every verdict bucket.
GPU RUN: `... --run --data_dir <node-local GOOG> [--ckpt_dir ... --ckpt_step ...]`.
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache, partial

import jax
import jax.numpy as jnp

# Engine (read-only use; never edited — gymnax_exchange is a shared public checkout).
import gymnax_exchange.jaxob.JaxOrderBookArrays as job  # noqa: F401  (kept for parity/debug)
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState

# Verdict codes.
V_VALID, V_PART_CANCEL, V_PART_FILL, V_PHANTOM, V_NO_CPARTY, V_MALFORMED = 0, 1, 2, 3, 4, 5
V_LABEL = {V_VALID: "valid", V_PART_CANCEL: "partial_cancel", V_PART_FILL: "partial_fill",
           V_PHANTOM: "phantom_cancel", V_NO_CPARTY: "no_counterparty_exec", V_MALFORMED: "malformed"}
HARD = (V_PHANTOM, V_NO_CPARTY, V_MALFORMED)      # "cannot be matched"
PARTIAL = (V_PART_CANCEL, V_PART_FILL)            # "partly applied"

# Engine message column layout (post msgs_to_jnp): [type, side(+/-1), qty, price, oid, tid, ts, tns].
M_TYPE, M_SIDE, M_QTY, M_PRICE, M_OID = 0, 1, 2, 3, 4
TRADE_PRICE_C, TRADE_QTY_C = 0, 1                 # engine trade row: col0 price, col1 signed matched qty


def _resting_qty(asks, bids):
    """Total displayed resting quantity across both sides (empty slots carry qty = -1)."""
    return jnp.sum(jnp.maximum(asks[:, 1], 0)) + jnp.sum(jnp.maximum(bids[:, 1], 0))


def _verdict_scan(sim, state0, eng_msgs, n_levels):
    """Per-message verdict for ONE rollout, read off the engine's real effect.

    sim       : OrderBook
    state0    : LobState the rollout starts from (post-conditioning, per-context)
    eng_msgs  : [T, 8] engine messages (output of inference.msgs_to_jnp)
    returns   : (verdicts [T] int32, l2 [T, 4*n_levels])  -- l2 for the trajectory-match gate.
    """
    def body(state, m):
        t, side, qty, price = m[M_TYPE], m[M_SIDE], m[M_QTY], m[M_PRICE]
        before = _resting_qty(state.asks, state.bids)
        # reset the trades buffer so post-state trades hold ONLY this step's fills
        st_in = LobState(state.asks, state.bids, jnp.full_like(state.trades, -1), state.key)
        st2 = sim.process_order_array(st_in, m)
        after = _resting_qty(st2.asks, st2.bids)
        tr = st2.trades
        traded = jnp.sum(jnp.where(tr[:, TRADE_PRICE_C] != -1, jnp.abs(tr[:, TRADE_QTY_C]), 0))
        removed = jnp.maximum(before - after, 0)

        malformed = (qty <= 0) | (price <= 0) | (t < 1) | (t > 4)
        is_cancel = (t == 2) | (t == 3)
        is_exec = (t == 4)
        v_cancel = jnp.where(removed <= 0, V_PHANTOM,
                             jnp.where(removed < qty, V_PART_CANCEL, V_VALID))
        v_exec = jnp.where(traded <= 0, V_NO_CPARTY,
                           jnp.where(traded < qty, V_PART_FILL, V_VALID))
        v = jnp.where(malformed, V_MALFORMED,
                      jnp.where(is_cancel, v_cancel,
                                jnp.where(is_exec, v_exec, V_VALID)))
        l2 = sim.get_L2_state(st2, n_levels)
        return st2, (v.astype(jnp.int32), l2)

    _, (verdicts, l2) = jax.lax.scan(body, state0, eng_msgs)
    return verdicts, l2


@lru_cache(maxsize=8)
def _batch_verdict_fn(n_levels):
    """Cached jitted vmap (keyed by n_levels) so repeated per-eval calls during post-training reuse the
    same compiled executable instead of re-tracing each time. `sim` is broadcast (in_axes=None)."""
    return jax.jit(jax.vmap(lambda sim, s0, em: _verdict_scan(sim, s0, em, n_levels),
                            in_axes=(None, 0, 0)))


def batch_verdicts(sim, states0, eng_msgs, n_levels):
    """vmap `_verdict_scan` over a batch of rollouts. eng_msgs: [N, T, 8] -> (verdicts [N, T], l2)."""
    return _batch_verdict_fn(int(n_levels))(sim, states0, eng_msgs)


# ----------------------------------------------------------------------------------------
# Summary / reporting
# ----------------------------------------------------------------------------------------
def summarize(verdicts, eng_msgs):
    """verdicts [N, T], eng_msgs [N, T, 8] -> nested dict of proportions (overall + by event type)."""
    import numpy as np
    v = np.asarray(verdicts).reshape(-1)
    t = np.asarray(eng_msgs[..., M_TYPE]).reshape(-1).astype(int)
    n = max(v.size, 1)

    def frac(mask):
        return float(np.mean(mask)) if mask.size else 0.0

    out = {
        "n_messages": int(v.size),
        "frac_valid": frac(v == V_VALID),
        "frac_partial": frac(np.isin(v, PARTIAL)),
        "frac_hard_invalid": frac(np.isin(v, HARD)),       # the headline "cannot be matched" rate
        "by_code": {V_LABEL[c]: frac(v == c) for c in V_LABEL},
        "by_type": {},
    }
    for tt, name in [(1, "new_limit"), (2, "cancel"), (3, "delete"), (4, "exec")]:
        tm = (t == tt)
        share = frac(tm)
        sub = v[tm]
        out["by_type"][name] = {
            "share_of_msgs": share,
            "n": int(tm.sum()),
            "frac_valid": float(np.mean(sub == V_VALID)) if sub.size else 0.0,
            "frac_partial": float(np.mean(np.isin(sub, PARTIAL))) if sub.size else 0.0,
            "frac_hard_invalid": float(np.mean(np.isin(sub, HARD))) if sub.size else 0.0,
            "codes": {V_LABEL[c]: (float(np.mean(sub == c)) if sub.size else 0.0) for c in V_LABEL},
        }
    return out


def rollout_invalidity_stats(sim, states0, msgs_decoded, inf_mod, n_levels=None):
    """Engine-truthful per-message LOB-validity summary of a generated rollout, for LIVE logging during
    post-training. `sim`: OrderBook; `states0`: per-context LobState the rollout started from; `msgs_decoded`:
    [N, T, F] decoded messages (the rollout's `g_out[0]`); `inf_mod`: the inference module (provides
    `msgs_to_jnp` + `l2_state_n`). Returns the same dict as `summarize`. Cheap at eval cadence (N is the
    small n_eval_ctx); never use `num_errors` (a no-op rate) for this — see module docstring."""
    nlev = inf_mod.l2_state_n if n_levels is None else n_levels
    eng = jax.vmap(inf_mod.msgs_to_jnp)(jnp.asarray(msgs_decoded))
    verdicts, _l2 = batch_verdicts(sim, states0, eng, nlev)
    return summarize(verdicts, eng)


def invalidity_line(stats, tag="invalidity"):
    """One-line human summary of `rollout_invalidity_stats`/`summarize` output, for training logs."""
    bt = stats["by_type"]
    return (f"[{tag}] valid {stats['frac_valid']:.2%} | partial {stats['frac_partial']:.2%} "
            f"| HARD-INVALID {stats['frac_hard_invalid']:.2%} "
            f"(phantom {stats['by_code']['phantom_cancel']:.2%}, "
            f"no-cparty {stats['by_code']['no_counterparty_exec']:.2%}; "
            f"cancel {bt['cancel']['frac_hard_invalid']:.1%} / exec {bt['exec']['frac_hard_invalid']:.1%} hard)")


def _print_report(tag, s):
    print(f"\n===== {tag} =====")
    print(f"  messages                 : {s['n_messages']}")
    print(f"  VALID                    : {s['frac_valid']:7.3%}")
    print(f"  PARTIAL (partly applied) : {s['frac_partial']:7.3%}   "
          f"[partial_cancel {s['by_code']['partial_cancel']:.3%} | partial_fill {s['by_code']['partial_fill']:.3%}]")
    print(f"  HARD-INVALID (unmatchable): {s['frac_hard_invalid']:7.3%}   "
          f"[phantom {s['by_code']['phantom_cancel']:.3%} | no_counterparty {s['by_code']['no_counterparty_exec']:.3%} "
          f"| malformed {s['by_code']['malformed']:.3%}]")
    print("  by event type (share | valid | partial | hard-invalid):")
    for name, d in s["by_type"].items():
        print(f"    {name:10s} {d['share_of_msgs']:6.2%} | {d['frac_valid']:6.2%} | "
              f"{d['frac_partial']:6.2%} | {d['frac_hard_invalid']:6.2%}")


def _unchanged_rate(l2):
    """Tier-1 cross-check: fraction of steps whose FULL L2 snapshot equals the previous step
    (== num_errors at this depth). l2: [N, T, W]."""
    unchanged = (l2[:, 1:] == l2[:, :-1]).all(axis=-1)
    return float(jnp.mean(unchanged.astype(jnp.float32)))


# ----------------------------------------------------------------------------------------
# CPU self-test (login-safe): craft messages hitting every verdict bucket.
# ----------------------------------------------------------------------------------------
def _eng_msg(t, side, qty, price, oid):
    return jnp.array([t, side, qty, price, oid, oid, 34000, 0], dtype=jnp.int32)


def _cpu_self_test():
    fails = []

    def chk(name, got, exp):
        ok = (got == exp)
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}: got={V_LABEL.get(int(got), got)} exp={V_LABEL[exp]}",
              flush=True)
        if not ok:
            fails.append(name)

    sim = OrderBook()
    # A simple two-sided book: bids at 99_00, asks at 101_00 (abs ticks), modest depth.
    l2init = jnp.array([1010000, 200, 990000, 200, 1020000, 100, 980000, 100], dtype=jnp.int32)
    state0 = sim.reset(l2init)
    n_levels = 2

    # Sequence designed so each message's verdict is known a priori.
    P_BID, P_ASK = 990000, 1010000
    msgs = jnp.stack([
        _eng_msg(1, 1, 100, P_BID, 5001),     # 0 new bid limit (adds 100 @ P_BID)            -> valid
        _eng_msg(2, 1, 40, P_BID, 5001),      # 1 cancel 40 of order 5001 (100 resting)        -> valid
        _eng_msg(2, 1, 1000, P_BID, 5001),    # 2 cancel 1000, only 60 remain                  -> partial_cancel
        _eng_msg(2, 1, 50, 990100, 99999),    # 3 cancel non-existent id, no init @ that price -> phantom
        _eng_msg(4, -1, 50, P_ASK, 7001),     # 4 buy exec crossing asks (>=200 resting)        -> valid (full fill)
        _eng_msg(4, -1, 100000, P_ASK, 7002), # 5 buy exec far exceeding remaining ask depth     -> partial_fill
        _eng_msg(4, -1, 50, 900000, 7003),    # 6 buy exec priced below ask -> no cross          -> no_counterparty
        _eng_msg(1, 1, 0, P_BID, 5002),       # 7 malformed (qty 0)                              -> malformed
    ])
    verdicts, _ = _verdict_scan(sim, state0, msgs, n_levels)
    verdicts = [int(x) for x in verdicts]
    print(f"[invalidity_audit] CPU self-test verdict codes = {[V_LABEL[c] for c in verdicts]}", flush=True)

    chk("(0) new limit",        verdicts[0], V_VALID)
    chk("(1) cancel within qty", verdicts[1], V_VALID)
    chk("(2) over-cancel",      verdicts[2], V_PART_CANCEL)
    chk("(3) phantom cancel",   verdicts[3], V_PHANTOM)
    chk("(4) exec full fill",   verdicts[4], V_VALID)
    chk("(5) exec partial fill", verdicts[5], V_PART_FILL)
    chk("(6) exec no counterparty", verdicts[6], V_NO_CPARTY)
    chk("(7) malformed",        verdicts[7], V_MALFORMED)

    # (B) cached batch path (used live during post-training) must reproduce the single-rollout verdicts.
    states2 = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), state0)
    bverd, _ = batch_verdicts(sim, states2, jnp.stack([msgs, msgs]), n_levels)
    ok_b = tuple(bverd.shape) == (2, int(msgs.shape[0])) and [int(x) for x in bverd[0]] == verdicts
    print(f"   [{'PASS' if ok_b else 'FAIL'}] (B) batch_verdicts matches single-rollout", flush=True)
    if not ok_b:
        fails.append("(B) batch_verdicts")

    print("\n[invalidity_audit] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


# ----------------------------------------------------------------------------------------
# GPU run: generate from the pretrained anchor, verdict gen + real baseline, print precise stats.
# ----------------------------------------------------------------------------------------
def _run(args):
    from ..tests.s3_es_rollout import _prep_real_batch

    import lob.inference_no_errcorr as inf
    l2n = inf.l2_state_n
    print(f"[invalidity_audit] host={os.uname().nodename} L2_STATE_N={l2n} "
          f"n_pop={args.n_pop} n_cond={args.n_cond} n_gen={args.n_gen}", flush=True)

    from ..config import DEFAULT as CFG
    ckpt_dir = args.ckpt_dir or CFG.paths.ckpt_dir
    ckpt_step = args.ckpt_step or CFG.paths.ckpt_step

    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, args.n_pop,
                         ckpt_dir=ckpt_dir, ckpt_step=ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels, wide_book_dir=args.wide_book_dir)
    sim = P["sim_init"]

    # --- generate (stock, unperturbed pretrained generator) ---
    out = inf.generate_batched(
        P["sim_init"], P["train_state"], P["model"], P["batchnorm"], P["encoder"],
        P["sample_top_n"], P["tick_size"], P["m_seq_inp"], P["b_seq_inp"], P["n_gen"],
        P["sim_states_init"], P["rngs"], P["init_hidden_batched"], True,
        P["init_time_batched"], False, None, P["valid_mask_array"])
    msgs_decoded, gen_l2_gb, num_errors, _tok, _bf = out

    # --- verdicts on GENERATED messages ---
    gen_eng = jax.vmap(inf.msgs_to_jnp)(msgs_decoded)                 # [N, T, 8]
    gen_verdicts, gen_l2_mine = batch_verdicts(sim, P["sim_states_init"], gen_eng, l2n)

    # GATE A: our replay reproduces generation's own L2 trajectory (verdicts are on the true path).
    traj_ok = bool(jnp.array_equal(jnp.asarray(gen_l2_mine), jnp.asarray(gen_l2_gb)))
    print(f"[gate A] our-replay L2 == generate_batched L2 : {traj_ok}", flush=True)

    # --- verdicts on REAL continuation (the 0-baseline correctness gate) ---
    real_eng = jax.vmap(inf.msgs_to_jnp)(P["m_seq_raw_cont"])        # [N, T, 8]
    real_verdicts, real_l2_mine = batch_verdicts(sim, P["sim_states_init"], real_eng, l2n)

    gen_s = summarize(gen_verdicts, gen_eng)
    real_s = summarize(real_verdicts, real_eng)

    # Tier-1 full-depth no-op rates (== num_errors at this depth).
    gen_noop = _unchanged_rate(jnp.asarray(gen_l2_gb))
    real_noop = _unchanged_rate(real_l2_mine)

    print("\n" + "=" * 78)
    print(f"[invalidity_audit] anchor={os.path.basename(str(ckpt_dir))} step={ckpt_step} "
          f"L2_STATE_N={l2n} contexts={args.n_pop} gen/ctx={args.n_gen}")
    print(f"  raw num_errors (top-{l2n} no-op) mean = {float(jnp.mean(num_errors)):.1f}/{args.n_gen} "
          f"= gen no-op {gen_noop:.3%}  vs  REAL no-op {real_noop:.3%}  (Tier-1; NOT invalidity)")
    print("=" * 78)
    _print_report("GENERATED (pretrained anchor)", gen_s)
    _print_report("REAL continuation [correctness gate: hard-invalid must be ~0]", real_s)

    gate_b = real_s["frac_hard_invalid"] < 0.01
    print(f"\n[gate B] REAL hard-invalid {real_s['frac_hard_invalid']:.3%} < 1% : {gate_b}")
    print(f"[gate A] trajectory match : {traj_ok}")
    print(f"\n>>> HEADLINE: {gen_s['frac_hard_invalid']:.3%} of generated messages CANNOT be matched "
          f"(phantom {gen_s['by_code']['phantom_cancel']:.3%} + no-counterparty {gen_s['by_code']['no_counterparty_exec']:.3%} "
          f"+ malformed {gen_s['by_code']['malformed']:.3%}); a further {gen_s['frac_partial']:.3%} partly apply.")
    if not (gate_b and traj_ok):
        print("\n[WARN] a correctness gate FAILED — do NOT trust the generated numbers until resolved.")

    import json
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"gen": gen_s, "real": real_s, "gen_noop": gen_noop, "real_noop": real_noop,
                       "gate_traj": traj_ok, "gate_real": gate_b,
                       "anchor": str(ckpt_dir), "step": ckpt_step, "L2_STATE_N": l2n}, f, indent=2)
        print(f"[invalidity_audit] wrote {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="True LOB invalidity rate of generated messages")
    ap.add_argument("--self_test", action="store_true", help="CPU verdict self-test (login-safe)")
    ap.add_argument("--run", action="store_true", help="GPU generation + audit (GH200)")
    ap.add_argument("--data_dir", default=None, help="node-local GOOG dir (SquashFS-staged)")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--ckpt_step", type=int, default=None)
    ap.add_argument("--n_pop", type=int, default=64, help="number of held-out contexts")
    ap.add_argument("--n_cond", type=int, default=500)
    ap.add_argument("--n_gen", type=int, default=500)
    ap.add_argument("--wide_levels", type=int, default=500, help="deep-book init depth")
    ap.add_argument("--wide_book_dir", default=None,
                    help="clean L500 deep-init snapshots (data/wide_L500_2026-02/GOOG). REQUIRED for a "
                         "faithful audit: a shallow init re-introduces the top-of-book-invisibility confound.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    if args.self_test:
        import sys
        sys.exit(1 if _cpu_self_test() else 0)
    if args.run:
        if not args.data_dir:
            raise SystemExit("--run requires --data_dir")
        if not args.wide_book_dir:
            raise SystemExit("--run requires --wide_book_dir (deep L500 init) — a shallow init would "
                             "spuriously inflate phantom-cancel/no-counterparty rates. Pass it explicitly.")
        _run(args)
    else:
        raise SystemExit("pass --self_test or --run")


if __name__ == "__main__":
    main()
