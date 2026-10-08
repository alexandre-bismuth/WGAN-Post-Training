"""Held-out test-set distributional eval panel — task #10 (the thesis-prelim arbiter).

WHY THIS EXISTS: every comparative number so far (W3, σ sweep) is lane-relative — composites
normalized to each lane's own first eval, scored on different context draws, single seed. This
script produces the confirmatory table: ALL rows (anchor / tuned-decoding bar / EGGROLL-proj
seeds / drift control) roll out on IDENTICAL held-out contexts (held-out MONTHS staged in
--data_dir) with common random numbers, scored with a DISTRIBUTIONAL panel (not the thin comp3:
its ret term is dead pathwise and event_l1 dominates), normalized by the ANCHOR row, with
bootstrap CIs over contexts.

Panel (gen vs real, pooled over contexts; lower = closer except noted):
  ret_w1_{1,10,100}   W1 between log-return distributions at 1/10/100-msg aggregation (bps)
  ret_ks_1            two-sample KS at the 1-msg scale
  absret_acf_l1       L1 between mean ACF curves of |log ret| (lags 1..20) — vol clustering
  spread_w1           W1 between best-ask−best-bid spread distributions (raw price units)
  touch_vol_w1_log    W1 between log1p(volume at touch) distributions (both sides pooled)
  depth_profile_l1    L1 between normalized mean-volume-per-level profiles (all levels, both sides)
  interarrival_w1_log W1 between log10(inter-event time) distributions (from decoded DTs/DTns)
  event_l1            normalized event-type histogram L1 (the legacy driver metric)
  unchanged_rate      fraction of gen messages leaving the top-`n_levels` book unchanged
                      (depth-50 staging => the deep inert-message proxy; real rate ~2% at 50)
Legacy eval_monitor.stylized_fact_metrics are also recorded per row for continuity with the
training histories.

GPU gates built in: (G1) merge_noop row — the proj-merge path fed with ANCHOR leaves must
reproduce the anchor row BIT-exactly (same program => bitwise semantics exist, TF32 caveat does
not apply); (G2) every EGGROLL payload key must exist in the anchor tree with matching shape
(apply_flat_subtree fails loudly).

CPU (login, no model): self-checks T1..T6 on the pure helpers. GPU: --run writes
<out_dir>/test_eval_results.json + test_eval_table.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

from ..config import DEFAULT as CFG
from . import eval_monitor as EM
from ..baselines.temp_topk import scaled_heads


# ----------------------------------------------------------------------------------------
# Pure metric helpers (CPU-testable, host-side numpy).
# ----------------------------------------------------------------------------------------
def w1(a, b, n_q=2048):
    """Quantile-interpolated Wasserstein-1 between two empirical samples (finite entries only)."""
    a = np.asarray(a, np.float64).ravel(); a = a[np.isfinite(a)]
    b = np.asarray(b, np.float64).ravel(); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    q = (np.arange(n_q) + 0.5) / n_q
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def ks(a, b):
    """Two-sample Kolmogorov–Smirnov statistic (finite entries only)."""
    a = np.sort(np.asarray(a, np.float64).ravel()); a = a[np.isfinite(a)]
    b = np.sort(np.asarray(b, np.float64).ravel()); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def acf_1d(x, lags):
    """Autocorrelation of one series at lags 1..lags (NaN-safe; demeaned)."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size < lags + 2:
        return np.full(lags, np.nan)
    x = x - x.mean()
    v = float(np.dot(x, x))
    if v <= 0:
        return np.full(lags, np.nan)
    return np.array([np.dot(x[:-k], x[k:]) / v for k in range(1, lags + 1)])


def mean_acf(series_2d, lags):
    """[n_series, T] -> mean ACF curve over series (per-series ACF, NaN-mean across series)."""
    curves = np.stack([acf_1d(s, lags) for s in np.asarray(series_2d)])
    return np.nanmean(curves, axis=0)


def log_returns(l2, agg=1):
    """[Q, T, W] l2 states -> [Q, n] log mid returns at `agg`-message aggregation (bps).
    Non-positive mids are masked to NaN (degenerate/empty-book guard)."""
    l2 = np.asarray(l2, np.float64)
    mid = (l2[..., 0] + l2[..., 2]) / 2.0
    mid = np.where(mid > 0, mid, np.nan)
    lm = np.log(mid[:, ::agg])
    return 1e4 * (lm[:, 1:] - lm[:, :-1])


def spreads(l2):
    """[Q, T, W] -> [Q, T] |best ask - best bid| in raw price units (non-positive prices -> NaN)."""
    l2 = np.asarray(l2, np.float64)
    pa, pb = l2[..., 0], l2[..., 2]
    s = np.abs(pa - pb)
    return np.where((pa > 0) & (pb > 0), s, np.nan)


def touch_vols_log(l2):
    """[Q, T, W] -> [Q, 2T] log1p volumes at the touch (best ask + best bid, clipped at 0)."""
    l2 = np.asarray(l2, np.float64)
    v = np.concatenate([np.clip(l2[..., 1], 0, None), np.clip(l2[..., 3], 0, None)], axis=-1)
    return np.log1p(v)


def depth_profile(l2):
    """[Q, T, W] -> [2 * n_levels] normalized mean-volume profile (ask levels then bid levels)."""
    l2 = np.asarray(l2, np.float64)
    va = np.clip(l2[..., 1::4], 0, None).mean(axis=(0, 1))     # [n_levels]
    vb = np.clip(l2[..., 3::4], 0, None).mean(axis=(0, 1))
    prof = np.concatenate([va, vb])
    tot = prof.sum()
    return prof / tot if tot > 0 else np.full_like(prof, np.nan)


def event_hist_l1(et_a, et_b):
    """Normalized event-type histogram L1 over types {1,2,3,4} (same formula as eval_monitor)."""
    def h(et):
        et = np.asarray(et).ravel()
        c = np.array([(et == t).sum() for t in (1, 2, 3, 4)], np.float64)
        return c / max(c.sum(), 1.0)
    return float(np.abs(h(et_a) - h(et_b)).sum())


def unchanged_rate(l2):
    """[Q, T, W] -> fraction of consecutive snapshot pairs with the FULL window unchanged."""
    l2 = np.asarray(l2)
    eq = (l2[:, 1:] == l2[:, :-1]).all(axis=-1)
    return float(eq.mean())


def interarrival_log(dts, dtns):
    """Decoded per-message delta times -> log10 seconds (non-finite / non-positive masked)."""
    dt = np.asarray(dts, np.float64) + 1e-9 * np.asarray(dtns, np.float64)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, np.nan)
    return np.log10(dt)


def panel(gen_l2, gen_et, real_l2, real_et, gen_dt_log=None, real_dt_log=None,
          aggs=(1, 10, 100), acf_lags=20):
    """The full distributional panel for one row. All inputs [Q, ...] with the SAME Q."""
    out = {}
    for a in aggs:
        out[f"ret_w1_{a}"] = w1(log_returns(gen_l2, a), log_returns(real_l2, a))
    out["ret_ks_1"] = ks(log_returns(gen_l2, 1), log_returns(real_l2, 1))
    ga = mean_acf(np.abs(log_returns(gen_l2, 1)), acf_lags)
    ra = mean_acf(np.abs(log_returns(real_l2, 1)), acf_lags)
    out["absret_acf_l1"] = float(np.nansum(np.abs(ga - ra)))
    out["spread_w1"] = w1(spreads(gen_l2), spreads(real_l2))
    out["touch_vol_w1_log"] = w1(touch_vols_log(gen_l2), touch_vols_log(real_l2))
    dpg, dpr = depth_profile(gen_l2), depth_profile(real_l2)
    out["depth_profile_l1"] = float(np.nansum(np.abs(dpg - dpr)))
    out["event_l1"] = event_hist_l1(gen_et, real_et)
    out["unchanged_rate"] = unchanged_rate(gen_l2)
    out["real_unchanged_rate"] = unchanged_rate(real_l2)
    if gen_dt_log is not None and real_dt_log is not None:
        gd, rd = np.asarray(gen_dt_log), np.asarray(real_dt_log)
        degenerate = (np.isfinite(gd).sum() < 100) or (np.isfinite(rd).sum() < 100)
        out["interarrival_w1_log"] = float("nan") if degenerate else w1(gd, rd)
    return out


# distance metrics only (ratios vs anchor are meaningful; rates are reported raw)
PANEL_DIST_KEYS = ("ret_w1_1", "ret_w1_10", "ret_w1_100", "ret_ks_1", "absret_acf_l1",
                   "spread_w1", "touch_vol_w1_log", "depth_profile_l1", "event_l1",
                   "interarrival_w1_log")


def bootstrap_ci(metric_fn, n_ctx, n_boot=500, seed=0, alpha=0.05):
    """Percentile CI of every scalar `metric_fn(idx)` returns, over context resamples idx."""
    rng = np.random.default_rng(seed)
    draws = {}
    for _ in range(n_boot):
        m = metric_fn(rng.integers(0, n_ctx, size=n_ctx))
        for k, v in m.items():
            draws.setdefault(k, []).append(v)
    return {k: (float(np.nanquantile(v, alpha / 2)), float(np.nanquantile(v, 1 - alpha / 2)))
            for k, v in draws.items()}


def apply_flat_subtree(params, flat):
    """Overwrite leaves of a nested-dict param tree from a flat {'a/b/c': leaf} payload.
    Copy-on-path (the input tree is never mutated); missing key or shape mismatch fails loudly
    (wrong checkpoint for this anchor/scope). This is merge_trainable without hs/es_map."""
    def set_path(node, parts, leaf):
        node = dict(node)
        k = parts[0]
        if k not in node:
            raise KeyError(f"payload key segment '{k}' not in param tree (have {list(node)[:8]}...)")
        if len(parts) == 1:
            if tuple(np.shape(node[k])) != tuple(np.shape(leaf)):
                raise ValueError(f"shape mismatch at '{k}': tree {np.shape(node[k])} vs payload {np.shape(leaf)}")
            node[k] = leaf
        else:
            node[k] = set_path(node[k], parts[1:], leaf)
        return node

    out = params
    for key, leaf in flat.items():
        out = set_path(out, key.split("/"), leaf)
    return out


def load_proj_payload(ckpt_dir):
    """Breadcrumb-only proj-checkpoint load (no ls — Lustre rule). Returns (flat payload, bc)."""
    from flax import serialization
    with open(os.path.join(ckpt_dir, "latest_checkpoint.json")) as f:
        bc = json.load(f)
    gen_file = bc.get("generator_proj")
    if not gen_file:
        raise ValueError(f"{ckpt_dir} is not a proj-scope checkpoint (no generator_proj in breadcrumb)")
    with open(os.path.join(ckpt_dir, gen_file), "rb") as f:
        flat = serialization.msgpack_restore(f.read())
    return flat, bc


# ----------------------------------------------------------------------------------------
# GPU run.
# ----------------------------------------------------------------------------------------
def run(args):
    from ..tests.s3_es_rollout import _prep_real_batch
    from ..training.train_eggroll_gan_s5 import _replay_real
    from ..es.es_generator import make_generate_es_sharded, tile_dirs_over_Q, grid_rngs

    Q = args.n_eval_ctx
    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, Q,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels, wide_book_dir=args.wide_book_dir)
    inf = P["inf"]
    ETi, DTsi, DTnsi = int(inf.EVENT_TYPE_i), int(inf.DTs_i), int(inf.DTns_i)
    kernel0, bias0 = P["kernel"], P["bias"]
    top_n = int(CFG.rollout.sample_top_n) if args.top_n is None else int(args.top_n)
    print(f"[test_eval] Q={Q} n_cond={args.n_cond} n_gen={args.n_gen} top_n={top_n} "
          f"wide_levels={args.wide_levels} wide_book_dir={args.wide_book_dir} "
          f"seed={args.seed} data={args.data_dir}", flush=True)

    CH = int(args.gen_chunk or 0)
    if CH > 0:
        assert args.shard == "off", "--gen_chunk is a single-device path (use --shard off)"
        assert Q % CH == 0, f"--gen_chunk {CH} must divide n_eval_ctx {Q} (one XLA program per slice)"

    def _sliced(tree, sl):
        return jax.tree_util.tree_map(lambda x: x[sl], tree)

    # The shared real side (identical contexts for every row).
    if CH > 0 and Q > CH:
        real_l2 = np.concatenate(
            [np.asarray(_replay_real(inf, P["sim_init"],
                                     _sliced(P["sim_states_init"], slice(s, s + CH)),
                                     P["m_seq_raw_cont"][s:s + CH]))
             for s in range(0, Q, CH)], axis=0)
    else:
        real_l2 = np.asarray(_replay_real(inf, P["sim_init"], P["sim_states_init"], P["m_seq_raw_cont"]))
    real_et = np.asarray(P["m_seq_raw_cont"][..., ETi]).astype(np.int32)
    real_dt = interarrival_log(P["m_seq_raw_cont"][..., DTsi], P["m_seq_raw_cont"][..., DTnsi])
    if args.save_rollouts:
        os.makedirs(args.out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(args.out_dir, "rollout_real.npz"),
                            msgs=np.asarray(P["m_seq_raw_cont"], np.float32),
                            l2=real_l2.astype(np.float32))

    # Rows: (name, head_pop, params). One gen build + one rng draw = CRN across all rows.
    anchor_head = {"kernel": kernel0[None], "bias": bias0[None]}
    rows = [("anchor", anchor_head, P["bb_params"])]
    rows.append((f"bar_t{args.bar_temp:g}_n{top_n}",
                 jax.tree_util.tree_map(lambda x: x, scaled_heads(kernel0, bias0, [args.bar_temp])),
                 P["bb_params"]))
    first_payload = None
    for spec in args.eggroll or []:
        name, _, d = spec.partition("=")
        flat, bc = load_proj_payload(d)
        first_payload = first_payload or flat
        merged = apply_flat_subtree(P["bb_params"], {k: jnp.asarray(v) for k, v in flat.items()})
        rows.append((name, anchor_head, merged))
        print(f"[test_eval] row '{name}': proj ckpt step {bc['step']} ({len(flat)} leaves, "
              f"meta sigma={bc.get('sigma')} lr={bc.get('lr')} fitness_control="
              f"{bc.get('fitness_control', 'none')}) <- {d}", flush=True)
    for spec in args.full_ckpt or []:
        # Full-checkpoint row (e.g. the continued-pretraining control): a complete
        # generator checkpoint rolled out in the SAME program as the anchor, so the
        # comparison is CRN-paired with zero cross-program jitter. Unlike --eggroll rows
        # it carries its own decoder head (full training moves the head too).
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
        print(f"[test_eval] row '{name}': FULL ckpt step {st} <- {d}", flush=True)
    if first_payload is not None:
        # (G1) merge no-op gate: anchor leaves through the SAME merge path must be bit-identical.
        anchor_resub = apply_flat_subtree(
            P["bb_params"], {k: _tree_get(P["bb_params"], k) for k in first_payload})
        rows.append(("merge_noop", anchor_head, anchor_resub))
    if args.rows:
        keep = {s.strip() for s in args.rows.split(",") if s.strip()}
        # anchor is always kept: every other row is normalized against it.
        rows = [r for r in rows if r[0] == "anchor"
                or any(r[0] == k or r[0].startswith(k) for k in keep)]
        print(f"[test_eval] --rows filter -> {[r[0] for r in rows]}", flush=True)

    gen = make_generate_es_sharded(P["model"], P["batchnorm"], P["encoder"], top_n,
                                   P["tick_size"], args.n_gen, P["sim_init"],
                                   P["valid_mask_array"], conditional=True, shard=args.shard)
    grid = dict(m=P["m_seq_inp"], b=P["b_seq_inp"], sim=P["sim_states_init"],
                ih=P["init_hidden_batched"], it=P["init_time_batched"])
    rng_grid = grid_rngs(jax.random.PRNGKey(args.seed + 1), 1, Q)        # G=1; CRN across rows

    def _gen_row(head_pop, ts):
        """gen() over all Q rollouts, optionally in --gen_chunk slices (exact, see flag help).
        The single-member head is tiled per chunk, never over the full Q: a full-Q tile of a
        [1, d_model, vocab] head is ~8 MB x Q (33 GiB at Q=4096) and OOMs before generation.
        All rows are G=1, so a CH-tile equals the CH-slice of the full tile bit-for-bit."""
        if CH <= 0 or Q <= CH:
            o = gen(tile_dirs_over_Q(head_pop, Q), ts, grid["m"], grid["b"], grid["sim"],
                    rng_grid, grid["ih"], grid["it"])
            return np.asarray(o[0]), np.asarray(o[1]), np.asarray(o[2])
        assert all(x.shape[0] == 1 for x in jax.tree_util.tree_leaves(head_pop)), \
            "chunked _gen_row assumes single-member heads"
        parts = []
        for s in range(0, Q, CH):
            sl = slice(s, s + CH)
            o = gen(tile_dirs_over_Q(head_pop, min(CH, Q - s)), ts, grid["m"][sl], grid["b"][sl],
                    _sliced(grid["sim"], sl), rng_grid[sl], _sliced(grid["ih"], sl),
                    grid["it"][sl])
            parts.append((np.asarray(o[0]), np.asarray(o[1]), np.asarray(o[2])))
        return tuple(np.concatenate([p[i] for p in parts], axis=0) for i in range(3))

    results, anchor_out = {}, None
    for name, head_pop, params in rows:
        ts = P["train_state"].replace(params=params)
        gmd, gl2, g_nerr = _gen_row(head_pop, ts)   # [Q, n_gen, msg_fields], [Q, n_gen, W], [Q]
        get_ = gmd[..., ETi].astype(np.int32)
        gdt = interarrival_log(gmd[..., DTsi], gmd[..., DTnsi])
        nerr = float(g_nerr.astype(np.float32).mean())

        if args.save_rollouts:
            # Persist the raw rollouts: LOB-Bench (task #13) scores these on CPU later with NO
            # extra GPU generation (decoded msgs [Q,T,14] + L2 books [Q,T,W], ~110 MB/row).
            os.makedirs(args.out_dir, exist_ok=True)
            np.savez_compressed(os.path.join(args.out_dir, f"rollout_{name}.npz"),
                                msgs=gmd.astype(np.float32), l2=gl2.astype(np.float32))
        if name == "anchor":
            anchor_out = (gl2.copy(), gmd.copy())
        if name == "merge_noop":
            bitexact = bool(np.array_equal(gl2, anchor_out[0]) and np.array_equal(gmd, anchor_out[1]))
            results[name] = dict(gate_bitexact=bitexact)
            print(f"[test_eval] (G1) merge_noop bit-exact vs anchor: {bitexact}", flush=True)
            continue

        pm = panel(gl2, get_, real_l2, real_et, gdt, real_dt)
        legacy = {k: float(v) for k, v in EM.stylized_fact_metrics(
            jnp.asarray(gl2), jnp.asarray(real_l2), jnp.asarray(get_),
            jnp.asarray(real_et), n_levels=inf.l2_state_n).items()}

        def row_metric(idx, gl2=gl2, get_=get_, gdt=gdt):
            return {k: v for k, v in panel(gl2[idx], get_[idx], real_l2[idx], real_et[idx],
                                           gdt[idx], real_dt[idx]).items() if k in PANEL_DIST_KEYS}
        cis = bootstrap_ci(row_metric, Q, n_boot=args.n_boot, seed=args.seed + 7)
        results[name] = dict(panel=pm, legacy=legacy, ci=cis, mean_num_errors=nerr)
        print(f"[test_eval] row '{name}': " + " ".join(f"{k}={v:.4g}" for k, v in pm.items()),
              flush=True)

    # Normalize the distance metrics by the anchor row; assemble the table.
    anchor_pm = results["anchor"]["panel"]
    for name, r in results.items():
        if "panel" in r:
            r["vs_anchor"] = {k: (r["panel"][k] / anchor_pm[k] if anchor_pm.get(k) else float("nan"))
                              for k in PANEL_DIST_KEYS if k in r["panel"]}

    os.makedirs(args.out_dir, exist_ok=True)
    # Context -> trading-day mapping (A/B day-split protocol). The idx draw is
    # PRNGKey(seed)-deterministic over the staged day files, so this mapping is portable
    # to every same-config eval's saved rollouts: split by day, re-score halves —
    # selection on A, sealed final on B.
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
        print(f"[test_eval] ctx_days.json: {len(set(_days))} day(s), "
              f"counts={dict(sorted(__import__('collections').Counter(_days).items()))}", flush=True)
    except Exception as _e:
        print(f"[test_eval] ctx->day mapping skipped: {_e}", flush=True)
    payload = dict(stage="test_eval", n_cond=args.n_cond, n_gen=args.n_gen, n_eval_ctx=Q,
                   seed=args.seed, n_boot=args.n_boot, wide_levels=args.wide_levels,
                   data_dir=args.data_dir, ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step,
                   bar_temp=args.bar_temp, top_n=top_n, rows={k: _jsonable(v) for k, v in results.items()})
    with open(os.path.join(args.out_dir, "test_eval_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    keys = [k for k in PANEL_DIST_KEYS if k in anchor_pm]
    lines = ["| row | " + " | ".join(keys) + " | unchanged@50 |",
             "|" + "---|" * (len(keys) + 2)]
    for name, r in results.items():
        if "panel" not in r:
            continue
        cells = []
        for k in keys:
            v, ci = r["panel"][k], r["ci"].get(k)
            cells.append(f"{v:.4g} [{ci[0]:.3g},{ci[1]:.3g}]" if ci else f"{v:.4g}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['panel']['unchanged_rate']:.3f} |")
    lines.append("")
    lines.append("| row | " + " | ".join(f"{k} /anchor" for k in keys) + " |")
    lines.append("|" + "---|" * (len(keys) + 1))
    for name, r in results.items():
        if "vs_anchor" in r:
            lines.append(f"| {name} | " + " | ".join(f"{r['vs_anchor'][k]:.3f}" for k in keys) + " |")
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, "test_eval_table.md"), "w") as f:
        f.write(f"# Held-out test-set panel (real_unchanged@50 = "
                f"{anchor_pm['real_unchanged_rate']:.3f})\n\n{table}\n")
    print("\n" + table, flush=True)
    gate_ok = results.get("merge_noop", {}).get("gate_bitexact", True)
    print(f"\n[test_eval] done -> {args.out_dir} (G1 merge_noop={'PASS' if gate_ok else 'FAIL'})",
          flush=True)
    return 0 if gate_ok else 3


def _tree_get(tree, flat_key):
    node = tree
    for k in flat_key.split("/"):
        node = node[k]
    return node


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


# ----------------------------------------------------------------------------------------
# Login-node self-checks (no model, no data).
# ----------------------------------------------------------------------------------------
def cpu_checks(seed=0):
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("[test_eval] CPU checks — distributional panel / merge / bootstrap", flush=True)
    rng = np.random.default_rng(seed)

    # (T1) W1/KS: identical -> 0; mean shift recovered; KS in [0,1] and ~1 for disjoint.
    a = rng.normal(0, 1, 20000)
    chk("(T1) w1(a,a)=0, w1 shift~|d|, ks bounds",
        w1(a, a) < 1e-12 and abs(w1(a, a + 3.0) - 3.0) < 0.05
        and ks(a, a) < 1e-12 and ks(a, a + 100.0) > 0.999,
        f"w1shift={w1(a, a + 3.0):.4f}")

    # (T2) ACF: iid ~0 at all lags; persistent |x| (AR-style) clearly nonzero at lag 1.
    iid = rng.normal(0, 1, (8, 4000))
    ar = np.cumsum(rng.normal(0, 1, (8, 4000)), axis=1)
    chk("(T2) mean_acf: iid~0, integrated~1",
        float(np.abs(mean_acf(iid, 5)).max()) < 0.08 and float(mean_acf(ar, 5)[0]) > 0.9)

    # (T3) l2 extraction on a synthetic book: layout [pa, va, pb, vb] x levels.
    Qc, T, L = 3, 50, 4
    l2 = np.zeros((Qc, T, 4 * L))
    l2[..., 0], l2[..., 2] = 1010.0, 990.0                     # spread 20, mid 1000
    l2[..., 1], l2[..., 3] = 7.0, 7.0
    for lv in range(1, L):
        l2[..., 4 * lv + 1] = l2[..., 4 * lv + 3] = 1.0
    sp = spreads(l2)
    dp = depth_profile(l2)
    chk("(T3) spreads / depth_profile / unchanged_rate on synthetic book",
        float(np.nanmax(np.abs(sp - 20.0))) == 0.0
        and abs(dp.sum() - 1.0) < 1e-12 and abs(dp[0] - 7.0 / 20.0) < 1e-12
        and unchanged_rate(l2) == 1.0)

    # (T4) returns + event hist: constant mid -> all ret 0; known histograms.
    r1 = log_returns(l2, 1)
    e_l1 = event_hist_l1(np.array([1, 1, 2, 4]), np.array([1, 1, 2, 4]))
    e_l2 = event_hist_l1(np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2]))
    chk("(T4) log_returns const-mid=0; event_l1 identical=0 / disjoint=2",
        float(np.nanmax(np.abs(r1))) == 0.0 and e_l1 == 0.0 and e_l2 == 2.0)

    # (T5) apply_flat_subtree: right leaf changed, original untouched, bad key/shape loud.
    tree = {"a": {"b": np.zeros((2, 3)), "c": np.ones(4)}, "d": np.full(2, 7.0)}
    new = apply_flat_subtree(tree, {"a/b": np.full((2, 3), 5.0)})
    ok = (float(new["a"]["b"].max()) == 5.0 and float(tree["a"]["b"].max()) == 0.0
          and new["a"]["c"] is tree["a"]["c"] and new["d"] is tree["d"])
    try:
        apply_flat_subtree(tree, {"a/zz": np.zeros(1)}); ok = False
    except KeyError:
        pass
    try:
        apply_flat_subtree(tree, {"a/b": np.zeros((9, 9))}); ok = False
    except ValueError:
        pass
    chk("(T5) apply_flat_subtree copy-on-path + loud failures", ok)

    # (T6) panel self-distance ~0 on synthetic data; bootstrap CI covers the point estimate.
    l2r = np.abs(rng.normal(1000, 5, (6, 200, 16))) + 1.0
    etr = rng.integers(1, 5, (6, 200))
    pm = panel(l2r, etr, l2r.copy(), etr.copy())
    self_zero = all(pm[k] < 1e-9 for k in PANEL_DIST_KEYS if k in pm and math.isfinite(pm[k]))
    l2g = l2r * (1 + 0.001 * rng.normal(size=l2r.shape))
    point = {k: v for k, v in panel(l2g, etr, l2r, etr).items() if k in PANEL_DIST_KEYS}
    cis = bootstrap_ci(lambda idx: {k: v for k, v in
                                    panel(l2g[idx], etr[idx], l2r[idx], etr[idx]).items()
                                    if k in PANEL_DIST_KEYS}, 6, n_boot=60, seed=seed)
    cover = all(not math.isfinite(point[k]) or (cis[k][0] <= point[k] * 1.5 and cis[k][1] >= point[k] * 0.5)
                for k in point)
    chk("(T6) panel self-distance=0; bootstrap CI brackets the estimate", self_zero and cover,
        f"ret_w1_1={point.get('ret_w1_1', float('nan')):.4g}")

    print("[test_eval] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


def main():
    ap = argparse.ArgumentParser(description="held-out test-set distributional eval panel")
    ap.add_argument("--run", action="store_true", help="run the GPU eval; else CPU checks only")
    ap.add_argument("--data_dir", default=None, help="node-local held-out-months data dir")
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "test_eval_out"))
    ap.add_argument("--eggroll", action="append", default=None, metavar="NAME=CKPT_DIR",
                    help="proj-scope checkpoint row (repeatable), e.g. es003_s0=/path/to/run/best")
    ap.add_argument("--full_ckpt", action="append", default=None, metavar="NAME=CKPT_DIR:STEP",
                    help="full-checkpoint row (repeatable): loads a complete generator "
                         "checkpoint (own decoder head) via the anchor loading path and "
                         "rolls it out in the same program — CRN-paired, no cross-program "
                         "jitter (used for the continued-pretraining control)")
    ap.add_argument("--bar_temp", type=float, default=0.8,
                    help="tuned-decoding bar temperature (W3 winner tau=0.8 @ default top_n)")
    ap.add_argument("--n_cond", type=int, default=500)
    ap.add_argument("--n_gen", type=int, default=500)
    ap.add_argument("--n_eval_ctx", type=int, default=256)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--top_n", type=int, default=None,
                    help="override CFG.rollout.sample_top_n for ALL rows (-1 = full sampling)")
    ap.add_argument("--rows", default=None,
                    help="comma list of row names (prefix match) to run; anchor always kept")
    ap.add_argument("--wide_book_dir", default=None,
                    help="dir of wide/linf book snapshots for sim init (get_dataset wide_book_dir); "
                         "without it the sim inits with only the L10 proc-book liquidity")
    ap.add_argument("--wide_levels", type=int, default=50,
                    help="L2 snapshot depth; 50 => unchanged_rate is the deep inert-message proxy")
    ap.add_argument("--shard", choices=["auto", "on", "off"], default="off")
    ap.add_argument("--gen_chunk", type=int, default=0,
                    help="run gen/replay in slices of this many rollouts (EXACT: rollouts are "
                         "vmapped + per-rollout independent, so slicing cannot change any rollout; "
                         "keeps the 1024-ctx tier inside the proven ~512-rollout/GPU regime). "
                         "0 = single call. Requires --shard off and n_eval_ctx %% gen_chunk == 0.")
    ap.add_argument("--save_rollouts", type=int, default=1,
                    help="persist per-row decoded msgs + L2 books (.npz) for LOB-Bench scoring")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fails = cpu_checks(seed=args.seed)
    if fails:
        print(f"[test_eval] CPU checks FAILED: {fails}"); sys.exit(1)
    if args.run:
        if not args.data_dir:
            print("[test_eval] --run requires --data_dir"); sys.exit(2)
        sys.exit(run(args))
    print("\n[test_eval] (skipped GPU eval — pass --run on the GH200; CPU checks PASSED)")


if __name__ == "__main__":
    main()
