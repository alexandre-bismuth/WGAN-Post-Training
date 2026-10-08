"""Block/cluster bootstrap CIs for LOB-Bench scores — the paper CI methodology (task 6.2).

WHY: lob_bench's own CIs resample the FLATTENED pool of per-message/per-event feature
observations (`metrics._bootstrap`), treating heavily autocorrelated within-rollout /
within-day observations as independent → CIs scale like 1/√(pool) instead of
1/√(rollouts) and are far too narrow (pseudoreplication; docs/reference/gotchas.md).
This module recomputes the SAME point estimates (verified against run_lobbench output)
and derives CIs by resampling WHOLE DAYS (primary) and WHOLE ROLLOUT PAIRS (secondary).

Conventions preserved from lob_bench (so points match bit-for-bit):
  * per-score tables mirror scoring.score_data: partitioning.score_real_gen →
    group_by_score with the score's own get_kwargs binning → per-observation table;
  * wasserstein: scores standardised ONCE on the full pooled table (their in-metric
    convention), W1 on raw standardised values per replicate;
  * l1: fixed groups (binned once on the full data), histogram L1/2 per replicate;
  * aggregate = mean over score names (the "WS-21"/"L1" table aggregates), computed
    per replicate → honest CI on the aggregate itself.

CRN pairing: rollout resampling draws SEQUENCE ids and keeps both the real and gen rows
of each drawn sequence (paired resample — deltas vs anchor stay CRN-consistent).

Outputs (JSON): per-score point + day-CI + rollout-CI; aggregate point + CIs; the
day-level vs rollout-level bootstrap variance components + their ratio (drives the
pre-registered 1024→2048 escalation for the reported rows,
docs/reference/preregistration_unseen_arm.md §5).

Run with the lobbench venv python on a COMPUTE node (same as run_lobbench):
    third_party/lobbench_venv/bin/python -m eggroll_gan.eval.block_bootstrap \
        --bench_dir <adapter out_dir> --row <row name> --out <json> \
        [--n_boot 500] [--check_against <lobbench_results.json>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from ..config import EXP_ROOT as _EXP
sys.path.insert(0, os.path.join(_EXP, "third_party", "lob_bench"))


def _seq_day(seq, i):
    d = getattr(seq, "date", None)
    if d is None:
        raise RuntimeError(f"sequence {i} has no .date — cannot day-cluster")
    return str(d)


def build_tagged_table(loader, score_name, score_config):
    """Mirror scoring.score_data but keep (seq_id, day) tags per observation.

    Returns dict of flat numpy arrays: score, group, is_real, seq, plus days list
    (seq_id → day). Nesting handling matches partitioning.get_score_table.
    """
    import partitioning
    from scoring import get_kwargs

    cfg = score_config
    if cfg.get("eval", None) is not None:      # conditional configs not needed here
        raise ValueError(f"{score_name}: conditional config not supported")

    scores_real, scores_gen = partitioning.score_real_gen(loader, cfg["fn"])
    kwargs = get_kwargs(cfg)
    group_scores = kwargs.pop("group_scores", True)
    if group_scores:
        groups_real, groups_gen = partitioning.group_by_score(
            scores_real, scores_gen, **kwargs)
    else:
        groups_real, groups_gen = scores_real, scores_gen

    rows_s, rows_g, rows_r, rows_q = [], [], [], []
    days = []

    def _push(vals, grps, is_real, seq_id):
        v = np.atleast_1d(np.asarray(vals, dtype=float)).ravel()
        g = np.atleast_1d(np.asarray(grps)).ravel()
        assert len(v) == len(g), (score_name, seq_id, len(v), len(g))
        rows_s.append(v)
        rows_g.append(g.astype(np.int64) if group_scores else np.zeros(len(v), np.int64))
        rows_r.append(np.full(len(v), is_real, dtype=bool))
        rows_q.append(np.full(len(v), seq_id, dtype=np.int64))

    for i, seq in enumerate(loader):
        days.append(_seq_day(seq, i))
        _push(scores_real[i], groups_real[i], True, i)
        sg, gg = scores_gen[i], groups_gen[i]
        # gen may be nested one level (per gen-series)
        if hasattr(sg, "__len__") and len(sg) and hasattr(sg[0], "__iter__"):
            for sij, gij in zip(sg, gg):
                _push(sij, gij, False, i)
        else:
            _push(sg, gg, False, i)

    tab = dict(score=np.concatenate(rows_s), group=np.concatenate(rows_g),
               is_real=np.concatenate(rows_r), seq=np.concatenate(rows_q))
    # standardise once on the full pooled table (lob_bench wasserstein convention)
    mu, sd = tab["score"].mean(), tab["score"].std()
    tab["z"] = (tab["score"] - mu) / (sd if sd > 0 else 1.0)
    return tab, days


def _w1(tab, sel):
    from scipy import stats
    r = tab["z"][sel & tab["is_real"]]
    g = tab["z"][sel & ~tab["is_real"]]
    if len(r) == 0 or len(g) == 0:
        return np.nan
    return float(stats.wasserstein_distance(r, g))


def _l1(tab, sel):
    """Clone of metrics.l1_by_group._calc_l1 on fixed groups."""
    r_g = tab["group"][sel & tab["is_real"]]
    g_g = tab["group"][sel & ~tab["is_real"]]
    if len(r_g) == 0 or len(g_g) == 0:
        return 1.0
    all_g = np.unique(tab["group"])
    hr = np.array([(r_g == v).sum() for v in all_g], dtype=float)
    hg = np.array([(g_g == v).sum() for v in all_g], dtype=float)
    hr /= hr.sum()
    hg /= hg.sum()
    return float(np.abs(hr - hg).sum() / 2.0)


def block_bootstrap_row(loader, n_boot=500, seed=12345):
    import run_bench as rb

    seqs = list(loader)
    tabs, day_of_seq = {}, None
    for name, cfg in rb.DEFAULT_SCORING_CONFIG.items():
        try:
            tabs[name], day_of_seq = build_tagged_table(loader, name, cfg)
        except Exception as e:
            print(f"[bboot] score '{name}' failed ({e!r}) — skipped", flush=True)
    names = sorted(tabs.keys())
    n_seq = len(seqs)
    days = sorted(set(day_of_seq))
    seqs_of_day = {d: np.array([i for i, dd in enumerate(day_of_seq) if dd == d])
                   for d in days}
    rng = np.random.default_rng(seed)

    # precompute per-score row masks by seq id
    def _sel_for(tab, seq_ids):
        # multiplicity-aware selection: rows of seq s repeated k times if s drawn k times
        counts = np.bincount(seq_ids, minlength=n_seq)
        reps = counts[tab["seq"]]
        idx = np.repeat(np.arange(len(tab["seq"])), reps)
        return idx

    def _metrics_for(seq_ids):
        out_w, out_l = [], []
        for name in names:
            tab = tabs[name]
            idx = _sel_for(tab, seq_ids)
            sub = {k: v[idx] for k, v in tab.items()}
            sel = np.ones(len(idx), dtype=bool)
            out_w.append(_w1(sub, sel))
            out_l.append(_l1(sub, sel))
        return np.array(out_w), np.array(out_l)

    # point estimates (all sequences once) — must match run_lobbench
    all_ids = np.arange(n_seq)
    point_w, point_l = _metrics_for(all_ids)

    # per-day aggregate points (pooled standardization/binning, day-subset rows) —
    # feeds the pre-registered per-month regime check without extra B passes
    per_day = {}
    for d in days:
        pw, pl = _metrics_for(seqs_of_day[d])
        per_day[d] = {"wasserstein": float(np.nanmean(pw)), "l1": float(np.nanmean(pl)),
                      "n_seq": int(len(seqs_of_day[d]))}

    boot_w_day = np.empty((n_boot, len(names)))
    boot_l_day = np.empty((n_boot, len(names)))
    boot_w_seq = np.empty((n_boot, len(names)))
    boot_l_seq = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        drawn_days = rng.choice(len(days), size=len(days), replace=True)
        ids_day = np.concatenate([seqs_of_day[days[j]] for j in drawn_days])
        boot_w_day[b], boot_l_day[b] = _metrics_for(ids_day)
        ids_seq = rng.integers(0, n_seq, size=n_seq)
        boot_w_seq[b], boot_l_seq[b] = _metrics_for(ids_seq)
        if (b + 1) % 50 == 0:
            print(f"[bboot] replicate {b + 1}/{n_boot}", flush=True)

    def _ci(mat, alpha=0.05):
        return np.nanpercentile(mat, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)

    agg = {}
    for label, point, bd, bs in (("wasserstein", point_w, boot_w_day, boot_w_seq),
                                 ("l1", point_l, boot_l_day, boot_l_seq)):
        agg_point = float(np.nanmean(point))
        agg_day = np.nanmean(bd, axis=1)     # aggregate per replicate
        agg_seq = np.nanmean(bs, axis=1)
        var_day, var_seq = float(np.nanvar(agg_day)), float(np.nanvar(agg_seq))
        agg[label] = {
            "point": agg_point,
            "ci_day": np.nanpercentile(agg_day, [2.5, 97.5]).tolist(),
            "ci_rollout": np.nanpercentile(agg_seq, [2.5, 97.5]).tolist(),
            "var_day": var_day, "var_rollout": var_seq,
            "rollout_over_day_var_ratio": (var_seq / var_day) if var_day > 0 else None,
            "per_score": {
                nm: {"point": float(point[i]),
                     "ci_day": _ci(bd)[:, i].tolist(),
                     "ci_rollout": _ci(bs)[:, i].tolist()}
                for i, nm in enumerate(names)},
        }
    # per-replicate aggregates (draws depend only on (seed, n_days, n_seq) -> two rows
    # bootstrapped in the same bench share draws; paired deltas = replicate differences)
    for label, bd, bs in (("wasserstein", boot_w_day, boot_w_seq), ("l1", boot_l_day, boot_l_seq)):
        agg[label]["replicates"] = {
            "day": np.nanmean(bd, axis=1).tolist(),
            "rollout": np.nanmean(bs, axis=1).tolist(),
            # per-feature replicate matrices (n_boot values per score name): the same
            # shared-draw property holds per feature, so paired per-feature delta CIs
            # (pick - anchor) can be formed downstream (paired_delta_from_bboot).
            "day_per_score": {nm: bd[:, i].tolist() for i, nm in enumerate(names)},
            "rollout_per_score": {nm: bs[:, i].tolist() for i, nm in enumerate(names)},
        }
    agg["per_day"] = per_day
    agg["meta"] = {"n_boot": n_boot, "n_seq": n_seq, "n_days": len(days),
                   "days": days, "seed": seed, "scores": names,
                   "alpha": 0.05,
                   "note": "primary CI = ci_day (day cluster bootstrap); "
                           "escalation rule reads rollout_over_day_var_ratio"}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", required=True)
    ap.add_argument("--row", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--check_against", default=None,
                    help="lobbench_results.json for point-estimate verification")
    args = ap.parse_args()

    import data_loading as dl
    row_dir = os.path.join(args.bench_dir, args.row)
    loader = dl.Simple_Loader(os.path.join(row_dir, "data_real"),
                              os.path.join(row_dir, "data_gen"),
                              os.path.join(row_dir, "data_cond"))
    for s in loader:
        s.materialize()

    res = block_bootstrap_row(loader, n_boot=args.n_boot)

    if args.check_against and os.path.isfile(args.check_against):
        ref = json.load(open(args.check_against)).get(args.row, {}).get("uncond", {})
        n_ok = n_bad = 0
        for nm, d in res["wasserstein"]["per_score"].items():
            rv = ref.get(nm, {}).get("wasserstein")
            if rv is None:
                continue
            if abs(rv - d["point"]) < 1e-6:
                n_ok += 1
            else:
                n_bad += 1
                print(f"[bboot] POINT MISMATCH {nm}: ours {d['point']:.8f} vs ref {rv:.8f}")
        res["meta"]["point_check"] = {"ok": n_ok, "mismatch": n_bad}
        print(f"[bboot] point check vs {args.check_against}: {n_ok} ok, {n_bad} mismatch")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    os.chmod(args.out, 0o664)
    w, l = res["wasserstein"], res["l1"]
    print(f"[bboot] {args.row}: WS agg {w['point']:.4f} day-CI [{w['ci_day'][0]:.4f},"
          f"{w['ci_day'][1]:.4f}] rollout-CI [{w['ci_rollout'][0]:.4f},{w['ci_rollout'][1]:.4f}] "
          f"var-ratio {w['rollout_over_day_var_ratio']}")
    print(f"[bboot] {args.row}: L1 agg {l['point']:.4f} day-CI [{l['ci_day'][0]:.4f},"
          f"{l['ci_day'][1]:.4f}] rollout-CI [{l['ci_rollout'][0]:.4f},{l['ci_rollout'][1]:.4f}]")


if __name__ == "__main__":
    main()
