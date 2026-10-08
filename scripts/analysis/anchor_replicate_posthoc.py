#!/usr/bin/env python
"""Post-hoc (disclosed) anchor program-replicate summary + cpt-vs-mean-anchor robustness line.

The three report jobs regenerated the SAME anchor weights under nominally identical seeds;
in exact arithmetic their rollouts would be bit-identical, and the observed spread is pure
TF32/XLA program jitter amplified by autoregressive divergence — so the anchor rows act as
independent program replicates. This script:

  1. reports anchor WS-21 as mean +- s.d. across replicates (n=3 primary / n=2 secondary),
     with a day-cluster bootstrap CI on the replicate MEAN (replicates are paired
     draw-for-draw across same-shape bboots, see paired_delta_from_bboot.py);
  2. recomputes the cpt_s0 delta against the replicate-mean anchor (the registered table
     compares it to the picks-job anchor alone — the lowest of the three jitter draws).

Seed deltas are NOT touched: within-job pairing cancels program jitter exactly and beats
any averaging. This is a presentation/robustness layer only (prereg Amendment 2 outcome
disclosed it as post-hoc).

Usage: third_party/lobbench_venv/bin/python scripts/analysis/anchor_replicate_posthoc.py
"""
from __future__ import annotations
import json
import os

import numpy as np

PT = "/lustre/projects/public/shared/post_training_GAN/panel28_evals"
JP, JN, JC, JS = 5506605, 5506610, 5506617, 5509802  # picks / null / cpt / s2new


def bb(job: int, tag: str) -> dict:
    p = f"{PT}/test_eval_{job}/lobbench_scores_{tag}/bboot_anchor.json"
    with open(p) as f:
        return json.load(f)


def summarize(name: str, anchors: list[dict], cpt: dict, out_prefix: str) -> None:
    metas = [a["meta"] for a in anchors] + [cpt["meta"]]
    for k in ("seed", "n_boot", "n_seq", "n_days"):
        assert len({m[k] for m in metas}) == 1, f"unpaired bootstraps: meta[{k}] differs"
    assert len({tuple(m["days"]) for m in metas}) == 1, "day universes differ"

    out = {"analysis": name, "n_replicates": len(anchors),
           "meta": {k: metas[0][k] for k in ("seed", "n_boot", "n_seq", "n_days", "alpha")}}
    print(f"\n== {name} ({len(anchors)} anchor replicates, "
          f"{metas[0]['n_seq']} windows / {metas[0]['n_days']} days) ==")
    for metric in ("wasserstein", "l1"):
        pts = np.array([a[metric]["point"] for a in anchors])
        mean, sd = float(pts.mean()), float(pts.std(ddof=1)) if len(pts) > 1 else 0.0
        m = {"anchor_points": pts.tolist(), "anchor_mean": mean, "anchor_sd": sd}
        d_point = cpt[metric]["point"] - mean
        m["cpt_point"] = cpt[metric]["point"]
        m["cpt_delta_vs_mean"] = d_point
        m["cpt_rel_vs_mean"] = d_point / mean
        for kind in ("day", "rollout"):
            reps = np.array([a[metric]["replicates"][kind] for a in anchors])  # (n, n_boot)
            mreps = reps.mean(axis=0)
            lo, hi = np.nanpercentile(mreps, [2.5, 97.5])
            m[f"mean_ci_{kind}"] = [float(lo), float(hi)]
            diff = np.asarray(cpt[metric]["replicates"][kind], dtype=float) - mreps
            dlo, dhi = np.nanpercentile(diff, [2.5, 97.5])
            m[f"cpt_ci_{kind}"] = [float(dlo), float(dhi)]
            m[f"cpt_excludes_zero_{kind}"] = bool(dhi < 0 or dlo > 0)
        out[metric] = m
        print(f"[{metric}] anchor mean {mean:.5f} +- {sd:.5f} (points: "
              + ", ".join(f"{p:.5f}" for p in pts)
              + f") day-CI on mean [{m['mean_ci_day'][0]:.5f},{m['mean_ci_day'][1]:.5f}]")
        print(f"[{metric}] cpt {cpt[metric]['point']:.5f} vs mean: {d_point:+.5f} "
              f"({100 * m['cpt_rel_vs_mean']:+.2f}%) "
              f"day-CI [{m['cpt_ci_day'][0]:+.5f},{m['cpt_ci_day'][1]:+.5f}] "
              f"rollout-CI [{m['cpt_ci_rollout'][0]:+.5f},{m['cpt_ci_rollout'][1]:+.5f}] "
              f"day-CI excludes 0: {m['cpt_excludes_zero_day']}")
    op = f"{PT}/pdeltas_final/{out_prefix}_anchor_replicates.json"
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    os.chmod(op, 0o664)
    print(f"-> {op}")


def load(job: int, tag: str, row: str) -> dict:
    with open(f"{PT}/test_eval_{job}/lobbench_scores_{tag}/bboot_{row}.json") as f:
        return json.load(f)


def seed_average(anchors_all: list[dict], picks: list[dict], paired_anchor: list[dict],
                 out_prefix: str) -> None:
    """Two seed-average formulations on the PRIMARY set.

    pure_avg:   mean(pick points) - mean(ALL anchor replicate points). Descriptive only:
                the anchor mean includes the null-job replicate (a program that produced
                no pick) and weights jobs differently on the two sides, so program jitter
                does NOT cancel — the bias is exactly (JP_anchor - JN_anchor)/3.
    paired_avg: mean over seeds of (pick - SAME-JOB anchor); jitter cancels exactly.
                Also reports the mean of per-seed RELATIVE deltas (the registered
                headline) with a day-cluster CI from per-replicate relative diffs.
    """
    metas = [a["meta"] for a in anchors_all + picks]
    for k in ("seed", "n_boot", "n_seq", "n_days"):
        assert len({m[k] for m in metas}) == 1, f"unpaired bootstraps: meta[{k}] differs"
    out = {"analysis": "PRIMARY seed-average formulations", "n_seeds": len(picks),
           "meta": {k: metas[0][k] for k in ("seed", "n_boot", "n_seq", "n_days", "alpha")}}
    print(f"\n== PRIMARY seed-average formulations ({len(picks)} seeds) ==")
    for metric in ("wasserstein", "l1"):
        a_pts = np.array([a[metric]["point"] for a in anchors_all])
        p_pts = np.array([p[metric]["point"] for p in picks])
        pa_pts = np.array([a[metric]["point"] for a in paired_anchor])
        m = {"pick_points": p_pts.tolist(), "anchor_points_all": a_pts.tolist(),
             "paired_anchor_points": pa_pts.tolist()}
        m["pure_avg"] = {"delta": float(p_pts.mean() - a_pts.mean()),
                         "rel": float((p_pts.mean() - a_pts.mean()) / a_pts.mean()),
                         "jitter_bias_vs_paired": float((a_pts[0] - a_pts[1]) / 3)}
        d_pts = p_pts - pa_pts
        m["paired_avg"] = {"delta": float(d_pts.mean()),
                           "rel_vs_weighted_anchor": float(d_pts.mean() / pa_pts.mean()),
                           "mean_of_rel_deltas": float((d_pts / pa_pts).mean()),
                           "sd_of_rel_deltas": float((d_pts / pa_pts).std(ddof=1))}
        for kind in ("day", "rollout"):
            a_reps = np.array([a[metric]["replicates"][kind] for a in anchors_all])
            p_reps = np.array([p[metric]["replicates"][kind] for p in picks])
            pa_reps = np.array([a[metric]["replicates"][kind] for a in paired_anchor])
            for key, diff in (("pure_avg", p_reps.mean(0) - a_reps.mean(0)),
                              ("paired_avg", (p_reps - pa_reps).mean(0)),
                              ("paired_rel", ((p_reps - pa_reps) / pa_reps).mean(0))):
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                tgt = m.setdefault(key, {})
                tgt[f"ci_{kind}"] = [float(lo), float(hi)]
                tgt[f"excludes_zero_{kind}"] = bool(hi < 0 or lo > 0)
        out[metric] = m
        pu, pv, pr = m["pure_avg"], m["paired_avg"], m["paired_rel"]
        print(f"[{metric}] pure_avg  Δ={pu['delta']:+.5f} ({100 * pu['rel']:+.2f}%) "
              f"day-CI [{pu['ci_day'][0]:+.5f},{pu['ci_day'][1]:+.5f}] "
              f"(jitter bias vs paired: {pu['jitter_bias_vs_paired']:+.5f})")
        print(f"[{metric}] paired_avg Δ={pv['delta']:+.5f} "
              f"({100 * pv['rel_vs_weighted_anchor']:+.2f}% vs weighted anchor; "
              f"mean-of-rels {100 * pv['mean_of_rel_deltas']:+.2f}% "
              f"+- {100 * pv['sd_of_rel_deltas']:.2f}%) "
              f"day-CI [{pv['ci_day'][0]:+.5f},{pv['ci_day'][1]:+.5f}] "
              f"rel day-CI [{100 * pr['ci_day'][0]:+.2f}%,{100 * pr['ci_day'][1]:+.2f}%] "
              f"day-CI excludes 0: {pv['excludes_zero_day']}")
    op = f"{PT}/pdeltas_final/{out_prefix}_seedavg.json"
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    os.chmod(op, 0o664)
    print(f"-> {op}")


def main():
    # cpt_s0_step3190 was generated by job JC under the row tag 'anchor' (fleet wrapper
    # names whatever checkpoint it is pointed at 'anchor').
    summarize("PRIMARY window-disjoint (testW)",
              [bb(j, "testW_anchor") for j in (JP, JN, JS)], bb(JC, "testW_anchor"), "P")
    summarize("SECONDARY day-held-out (daysB)",
              [bb(j, "daysB_anchor") for j in (JP, JN)], bb(JC, "daysB_anchor"), "B")
    picks = [load(JP, "testW_s0pick", "s0pick"), load(JP, "testW_s1pick", "s1pick"),
             load(JS, "testW_s2pick_st0035", "s2pick_st0035")]
    paired_anchor = [bb(JP, "testW_anchor"), bb(JP, "testW_anchor"), bb(JS, "testW_anchor")]
    seed_average([bb(j, "testW_anchor") for j in (JP, JN, JS)], picks, paired_anchor, "P")


if __name__ == "__main__":
    main()
