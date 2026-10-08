#!/usr/bin/env python
"""Aggregate the single-program B re-scoring bboot files into figure data + markdown.

Reads the per-row block_bootstrap outputs (anchor + N seed picks + optional control rows,
e.g. the CPT control and the selection-matched null pick), forms the paired seed-average
and per-control deltas from the shared bootstrap draws, and writes:
  - <figd>/rescoreB_data.json   (feeds scripts/analysis/plot_unseen_paper_figures.py)
  - <outmd>                     (human-readable results table)

Row sets are arbitrary: --seed_rows lists the treated picks (averaged), --control_rows
lists rows that only get individual paired deltas (missing control bboots are skipped).
Defaults reproduce the legacy 3-seed unseen-arm layout. The legacy fig["cpt"] alias is
kept whenever a cpt_s0 control is present so existing plot scripts keep working.

PRIMARY CI is the paired WINDOW-level block bootstrap (whole sealed windows resampled with
replacement, real/gen pairing preserved, deltas replicate-wise from shared draws). Day-cluster
CIs are carried as a robustness secondary. Pure json+numpy — login-node safe.

bboot layout: normally <D>/lobbench_scores_<tag_prefix><row>/bboot_<row>.json (tag_prefix
default testW_). Pass --flat_tag to read a single shared directory
<D>/lobbench_scores_<flat_tag>/bboot_<row>.json instead.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SEEDS = ["s0pick", "s1pick", "s2pick"]
CONTROLS = ["cpt_s0"]
TAG_PREFIX = "testW_"
UNIVERSE = "window-disjoint positions 512-2047 (1536 windows, 28 days, Jan+Feb 2026)"
TITLE = "Unseen arm — single-program B re-scoring (window-disjoint primary)"


def _ci(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def _bboot_path(scores_root, row, flat_tag):
    tag = flat_tag if flat_tag is not None else f"{TAG_PREFIX}{row}"
    return os.path.join(scores_root, f"lobbench_scores_{tag}", f"bboot_{row}.json")


def aggregate(scores_root, gen_job, score_job, flat_tag=None):
    def load(row):
        return json.load(open(_bboot_path(scores_root, row, flat_tag)))

    controls = [c for c in CONTROLS
                if os.path.isfile(_bboot_path(scores_root, c, flat_tag))]
    rows = ["anchor"] + SEEDS + controls
    B = {r: load(r) for r in rows}
    meta = B["anchor"]["meta"]
    for r in rows[1:]:
        assert B[r]["meta"]["days"] == meta["days"] and B[r]["meta"]["n_seq"] == meta["n_seq"], \
            f"{r}: bootstrap universe differs from anchor — pairing invalid"
    names = meta["scores"]

    fig = {"meta": {"gen_job": gen_job, "score_job": score_job,
                    "universe": UNIVERSE,
                    "n_boot": meta["n_boot"], "n_seq": meta["n_seq"],
                    "n_days": meta["n_days"], "features": names},
           "rows": {}, "seed_avg": {}, "per_feature": {}}
    # boot_median_*: WS/L1 are positively-biased plug-in estimators under resampling, so
    # the full-sample point can fall outside its own percentile CI. Figures therefore show
    # the bootstrap MEDIAN (inside the percentile CI by construction); tables keep the raw
    # point (bit-matches LOB-Bench). Paired deltas are unaffected (the bias cancels).
    for r, j in B.items():
        fig["rows"][r] = {m: {"point": j[m]["point"], "ci_day": j[m]["ci_day"],
                              "ci_rollout": j[m]["ci_rollout"],
                              "boot_median_day": float(np.nanmedian(
                                  np.asarray(j[m]["replicates"]["day"], float))),
                              "boot_median_rollout": float(np.nanmedian(
                                  np.asarray(j[m]["replicates"]["rollout"], float)))}
                          for m in ("wasserstein", "l1")}

    # 3-seed average: replicate-wise mean across seeds shares the anchor's draws, so the
    # average row gets an honest paired CI (absolute AND delta-vs-anchor).
    for m in ("wasserstein", "l1"):
        pts = np.array([B[r][m]["point"] for r in SEEDS])
        fa = {}
        for kind in ("day", "rollout"):
            reps = np.mean([np.asarray(B[r][m]["replicates"][kind], float) for r in SEEDS], axis=0)
            ra = np.asarray(B["anchor"][m]["replicates"][kind], float)
            fa[f"ci_{kind}"] = _ci(reps)
            fa[f"boot_median_{kind}"] = float(np.nanmedian(reps))
            fa[f"seed_boot_median_{kind}"] = {
                r: float(np.nanmedian(np.asarray(B[r][m]["replicates"][kind], float)))
                for r in SEEDS}
            fa[f"delta_ci_{kind}"] = _ci(reps - ra)
            fa[f"delta_excludes_zero_{kind}"] = bool(fa[f"delta_ci_{kind}"][1] < 0
                                                     or fa[f"delta_ci_{kind}"][0] > 0)
        ap = B["anchor"][m]["point"]
        fa.update(point=float(pts.mean()), seed_points={r: float(p) for r, p in zip(SEEDS, pts)},
                  anchor_point=ap, delta_point=float(pts.mean() - ap),
                  rel_delta=float((pts.mean() - ap) / ap))
        fig["seed_avg"][m] = fa

    # Per-seed paired deltas vs the same in-program anchor (replaces the standalone
    # paired_delta_from_bboot pass — same shared-draw arithmetic, same source files).
    fig["seed_delta"] = {}
    for r in SEEDS:
        fig["seed_delta"][r] = {}
        for m in ("wasserstein", "l1"):
            rp, ap = B[r][m]["point"], B["anchor"][m]["point"]
            sd = {"delta_point": float(rp - ap), "rel_delta": float((rp - ap) / ap)}
            for kind in ("day", "rollout"):
                rr = np.asarray(B[r][m]["replicates"][kind], float)
                ra = np.asarray(B["anchor"][m]["replicates"][kind], float)
                sd[f"delta_ci_{kind}"] = _ci(rr - ra)
                sd[f"delta_excludes_zero_{kind}"] = bool(sd[f"delta_ci_{kind}"][1] < 0
                                                         or sd[f"delta_ci_{kind}"][0] > 0)
            fig["seed_delta"][r][m] = sd

    # Control rows (CPT, selection-null pick, ...): paired delta vs the SAME in-program
    # anchor — jitter-free by construction (shared draws, one program).
    for c in controls:
        for m in ("wasserstein", "l1"):
            cp, ap = B[c][m]["point"], B["anchor"][m]["point"]
            fc = {"point": cp, "anchor_point": ap, "delta_point": float(cp - ap),
                  "rel_delta": float((cp - ap) / ap)}
            for kind in ("day", "rollout"):
                rc = np.asarray(B[c][m]["replicates"][kind], float)
                ra = np.asarray(B["anchor"][m]["replicates"][kind], float)
                fc[f"delta_ci_{kind}"] = _ci(rc - ra)
                fc[f"delta_excludes_zero_{kind}"] = bool(fc[f"delta_ci_{kind}"][1] < 0
                                                         or fc[f"delta_ci_{kind}"][0] > 0)
            fig.setdefault("controls", {}).setdefault(c, {})[m] = fc
    if "cpt_s0" in controls:                    # legacy alias for existing plot scripts
        fig["cpt"] = fig["controls"]["cpt_s0"]

    # Per-feature: anchor + per-seed + seed-avg points, and the seed-avg paired delta CI.
    for m in ("wasserstein", "l1"):
        per = {}
        for nm in names:
            arow = B["anchor"][m]["per_score"][nm]
            spts = {r: B[r][m]["per_score"][nm]["point"] for r in SEEDS}
            sa = float(np.mean(list(spts.values())))
            rec = {"anchor_point": arow["point"], "anchor_ci_day": arow["ci_day"],
                   "anchor_ci_rollout": arow["ci_rollout"],
                   "seed_points": spts, "seed_avg_point": sa,
                   "delta_point": sa - arow["point"],
                   "rel_delta": (sa - arow["point"]) / arow["point"] if arow["point"] else None}
            for kind in ("day", "rollout"):
                key = f"{kind}_per_score"
                if key not in B["anchor"][m]["replicates"]:
                    continue
                reps = np.mean([np.asarray(B[r][m]["replicates"][key][nm], float)
                                for r in SEEDS], axis=0)
                ra = np.asarray(B["anchor"][m]["replicates"][key][nm], float)
                rec[f"seed_avg_ci_{kind}"] = _ci(reps)
                rec[f"seed_avg_boot_median_{kind}"] = float(np.nanmedian(reps))
                rec[f"anchor_boot_median_{kind}"] = float(np.nanmedian(ra))
                rec[f"delta_ci_{kind}"] = _ci(reps - ra)
                rec[f"delta_excludes_zero_{kind}"] = bool(rec[f"delta_ci_{kind}"][1] < 0
                                                          or rec[f"delta_ci_{kind}"][0] > 0)
            per[nm] = rec
        fig["per_feature"][m] = per
    return fig, rows, meta


def render_markdown(fig, rows, meta, gen_job, score_job):
    W = fig["seed_avg"]["wasserstein"]
    L1 = fig["seed_avg"]["l1"]
    n_ctrl = len(fig.get("controls", {}))
    lines = [f"# {TITLE}", "",
             f"Generation job {gen_job} (anchor + {len(SEEDS)} seed picks + {n_ctrl} "
             f"control row(s) in ONE program, merge_noop bit-exact); scoring job "
             f"{score_job}. One lob_bench evaluation per row over the identical universe "
             f"({fig['meta']['universe']}). PRIMARY CI = paired WINDOW-level "
             f"block bootstrap (n_boot {meta['n_boot']}: whole sealed windows resampled with "
             f"replacement, real/gen pairing preserved, deltas replicate-wise from shared "
             f"draws). Day-cluster CIs are reported as a robustness secondary.", "",
             "| row | WS-21 | window-CI | Mean L1 | window-CI |", "|---|---|---|---|---|"]
    for r in rows:
        w, l = fig["rows"][r]["wasserstein"], fig["rows"][r]["l1"]
        lines.append(f"| {r} | {w['point']:.5f} "
                     f"| [{w['ci_rollout'][0]:.4f},{w['ci_rollout'][1]:.4f}] "
                     f"| {l['point']:.5f} "
                     f"| [{l['ci_rollout'][0]:.4f},{l['ci_rollout'][1]:.4f}] |")
    if fig.get("seed_delta"):
        lines += ["", "Per-seed paired deltas (vs the same in-program anchor):", "",
                  "| seed | dWS-21 | rel | window-CI | excl. 0 | dL1 | rel | window-CI | excl. 0 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in SEEDS:
            w = fig["seed_delta"][r]["wasserstein"]
            l = fig["seed_delta"][r]["l1"]
            lines.append(
                f"| {r} | {w['delta_point']:+.5f} | {100*w['rel_delta']:+.2f}% "
                f"| [{w['delta_ci_rollout'][0]:+.4f},{w['delta_ci_rollout'][1]:+.4f}] "
                f"| {w['delta_excludes_zero_rollout']} "
                f"| {l['delta_point']:+.5f} | {100*l['rel_delta']:+.2f}% "
                f"| [{l['delta_ci_rollout'][0]:+.4f},{l['delta_ci_rollout'][1]:+.4f}] "
                f"| {l['delta_excludes_zero_rollout']} |")
    lines += ["", f"**{len(SEEDS)}-seed average**: WS-21 {W['point']:.5f} [{W['ci_rollout'][0]:.4f},"
              f"{W['ci_rollout'][1]:.4f}], delta vs anchor {W['delta_point']:+.5f} "
              f"({100*W['rel_delta']:+.2f}%), paired window-CI [{W['delta_ci_rollout'][0]:+.4f},"
              f"{W['delta_ci_rollout'][1]:+.4f}], excludes 0: {W['delta_excludes_zero_rollout']} "
              f"(day-cluster robustness: [{W['delta_ci_day'][0]:+.4f},{W['delta_ci_day'][1]:+.4f}], "
              f"excludes 0: {W['delta_excludes_zero_day']})",
              f"Mean L1 {L1['point']:.5f} [{L1['ci_rollout'][0]:.4f},{L1['ci_rollout'][1]:.4f}], "
              f"delta {L1['delta_point']:+.5f} ({100*L1['rel_delta']:+.2f}%), paired window-CI "
              f"[{L1['delta_ci_rollout'][0]:+.4f},{L1['delta_ci_rollout'][1]:+.4f}], excludes 0: "
              f"{L1['delta_excludes_zero_rollout']} (day-cluster robustness: "
              f"[{L1['delta_ci_day'][0]:+.4f},{L1['delta_ci_day'][1]:+.4f}], excludes 0: "
              f"{L1['delta_excludes_zero_day']})", ""]
    for c, fc in fig.get("controls", {}).items():
        CW, CL = fc["wasserstein"], fc["l1"]
        lines += [f"**{c} control (same program, jitter-free)**: WS-21 {CW['point']:.5f}, "
                  f"delta vs anchor {CW['delta_point']:+.5f} ({100*CW['rel_delta']:+.2f}%), "
                  f"paired window-CI [{CW['delta_ci_rollout'][0]:+.4f},"
                  f"{CW['delta_ci_rollout'][1]:+.4f}], excludes 0: "
                  f"{CW['delta_excludes_zero_rollout']}; Mean L1 delta {CL['delta_point']:+.5f} "
                  f"({100*CL['rel_delta']:+.2f}%), window-CI [{CL['delta_ci_rollout'][0]:+.4f},"
                  f"{CL['delta_ci_rollout'][1]:+.4f}], excludes 0: "
                  f"{CL['delta_excludes_zero_rollout']}", ""]
    lines += ["## Per-feature seed-avg deltas (WS), sorted", "",
              "| feature | anchor | seed-avg | delta | rel | paired window-CI | excl. 0 |",
              "|---|---|---|---|---|---|---|"]
    pf = fig["per_feature"]["wasserstein"]
    for nm in sorted(pf, key=lambda n: pf[n]["delta_point"]):
        p = pf[nm]
        cir = p.get("delta_ci_rollout", [float("nan")] * 2)
        lines.append(f"| {nm} | {p['anchor_point']:.4f} | {p['seed_avg_point']:.4f} | "
                     f"{p['delta_point']:+.4f} | {100*(p['rel_delta'] or 0):+.1f}% | "
                     f"[{cir[0]:+.4f},{cir[1]:+.4f}] | "
                     f"{p.get('delta_excludes_zero_rollout', '?')} |")
    lines += ["", "See also: [`native_lobbench_cis.md`](native_lobbench_cis.md) — LOB-Bench's "
              "own summary CIs for these rows (99%, marginal/unpaired), for benchmark "
              "comparability."]
    return "\n".join(lines) + "\n"


def main():
    global SEEDS, CONTROLS, TAG_PREFIX, UNIVERSE, TITLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_root", required=True,
                    help="test_eval dir holding the lobbench_scores_* subdirs")
    ap.add_argument("--fig_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--gen_job", default="?")
    ap.add_argument("--score_job", default="?")
    ap.add_argument("--flat_tag", default=None,
                    help="read all rows from lobbench_scores_<flat_tag>/bboot_<row>.json "
                         "instead of per-row testW_<row> dirs (recovery mode)")
    ap.add_argument("--seed_rows", default=",".join(SEEDS),
                    help="comma list of treated pick rows (averaged + per-seed deltas)")
    ap.add_argument("--control_rows", default=",".join(CONTROLS),
                    help="comma list of control rows (individual paired deltas; "
                         "missing bboots are skipped)")
    ap.add_argument("--tag_prefix", default=TAG_PREFIX,
                    help="lobbench_scores_<prefix><row> dir prefix (default testW_)")
    ap.add_argument("--universe", default=UNIVERSE, help="universe description for meta/md")
    ap.add_argument("--title", default=TITLE, help="markdown title line")
    args = ap.parse_args()

    SEEDS = [r for r in args.seed_rows.split(",") if r]
    CONTROLS = [r for r in args.control_rows.split(",") if r]
    TAG_PREFIX = args.tag_prefix
    UNIVERSE = args.universe
    TITLE = args.title

    fig, rows, meta = aggregate(args.scores_root, args.gen_job, args.score_job, args.flat_tag)
    os.makedirs(os.path.dirname(os.path.abspath(args.fig_json)), exist_ok=True)
    with open(args.fig_json, "w") as f:
        json.dump(fig, f, indent=1)
    os.chmod(args.fig_json, 0o664)
    md = render_markdown(fig, rows, meta, args.gen_job, args.score_job)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md)
    os.chmod(args.out_md, 0o664)
    print(md)


if __name__ == "__main__":
    main()
