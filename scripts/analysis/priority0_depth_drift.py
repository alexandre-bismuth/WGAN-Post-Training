#!/usr/bin/env python3
"""Priority-0 diagnostic (per-run): is the post-training "tie" within eval noise, and is
there a free-running exposure-bias gap (and does post-training close it)?

Reuses the eval's EXACT feature/metric definitions (w1, log_returns, spreads from
eggroll_gan/eval/test_eval.py) so results are faithful to the held-out panel.

Reads ONLY existing saved artifacts from a test_eval run dir (no GPU, login-safe):
  <run>/lobbench_scores/lobbench_results.json   -> Part A (marginal CI + per-feature trade)
  <run>/test_eval_results.json                  -> ret_w1_1/10/100 horizon (panel)
  <run>/rollout_{real,<row>}.npz                -> Part B (depth-resolved drift)
Rows are auto-discovered from the rollout_*.npz present (excl. real, merge_noop).

Usage:  python3 scripts/analysis/priority0_depth_drift.py [RUN_DIR]
Default RUN_DIR = runs/test_eval_5287549 (GRPO+stylized soup, held-out 2026-02).
For the cross-arm consolidation see priority0_cross_arm.py.
"""
import json, os, sys
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/test_eval_5287549"
SKIP = {"real", "merge_noop"}


# ---- eval's exact definitions (mirror eggroll_gan/eval/test_eval.py:53,94,104) ----
def w1(a, b, n_q=2048):
    a = np.asarray(a, np.float64).ravel(); a = a[np.isfinite(a)]
    b = np.asarray(b, np.float64).ravel(); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    q = (np.arange(n_q) + 0.5) / n_q
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _mid(l2):
    m = (l2[..., 0] + l2[..., 2]) / 2.0
    return np.where(m > 0, m, np.nan)


def ret1(l2):                       # (Q,T-1) one-step log mid-returns (bps)
    lm = np.log(_mid(np.asarray(l2, np.float64)))
    return 1e4 * (lm[:, 1:] - lm[:, :-1])


def spread(l2):                     # (Q,T) best-ask - best-bid (price units)
    l2 = np.asarray(l2, np.float64)
    pa, pb = l2[..., 0], l2[..., 2]
    return np.where((pa > 0) & (pb > 0), np.abs(pa - pb), np.nan)


def discover_rows(run):
    f = []
    for x in os.listdir(run):
        if x.startswith("rollout_") and x.endswith(".npz"):
            n = x[len("rollout_"):-4]
            if n not in SKIP:
                f.append(n)
    return sorted(f)


def part_a_marginal(run):
    p = f"{run}/lobbench_scores/lobbench_results.json"
    if not os.path.exists(p):
        print("  (no lobbench_results.json)"); return
    d = json.load(open(p))
    rows = list(d.keys())
    print("== PART A: marginal LOB-Bench aggregate (point [95% bootstrap CI]) ==")
    fmt = lambda e: f"{e[0]:.4f} [{e[1][0]:.4f}, {e[1][1]:.4f}]"
    S = {r: (d[r]["summary"]["l1"][0], d[r]["summary"]["wasserstein"][0]) for r in rows}
    for r in rows:
        print(f"  {r:<16} mean-L1 {fmt(S[r][0]):<28} WS {fmt(S[r][1])}")
    a_l1, a_ws = S["anchor"]
    ov = lambda a, b: not (a[1][1] < b[1][0] or b[1][1] < a[1][0])
    print("  -- CI-overlap vs anchor (overlap=True => within eval noise) --")
    for r in rows:
        if r == "anchor":
            continue
        print(f"     {r:<16} L1 d={S[r][0][0]-a_l1[0]:+.4f} ov={ov(a_l1,S[r][0])} | "
              f"WS d={S[r][1][0]-a_ws[0]:+.4f} ov={ov(a_ws,S[r][1])}")
    # per-feature trade vs anchor for each treatment row (skip baselines/controls)
    a = d["anchor"]
    for r in rows:
        if r == "anchor" or any(t in r for t in ("bar", "ctrl", "null", "noop")):
            continue
        diffs = []
        for sect in ("uncond", "cond"):
            for f in a.get(sect, {}):
                av = a[sect][f].get("wasserstein")
                gv = d[r].get(sect, {}).get(f, {}).get("wasserstein")
                if isinstance(av, (int, float)) and isinstance(gv, (int, float)):
                    diffs.append((f, gv - av))
        diffs.sort(key=lambda x: x[1])
        wins = sum(1 for _, dl in diffs if dl < 0)
        print(f"  -- {r}: WS better on {wins}/{len(diffs)} features --")
        print(f"     wins : {', '.join(f'{f}({dl:+.2f})' for f,dl in diffs[:3])}")
        print(f"     losses: {', '.join(f'{f}({dl:+.2f})' for f,dl in diffs[-3:])}")


def part_b_horizon(run):
    p = f"{run}/test_eval_results.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))["rows"]
    print("\n== PART B(i): return-W1 vs aggregation horizon (panel; diffusive null=10x) ==")
    for row in d:
        if row in SKIP:
            continue
        pn = d[row]["panel"]
        r1, r100 = pn.get("ret_w1_1"), pn.get("ret_w1_100")
        if r1 and r100:
            print(f"  {row:<16} h1={r1:.4g} h100={r100:.4g} growth={r100/r1:.0f}x")


def part_b_depth(run):
    rows = discover_rows(run)
    real = np.load(f"{run}/rollout_real.npz")["l2"].astype(np.float64)
    R = {n: np.load(f"{run}/rollout_{n}.npz")["l2"].astype(np.float64) for n in rows}
    rR, sR = ret1(real), spread(real)
    Rret = {n: ret1(v) for n, v in R.items()}
    Rspr = {n: spread(v) for n, v in R.items()}
    Qh, T = real.shape[0] // 2, real.shape[1]
    bins = [(i, min(i + 50, T)) for i in range(0, T, 50)]
    print("\n== PART B(ii): depth-resolved 1-step drift  W1(model_t || real_t) ==")
    for feat, Rf, Sf, unit in (("RET(bps)", Rret, rR, 4), ("SPREAD", Rspr, sR, 1)):
        print(f"  [{feat}]  depth | {'floor':>8} " + " ".join(f"{n:>10}" for n in rows))
        for (a, b) in bins:
            fl = w1(Sf[:Qh, a:b], Sf[Qh:, a:b])
            vals = {n: w1(Rf[n][:, a:b], Sf[:, a:b]) for n in rows}
            print(f"   {a:>9}-{b:<3} {fl:>8.{unit}f} "
                  + " ".join(f"{vals[n]:>10.{unit}f}" for n in rows))


if __name__ == "__main__":
    print(f"# Priority-0 diagnostic on {RUN}\n")
    part_a_marginal(RUN)
    part_b_horizon(RUN)
    part_b_depth(RUN)
