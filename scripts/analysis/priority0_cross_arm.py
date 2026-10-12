#!/usr/bin/env python3
"""Priority-0 CROSS-ARM consolidation: does the depth-resolved exposure-bias picture
(marginal tie / null vs real spread-drift repair / OFI-vs-spread trade) generalize across
the full {EGGROLL, GRPO} x {backbone, stylized} grid?

Login-safe, no GPU. Reads existing held-out eval artifacts only. Within-run CRN holds
(same contexts), so all treatment-vs-anchor and vs-control ratios are computed WITHIN a run.

Usage: python3 scripts/analysis/priority0_cross_arm.py
"""
import json, os
import numpy as np

# (run dir, arm, critic, note)
RUNS = [
    ("test_eval_5267775", "EGGROLL", "backbone", "Muon/AdamW soups"),
    ("test_eval_5223670", "EGGROLL", "backbone", "pilot per-seed"),
    ("test_eval_5286364", "EGGROLL", "stylized", "per-seed"),
    ("test_eval_5297033", "GRPO",    "backbone", "soup (vs own null)"),
    ("test_eval_5286365", "GRPO",    "stylized", "per-seed"),
    ("test_eval_5287549", "GRPO",    "stylized", "soup"),
]
BASE = "runs"
SKIP = {"real", "merge_noop"}
EXPO = ("spread", "spread | time", "spread | volatility", "log_time_to_cancel",
        "log_inter_arrival_time", "limit_bid_order_depth", "limit_ask_order_depth",
        "bid_cancellation_depth", "ask_cancellation_depth")
OFI = ("ofi", "ofi_up", "ofi_stay", "ofi_down")


def w1(a, b, n_q=2048):
    a = np.asarray(a, np.float64).ravel(); a = a[np.isfinite(a)]
    b = np.asarray(b, np.float64).ravel(); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    q = (np.arange(n_q) + 0.5) / n_q
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def spread(l2):
    l2 = np.asarray(l2, np.float64); pa, pb = l2[..., 0], l2[..., 2]
    return np.where((pa > 0) & (pb > 0), np.abs(pa - pb), np.nan)


def deep_spread_w1(run, rows):
    """W1(model spread || real spread) over the deepest depth bin (msgs 450-500)."""
    real = spread(np.load(f"{run}/rollout_real.npz")["l2"])
    out = {}
    for n in rows:
        s = spread(np.load(f"{run}/rollout_{n}.npz")["l2"])
        out[n] = w1(s[:, 450:500], real[:, 450:500])
    return out


def grp_delta(d, treat, feats):
    """mean (treat - anchor) WS over a feature group (negative = treat better)."""
    a, t, ds = d["anchor"], d[treat], []
    for sect in ("uncond", "cond"):
        for f in a.get(sect, {}):
            if f in feats:
                av = a[sect][f].get("wasserstein")
                tv = t.get(sect, {}).get(f, {}).get("wasserstein")
                if isinstance(av, (int, float)) and isinstance(tv, (int, float)):
                    ds.append(tv - av)
    return float(np.mean(ds)) if ds else float("nan")


print("# Priority-0 cross-arm consolidation  ({EGGROLL,GRPO} x {backbone,stylized})\n")
hdr = (f"{'arm':<8}{'critic':<10}{'row':<16}{'mL1 Δ':>8}{'ov':>4}{'WS Δ':>8}{'ov':>4}"
       f"{'spr@deep':>10}{'/anchor':>9}{'/ctrl':>8}{'OFI Δ':>9}{'EXPO Δ':>9}")
print(hdr); print("-" * len(hdr))

for run, arm, critic, note in RUNS:
    rd = f"{BASE}/{run}"
    if not os.path.isdir(rd):
        print(f"{arm:<8}{critic:<10}MISSING {run}"); continue
    d = json.load(open(f"{rd}/lobbench_scores/lobbench_results.json"))
    allrows = [r[len("rollout_"):-4] for r in os.listdir(rd)
               if r.startswith("rollout_") and r.endswith(".npz") and r[len("rollout_"):-4] not in SKIP]
    treats = [r for r in allrows if not any(t in r for t in ("bar", "ctrl", "null"))]
    ctrls = [r for r in allrows if "ctrl" in r]
    dsw = deep_spread_w1(rd, sorted(set(treats + ctrls + ["anchor"])))
    anc_deep = dsw.get("anchor", float("nan"))
    ctrl_deep = np.mean([dsw[c] for c in ctrls]) if ctrls else float("nan")
    aL1 = d["anchor"]["summary"]["l1"][0]; aWS = d["anchor"]["summary"]["wasserstein"][0]
    ov = lambda a, b: "Y" if not (a[1][1] < b[1][0] or b[1][1] < a[1][0]) else "."
    print(f"  [{run} | {arm} x {critic} | {note}; anchor spr@deep={anc_deep:.0f}, ctrl spr@deep={ctrl_deep:.0f}]")
    for r in sorted(treats):
        L1 = d[r]["summary"]["l1"][0]; WS = d[r]["summary"]["wasserstein"][0]
        sd = dsw[r]
        print(f"{arm:<8}{critic:<10}{r:<16}{L1[0]-aL1[0]:>+8.4f}{ov(aL1,L1):>4}"
              f"{WS[0]-aWS[0]:>+8.4f}{ov(aWS,WS):>4}{sd:>10.0f}{sd/anc_deep:>9.2f}"
              f"{sd/ctrl_deep:>8.2f}{grp_delta(d,r,OFI):>+9.3f}{grp_delta(d,r,EXPO):>+9.3f}")
print("\nLegend: Δ = vs anchor (negative=better); ov=CI overlaps anchor (Y=within noise).")
print("spr@deep = spread W1 vs real over msgs 450-500; /anchor<1 and /ctrl<1 => repairs free-running drift.")
print("OFI Δ, EXPO Δ = mean WS delta vs anchor over order-flow vs exposure-bias feature groups.")
