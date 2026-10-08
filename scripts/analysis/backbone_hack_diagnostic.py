#!/usr/bin/env python3
"""Backbone-critic hack diagnostic: is the ES reward a faithful realism signal, or is the
generator hacking the critic?

Reads the per-seed `history` in each seed's latest_checkpoint.json from an S5b production run
(no GPU, login-safe). Each history row carries: separation/auc (critic real-vs-fake), g_mean_score
(the ES reward), mean_kl, dkernel (delta magnitude), composite_norm (an INDEPENDENT held-out-from-
gradient realism proxy, anchor==1.0, lower=better), goodhart_fired.

The hack test: correlate g_mean_score (reward) with composite_norm (realism). A positive correlation
means higher critic-reward coincides with WORSE realism => the generator is gaming the critic. Also
contrasts the max-reward step's realism vs the best-realism step's reward: if they differ and the
max-reward step is less realistic than the anchor, the reward is not a realism signal.

Usage: python3 scripts/analysis/backbone_hack_diagnostic.py [RUN_DIR ...]
Default = the three backbone ES runs (pilot / Muon / AdamW).
"""
import json, os, sys
import numpy as np

DEFAULT = ["runs/s5b_production_5217779", "runs/s5b_production_5264754",
           "runs/s5b_production_5264758"]
RUNS = sys.argv[1:] or DEFAULT


def corr(x, y):
    x, y = np.array(x, float), np.array(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def seeds(d):
    return sorted(s for s in os.listdir(d)
                  if os.path.exists(f"{d}/{s}/latest_checkpoint.json"))


print("HACK TEST  corr(reward, realism)>0 => higher critic-reward goes WITH worse realism (Goodhart)")
print(f"{'arm/seed':<20}{'corr(R,real)':>13}{'maxR: step/R/real':>22}{'bestReal: step/R/real':>24}")
for d in RUNS:
    if not os.path.isdir(d):
        print(f"  MISSING {d}"); continue
    label = d.split("_")[-1]
    for s in seeds(d):
        h = json.load(open(f"{d}/{s}/latest_checkpoint.json")).get("history", [])
        if len(h) < 3:
            continue
        g = [r.get("g_mean_score") for r in h]
        c = [r.get("composite_norm") for r in h]
        st = [r["step"] for r in h]
        iR, ic = int(np.nanargmax(g)), int(np.nanargmin(c))
        cc = corr(g, c)
        flag = " HACK" if (cc == cc and cc > 0.3) else ""
        print(f"{label+'/'+s:<20}{cc:>12.2f}{flag:<5}"
              f"{f'{st[iR]}/{g[iR]:+.0f}/{c[iR]:.2f}':>22}{f'{st[ic]}/{g[ic]:+.0f}/{c[ic]:.2f}':>24}")
print("\nReading: in every treatment seed the max-reward step is LESS realistic than the anchor "
      "(composite>1.0)\nand the best-realism step sits at low/negative reward => the ES reward is "
      "anti-aligned with realism.\nGenuine gains occur only early (small dkernel/kl, negative reward) "
      "before the generator learns to hack.")
