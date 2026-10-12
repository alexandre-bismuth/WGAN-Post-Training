#!/usr/bin/env python
"""Critic cross-entropy curves across post-training runs (display figure).

Reads the eval-history breadcrumbs (`eval_history_<tag>.json` — a copy of the run's
latest_checkpoint.json holding the in-loop eval `history`) and plots the critic
cross-entropy side-diagnostic vs ES step, one line per run. CE is the symmetrised,
orientation-free logistic CE of the critic's real/fake scores: ~ln2≈0.693 at chance
(critic cannot separate), → 0 when fully separable. It is a DIAGNOSTIC, not a
selection signal (selection = held-out LOB-Bench WS-21).

Usage:
    python scripts/analysis/plot_ce_curves.py label=path/to/eval_history.json ... \
        [--out docs/figures/ce_curves.png]

Login-safe: pure json + matplotlib, no JAX.
"""
from __future__ import annotations
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = math.log(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="label=path pairs (path -> eval_history json)")
    ap.add_argument("--out", default="docs/figures/ce_curves.png")
    ap.add_argument("--title", default="Critic cross-entropy during EGGROLL post-training")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for spec in args.specs:
        label, path = spec.split("=", 1)
        with open(path) as f:
            hist = json.load(f)["history"]
        steps = [r["step"] for r in hist if "cross_entropy" in r]
        ce = [r["cross_entropy"] for r in hist if "cross_entropy" in r]
        ax.plot(steps, ce, marker="o", ms=3.5, lw=1.6, label=label)
    ax.axhline(LN2, color="grey", lw=1.0, ls="--", alpha=0.8)
    ax.text(0.995, LN2 + 0.004, "chance (ln 2)", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=8, color="grey")
    ax.set_xlabel("ES step")
    ax.set_ylabel("critic cross-entropy (nats)")
    ax.set_title(args.title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
