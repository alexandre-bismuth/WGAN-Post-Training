"""Paper figure for §Compounding error (single-column, Okabe-Ito, no title).

Reads a compounding_error.json produced by eggroll_gan.eval.compounding_error_analysis and
plots ONLY the floor-subtracted, day-binned curves with their day-clustered bootstrap 95%
CI bands (the working figure additionally overlays the raw normalized curves, which reads
as a second detached group — dropped here; the floor becomes the zero line).

Login-node safe: one small JSON in, one PNG out.

  python3 scripts/analysis/plot_compounding_paper.py \
      --json runs/compounding_error_5858607/compounding_error.json \
      --out paper/figures/compounding_error.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_unseen_paper_figures import C_ANCHOR, C_POST, C_WORSE  # style + rcParams

# Entity-stable series spec: name -> (legend label, color, linestyle, linewidth, band
# alpha). Names resolve against curves.binned (single rows) first, then curves.groups
# (seed-mean curves from --groups). Anchor black; controls thin grey (dashed vs dotted);
# treated curves Okabe-Ito color.
SERIES = {
    "anchor":   ("Pre-trained anchor",        "black",   "-",  1.8, 0.07),
    "eggroll":  ("Post-trained (10-seed mean)", C_POST,  "-",  1.8, 0.09),
    "null":     ("Selection-null",            "0.45",    "--", 1.1, 0.0),   # controls: no band
    "cpt":      ("Continued pre-training",    "0.45",    ":",  1.1, 0.0),
}
GRPO_SERIES = ("GRPO (10-seed mean)", "#56B4E9", "-", 1.6, 0.09)  # opt-in via --with_grpo
_ = C_ANCHOR, C_WORSE  # style module import also applies the shared rcParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ymax", type=float, default=0.13,
                    help="y-axis top (legend headroom); tune per data vintage")
    ap.add_argument("--with_grpo", action="store_true",
                    help="also draw the GRPO 10-seed mean curve")
    args = ap.parse_args()

    d = json.load(open(args.json))
    binned = d["curves"]["binned"]
    groups = d["curves"].get("groups", {})
    n_msgs = d["n_msgs"]
    n_bins = len(next(iter(binned.values()))["point"])
    centers = (np.arange(n_bins) + 0.5) * (n_msgs / n_bins)

    series = dict(SERIES)
    if args.with_grpo:
        order = list(series.items())
        order.insert(2, ("grpo", GRPO_SERIES))   # after eggroll, before controls
        series = dict(order)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.axhline(0, color="0.2", lw=0.8, ls=":", zorder=1)
    ax.annotate("real-vs-real floor", (0.99, 0.0), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                fontsize=7.5, color="0.35")
    for row, (label, color, ls, lw, ba) in series.items():
        b = binned.get(row) or groups.get(row, {}).get("binned")
        if b is None:
            continue
        # Day-resampling inflates the (convex) KL estimator, so the bootstrap replicate
        # distribution sits above the full-sample point estimate and raw percentile bands
        # can exclude their own curve. Keep the bootstrap WIDTH, recentre on the point.
        p = np.asarray(b["point"])
        if ba > 0:
            half = (np.asarray(b["hi"]) - np.asarray(b["lo"])) / 2.0
            ax.fill_between(centers, p - half, p + half, color=color, alpha=ba, lw=0, zorder=2)
        ax.plot(centers, p, color=color, ls=ls, lw=lw, label=label, zorder=3)
    ax.set_xlabel("Generated message index")
    ax.set_ylabel(r"KL(true$\,\|\,$gen)$\,/\,H$(true), floor-subtracted")
    ax.set_xlim(0, n_msgs)
    ax.set_ylim(top=args.ymax)   # fixed legend headroom
    ax.legend(ncol=2, loc="upper left", handlelength=1.8, columnspacing=1.0,
              borderaxespad=0.2)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
