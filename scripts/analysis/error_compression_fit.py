"""Error-compression power-law fit + paper figure (§Absolute value variations).

Per-feature anchor WS level (x) vs seed-averaged relative change in % (y) on a chain TEST
dir, fit with the saturating power law y = y0 + A*x^p. Numpy-only estimator: (y0, A) are
profiled out by exact least squares on a fine p-grid — deterministic, and reproduces the
scipy multi-start curve_fit numbers of fig_feature_compression (unseen arm: R^2 0.741).

Login-node safe: reads one small JSON per row, no rollouts. Regenerate for the 10-seed run
by pointing --test_dir at its test_eval dir (pick rows are auto-discovered).

  python3 scripts/analysis/error_compression_fit.py \
      --test_dir /lus/.../panel28_evals/test_eval_5716072 --out paper/figures/abs_variations.png
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_unseen_paper_figures import C_POST, FEAT_LABEL   # style + labels stay in one place

RAW_DEFAULT = "/lustre/projects/public/shared/post_training_GAN/panel28_evals/test_eval_5716072"
LEARNED_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..",
                               "docs", "figures", "unseen_arm", "rescoreB_data.json")


def fit_pow(x, y, p_grid=np.linspace(-3.0, 3.0, 1201)):
    """min ||y - (y0 + A x^p)||^2 over (y0, A, p); linear in (y0, A) at fixed p."""
    best = None
    for p in p_grid:
        if abs(p) < 1e-9:
            continue
        X = np.column_stack([np.ones_like(x), np.power(x, p)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(((y - X @ coef) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, coef[0], coef[1], p)
    rss, y0, A, p = best
    r2 = 1.0 - rss / float(((y - y.mean()) ** 2).sum())
    return r2, y0, A, p


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def load_chain_test(test_dir):
    """Anchor per-feature WS + seed-avg rel delta (%) from a chain TEST dir's score JSONs."""
    def uncond(d):
        j = json.load(open(glob.glob(os.path.join(d, "lobbench_results.json"))[0]))
        return j[next(iter(j))]["uncond"]
    anch = uncond(os.path.join(test_dir, "lobbench_scores_testW_anchor"))
    pick_dirs = sorted(d for d in glob.glob(os.path.join(test_dir, "lobbench_scores_testW_*"))
                       if re.search(r"_s\d+pick$", d))
    assert pick_dirs, f"no *_s<N>pick score dirs in {test_dir}"
    picks = [uncond(d) for d in pick_dirs]
    feats = sorted(anch)
    x = np.array([anch[f]["wasserstein"] for f in feats])
    y = np.array([100.0 * (np.mean([pk[f]["wasserstein"] for pk in picks]) - anch[f]["wasserstein"])
                  / anch[f]["wasserstein"] for f in feats])
    return feats, x, y, len(picks)


def load_fig_json(path):
    """Anchor per-feature WS + seed-avg rel delta (%) from a rescoreB FIG_JSON."""
    j = json.load(open(path))
    pf = j["per_feature"]["wasserstein"]
    feats = sorted(pf)
    x = np.array([pf[f]["anchor_point"] for f in feats])
    y = np.array([100.0 * pf[f]["rel_delta"] for f in feats])
    n_seeds = len(pf[feats[0]].get("seed_points", {})) or j["meta"].get("n_seeds", 0)
    return feats, x, y, n_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default=RAW_DEFAULT)
    ap.add_argument("--fig_json", default=None,
                    help="rescoreB FIG_JSON to fit/plot instead of a chain TEST dir")
    ap.add_argument("--learned_json", default=LEARNED_DEFAULT,
                    help="rescoreB_data.json for the learned-critic comparison line (console only)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..",
                                                  "paper", "figures", "abs_variations.png"))
    args = ap.parse_args()

    if args.fig_json:
        feats, x, y, n_seeds = load_fig_json(args.fig_json)
    else:
        feats, x, y, n_seeds = load_chain_test(args.test_dir)
    r2, y0, A, p = fit_pow(x, y)
    rho = spearman(x, y)
    print(f"[raw arm] n_feats={len(feats)} n_seeds={n_seeds}  R2={r2:.3f}  rho={rho:.3f}  "
          f"fit: y = {y0:.1f} + {A:.2f} * x^{p:.2f}")

    if args.learned_json and os.path.isfile(args.learned_json):
        pf = json.load(open(args.learned_json))["per_feature"]["wasserstein"]
        xl = np.array([pf[f]["anchor_point"] for f in sorted(pf)])
        yl = np.array([100.0 * (pf[f]["rel_delta"] or 0.0) for f in sorted(pf)])
        r2l, *_ = fit_pow(xl, yl)
        print(f"[learned arm] R2={r2l:.3f}  rho={spearman(xl, yl):.3f}  (comparison line for the text)")

    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    xs = np.geomspace(x.min() * 0.85, x.max() * 1.18, 200)
    ax.plot(xs, y0 + A * np.power(xs, p), color="0.25", lw=1.2, zorder=2)
    ax.plot(x, y, "o", ms=4.5, color=C_POST, mec=C_POST, zorder=4)
    # OFI is the lowest point, so a downward offset drove its label into the x-axis and
    # into inter-arrival's label; place it to the LEFT at the same height, and send
    # inter-arrival's to the right, where the panel is empty at that depth.
    ann = {"spread": (4, 5, "left"), "log_inter_arrival_time": (6, -10, "left"),
           "ofi": (-8, -3, "right"), "bid_volume": (5, 2, "left")}
    for nm, (dx, dy, ha) in ann.items():
        if nm in feats:
            i = feats.index(nm)
            ax.annotate(FEAT_LABEL.get(nm, nm), (x[i], y[i]), fontsize=6.5, color="0.35",
                        xytext=(dx, dy), textcoords="offset points", ha=ha)
    sgn = "$-$" if A < 0 else "$+$"
    rho_s = f"$-${abs(rho):.2f}" if rho < 0 else f"{rho:.2f}"
    y0_s = f"$-${abs(y0):.1f}" if y0 < 0 else f"{y0:.1f}"
    ax.text(0.97, 0.95, f"$R^2$ = {r2:.2f}, Spearman $\\rho$ = {rho_s}\n"
            f"fit: $y$ = {y0_s} {sgn} {abs(A):.2f}$\\,x^{{{p:.2f}}}$",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", color="0.25")
    ax.set_xscale("log")
    # Labels are kept short enough to sit inside the figure box: the long forms
    # overflowed the single-column width in the paper build.
    ax.set_xlabel("Wasserstein distance per-feature after pretraining (log scale)",
                  fontsize=7.5)
    ax.set_ylabel(f"Relative change (%, {n_seeds}-seed avg)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
