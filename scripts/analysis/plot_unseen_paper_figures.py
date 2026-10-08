#!/usr/bin/env python
"""Paper figures for the unseen arm (login-safe: pure json + numpy + matplotlib).

Inputs (each figure renders only when its input exists):
  A. docs/figures/unseen_arm/rescoreB_data.json      (single-program B re-scoring, job R2)
       -> fig_headline_bars      WS-21 + Mean L1: anchor vs 3-seed average, paired
                                 WINDOW-level bootstrap 95% CIs, one program, one universe
       -> fig_feature_top5       top-5 improved / top-5 worsened WS features, anchor vs
                                 3-seed average with 95% CIs
  B. runs/gen_ce_curve_<job>/gen_ce_curve.json       (generator CE job R3; --ce_json)
     + runs/s5b_mnode_549338{7,8,9}/eval_history_unseen_s{0,1,2}.json (training histories)
       -> fig_training_dynamics  left axis: held-out generator CE (nats/token);
                                 right axis: critic real-vs-fake AUC. 3-seed mean with
                                 95% t-CIs (n=3, t=4.303).

Also prints the OFI gate for the probe-port decision: seed-avg paired WS delta on the
`ofi` feature, its window-level CI, and the |z| >= 2 verdict.

Outputs: docs/figures/unseen_arm/<fig>.png + .pdf (300 dpi).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGD_DEFAULT = os.path.join(EXP, "docs", "figures", "unseen_arm")

C_ANCHOR = "#8C8C8C"      # neutral grey — the frozen pretrained model
C_POST = "#0072B2"        # colourblind-safe blue — adversarially post-trained
C_WORSE = "#D55E00"       # vermillion accent — regressions
T95_N3 = 4.302652729911275  # two-sided 97.5% Student-t quantile, dof = 2

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# lob_bench feature names -> compact display labels
FEAT_LABEL = {
    "spread": "spread", "orderbook_imbalance": "book imbalance",
    "log_inter_arrival_time": "inter-arrival time", "log_time_to_cancel": "time to cancel",
    "ask_volume_touch": "ask touch volume", "bid_volume_touch": "bid touch volume",
    "ask_volume": "ask volume (10 lvl)", "bid_volume": "bid volume (10 lvl)",
    "limit_ask_order_depth": "ask limit depth", "limit_bid_order_depth": "bid limit depth",
    "ask_cancellation_depth": "ask cancel depth", "bid_cancellation_depth": "bid cancel depth",
    "limit_ask_order_levels": "ask limit levels", "limit_bid_order_levels": "bid limit levels",
    "ask_cancellation_levels": "ask cancel levels", "bid_cancellation_levels": "bid cancel levels",
    "vol_per_min": "volume / minute", "ofi": "order-flow imbalance (OFI)",
    "ofi_up": "OFI | mid up", "ofi_stay": "OFI | mid flat", "ofi_down": "OFI | mid down",
}


def _ci_whisker(ax, x, ci, color, lw=1.0, cap=0.05):
    """Draw the CI as an explicit [lo, hi] whisker with caps at position x.

    Decoupled from the point estimate: bootstrap CIs for divergence statistics (L1) can be
    biased so the point sits just outside its own interval — a from-point error bar would go
    negative. This always draws the true [lo, hi]."""
    lo, hi = ci
    ax.plot([x, x], [lo, hi], color=color, lw=lw, zorder=6, solid_capstyle="butt")
    for y in (lo, hi):
        ax.plot([x - cap, x + cap], [y, y], color=color, lw=lw, zorder=6)


def _axes_mid(fig, axes):
    """Horizontal centre of the axes span in figure coordinates. Suptitles/captions placed
    at figure x=0.5 look off-centre because the y-label pads only the left side."""
    fig.canvas.draw()
    x0 = min(a.get_position().x0 for a in axes)
    x1 = max(a.get_position().x1 for a in axes)
    return 0.5 * (x0 + x1)


def _save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{stem}.png")
    fig.savefig(p)
    try:
        os.chmod(p, 0o664)
    except OSError:
        pass
    plt.close(fig)
    print(f"[plot] wrote {p}")


def _panel_label(ax, s):
    """Paper-convention bold panel tag, outside the top-left corner."""
    ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ----------------------------------------------------------------------------------------
# Figure A1 — headline bars (WS-21 + Mean L1, anchor vs 3-seed average)
# ----------------------------------------------------------------------------------------
def fig_headline(data, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8), gridspec_kw={"wspace": 0.34})
    ylabels = {"wasserstein": "WS-21 (lower is better)",
               "l1": "Mean L1 (lower is better)"}
    for pane, (ax, metric) in zip("ab", zip(axes, ("wasserstein", "l1"))):
        anc = data["rows"]["anchor"][metric]
        sa = data["seed_avg"][metric]
        a_bar = anc.get("boot_median_rollout", anc["point"])
        s_bar = sa.get("boot_median_rollout", sa["point"])
        xs = [0.0, 0.62]
        ax.bar(xs[0], a_bar, width=0.42, color=C_ANCHOR, label="pretrained anchor")
        ax.bar(xs[1], s_bar, width=0.42, color=C_POST,
               label="post-trained (3-seed avg)")
        _ci_whisker(ax, xs[0], anc["ci_rollout"], "#333333", cap=0.06)
        _ci_whisker(ax, xs[1], sa["ci_rollout"], "#333333", cap=0.06)
        lo = min(anc["ci_rollout"][0], sa["ci_rollout"][0])
        hi = max(anc["ci_rollout"][1], sa["ci_rollout"][1])
        pad = 0.12 * (hi - lo)
        ax.text(xs[1], sa["ci_rollout"][1] + 0.45 * pad,
                f"{100 * sa['rel_delta']:+.1f}%", ha="center", va="bottom",
                fontsize=8, color=C_POST, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(["anchor", "post-trained"])
        ax.set_ylabel(ylabels[metric])
        ax.set_ylim(lo - pad, hi + 2.2 * pad)
        _panel_label(ax, f"({pane})")
    handles, labels = axes[0].get_legend_handles_labels()
    xm = _axes_mid(fig, axes)
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(xm, 1.09))
    _save(fig, outdir, "fig_headline_bars")


# ----------------------------------------------------------------------------------------
# Figure A2 — per-feature top-5 improvements / top-5 regressions (WS)
# ----------------------------------------------------------------------------------------
def fig_feature_top5(data, outdir, k=5):
    pf = data["per_feature"]["wasserstein"]
    order = sorted(pf, key=lambda n: pf[n]["delta_point"])
    improved, worsened = order[:k], order[-k:][::-1]

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.2))
    for pane, (ax, feats, accent) in zip("ab", (
            (axes[0], improved, C_POST),
            (axes[1], worsened, C_WORSE))):
        xs = np.arange(len(feats), dtype=float)
        w = 0.36
        for i, nm in enumerate(feats):
            p = pf[nm]
            a_ci = p.get("anchor_ci_rollout", [p["anchor_point"]] * 2)
            s_ci = p.get("seed_avg_ci_rollout", [p["seed_avg_point"]] * 2)
            a_bar = p.get("anchor_boot_median_rollout", p["anchor_point"])
            s_bar = p.get("seed_avg_boot_median_rollout", p["seed_avg_point"])
            ax.bar(xs[i] - w / 2, a_bar, width=w, color=C_ANCHOR)
            ax.bar(xs[i] + w / 2, s_bar, width=w, color=accent)
            _ci_whisker(ax, xs[i] - w / 2, a_ci, "#333333", lw=0.9, cap=0.05)
            _ci_whisker(ax, xs[i] + w / 2, s_ci, "#333333", lw=0.9, cap=0.05)
            top = max(a_ci[1], s_ci[1])
            ax.text(xs[i], top * 1.03, f"{100 * (p['rel_delta'] or 0):+.0f}%",
                    ha="center", fontsize=7.5, color=accent, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([FEAT_LABEL.get(n, n) for n in feats], fontsize=7.5)
        ax.set_ylabel("Wasserstein distance")
        ax.margins(y=0.18)
        _panel_label(ax, f"({pane})")
    fig.tight_layout()
    from matplotlib.patches import Patch
    xm = _axes_mid(fig, axes)
    fig.legend(handles=[Patch(color=C_ANCHOR, label="pretrained anchor"),
                        Patch(color=C_POST, label="post-trained (3-seed avg), improved"),
                        Patch(color=C_WORSE, label="post-trained (3-seed avg), worsened")],
               loc="upper center", ncol=3, bbox_to_anchor=(xm, 1.02))
    _save(fig, outdir, "fig_feature_top5")


# ----------------------------------------------------------------------------------------
# Figure A3 — error compression: anchor WS level vs post-training change
# ----------------------------------------------------------------------------------------
def fig_feature_compression(data, outdir):
    """Anchor per-feature WS level vs relative change after post-training. Power-law
    fit y = y0 + A*x^p (multi-start curve_fit): the response saturates — high-error
    features improve toward a common floor while features near the eval noise floor
    swing wildly in relative terms — so a saturating, heavy-tail-compatible form fits
    better than a straight line in ln(x) (R^2 0.74 vs 0.63; exponential decay fits
    identically but its decay scale is unidentified, pinned at the data's left edge).
    R^2 (of the fit) and Spearman rho (rank-based, transform-invariant) quoted
    in-plot. One marker class for all features."""
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    from scipy.optimize import curve_fit
    pf = data["per_feature"]["wasserstein"]
    names = sorted(pf)
    a = np.array([pf[n]["anchor_point"] for n in names])
    rel = np.array([100.0 * (pf[n]["rel_delta"] or 0.0) for n in names])

    def _pow(xv, y0, A, p):
        return y0 + A * np.power(xv, p)

    best = None
    for p0 in ([0.0, 1.0, -1.2], [0.0, -10.0, -0.3], [-20.0, 30.0, 0.3]):
        try:
            popt, _ = curve_fit(_pow, a, rel, p0=p0, maxfev=20000)
            rss = float(((rel - _pow(a, *popt)) ** 2).sum())
            if best is None or rss < best[0]:
                best = (rss, popt)
        except RuntimeError:
            pass
    rss, (y0, A, p) = best
    r2 = 1.0 - rss / ((rel - rel.mean()) ** 2).sum()
    rho = _spearman(a, rel)

    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    xlo, xhi = a.min() * 0.85, a.max() * 1.18
    xs = np.geomspace(xlo, xhi, 200)
    ax.plot(xs, _pow(xs, y0, A, p), color="0.25", lw=1.2, zorder=2)
    ax.plot(a, rel, "o", ms=4.5, color=C_POST, mec=C_POST, zorder=4)
    ann = {"bid_cancellation_levels": (4, 5, "left"),
           "log_inter_arrival_time": (4, 5, "left"),
           "ofi": (-4, -12, "right"),
           "log_time_to_cancel": (-4, 6, "right")}
    for nm, (dx, dy, ha) in ann.items():
        i = names.index(nm)
        ax.annotate(FEAT_LABEL.get(nm, nm), (a[i], rel[i]), fontsize=6.5, color="0.35",
                    xytext=(dx, dy), textcoords="offset points", ha=ha)
    sgn = "$-$" if A < 0 else "$+$"
    rho_s = (f"$-${abs(rho):.2f}" if rho < 0 else f"{rho:.2f}")
    y0_s = (f"$-${abs(y0):.1f}" if y0 < 0 else f"{y0:.1f}")
    ax.text(0.97, 0.95, f"$R^2$ = {r2:.2f}, Spearman $\\rho$ = {rho_s}\n"
            f"fit: $y$ = {y0_s} {sgn} {abs(A):.2f}$\\,x^{{{p:.2f}}}$",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", color="0.25")
    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5) if xlo <= t <= xhi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("anchor per-feature Wasserstein distance (log scale)")
    ax.set_ylabel("relative change after post-training (%)")
    fig.tight_layout()
    _save(fig, outdir, "fig_feature_compression")


# ----------------------------------------------------------------------------------------
# OFI gate for the probe-port decision
# ----------------------------------------------------------------------------------------
def ofi_gate(data):
    p = data["per_feature"]["wasserstein"].get("ofi")
    if not p or "delta_ci_rollout" not in p:
        print("[ofi-gate] no per-feature ofi delta available")
        return
    lo, hi = p["delta_ci_rollout"]
    se = (hi - lo) / (2 * 1.959963984540054)
    z = p["delta_point"] / se if se > 0 else float("nan")
    verdict = "IMPROVED > 2 sigma" if (z <= -2) else (
        "WORSENED > 2 sigma" if z >= 2 else "not significant at 2 sigma")
    print(f"[ofi-gate] ofi WS: anchor {p['anchor_point']:.4f} -> seed-avg "
          f"{p['seed_avg_point']:.4f}, delta {p['delta_point']:+.4f} "
          f"({100 * (p['rel_delta'] or 0):+.1f}%), window-CI [{lo:+.4f},{hi:+.4f}], "
          f"z = {z:+.2f} -> {verdict}")


# ----------------------------------------------------------------------------------------
# Figure B — training dynamics: generator CE (left) vs critic AUC (right)
# ----------------------------------------------------------------------------------------
def _seed_band(mat):
    """[n_seed, n_steps] -> (mean, half-width of the 95% t-CI, n=3)."""
    m = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0, ddof=1)
    return m, T95_N3 * sd / np.sqrt(mat.shape[0])


def fig_training_dynamics(ce_json, hist_paths, outdir):
    ce = json.load(open(ce_json)) if ce_json and os.path.isfile(ce_json) else None
    hists = [json.load(open(p))["history"] for p in hist_paths if os.path.isfile(p)]
    if not hists:
        print("[plot] no eval histories — skipping training-dynamics figure")
        return
    steps_h = [r["step"] for r in hists[0]]
    auc = np.array([[r["auc"] for r in h] for h in hists], float)
    auc_m, auc_e = _seed_band(auc)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)

    if ce is not None:
        lanes = sorted({v["lane"] for v in ce["rows"].values() if v.get("lane")})
        steps_c = [0] + sorted({v["step"] for v in ce["rows"].values() if v["step"] > 0})
        anchor_ce = ce["rows"]["anchor"]["ce_mean"]
        mat = np.full((len(lanes), len(steps_c)), np.nan)
        for i, ln in enumerate(lanes):
            mat[i, 0] = anchor_ce
            for j, st in enumerate(steps_c[1:], start=1):
                r = ce["rows"].get(f"{ln}_st{st:04d}")
                if r:
                    mat[i, j] = r["ce_mean"]
        ce_m, ce_e = _seed_band(mat)
        ax.plot(steps_c, ce_m, "-o", ms=3, color=C_POST, label="generator CE (held-out)")
        ax.fill_between(steps_c, ce_m - ce_e, ce_m + ce_e, color=C_POST, alpha=0.18, lw=0)
        ax.axhline(anchor_ce, color=C_POST, lw=0.8, ls=":", alpha=0.7)
        ax.annotate("anchor", (steps_c[-1], anchor_ce), fontsize=7, color=C_POST,
                    xytext=(-2, 4), textcoords="offset points", ha="right")
        ax.set_ylabel("generator CE on real data (nats/token)", color=C_POST)
        ax.tick_params(axis="y", colors=C_POST)
    else:
        ax.set_ylabel("generator CE (pending)", color=C_POST)
        ax.set_yticks([])
        print("[plot] CE json missing — left axis left empty")

    ax2.plot(steps_h, auc_m, "-s", ms=3, color=C_WORSE, label="critic AUC (real vs generated)")
    ax2.fill_between(steps_h, auc_m - auc_e, auc_m + auc_e, color=C_WORSE, alpha=0.18, lw=0)
    ax2.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax2.annotate("chance (0.5)", (steps_h[-1], 0.5), fontsize=7, color="0.45",
                 xytext=(-2, -10), textcoords="offset points", ha="right")
    ax2.set_ylabel("critic real-vs-generated AUC", color=C_WORSE)
    ax2.tick_params(axis="y", colors=C_WORSE)

    ax.set_xlabel("ES post-training step")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=7.5,
              bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    _save(fig, outdir, "fig_training_dynamics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore_json", default=os.path.join(FIGD_DEFAULT, "rescoreB_data.json"))
    ap.add_argument("--ce_json", default=None, help="gen_ce_curve.json from the CE job")
    ap.add_argument("--hist", nargs="*", default=[
        os.path.join(EXP, "runs", f"s5b_mnode_{j}", f"eval_history_unseen_s{s}.json")
        for s, j in ((0, 5493387), (1, 5493388), (2, 5493389))])
    ap.add_argument("--outdir", default=FIGD_DEFAULT)
    args = ap.parse_args()

    if os.path.isfile(args.rescore_json):
        data = json.load(open(args.rescore_json))
        fig_headline(data, args.outdir)
        fig_feature_top5(data, args.outdir)
        fig_feature_compression(data, args.outdir)
        ofi_gate(data)
    else:
        print(f"[plot] {args.rescore_json} missing — headline/feature figures skipped "
              "(run the rescoreB chain first)")
    fig_training_dynamics(args.ce_json, args.hist, args.outdir)


if __name__ == "__main__":
    main()
