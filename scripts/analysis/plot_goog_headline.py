#!/usr/bin/env python
"""Headline figures for the GOOG-arm sealed result (2026-07-02).

Fig 1  goog_arm_headline_bars.png — sealed B-day LOB-Bench aggregates (WS-21 | MEAN-L1)
       for anchor / shuffle-null winner / soup, with bootstrap 95% CIs. Small multiples,
       one metric per panel (one axis each), zero baseline, identity on the x labels
       (color is emphasis only: neutral grays for controls, accent blue for the soup).

Fig 2  goog_arm_ce_curve.png — critic cross-entropy over post-training (mean ± s.d.
       across the 3 real seeds), chance line at ln 2, anchor init at step 0, and the
       A-day winner-checkpoint window (steps 15–25) marked.

Reads docs/figures/goog_headline_data.json (written by the extraction step).
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(EXP, "docs", "figures")
DATA = json.load(open(os.path.join(FIG, "goog_headline_data.json")))

# ---- palette (dataviz reference instance, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"          # categorical slot 1 — the treatment (soup)
GRAY_ANCHOR = "#a5a49d"   # neutral bar for the anchor baseline
GRAY_NULL = "#cfcec7"     # lighter neutral for the shuffle null
GOOD = "#006300"          # success-delta text

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
})


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ================= Fig 1: sealed B-day bars (one per arm) =================
# Bars: anchor / shuffle null (single models, bootstrap 95% CI) vs the EGGROLL treatment
# reported the standard way — MEAN across the 3 seeds' sealed winners, error bar = ±1 s.d.
# across seeds. No soup (failed sealed on the SP500 arm). Standard matplotlib colors
# (C0/C1/C2).
metrics = [("WS-21 (Wasserstein aggregate)", "ws"),
           ("MEAN-L1 (21-feature aggregate)", "l1")]
C0, C1, C2 = "#1f77b4", "#ff7f0e", "#2ca02c"


def bars_figure(anchor, null_b, seeds_b, post_label, fname, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, key) in zip(axes, metrics):
        style_ax(ax)
        xs = np.arange(3)
        a_v, a_ci = anchor[key], anchor[key + "_ci"]
        n_v, n_ci = null_b[key], null_b[key + "_ci"]
        m_v, m_sd = seeds_b[key + "_mean"], seeds_b[key + "_sd"]
        vals = [a_v, n_v, m_v]
        ax.bar(xs, vals, width=0.55, color=[C0, C1, C2], zorder=2)
        ax.errorbar(xs[:2], vals[:2],
                    yerr=[[a_v - a_ci[0], n_v - n_ci[0]], [a_ci[1] - a_v, n_ci[1] - n_v]],
                    fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)
        ax.errorbar([xs[2]], [m_v], yerr=[[m_sd], [m_sd]], fmt="none", ecolor=INK2,
                    elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)
        ax.scatter([xs[2]] * 3, seeds_b[key + "_vals"], s=14, facecolor=SURFACE,
                   edgecolor=INK, linewidth=0.8, zorder=4)
        for x, v in zip(xs, vals):
            ax.text(x, 0.012, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=8, color="#ffffff", fontweight="bold", zorder=5)
        d = 100 * (m_v - a_v) / a_v
        ax.text(xs[2], m_v + m_sd + 0.012, f"{d:+.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=GOOD if d < 0 else "#c23b3b")
        ax.set_xticks(xs)
        ax.set_xticklabels(["Anchor\n(pretrained)", "Shuffle null\n(best ckpt)",
                            f"{post_label}\n(mean of 3 seeds)"], fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK, pad=8)
        ax.set_ylim(0, max(a_ci[1], n_ci[1], m_v + m_sd) * 1.22)
    fig.suptitle("Sealed B-day LOB-Bench — GOOG Feb-2026  (selection on A-days; B scored once; lower = better)",
                 fontsize=10, color=INK, y=1.00)
    fig.text(0.5, 0.005, subtitle + "\n"
             "anchor/null bars: bootstrap 95% CI · treatment bar: mean ± 1 s.d. across 3 seeds (open circles = individual seeds)",
             ha="center", fontsize=7.0, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = os.path.join(FIG, fname)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


bars_figure(DATA["anchor"], DATA["null_B"], DATA["seeds_B"],
            "EGGROLL post-trained",
            "goog_arm_headline_bars.png",
            "SP500-pretrained anchor (s28730) · GOOG Jan-2026 EGGROLL post-training · 131 B-day contexts")
if "sp500" in DATA:
    SP = DATA["sp500"]
    bars_figure(SP["anchor"], SP["null_B"], SP["seeds_B"],
                "EGGROLL post-trained",
                "sp500_arm_headline_bars.png",
                "SP500-pretrained anchor (s28730) · 478-ticker SP500 Jan-2026 EGGROLL post-training · 131 B-day contexts")
if "ganchor" in DATA:
    GA = DATA["ganchor"]
    bars_figure(GA["anchor"], GA["null_B"], GA["seeds_B"],
                "EGGROLL post-trained",
                "ganchor_arm_headline_bars.png",
                "GOOG-only-pretrained anchor (j5474870) · GOOG Jan-2026 EGGROLL post-training · 131 B-day contexts")

# ================= Fig 2: CE curve =================
LN2 = math.log(2)
steps = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def ce_band(key):
    seed_ce = DATA[key]
    m, s = [], []
    for st in steps:
        if st == 0:
            m.append(LN2); s.append(0.0)   # anchor init: untrained critic == chance by construction
        else:
            vs = [seed_ce[k][str(st)] for k in ("s0", "s1", "s2")]
            m.append(float(np.mean(vs))); s.append(float(np.std(vs)))
    return np.array(m), np.array(s)


mean, sd = ce_band("ce")
xs = np.array(steps)
HAVE_SP500 = "ce_sp500" in DATA
if HAVE_SP500:
    mean2, sd2 = ce_band("ce_sp500")
AQUA = "#1baf7a"          # categorical slot 2 — the SP500-corpus arm

fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
style_ax(ax)
# A-day winner-checkpoint window
ax.axvspan(15, 25, color=GRID, alpha=0.55, zorder=1)
ax.text(20, 0.512, "A-day winner\ncheckpoints (15–25)", ha="center", va="bottom",
        fontsize=7.5, color=MUTED)
# chance line
ax.axhline(LN2, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
ax.text(57.5, LN2 + 0.006, "chance (ln 2 ≈ 0.693)", ha="right", va="bottom",
        fontsize=7.5, color=MUTED)
# band + line + points (GOOG-post arm = blue; SP500-corpus arm = aqua)
ax.fill_between(xs, mean - sd, mean + sd, color=BLUE, alpha=0.15, linewidth=0, zorder=2)
ax.plot(xs, mean, color=BLUE, linewidth=2.0, zorder=3, label="GOOG-post arm (3 seeds)")
ax.scatter(xs, mean, s=16, color=BLUE, zorder=4)
if HAVE_SP500:
    ax.fill_between(xs, mean2 - sd2, mean2 + sd2, color=AQUA, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(xs, mean2, color=AQUA, linewidth=2.0, zorder=3, label="SP500-corpus arm (3 seeds)")
    ax.scatter(xs, mean2, s=16, color=AQUA, zorder=4)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK2)
# anchor init marker
ax.scatter([0], [LN2], s=42, facecolor=SURFACE, edgecolor=INK, linewidth=1.2, zorder=5)
ax.annotate("anchor init\n(critic at chance)", xy=(0, LN2), xytext=(2.2, 0.727),
            fontsize=7.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.8))
ax.set_xlim(-1.5, 58)
ax.set_ylim(0.485, 0.82)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
ax.set_xlabel("EGGROLL post-training step")
ax.set_ylabel("critic cross-entropy")
ax.set_title("Critic cross-entropy over post-training — mean ± s.d. across 3 seeds per arm",
             fontsize=10, color=INK, pad=8)
fig.text(0.5, 0.005,
         "below ln 2: critic separates real vs generated · above ln 2: critic exploited · bands = ±1 s.d. (s0/s1/s2)",
         ha="center", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out2 = os.path.join(FIG, "goog_arm_ce_curve.png")
fig.savefig(out2, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("wrote", out2)
