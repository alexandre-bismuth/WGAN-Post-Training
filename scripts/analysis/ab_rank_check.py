#!/usr/bin/env python
"""A-vs-B selection-integrity check + full sealed B-day table (GOOG arm or SP500 arm).

Reads the two consolidated winners JSONs (ab_selection_<A|B>_winners.json written by
consolidate_ab_selection.py) and reports:
  1. per-seed Spearman rank correlation of the step->WS-21 curve between splits
     (did A-day selection pick steps that are genuinely good on B, or was it split-lucky?)
  2. the sealed B-day values AT the A-selected steps (the honest per-seed numbers)
  3. the regret vs the (illegal) B-oracle step — how much was left on the table.

Usage: python ab_rank_check.py --a docs/results/robust_critic/selection_tables/ab_selection_A_winners.json \
                               --b docs/results/robust_critic/selection_tables/ab_selection_B_winners.json \
                               [--out docs/results/robust_critic/selection_tables/ab_integrity.md]
"""
from __future__ import annotations
import argparse
import json


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A = json.load(open(args.a))
    B = json.load(open(args.b))

    lines = ["# A/B selection integrity — rank correlation + sealed values at A-picked steps", ""]
    anchB = B.get("anchor", {})
    if anchB:
        lines.append(f"**anchor (B-days):** WS-21={anchB.get('ws', float('nan')):.4f}  "
                     f"L1={anchB.get('l1', float('nan')):.4f}")
        lines.append("")
    lines.append("| seed | Spearman ρ (A vs B, WS-21 over steps) | A-picked step | B WS-21 @A-pick | B-oracle step | B-oracle WS-21 | regret |")
    lines.append("|---|---|---|---|---|---|---|")
    for seed in A["per_seed"]:
        ca = A["per_seed"][seed]
        cb = B["per_seed"].get(seed, {})
        steps = sorted(set(ca) & set(cb), key=int)
        if not steps:
            continue
        wsa = [ca[s]["ws"] for s in steps]
        wsb = [cb[s]["ws"] for s in steps]
        rho = spearman(wsa, wsb)
        pick = str(A["winners"][seed]["step"])
        b_at_pick = cb.get(pick, {}).get("ws", float("nan"))
        oracle = min(cb, key=lambda s: cb[s]["ws"])
        b_oracle = cb[oracle]["ws"]
        lines.append(f"| {seed} | {rho:.3f} | {pick} | {b_at_pick:.4f} | {oracle} | "
                     f"{b_oracle:.4f} | {b_at_pick - b_oracle:+.4f} |")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)


if __name__ == "__main__":
    main()
