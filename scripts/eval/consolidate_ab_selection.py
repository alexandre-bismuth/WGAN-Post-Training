#!/usr/bin/env python
"""Consolidate A/B-split LOB-Bench scores into a per-seed step->WS-21 table + argmin winner.

Reads the per-dir outputs that ab_split_scoring.py writes
(runs/test_eval_<jid>/lobbench_scores_days<split>/lobbench_results.json) and, for each
seed, merges the rows across its (typically two) eval dirs into one 10-step curve. Row
tags map to steps intrinsically: bs05->5 ... bs50->50, bbest->40 (s0's step-40 rollout),
anchor->reference. Selection = argmin WS-21 over the proj steps on the SELECTION split.

WS-21 and MEAN(L1) are extracted the same way ab_split_scoring.py builds its table:
  WS-21    = results[row]["summary"]["wasserstein"][0][0]
  MEAN(L1) = mean over uncond stats of results[row]["uncond"][stat]["l1"]

Usage:
  python consolidate_ab_selection.py --split A \
     --seeds "s0=5459924,5464409 s1=5474032,5474079 s2=5474082,5474086 shuf0=5474089,5474093" \
     --runs_dir runs --out docs/results/robust_critic/selection_tables/ab_selection_A.md
"""
from __future__ import annotations
import argparse, json, os, re, sys
import math


def row_to_step(row: str):
    if row == "anchor":
        return "anchor"
    if row == "bbest":
        return 40
    m = re.fullmatch(r"bs0*(\d+)", row)
    if m:
        return int(m.group(1))
    # unseen-arm row tags: unseen_<lane>_st00NN (e.g. unseen_s0_st0025)
    m = re.fullmatch(r".*_st0*(\d+)", row)
    return int(m.group(1)) if m else None


def extract(res: dict):
    """(WS-21, MEAN-L1) from one row's score_row output, mirroring ab_split_scoring."""
    summ = res.get("summary") or {}
    ws = float("nan")
    w = summ.get("wasserstein")
    if isinstance(w, list) and w and isinstance(w[0], list) and w[0]:
        ws = float(w[0][0])
    u = res.get("uncond") or {}
    l1s = [u[s]["l1"] for s in u if isinstance(u[s], dict) and "l1" in u[s]]
    l1 = float(sum(l1s) / len(l1s)) if l1s else float("nan")
    return ws, l1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["A", "B"])
    ap.add_argument("--seeds", required=True,
                    help='space list: "s0=5459924,5464409 s1=5474032,5474079 ..."')
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out", required=True, help="markdown table output path")
    args = ap.parse_args()

    seed_map = {}
    for tok in args.seeds.split():
        seed, jids = tok.split("=")
        seed_map[seed] = jids.split(",")

    per_seed = {}          # seed -> {step: {"ws":, "l1":}}
    anchor_ref = {}        # any dir's anchor row
    missing = []
    for seed, jids in seed_map.items():
        curve = {}
        for jid in jids:
            p = os.path.join(args.runs_dir, f"test_eval_{jid}",
                             f"lobbench_scores_days{args.split}", "lobbench_results.json")
            if not os.path.exists(p):
                missing.append(p)
                continue
            with open(p) as f:
                res = json.load(f)
            for row, rr in res.items():
                st = row_to_step(row)
                if st is None:
                    continue
                ws, l1 = extract(rr)
                if st == "anchor":
                    anchor_ref.setdefault("ws", ws); anchor_ref.setdefault("l1", l1)
                else:
                    curve[st] = {"ws": ws, "l1": l1}
        per_seed[seed] = curve

    # ---- build table + argmin winners ----
    steps = sorted({s for c in per_seed.values() for s in c})
    lines = [f"# A/B-split selection — split {args.split}  (WS-21, lower = better)", ""]
    if anchor_ref:
        lines.append(f"**anchor (reference):** WS-21={anchor_ref.get('ws', float('nan')):.4f}  "
                     f"MEAN-L1={anchor_ref.get('l1', float('nan')):.4f}")
        lines.append("")
    hdr = "| seed | " + " | ".join(f"s{st}" for st in steps) + " | **argmin step** | **best WS-21** |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(steps) + 3))
    winners = {}
    for seed, curve in per_seed.items():
        cells = []
        for st in steps:
            v = curve.get(st, {}).get("ws")
            cells.append(f"{v:.4f}" if v is not None and not math.isnan(v) else "—")
        valid = {st: c["ws"] for st, c in curve.items()
                 if c.get("ws") is not None and not math.isnan(c["ws"])}
        if valid:
            best_step = min(valid, key=valid.get)
            winners[seed] = {"step": best_step, "ws": valid[best_step],
                             "l1": curve[best_step]["l1"]}
            wcell = f"**{best_step}**"; bcell = f"**{valid[best_step]:.4f}**"
        else:
            wcell = "—"; bcell = "—"
        lines.append(f"| {seed} | " + " | ".join(cells) + f" | {wcell} | {bcell} |")

    lines += ["", "## Per-seed A-day winners (checkpoints to soup)"]
    for seed, w in winners.items():
        lines.append(f"- **{seed}**: step {w['step']}  (WS-21 {w['ws']:.4f}, MEAN-L1 {w['l1']:.4f})")
    if missing:
        lines += ["", "## MISSING score files (scored 0 rows here)"] + [f"- {m}" for m in missing]

    table = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(table)
    # also dump machine-readable winners next to the md
    with open(os.path.splitext(args.out)[0] + "_winners.json", "w") as f:
        json.dump({"split": args.split, "anchor": anchor_ref,
                   "winners": winners, "per_seed": per_seed}, f, indent=2)
    print(table)
    if missing:
        print(f"[consolidate] WARNING: {len(missing)} score file(s) missing", file=sys.stderr)


if __name__ == "__main__":
    main()
