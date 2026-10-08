#!/usr/bin/env python
"""A/B day-split LOB-Bench scoring over saved test_eval rollouts (protocol of 2026-07-02).

Selection-on-A / sealed-final-on-B: the 256 CRN eval contexts are split BY TRADING DAY
(windows within a day share regime info — day is the honest exchangeability unit):

    A (selection) = 2026-02-02, 2026-02-04, 2026-02-06, 2026-02-10   (interleaved,
    B (final)     = 2026-02-03, 2026-02-05, 2026-02-09, 2026-02-11    balances weeks/weekdays)

Per-seed WS-21 argmin over the 5-step checkpoints is computed on A ONLY; the winners +
soup get ONE B pass as the headline (unbiased by selection). Requires ctx_days.json (the
deterministic ctx->day mapping dumped by test_eval; portable across same-config evals —
rollout_real.npz is byte-identical across jobs).

Run with the lobbench venv python (pandas), CSV farm on node-local scratch (never Lustre):

    third_party/lobbench_venv/bin/python scripts/eval/ab_split_scoring.py \
        --eval_dir runs/test_eval_<jid> --ctx_days runs/test_eval_<jid2>/ctx_days.json \
        --split A --rows anchor,bs05,... --work /tmp/ab_<jid> [--out_tag daysA]

Output: <eval_dir>/lobbench_scores_<out_tag>/lobbench_results.json + lobbench_table.md
(same format as the full-panel scorer; summary = aggregate L1 / WS-21 with bootstrap CIs).
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, EXP)

# Default = the ORIGINAL Feb-only 8-day protocol (2026-07-02 sealed-arm results reproduce
# bit-for-bit). The 28-day paper panel passes --split_json docs/results/unseen_arm/panel_ab_split.json
# (committed 2026-07-03 with seed 20260704, BEFORE any model scoring on that panel).
SPLITS = {
    "A": ("2026-02-02", "2026-02-04", "2026-02-06", "2026-02-10"),
    "B": ("2026-02-03", "2026-02-05", "2026-02-09", "2026-02-11"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="test_eval out dir holding rollout_*.npz")
    ap.add_argument("--ctx_days", required=True, help="ctx_days.json (any same-config eval)")
    ap.add_argument("--split", required=True, choices=sorted(SPLITS) + ["ALL"],
                    help="A / B day subset, or ALL = union of both (window-level protocols)")
    ap.add_argument("--rows", required=True, help="comma list of row tags (rollout_<tag>.npz)")
    ap.add_argument("--work", required=True, help="scratch dir for the CSV farm (node-local)")
    ap.add_argument("--out_tag", default=None, help="default: days<split>")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--ticker", default="GOOG")
    ap.add_argument("--split_json", default=None,
                    help="pre-committed split file (panel_ab_split.json: A_selection/B_sealed "
                         "day lists) — overrides the built-in Feb-only SPLITS")
    ap.add_argument("--drop_first", type=int, default=0,
                    help="exclude the first N draw positions (rollout array indices < N) — "
                         "window-level protocol: positions 0..N-1 of the seed-0 CRN draw were "
                         "consumed by checkpoint selection (prefix-nesting verified 2026-07-05), "
                         "so the test set is the disjoint remainder")
    ap.add_argument("--by_day", action="store_true",
                    help="convert each context under its TRUE trading day (CSV date = ctx day) "
                         "so seq.date is real and eggroll_gan.eval.block_bootstrap can "
                         "day-cluster; without it every CSV gets the synthetic default date "
                         "and the day-cluster bootstrap silently degenerates to 1 day. "
                         "Pooled point estimates are order-invariant -> identical either way.")
    args = ap.parse_args()

    from eggroll_gan.eval.lobbench_adapter import convert_row
    from eggroll_gan.eval.run_lobbench import score_row, _jsonable

    splits = SPLITS
    if args.split_json:
        with open(args.split_json) as f:
            sj = json.load(f)
        splits = {"A": tuple(sj["A_selection"]), "B": tuple(sj["B_sealed"])}
        print(f"[ab] split file {args.split_json} (committed: {sj.get('committed', '?')}): "
              f"A={len(splits['A'])}d B={len(splits['B'])}d", flush=True)

    with open(args.ctx_days) as f:
        cd = json.load(f)
    days = np.asarray(cd["day"])
    want = splits["A"] + splits["B"] if args.split == "ALL" else splits[args.split]
    sel = np.flatnonzero(np.isin(days, want))
    if args.drop_first:
        n0 = len(sel)
        sel = sel[sel >= args.drop_first]
        print(f"[ab] drop_first {args.drop_first}: {n0} -> {len(sel)} contexts "
              f"(selection-window-disjoint)", flush=True)
    n_real = np.load(os.path.join(args.eval_dir, "rollout_real.npz"))["msgs"].shape[0]
    assert len(days) == n_real, f"ctx_days len {len(days)} != rollout Q {n_real} — wrong mapping?"
    assert len(sel) > 0, f"no contexts on split {args.split} days {want}"
    print(f"[ab] split {args.split}: {len(sel)}/{len(days)} contexts on {want}", flush=True)

    real = np.load(os.path.join(args.eval_dir, "rollout_real.npz"))
    real_sub = {"msgs": real["msgs"][sel], "l2": real["l2"][sel]}

    out_tag = args.out_tag or f"days{args.split}"
    bench = os.path.join(args.work, f"bench_{out_tag}")
    results = {}
    rows = [r for r in args.rows.split(",") if r]
    for row in rows:
        p = os.path.join(args.eval_dir, f"rollout_{row}.npz")
        if not os.path.exists(p):
            print(f"[ab] WARN no rollout for '{row}' — skipped", flush=True)
            continue
        g = np.load(p)
        gen_sub = {"msgs": g["msgs"][sel], "l2": g["l2"][sel]}
        if args.by_day:
            for d in sorted({str(x) for x in days[sel]}):
                sd = sel[days[sel] == d]  # derive from sel so drop_first composes
                convert_row({"msgs": real["msgs"][sd], "l2": real["l2"][sd]},
                            {"msgs": g["msgs"][sd], "l2": g["l2"][sd]},
                            bench, row, ticker=args.ticker, date=d, levels=args.levels)
        else:
            convert_row(real_sub, gen_sub, bench, row, ticker=args.ticker, levels=args.levels)
        print(f"[ab] scoring '{row}' ({len(sel)} pairs{' by-day' if args.by_day else ''}) ...",
              flush=True)
        results[row] = score_row(os.path.join(bench, row), skip_cond=True)

    out_dir = os.path.join(args.eval_dir, f"lobbench_scores_{out_tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "lobbench_results.json"), "w") as f:
        json.dump(_jsonable(results), f, indent=2)

    ok = [r for r in rows if "uncond" in results.get(r, {})]
    if ok:
        names = list(results[ok[0]]["uncond"].keys())
        lines = ["| row | " + " | ".join(names) + " | MEAN(L1) | WS-21 |",
                 "|" + "---|" * (len(names) + 3)]
        for r in ok:
            u = results[r]["uncond"]
            vals = [u[s].get("l1", float("nan")) for s in names]
            s = results[r].get("summary") or {}
            ws = s.get("wasserstein", [[float("nan")]])[0][0] if s else float("nan")
            lines.append(f"| {r} | " + " | ".join(f"{v:.4f}" for v in vals)
                         + f" | {np.nanmean(vals):.4f} | {ws:.4f} |")
        table = "\n".join(lines)
        with open(os.path.join(out_dir, "lobbench_table.md"), "w") as f:
            f.write(f"# LOB-Bench L1 (split {args.split} = {', '.join(want)}; lower = better)\n\n"
                    + table + "\n")
        print("\n" + table, flush=True)
    print(f"[ab] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
