"""LOB-Bench scoring runner — score adapter output with the published benchmark.

Run with the lobbench venv python (pandas/scipy/statsmodels), on a COMPUTE node (pandas over
256 seqs x ~20 score fns x N rows is not login-node work):

    third_party/lobbench_venv/bin/python -m eggroll_gan.run_lobbench \
        --bench_dir <adapter out_dir> --out_dir <results dir> [--skip_cond 0] [--skip_impact 1]

Per row (= per generative arm): Simple_Loader over the row's data_real/data_gen/data_cond,
`scoring.run_benchmark` with lob_bench's OWN DEFAULT_SCORING_CONFIG (+ conditional config) and
L1/Wasserstein metrics, then `scoring.summary_stats` (bootstrap CIs, paper aggregate). Output:
lobbench_results.json + lobbench_table.md (rows x scores, L1 distance; lower = better).

Impact-response curves (--skip_impact 0) are attempted per row and reported if the event mix
supports them (needs executions in both real and gen streams).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

from ..config import EXP_ROOT as _EXP
sys.path.insert(0, os.path.join(_EXP, "third_party", "lob_bench"))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def score_row(row_dir, skip_cond=False):
    import data_loading as dl
    import scoring
    import run_bench as rb

    loader = dl.Simple_Loader(os.path.join(row_dir, "data_real"),
                              os.path.join(row_dir, "data_gen"),
                              os.path.join(row_dir, "data_cond"))
    for s in loader:
        s.materialize()

    def per_score(config):
        """Score-by-score so one degenerate score (empty array — e.g. time_to_cancel on a
        stream whose cancels never match a prior order id) cannot kill the whole row."""
        sc, failed = {}, {}
        for name, cfg in config.items():
            try:
                s, _, _ = scoring.run_benchmark(loader, {name: cfg},
                                                default_metric=rb.DEFAULT_METRICS)
                sc[name] = s[name]
            except Exception as e:
                failed[name] = repr(e)
                print(f"[lobbench]   score '{name}' failed ({e!r}) — skipped", flush=True)
        return sc, failed

    out = {}
    scores, failed = per_score(rb.DEFAULT_SCORING_CONFIG)
    out["uncond"] = {name: {m: float(v[0]) for m, v in d.items()} for name, d in scores.items()}
    out["failed_scores"] = failed
    try:
        out["summary"] = _jsonable(scoring.summary_stats(scores, bootstrap=True))
    except Exception as e:
        print(f"[lobbench] summary_stats failed ({e!r})", flush=True)
        out["summary"] = None
    if not skip_cond:
        scores_c, failed_c = per_score(rb.DEFAULT_SCORING_CONFIG_COND)
        out["cond"] = {name: {m: float(v[0]) for m, v in d.items()}
                       for name, d in scores_c.items()}
        out["failed_scores_cond"] = failed_c
    return out


def impact_row(row_dir):
    import data_loading as dl
    import impact
    loader = dl.Simple_Loader(os.path.join(row_dir, "data_real"),
                              os.path.join(row_dir, "data_gen"),
                              os.path.join(row_dir, "data_cond"))
    for s in loader:
        s.materialize()
    return impact.impact_compare(loader)


def main():
    ap = argparse.ArgumentParser(description="run LOB-Bench over adapter output")
    ap.add_argument("--bench_dir", required=True, help="adapter out_dir (one subdir per row)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rows", nargs="*", default=None, help="default: rows in adapter_summary.json")
    ap.add_argument("--skip_cond", type=int, default=0)
    ap.add_argument("--skip_impact", type=int, default=1,
                    help="impact-response needs exec-rich streams; default off for the prelim table")
    args = ap.parse_args()

    if args.rows:
        rows = args.rows
    else:
        with open(os.path.join(args.bench_dir, "adapter_summary.json")) as f:
            rows = list(json.load(f).keys())

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for row in rows:
        rd = os.path.join(args.bench_dir, row)
        print(f"[lobbench] scoring row '{row}' ...", flush=True)
        try:
            results[row] = score_row(rd, skip_cond=bool(args.skip_cond))
        except Exception:
            print(f"[lobbench] row '{row}' FAILED:\n{traceback.format_exc()}", flush=True)
            results[row] = {"error": traceback.format_exc().splitlines()[-1]}
            continue
        if not args.skip_impact:
            try:
                curves, score = impact_row(rd)
                results[row]["impact_score"] = _jsonable(score)
            except Exception:
                print(f"[lobbench] impact failed for '{row}' (non-fatal)", flush=True)

    with open(os.path.join(args.out_dir, "lobbench_results.json"), "w") as f:
        json.dump(_jsonable(results), f, indent=2)

    # Markdown table: one row per arm, one column per score (L1), + the summary aggregate.
    ok_rows = [r for r in rows if "uncond" in results.get(r, {})]
    if ok_rows:
        score_names = list(results[ok_rows[0]]["uncond"].keys())
        lines = ["| row | " + " | ".join(score_names) + " | MEAN(L1) |",
                 "|" + "---|" * (len(score_names) + 2)]
        for r in ok_rows:
            u = results[r]["uncond"]
            vals = [u[s].get("l1", float("nan")) for s in score_names]
            lines.append(f"| {r} | " + " | ".join(f"{v:.4f}" for v in vals)
                         + f" | {np.nanmean(vals):.4f} |")
        table = "\n".join(lines)
        with open(os.path.join(args.out_dir, "lobbench_table.md"), "w") as f:
            f.write("# LOB-Bench L1 distances (lower = better)\n\n" + table + "\n")
        print("\n" + table, flush=True)
    print(f"[lobbench] done -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
