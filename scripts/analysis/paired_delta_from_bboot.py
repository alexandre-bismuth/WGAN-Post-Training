#!/usr/bin/env python
"""Paired block-bootstrap delta between two rows (e.g. pick - anchor) from bboot JSONs.

block_bootstrap draws its day/rollout resamples from a fixed seed with an RNG-consumption
order that depends only on (seed, n_days, n_seq) — NOT on the row's values. Two rows
bootstrapped over the same context universe (same days list, same n_seq, same seed,
same n_boot) therefore share their draws replicate-for-replicate, and the honest CI on
the DIFFERENCE is the percentile band of the per-replicate differences (day-cluster
primary, rollout secondary), exactly as pre-registered
(docs/reference/preregistration_unseen_arm.md §5).

Within one job the pairing is CRN-exact. Across jobs (e.g. cpt_s0 vs another job's
anchor) it is valid iff the two ctx_days sets are identical (the fleet's seed-0 CRN
guarantees this; this script asserts days/n_seq/seed/n_boot equality and refuses
otherwise).

Usage:
  python scripts/analysis/paired_delta_from_bboot.py \
      --row <bboot_s1pick.json> --ref <bboot_anchor.json> [--label "s1pick - anchor"] \
      [--out <json>]
"""
from __future__ import annotations
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, help="bboot JSON of the treatment row")
    ap.add_argument("--ref", required=True, help="bboot JSON of the reference row (anchor)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A = json.load(open(args.row))
    R = json.load(open(args.ref))
    ma, mr = A["meta"], R["meta"]
    for k in ("seed", "n_boot", "n_seq", "n_days"):
        assert ma[k] == mr[k], f"unpaired bootstraps: meta[{k}] {ma[k]} != {mr[k]}"
    assert ma["days"] == mr["days"], "unpaired bootstraps: day universes differ"
    if ma["scores"] != mr["scores"]:
        print(f"[pdelta] WARN score sets differ: {set(ma['scores']) ^ set(mr['scores'])} "
              f"— aggregate deltas compare different metric mixes")

    label = args.label or f"{args.row} - {args.ref}"
    out = {"label": label, "row": args.row, "ref": args.ref,
           "meta": {k: ma[k] for k in ("seed", "n_boot", "n_seq", "n_days", "alpha")}}
    for metric in ("wasserstein", "l1"):
        d_point = A[metric]["point"] - R[metric]["point"]
        res = {"point": d_point,
               "rel_point": d_point / R[metric]["point"] if R[metric]["point"] else None}
        for kind in ("day", "rollout"):
            da = np.asarray(A[metric]["replicates"][kind], dtype=float)
            dr = np.asarray(R[metric]["replicates"][kind], dtype=float)
            diff = da - dr
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            res[f"ci_{kind}"] = [float(lo), float(hi)]
            res[f"excludes_zero_{kind}"] = bool(hi < 0 or lo > 0)
        # Per-feature paired deltas: available when both bboots carry the per-score
        # replicate matrices (block_bootstrap "day_per_score"/"rollout_per_score").
        # Same shared-draw argument as the aggregate, applied feature-by-feature.
        per = {}
        for kind in ("day", "rollout"):
            pa = A[metric]["replicates"].get(f"{kind}_per_score")
            pr = R[metric]["replicates"].get(f"{kind}_per_score")
            if not (pa and pr):
                continue
            for nm in sorted(set(pa) & set(pr)):
                p_row = A[metric]["per_score"][nm]["point"]
                p_ref = R[metric]["per_score"][nm]["point"]
                diff = np.asarray(pa[nm], dtype=float) - np.asarray(pr[nm], dtype=float)
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                d = per.setdefault(nm, {
                    "point": p_row - p_ref,
                    "rel_point": (p_row - p_ref) / p_ref if p_ref else None,
                    "row_point": p_row, "ref_point": p_ref})
                d[f"ci_{kind}"] = [float(lo), float(hi)]
                d[f"excludes_zero_{kind}"] = bool(hi < 0 or lo > 0)
        if per:
            res["per_score"] = per
        out[metric] = res
        print(f"[pdelta] {label} [{metric}] Δ={d_point:+.4f} "
              f"({100 * res['rel_point']:+.1f}% vs ref) "
              f"day-CI [{res['ci_day'][0]:+.4f},{res['ci_day'][1]:+.4f}] "
              f"rollout-CI [{res['ci_rollout'][0]:+.4f},{res['ci_rollout'][1]:+.4f}] "
              f"day-CI excludes 0: {res['excludes_zero_day']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        import os
        os.chmod(args.out, 0o664)


if __name__ == "__main__":
    main()
