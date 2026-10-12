"""Summarize a Muon lr-calibration run for the production-lr decision.

Reads each lane's breadcrumb (latest_checkpoint.json) written by _run_eggroll_production.sbatch with
SOLVER=muon. Each breadcrumb carries `composites` (normalised composite trajectory, lower=better),
`history` (per-eval dicts incl. `mean_kl`, `mean_num_errors`, `separation`), and meta (`lr`,
`solver`, `diverged`, `best_composite`). No log-text parsing — all structured JSON.

Decision rule (mirrors the prior sigma-sweep): pick the lr with the LOWEST best composite among the
lanes whose KL-to-anchor stays BOUNDED (no spike) and that did not diverge. A lane that wins on
composite only by letting KL blow up is rejected (Goodhart / trust-region violation).

CPU-only, login-node safe.
    python -m eggroll_gan.eval.calib_report --run_dir runs/s5b_production_<jobid>
    python -m eggroll_gan.eval.calib_report --selftest
"""
import argparse
import json
import math
import os
import re
import sys


def _lane_lr(bc, lane_name):
    """lr from the breadcrumb meta, else parsed from the lane tag (m_lr010 -> 0.010, m_lr0005 ...)."""
    lr = bc.get("lr")
    if lr is not None:
        return float(lr)
    m = re.search(r"lr([0-9]+)", lane_name)
    if m:                                   # m_lr001 -> 0.001, m_lr010 -> 0.01, m_lr020 -> 0.02
        digits = m.group(1)
        return float("0." + digits.lstrip("0").rjust(len(digits), "0")) if digits else float("nan")
    return float("nan")


def _finite(xs):
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def summarize_lane(lane_dir, lane_name):
    bc_path = os.path.join(lane_dir, "latest_checkpoint.json")
    if not os.path.isfile(bc_path):
        return None
    with open(bc_path) as f:
        bc = json.load(f)
    comps = _finite(bc.get("composites", []))
    hist = bc.get("history", []) or []
    kls = _finite([h.get("mean_kl") for h in hist])
    nerr = _finite([h.get("mean_num_errors") for h in hist])
    n_raw = len(bc.get("composites", []))
    return {
        "lane": lane_name,
        "lr": _lane_lr(bc, lane_name),
        "solver": bc.get("solver", "?"),
        "sigma": bc.get("sigma"),
        "step": bc.get("step"),
        "n_evals": len(comps),
        "best_comp": min(comps) if comps else float("nan"),
        "final_comp": comps[-1] if comps else float("nan"),
        "kl_max": max(kls) if kls else float("nan"),
        "kl_final": kls[-1] if kls else float("nan"),
        "nerr_final": nerr[-1] if nerr else float("nan"),
        "diverged": bool(bc.get("diverged", False)),
        "nonfinite": len(comps) != n_raw,          # a dropped (NaN) composite eval
    }


def collect(run_dir):
    rows = []
    for name in sorted(os.listdir(run_dir)):       # one known dir; lanes + logs only
        d = os.path.join(run_dir, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "latest_checkpoint.json")):
            r = summarize_lane(d, name)
            if r:
                rows.append(r)
    return sorted(rows, key=lambda r: (math.isnan(r["lr"]), r["lr"]))


def render(rows, kl_max_bound):
    if not rows:
        return "no lanes with a breadcrumb found"
    hdr = ("| lane | lr | solver | best_comp | final_comp | kl_max | kl_final | "
           "nerr_final | n_evals | stable |")
    sep = "|" + "---|" * 10
    lines = [hdr, sep]
    for r in rows:
        stable = (not r["diverged"]) and (not r["nonfinite"]) and \
                 math.isfinite(r["kl_max"]) and r["kl_max"] <= kl_max_bound and \
                 math.isfinite(r["best_comp"])
        r["_stable"] = stable
        flag = "" if stable else (" DIVERGED" if r["diverged"]
                                  else " KL>{:.0f}".format(kl_max_bound) if (math.isfinite(r["kl_max"]) and r["kl_max"] > kl_max_bound)
                                  else " NONFINITE")
        lines.append(f"| {r['lane']} | {r['lr']:.4g} | {r['solver']} | {r['best_comp']:.4f} | "
                     f"{r['final_comp']:.4f} | {r['kl_max']:.3f} | {r['kl_final']:.3f} | "
                     f"{r['nerr_final']:.3f} | {r['n_evals']} | {'yes' if stable else 'no' + flag} |")
    table = "\n".join(lines)

    stable = [r for r in rows if r["_stable"]]
    if stable:
        best = min(stable, key=lambda r: r["best_comp"])
        rec = (f"\nRECOMMENDED production lr = {best['lr']:.4g}  "
               f"(best_comp {best['best_comp']:.4f} at bounded KL {best['kl_max']:.2f} nats). "
               f"Decision rule: lowest best composite among KL-bounded (<= {kl_max_bound:g}), "
               f"non-diverged lanes.")
    else:
        rec = (f"\nNO lane satisfied the KL bound (<= {kl_max_bound:g}) without diverging — "
               f"re-probe with smaller lr / larger kl_coef before committing the 3-seed run.")
    return table + "\n" + rec


def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp()
    # synthetic lanes: a good mid lr, a too-cold lr, a too-hot (KL-spiking) lr
    lanes = {
        "m_lr001": dict(lr=0.001, comps=[1.0, 0.98, 0.97], kls=[0.5, 0.6, 0.7], div=False),
        "m_lr003": dict(lr=0.003, comps=[1.0, 0.90, 0.85], kls=[1.0, 2.0, 3.0], div=False),
        "m_lr020": dict(lr=0.02, comps=[1.0, 0.80, 0.70], kls=[5.0, 18.0, 26.0], div=False),
    }
    for name, spec in lanes.items():
        d = os.path.join(tmp, name)
        os.makedirs(d)
        bc = {"stage": "S5", "step": 30, "solver": "muon", "sigma": 0.003, "lr": spec["lr"],
              "composites": spec["comps"], "diverged": spec["div"],
              "best_composite": min(spec["comps"]),
              "history": [{"step": (i + 1) * 10, "mean_kl": k, "mean_num_errors": 0.3}
                          for i, k in enumerate(spec["kls"])],
              "generator_proj": "x.msgpack"}
        with open(os.path.join(d, "latest_checkpoint.json"), "w") as f:
            json.dump(bc, f)
    os.makedirs(os.path.join(tmp, "logs"))         # must be ignored (no breadcrumb)
    rows = collect(tmp)
    out = render(rows, kl_max_bound=10.0)
    print(out)
    # lr=0.02 has best comp (0.70) but KL spikes to 26 -> must NOT be recommended; lr=0.003 should win.
    ok = (len(rows) == 3 and "RECOMMENDED production lr = 0.003" in out
          and any(r["lane"] == "m_lr020" and not r["_stable"] for r in rows))
    print("[calib_report selftest]", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="summarize a muon lr-calibration run")
    ap.add_argument("--run_dir", help="runs/s5b_production_<jobid> (lane subdirs with breadcrumbs)")
    ap.add_argument("--kl_max", type=float, default=10.0,
                    help="max acceptable KL-to-anchor (nats) for a lane to qualify (default 10)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.run_dir:
        ap.error("--run_dir is required (or use --selftest)")
    rows = collect(args.run_dir)
    print(render(rows, args.kl_max))
    return 0


if __name__ == "__main__":
    sys.exit(main())
