#!/usr/bin/env python3
"""Post-hoc checkpoint selection for a no-early-stop EGGROLL-proj production run.

The in-training Goodhart composite is `mean(book_l1, ret, moment_l1, event_l1)` ratios-to-baseline
(eval_monitor.normalize_composite). At small `n_eval_ctx` the `moment_l1` term is sampling-noise
dominated (observed swing ~60x across evals), so selecting `best/` by the raw composite argmin picks
a moment-noise-lucky step whose STABLE stylized facts (book/ret/event) are often no better than the
anchor. This script re-selects each lane's checkpoint by a DE-NOISED (moment_l1 excluded) and SMOOTHED
(trailing moving-average) composite over the run's logged eval history, restricted to steps that have a
loadable step-keyed checkpoint (train with --keep_step_ckpts). It copies the chosen step into
<lane>/selected/ and writes <lane>/selection.json. Point the eval harness at <lane>/selected.

No JAX needed — pure stdlib (json/os/shutil). Read-only except the optional --copy.
"""
import argparse, json, os, shutil

EPS = 1e-8
DEFAULT_TERMS = ("book", "ret", "event")   # de-noised: drop the noisy 'moment'; 'mid' already excluded


def term_vals(m, b):
    return {
        "book":   m["book_l1"]   / max(b["book_l1"], EPS),
        "ret":    (1.0 - m["ret_corr"]) / max(1.0 - b["ret_corr"], EPS),
        "moment": m["moment_l1"] / max(b["moment_l1"], EPS),
        "event":  m["event_l1"]  / max(b["event_l1"], EPS),
    }


def trailing_sma(xs, k):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k + 1)
        out.append(sum(xs[lo:i + 1]) / (i + 1 - lo))
    return out


def lane_trajectory(lane_dir, terms):
    bc = json.load(open(os.path.join(lane_dir, "latest_checkpoint.json")))
    base, H = bc["ev_baseline"], bc["history"]
    steps, comp = [], []
    for h in H:
        m = {key[3:]: h[key] for key in h if key.startswith("ev_") and key != "ev_composite"}
        tv = term_vals(m, base)
        steps.append(int(h["step"]))
        comp.append(sum(tv[t] for t in terms) / len(terms))
    return steps, comp


def has_ckpt(lane_dir, step):
    return os.path.isfile(os.path.join(lane_dir, f"step{step:04d}", "latest_checkpoint.json"))


def select_lane(lane_dir, terms, smooth, require_ckpt):
    steps, comp = lane_trajectory(lane_dir, terms)
    sm = trailing_sma(comp, smooth)
    # candidate indices: those with a loadable step-keyed ckpt (if required), else all
    cand = [i for i in range(len(steps)) if (not require_ckpt or has_ckpt(lane_dir, steps[i]))]
    if not cand:
        return None, steps, comp, sm
    bi = min(cand, key=lambda i: sm[i])
    return bi, steps, comp, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod_dir", required=True, help="dir holding the per-lane subdirs")
    ap.add_argument("--lanes", default="es003_s0,es003_s1,es003_s2,ctrl_shuf")
    ap.add_argument("--terms", default=",".join(DEFAULT_TERMS),
                    help="composite terms for selection (subset of book,ret,moment,event)")
    ap.add_argument("--smooth", type=int, default=3, help="trailing moving-average window (evals)")
    ap.add_argument("--copy", action="store_true",
                    help="copy the selected step-keyed ckpt to <lane>/selected/ + write selection.json")
    ap.add_argument("--target", default="selected", help="subdir name to copy the selection into")
    args = ap.parse_args()
    terms = tuple(t.strip() for t in args.terms.split(",") if t.strip())

    for lane in args.lanes.split(","):
        lane = lane.strip()
        lane_dir = os.path.join(args.prod_dir, lane)
        if not os.path.isfile(os.path.join(lane_dir, "latest_checkpoint.json")):
            print(f"[{lane}] no latest_checkpoint.json — skipping")
            continue
        bi, steps, comp, sm = select_lane(lane_dir, terms, args.smooth, require_ckpt=args.copy)
        print(f"\n===== {lane} =====")
        print(f"  terms={terms} smooth={args.smooth}")
        print("  step:  " + " ".join(f"{s:>5d}" for s in steps))
        print("  deno:  " + " ".join(f"{c:>5.2f}" for c in comp))
        print("  smth:  " + " ".join(f"{c:>5.2f}" for c in sm))
        if bi is None:
            print(f"  !! no step-keyed ckpt available (train with --keep_step_ckpts); "
                  f"raw-history argmin would be step {steps[min(range(len(sm)), key=lambda i: sm[i])]}")
            continue
        sel_step = steps[bi]
        print(f"  --> SELECT step {sel_step}  deno {comp[bi]:.3f}  smoothed {sm[bi]:.3f}  "
              f"(baseline=1.0; <1 better)")
        if args.copy:
            src = os.path.join(lane_dir, f"step{sel_step:04d}")
            dst = os.path.join(lane_dir, args.target)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            with open(os.path.join(lane_dir, "selection.json"), "w") as f:
                json.dump({"lane": lane, "selected_step": sel_step, "deno": comp[bi],
                           "smoothed": sm[bi], "terms": list(terms), "smooth": args.smooth,
                           "src": src, "dst": dst}, f, indent=2)
            print(f"  copied {src} -> {dst}  (+ selection.json)")


if __name__ == "__main__":
    main()
