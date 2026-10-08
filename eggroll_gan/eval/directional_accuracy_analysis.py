"""Directional accuracy analysis — CPU-side scoring of the mid-price dumps.

Consumes a runs/directional_accuracy_<job>/ dir (mids_real.npz, mids_<row>.npz,
mid0_sim.npy, ctx_days.json, directional_accuracy_gen.json). Per horizon h the task is
THREE-WAY classification of the mid-price move over h messages:

  y_true_i = sign(real[i, h] - real[i, 0])   in {down=-1, flat=0, up=+1}
  y_pred_i = sign(gen[i, h-1] - m0_i)        the model's own rollout path

with m0 = mid0_sim by default (the sim book's own boundary mid — self-consistent with the
generated path; the gen job verifies it sits on top of real[:, 0]). Windows where either
side is NaN (day-file end, empty sim book) are dropped per-horizon and counted.

Reported metrics, per horizon, for exactly three headline rows:
  majority   — the constant-class predictor (most frequent TRUE class); its accuracy is
               the max class rate and its macro F1 the constant-predictor F1. Inside the
               bootstrap the majority class is re-chosen per replicate.
  anchor     — the pretrained generator.
  seed mean  — the 10 post-trained headline picks, averaged.
Metrics: 3-class directional accuracy (acc3) and macro F1 over the classes with true
support (a class absent from the truth at some horizon contributes no F1 term rather than
a hard 0 for every predictor); per-class F1 and the 2-class (non-flat) accuracy are kept
in the JSON for reference. Model comparisons are PAIRED: delta vs anchor on the
intersection of both rows' valid windows. CIs: day-clustered bootstrap (resample panel
days with replacement, ONE shared draw matrix across rows/horizons so deltas cancel day
noise) — the rescoreB convention.

Outputs directional_accuracy.json + fig_directional_accuracy.png (acc3 and macro-F1 vs
horizon: majority baseline / anchor / seed band+mean) + a printed markdown table.

Pure numpy/matplotlib; login-node-light or the same job's CPU tail.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

CLASSES = (-1, 0, 1)
EPS = 1e-12


def directions(real, gen, mid0, h, flat_band=0.0):
    """Per-window (valid, y_true, y_pred) at horizon h. real [X, H+1], gen [X, H],
    mid0 [X]. flat_band: |move| <= band counts as flat (raw price units; 0 = exact)."""
    rt = real[:, h] - real[:, 0]
    gn = gen[:, h - 1] - mid0
    valid = np.isfinite(rt) & np.isfinite(gn)
    y_true = np.where(np.abs(rt) <= flat_band, 0, np.sign(rt)).astype(np.int8)
    y_pred = np.where(np.abs(gn) <= flat_band, 0, np.sign(gn)).astype(np.int8)
    return valid, y_true, y_pred


def _wsum(wv, ind):
    return (wv * ind[None, :]).sum(axis=1)


def stats_bundle(valid, y_true, y_pred, W=None):
    """Vectorized metric bundle. W [B, X] day-bootstrap weights (None -> single
    point-estimate replicate). Returns dict of [B] arrays: acc3, macro_f1, acc2,
    f1_down/flat/up, plus class true-rates. Macro F1 averages over classes with true
    support in the (weighted) replicate."""
    if W is None:
        W = np.ones((1, valid.shape[0]))
    wv = W * valid[None, :].astype(np.float64)
    n = wv.sum(axis=1)
    n = np.where(n > 0, n, np.nan)
    out = dict(n=wv.sum(axis=1), acc3=_wsum(wv, (y_pred == y_true)) / n)
    f1s, rates = [], []
    for c in CLASSES:
        tp = _wsum(wv, (y_pred == c) & (y_true == c))
        pp = _wsum(wv, y_pred == c)
        ap = _wsum(wv, y_true == c)
        f1 = 2 * tp / np.maximum(pp + ap, EPS)          # = 2PR/(P+R) without 0/0 hazards
        f1s.append(np.where(ap > 0, f1, np.nan))        # no-true-support class: no term
        rates.append(ap / n)
        out[f"f1_{'down' if c < 0 else 'flat' if c == 0 else 'up'}"] = f1
    out["macro_f1"] = np.nanmean(np.stack(f1s), axis=0)
    out["rate_down"], out["rate_flat"], out["rate_up"] = rates
    nf = _wsum(wv, y_true != 0)
    out["acc2"] = _wsum(wv, (y_true != 0) & (y_pred == y_true)) / np.where(nf > 0, nf, np.nan)
    return out


def majority_bundle(valid, y_true, W=None):
    """The constant-majority-class predictor, class re-chosen per replicate: acc3 = max
    class rate; its F1 = 2*ap/(n+ap) for the majority class, 0 for the other supported
    classes; macro over supported classes."""
    if W is None:
        W = np.ones((1, valid.shape[0]))
    wv = W * valid[None, :].astype(np.float64)
    n = wv.sum(axis=1)
    n = np.where(n > 0, n, np.nan)
    ap = np.stack([_wsum(wv, y_true == c) for c in CLASSES])            # [3, B]
    maj = np.argmax(ap, axis=0)                                         # [B]; ties -> first
    ap_maj = np.take_along_axis(ap, maj[None, :], axis=0)[0]
    # One-hot on the ARGMAX INDEX, not on the count value: when two classes tie for the
    # top count, a value comparison marks BOTH as predicted, which a constant predictor
    # cannot do and which inflates macro F1 (seen at h=975 on job 5918381: 0.433 vs a
    # ~0.218 baseline). argmax's first-wins tie-break is the constant predictor.
    is_maj = np.arange(len(CLASSES))[:, None] == maj[None, :]
    f1_c = np.where(is_maj, 2 * ap / np.maximum(n + ap, EPS), 0.0)
    macro = np.nanmean(np.where(ap > 0, f1_c, np.nan), axis=0)
    out = dict(n=wv.sum(axis=1), acc3=ap_maj / n, macro_f1=macro,
               majority_class=[int(CLASSES[m]) for m in maj])
    nf = ap[0] + ap[2]
    out["acc2"] = np.maximum(ap[0], ap[2]) / np.where(nf > 0, nf, np.nan)
    return out


def day_boot_matrix(days, n_boot, seed):
    """[n_boot, X] multiplicity weights from resampling the unique days with replacement.
    ONE shared matrix for every row/horizon so paired deltas cancel day noise."""
    uniq = sorted(set(days))
    day_ix = np.array([uniq.index(d) for d in days])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    counts = np.stack([np.bincount(dr, minlength=len(uniq)) for dr in draws])
    return counts[:, day_ix].astype(np.float64)                         # [n_boot, X]


def ci(vals, lo=2.5, hi=97.5):
    v = np.asarray(vals, np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def _pt_ci(point, reps):
    c = ci(reps)
    return dict(value=float(point), ci=c)


def load_dirs(dirs):
    """Merge one or more per-job dump dirs (split-mode runs) into a single row table.
    Every dir must carry the SAME universe: identical config, bit-identical mids_real /
    mid0_sim / ctx_days (the window draw and boundary replay are seed-deterministic, so a
    mismatch means the jobs did not run the same program config — fail loudly)."""
    metas, reals, mid0s, dayss = [], [], [], []
    for d in dirs:
        with open(os.path.join(d, "directional_accuracy_gen.json")) as f:
            metas.append(json.load(f))
        reals.append(np.load(os.path.join(d, "mids_real.npz"))["mids"])
        mid0s.append(np.load(os.path.join(d, "mid0_sim.npy")))
        with open(os.path.join(d, "ctx_days.json")) as f:
            dayss.append(json.load(f)["day"])
    meta = metas[0]
    for d, m in zip(dirs[1:], metas[1:]):
        for k in ("n_cond", "n_gen", "horizon_max", "n_eval_ctx", "seed", "top_n",
                  "tick_size", "wide_levels", "ckpt_step"):
            assert m.get(k) == meta.get(k), f"merge mismatch on '{k}' in {d}"
    for d, r, m0, dy in zip(dirs[1:], reals[1:], mid0s[1:], dayss[1:]):
        assert np.array_equal(r, reals[0], equal_nan=True), f"mids_real differs in {d}"
        assert np.array_equal(m0, mid0s[0]), f"mid0_sim differs in {d}"
        assert dy == dayss[0], f"ctx_days differs in {d}"
    row_dir = {}
    for d, m in zip(dirs, metas):
        for r in m["rows"]:
            if r == "merge_noop":
                continue
            assert r not in row_dir, f"row '{r}' appears in both {row_dir[r]} and {d}"
            row_dir[r] = d
    assert "anchor" in row_dir, "no dir carries the anchor row"
    rows = ["anchor"] + [r for r in row_dir if r != "anchor"]
    meta = dict(meta, rows=rows)
    return meta, reals[0], mid0s[0], dayss[0], rows, row_dir


def main():
    ap = argparse.ArgumentParser(description="directional accuracy analysis (CPU)")
    ap.add_argument("--dir", required=True,
                    help="dump dir, or comma list of per-row split dirs (merged; outputs "
                         "land in the first)")
    ap.add_argument("--horizons", default=None,
                    help="comma ints; default = the gen meta's horizons")
    ap.add_argument("--curve_step", type=int, default=25)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--flat_band", type=float, default=0.0)
    ap.add_argument("--m0", choices=["sim", "real"], default="sim")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.dir.split(",") if d.strip()]
    out_dir = dirs[0]
    meta, real, mid0_sim, days, rows, row_dir = load_dirs(dirs)
    real = real.astype(np.float64)
    mid0_sim = mid0_sim.astype(np.float64)
    H = int(meta["horizon_max"])
    horizons = ([int(h) for h in args.horizons.split(",")] if args.horizons
                else [int(h) for h in meta["horizons"]])
    assert all(1 <= h <= H for h in horizons), f"horizons {horizons} outside [1, {H}]"

    mid0 = mid0_sim if args.m0 == "sim" else real[:, 0]
    d0 = np.abs(mid0_sim - real[:, 0])
    tick = float(meta.get("tick_size", 100))
    seeds = [r for r in rows if r != "anchor"]
    gens = {r: np.load(os.path.join(row_dir[r], f"mids_{r}.npz"))["mids"].astype(np.float64)
            for r in rows}
    W = day_boot_matrix(days, args.n_boot, args.boot_seed)

    out = dict(config=vars(args), meta=dict(meta, rows=rows), n_days=len(set(days)),
               boundary=dict(mean_abs=float(np.nanmean(d0)), max_abs=float(np.nanmax(d0)),
                             frac_gt_half_tick=float(np.mean(d0 > tick / 2))),
               horizons={}, curve={})

    for h in horizons:
        vs = {r: directions(real, gens[r], mid0, h, args.flat_band) for r in rows}
        pt = {r: stats_bundle(*vs[r]) for r in rows}
        bs = {r: stats_bundle(*vs[r], W=W) for r in rows}
        hout = {}
        for r in rows:
            hout[r] = {k: float(pt[r][k][0]) for k in
                       ("n", "acc3", "macro_f1", "acc2", "f1_down", "f1_flat", "f1_up",
                        "rate_down", "rate_flat", "rate_up")}
            hout[r]["acc3_ci"] = ci(bs[r]["acc3"])
            hout[r]["macro_f1_ci"] = ci(bs[r]["macro_f1"])

        # Majority baseline on the real-side-valid universe (predictor-independent).
        v_real = np.isfinite(real[:, h]) & np.isfinite(real[:, 0])
        y_real = np.where(np.abs(real[:, h] - real[:, 0]) <= args.flat_band, 0,
                          np.sign(real[:, h] - real[:, 0])).astype(np.int8)
        mpt = majority_bundle(v_real, y_real)
        mbs = majority_bundle(v_real, y_real, W=W)
        hout["majority"] = dict(
            n=float(mpt["n"][0]), acc3=float(mpt["acc3"][0]),
            macro_f1=float(mpt["macro_f1"][0]), acc2=float(mpt["acc2"][0]),
            majority_class=mpt["majority_class"][0],
            acc3_ci=ci(mbs["acc3"]), macro_f1_ci=ci(mbs["macro_f1"]))

        # Paired deltas vs anchor on the intersection of valid windows; the shared
        # bootstrap draw matrix makes each delta CI a paired-replicate CI.
        deltas = {}
        d_reps = {"acc3": [], "macro_f1": []}
        for r in seeds:
            both = vs["anchor"][0] & vs[r][0]
            p_r = stats_bundle(both, vs[r][1], vs[r][2])
            p_a = stats_bundle(both, vs["anchor"][1], vs["anchor"][2])
            b_r = stats_bundle(both, vs[r][1], vs[r][2], W=W)
            b_a = stats_bundle(both, vs["anchor"][1], vs["anchor"][2], W=W)
            deltas[r] = {}
            for k in ("acc3", "macro_f1"):
                reps = b_r[k] - b_a[k]
                d_reps[k].append(reps)
                deltas[r][f"delta_{k}"] = _pt_ci(p_r[k][0] - p_a[k][0], reps)
                deltas[r][f"delta_{k}"]["sig"] = bool(
                    np.prod(np.sign(deltas[r][f"delta_{k}"]["ci"])) > 0)
        if seeds:
            sm = {}
            for k in ("acc3", "macro_f1"):
                mean_pt = float(np.mean([deltas[r][f"delta_{k}"]["value"] for r in seeds]))
                mean_reps = np.nanmean(np.stack(d_reps[k]), axis=0)
                sm[f"delta_{k}"] = _pt_ci(mean_pt, mean_reps)
                sm[f"delta_{k}"]["sig"] = bool(np.prod(np.sign(sm[f"delta_{k}"]["ci"])) > 0)
                sm[f"{k}_seed_mean"] = float(np.mean([hout[r][k] for r in seeds]))
                sm[f"{k}_seed_mean_ci"] = ci(np.nanmean(
                    np.stack([bs[r][k] for r in seeds]), axis=0))
            deltas["_seed_mean"] = sm
        out["horizons"][str(h)] = dict(rows=hout, delta_vs_anchor=deltas)

    # Full curves vs horizon (point estimates only; cheap, every curve_step).
    grid = sorted(set(list(range(args.curve_step, H + 1, args.curve_step)) + horizons))
    for r in rows:
        b = [stats_bundle(*directions(real, gens[r], mid0, h, args.flat_band)) for h in grid]
        out["curve"][r] = dict(h=grid, acc3=[float(s["acc3"][0]) for s in b],
                               macro_f1=[float(s["macro_f1"][0]) for s in b])
    mb = []
    for h in grid:
        v = np.isfinite(real[:, h]) & np.isfinite(real[:, 0])
        y = np.where(np.abs(real[:, h] - real[:, 0]) <= args.flat_band, 0,
                     np.sign(real[:, h] - real[:, 0])).astype(np.int8)
        mb.append(majority_bundle(v, y))
    out["curve"]["majority"] = dict(h=grid, acc3=[float(s["acc3"][0]) for s in mb],
                                    macro_f1=[float(s["macro_f1"][0]) for s in mb])

    with open(os.path.join(out_dir, "directional_accuracy.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- printed summary: the three headline rows, both metrics
    print(f"\n## Three-way directional accuracy + macro F1 (m0={args.m0}, "
          f"flat_band={args.flat_band}, X={real.shape[0]}, {len(set(days))} days, "
          f"{args.n_boot} day-boot reps)\n")
    print("| horizon | row | acc3 [CI] | macro F1 [CI] | Δacc3 vs anchor [CI] | ΔF1 vs anchor [CI] |")
    print("|---|---|---|---|---|---|")
    for h in horizons:
        hh = out["horizons"][str(h)]
        m, a = hh["rows"]["majority"], hh["rows"]["anchor"]
        sm = hh["delta_vs_anchor"].get("_seed_mean", {})

        def _f(v, c):
            return f"{v:.4f} [{c[0]:.4f},{c[1]:.4f}]"

        def _d(dd):
            s = "*" if dd.get("sig") else " "
            return f"{dd['value']:+.4f} [{dd['ci'][0]:+.4f},{dd['ci'][1]:+.4f}]{s}"

        print(f"| {h} | majority (class {m['majority_class']:+d}) | "
              f"{_f(m['acc3'], m['acc3_ci'])} | {_f(m['macro_f1'], m['macro_f1_ci'])} | — | — |")
        print(f"| {h} | anchor | {_f(a['acc3'], a['acc3_ci'])} | "
              f"{_f(a['macro_f1'], a['macro_f1_ci'])} | — | — |")
        if sm:
            print(f"| {h} | post-trained ({len(seeds)}-seed mean) | "
                  f"{_f(sm['acc3_seed_mean'], sm['acc3_seed_mean_ci'])} | "
                  f"{_f(sm['macro_f1_seed_mean'], sm['macro_f1_seed_mean_ci'])} | "
                  f"{_d(sm['delta_acc3'])} | {_d(sm['delta_macro_f1'])} |")
    nsig = {h: sum(1 for r in seeds
                   if out["horizons"][str(h)]["delta_vs_anchor"][r]["delta_acc3"]["sig"])
            for h in horizons}
    print(f"\nper-seed Δacc3 significant: " +
          ", ".join(f"h={h}: {nsig[h]}/{len(seeds)}" for h in horizons))
    b = out["boundary"]
    print(f"boundary |mid0_sim-real0|: mean {b['mean_abs']:.2f} max {b['max_abs']:.0f} "
          f"frac>half-tick {b['frac_gt_half_tick']:.4f}")

    # ---- figure (working matplotlib; styling pass later)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for ax, k, ttl in zip(axes, ("acc3", "macro_f1"),
                              ("3-class directional accuracy", "macro F1")):
            if seeds:
                sa = np.array([out["curve"][r][k] for r in seeds], np.float64)
                ax.fill_between(grid, np.nanmin(sa, 0), np.nanmax(sa, 0),
                                alpha=0.25, color="#1f77b4", lw=0,
                                label=f"post-trained seeds (n={len(seeds)}, min-max)")
                ax.plot(grid, np.nanmean(sa, 0), color="#1f77b4", lw=2,
                        label="post-trained mean")
            ax.plot(grid, out["curve"]["anchor"][k], color="black", lw=2, label="anchor")
            ax.plot(grid, out["curve"]["majority"][k], color="gray", lw=1.5, ls="--",
                    label="majority baseline")
            for h in horizons:
                ax.axvline(h, color="gray", lw=0.5, alpha=0.4)
            ax.set_xlabel("horizon (messages)")
            ax.set_ylabel(ttl)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_directional_accuracy.png"), dpi=150)
        print(f"figure -> {os.path.join(out_dir, 'fig_directional_accuracy.png')}")
    except Exception as e:
        print(f"figure skipped: {e}")


if __name__ == "__main__":
    main()
