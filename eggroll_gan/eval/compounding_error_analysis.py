"""Compounding-error analysis — per-position divergence curves from saved token streams.

Consumes compounding_error.py's output dir (tokens_real.npz + tokens_<row>.npz +
ctx_days.json) and produces the divergence-vs-position curves that test the project
premise: does post-training reduce autoregressive compounding error?

Per continuation position t (message m = t//26, phase i = t%26) it compares the empirical
marginal of TRUE tokens vs GENERATED tokens across the X contexts, over the phase's valid
sub-alphabet plus one catch-all OTHER bucket (padding/specials; also NA where it is
illegal, i.e. phases < 16):
  KL(p_t || q_t)      forward KL, add-alpha smoothed (alpha = 1/K per cell)
  JS(p_t, q_t)        bounded robustness companion
  H(p_t)              entropy of the true marginal — the per-position normalizer

Estimator design — every divergence is a SYMMETRIZED CROSS-HALF comparison on a
day-stratified half split (A, B) of the contexts:
  model KL_t = 1/2 [ KL(real_A || gen_B) + KL(real_B || gen_A) ]
  floor_t    = 1/2 [ KL(real_A || real_B) + KL(real_B || real_A) ]
Both compare disjoint-context X/2-vs-X/2 streams, so the finite-sample KL bias is
IDENTICAL between model and floor (it cancels in model - floor) and the model side never
shares contexts with the real side it is scored against (kills the paired-context
correlation that would bias model KL downward).

Normalization (the "std of the true data at each position" request, categorical form):
  kl_norm  = KL / H              error as a fraction of the position's intrinsic uncertainty
  excess   = (KL - floor) / H    0 means indistinguishable from sampling noise
Phases whose true data is ~deterministic (median H < --h_eps: the Delta-t-seconds token,
the price/size high digits) are excluded from normalized aggregates and reported raw.

Phase groups: HEADLINE = phases 0-10 (the sampled "new message" block); REF = 16-25
(reported separately — real data is a ~50/50 NA/numeric mixture there while generations
are ~100% numeric, a large but t-independent defect that would constant-offset the pooled
curve); phases 11-15 (absolute time) are never sampled by the generator and are excluded.

Literal z-drift companion (ordinal fields decoded straight from token arithmetic):
  z(m) = (mean_gen(m) - mean_true(m)) / std_true(m)   for price_rel, size, log10(dt).

CIs: day-clustered bootstrap (resample the panel days with replacement) on message-BINNED
curves — per-day sufficient statistics are binned counts, so resampling is exact and cheap.

Pure numpy + matplotlib; no jax. Login-node-light or GPU-job tail.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

MSG_LEN = 26
VOCAB = 2112
NA_TOK = 2

# Phase table (26-token message layout, lobmamba/lob/encoding.py): phase -> (name, first
# token id of the field's sub-alphabet, alphabet size, ordinal?). NA (id 2) is legal only
# at phases >= 16. Phases 11-15 (absolute time) are deterministically FILLED from Delta-t
# during generation — never sampled — and are excluded everywhere.
PHASES = {
    0: ("event_type", 1004, 4, False),
    1: ("direction", 2110, 2, False),
    2: ("price_sign", 2108, 2, False),
    3: ("price_hi", 1108, 1000, True),
    4: ("price_lo", 1108, 1000, True),
    5: ("size_hi", 1008, 100, True),
    6: ("size_lo", 1008, 100, True),
    7: ("dt_s", 4, 1000, True),
    8: ("dt_ns_hi", 4, 1000, True),
    9: ("dt_ns_mid", 4, 1000, True),
    10: ("dt_ns_lo", 4, 1000, True),
    16: ("ref_sign", 2108, 2, False),
    17: ("ref_price_hi", 1108, 1000, True),
    18: ("ref_price_lo", 1108, 1000, True),
    19: ("ref_size_hi", 1008, 100, True),
    20: ("ref_size_lo", 1008, 100, True),
    21: ("ref_time_s_hi", 4, 1000, True),
    22: ("ref_time_s_lo", 4, 1000, True),
    23: ("ref_time_ns_hi", 4, 1000, True),
    24: ("ref_time_ns_mid", 4, 1000, True),
    25: ("ref_time_ns_lo", 4, 1000, True),
}
HEADLINE_PHASES = list(range(0, 11))
REF_PHASES = list(range(16, 26))
FIELD_GROUPS = {                       # figure panels: phase groups of the new-message block
    "event_type": [0], "direction": [1], "price": [2, 3, 4],
    "size": [5, 6], "delta_t": [7, 8, 9, 10],
}

# Entity-stable series colors (dataviz fixed categorical order, light mode); the floor is
# neutral, never a series hue.
SLOT_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
            "#e34948"]
FLOOR_HEX = "#757570"


def phase_lut(i, na_legal):
    """Vocab-size lookup: token id -> local code in [0, K); K-1 is the OTHER bucket
    (anything outside the phase alphabet — padding, specials, illegal NA)."""
    name, off, size, _ = PHASES[i]
    ids = list(range(off, off + size)) + ([NA_TOK] if na_legal else [])
    K = len(ids) + 1
    lut = np.full(VOCAB, K - 1, np.int32)
    lut[np.asarray(ids, np.int64)] = np.arange(len(ids), dtype=np.int32)
    return lut, K


def counts_by_msg(tokens, i, lut, K, n_msgs):
    """[X, T] tokens -> [n_msgs, K] counts at phase i (columns i, i+26, ...)."""
    codes = lut[tokens[:, i::MSG_LEN].astype(np.int64)]          # [X, n_msgs]
    out = np.zeros((n_msgs, K), np.int64)
    np.add.at(out, (np.broadcast_to(np.arange(n_msgs), codes.shape), codes), 1)
    return out


def counts_by_day_bin(tokens, i, lut, K, day_idx, n_days, bins):
    """[X, T] tokens -> [n_days, n_bins, K] binned counts at phase i (bootstrap stats)."""
    n_msgs = tokens.shape[1] // MSG_LEN
    codes = lut[tokens[:, i::MSG_LEN].astype(np.int64)]          # [X, n_msgs]
    b = np.minimum(np.arange(n_msgs) // (n_msgs // bins), bins - 1)
    out = np.zeros((n_days, bins, K), np.int64)
    np.add.at(out, (np.broadcast_to(day_idx[:, None], codes.shape),
                    np.broadcast_to(b, codes.shape), codes), 1)
    return out


def _smooth_probs(c):
    c = np.asarray(c, np.float64)
    K = c.shape[-1]
    return (c + 1.0 / K) / (c.sum(-1, keepdims=True) + 1.0)


def kl_js_h(c_p, c_q):
    """Counts [..., K] x2 -> (KL(p||q), JS, H(p)) along the last axis. KL/JS use add-1/K
    smoothed probabilities (finite by construction); H is the PLUG-IN entropy of p — it is
    only a normalizer, and smoothing would inflate it for near-constant positions (a
    constant column must read H=0 so the low-entropy guard catches it)."""
    p, q = _smooth_probs(c_p), _smooth_probs(c_q)
    kl = np.sum(p * (np.log(p) - np.log(q)), -1)
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * (np.log(p) - np.log(m)), -1) \
        + 0.5 * np.sum(q * (np.log(q) - np.log(m)), -1)
    pt = np.asarray(c_p, np.float64)
    pt = pt / np.maximum(pt.sum(-1, keepdims=True), 1e-12)
    h = -np.sum(np.where(pt > 0, pt * np.log(np.maximum(pt, 1e-300)), 0.0), -1)
    return kl, js, h


def rolling(x, w):
    x = np.asarray(x, np.float64)
    if w <= 1 or x.size < w:
        return x
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    pad = np.full(x.size - c.size, np.nan)
    return np.concatenate([pad[: (x.size - c.size + 1) // 2], c,
                           pad[: (x.size - c.size) // 2]])


def decode_fields(tokens):
    """[X, T] tokens -> dict of per-message numeric fields (NaN where any digit is outside
    its alphabet). Straight token arithmetic — no jax, mirrors encoding.py offsets."""
    X, T = tokens.shape
    n_msgs = T // MSG_LEN
    tk = tokens.reshape(X, n_msgs, MSG_LEN).astype(np.int64)

    def dig(ph, off, size):
        v = tk[:, :, ph] - off
        ok = (v >= 0) & (v < size)
        return np.where(ok, v, 0).astype(np.float64), ok

    sgn, ok0 = dig(2, 2108, 2)
    hi, ok1 = dig(3, 1108, 1000)
    lo, ok2 = dig(4, 1108, 1000)
    price = np.where(ok0 & ok1 & ok2, (2 * sgn - 1) * (hi * 1000 + lo), np.nan)
    shi, ok3 = dig(5, 1008, 100)
    slo, ok4 = dig(6, 1008, 100)
    size = np.where(ok3 & ok4, shi * 100 + slo, np.nan)
    ds, ok5 = dig(7, 4, 1000)
    d1, ok6 = dig(8, 4, 1000)
    d2, ok7 = dig(9, 4, 1000)
    d3, ok8 = dig(10, 4, 1000)
    dt = np.where(ok5 & ok6 & ok7 & ok8, ds + (d1 * 1e6 + d2 * 1e3 + d3) * 1e-9, np.nan)
    return {"price_rel": price, "size": size, "log10_dt": np.log10(dt + 1e-9)}


def zdrift(real_f, gen_f):
    """Per-message z: (mean_gen - mean_true) / std_true, NaN-safe."""
    mt, mg = np.nanmean(real_f, 0), np.nanmean(gen_f, 0)
    st = np.nanstd(real_f, 0)
    return np.where(st > 0, (mg - mt) / st, np.nan)


def stratified_halves(day_idx):
    """Alternate contexts within each day -> two ~equal halves with identical day mix."""
    order = np.lexsort((np.arange(day_idx.size), day_idx))
    a = np.zeros(day_idx.size, bool)
    a[order[::2]] = True
    return a, ~a


def main():
    ap = argparse.ArgumentParser(description="compounding-error curves from token dumps")
    ap.add_argument("--dir", required=True, help="compounding_error.py output dir")
    ap.add_argument("--out_dir", default=None, help="default: same as --dir")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--h_eps", type=float, default=0.05,
                    help="min median true-entropy (nats) for a phase to enter normalized aggregates")
    ap.add_argument("--groups", default="",
                    help="seed groups 'name=row1,row2;name2=...' -> mean curve over member "
                         "rows with the SAME shared bootstrap draws (paired mean CI)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = args.out_dir or args.dir
    os.makedirs(out_dir, exist_ok=True)

    real = np.load(os.path.join(args.dir, "tokens_real.npz"))["tokens"]
    X, T = real.shape
    n_msgs = T // MSG_LEN
    rows = sorted(f[len("tokens_"):-len(".npz")]
                  for f in os.listdir(args.dir)
                  if f.startswith("tokens_") and f.endswith(".npz") and f != "tokens_real.npz")
    rows = sorted(rows, key=lambda r: (r != "anchor", r))        # anchor first, stable
    gens = {r: np.load(os.path.join(args.dir, f"tokens_{r}.npz"))["tokens"] for r in rows}
    for r, g in gens.items():
        assert g.shape == real.shape, f"row '{r}' shape {g.shape} != real {real.shape}"

    with open(os.path.join(args.dir, "ctx_days.json")) as f:
        cd = json.load(f)
    days = np.asarray(cd["day"])
    uday = sorted(set(days.tolist()))
    day_idx = np.searchsorted(np.asarray(uday), days)
    n_days = len(uday)
    print(f"[analysis] X={X} n_msgs={n_msgs} rows={rows} days={n_days}", flush=True)

    half_a, half_b = stratified_halves(day_idx)
    kept = HEADLINE_PHASES + REF_PHASES
    res = {"n_ctx": X, "n_msgs": n_msgs, "rows": rows, "days": uday, "h_eps": args.h_eps}
    per_phase = {}                                   # phase -> dict of per-message curves
    boot_counts = {}                                 # phase -> {stream: [D, B, K]}
    other_frac = {}
    for i in kept:
        na_legal = i >= 16
        lut, K = phase_lut(i, na_legal)
        c_ra = counts_by_msg(real[half_a], i, lut, K, n_msgs)
        c_rb = counts_by_msg(real[half_b], i, lut, K, n_msgs)
        c_real = c_ra + c_rb
        f1, _, _ = kl_js_h(c_ra, c_rb)
        f2, _, _ = kl_js_h(c_rb, c_ra)
        _, _, h = kl_js_h(c_real, c_real)
        d = {"floor": 0.5 * (f1 + f2), "H": h, "kl": {}, "js": {}}
        other_frac[i] = {"real": float(c_real[:, -1].sum() / c_real.sum())}
        bc = {"real": (counts_by_day_bin(real[half_a], i, lut, K, day_idx[half_a],
                                         n_days, args.bins),
                       counts_by_day_bin(real[half_b], i, lut, K, day_idx[half_b],
                                         n_days, args.bins))}
        for r in rows:
            c_ga = counts_by_msg(gens[r][half_a], i, lut, K, n_msgs)
            c_gb = counts_by_msg(gens[r][half_b], i, lut, K, n_msgs)
            k1, j1, _ = kl_js_h(c_ra, c_gb)
            k2, j2, _ = kl_js_h(c_rb, c_ga)
            d["kl"][r], d["js"][r] = 0.5 * (k1 + k2), 0.5 * (j1 + j2)
            other_frac[i][r] = float((c_ga[:, -1].sum() + c_gb[:, -1].sum())
                                     / (c_ga.sum() + c_gb.sum()))
            bc[r] = (counts_by_day_bin(gens[r][half_a], i, lut, K, day_idx[half_a],
                                       n_days, args.bins),
                     counts_by_day_bin(gens[r][half_b], i, lut, K, day_idx[half_b],
                                       n_days, args.bins))
        per_phase[i] = d
        boot_counts[i] = bc

    # Phase health report: OTHER fractions + low-entropy exclusions.
    med_h = {i: float(np.median(per_phase[i]["H"])) for i in kept}
    low_h = [i for i in kept if med_h[i] < args.h_eps]
    print(f"[analysis] low-entropy phases (excluded from normalized aggregates): "
          f"{[(i, PHASES[i][0], round(med_h[i], 4)) for i in low_h]}", flush=True)
    bad_real = {i: v["real"] for i, v in other_frac.items() if v["real"] > 1e-3}
    if bad_real:
        print(f"[analysis] WARN: real OTHER-bucket fraction > 0.1% at phases {bad_real}",
              flush=True)

    def agg(phases, stream, stat="kl", normalized=True, floor_sub=False):
        """Mean over usable phases of the per-message curve; [n_msgs]."""
        use = [i for i in phases if i not in low_h] if normalized else list(phases)
        cur = []
        for i in use:
            v = per_phase[i]["floor"].copy() if stream == "floor" else per_phase[i][stat][stream].copy()
            if floor_sub and stream != "floor":
                v = v - per_phase[i]["floor"]
            cur.append(v / per_phase[i]["H"] if normalized else v)
        return np.mean(np.stack(cur), 0) if cur else np.full(n_msgs, np.nan)

    # Day-clustered bootstrap on BINNED curves (per-day binned counts are exact suff.
    # stats). ONE draw matrix shared by every row so cross-row comparisons are paired.
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, n_days, size=(args.n_boot, n_days))

    def _sym_norm(cra, crb, cga, cgb):
        k1, _, _ = kl_js_h(cra, cgb)
        k2, _, _ = kl_js_h(crb, cga)
        _, _, h = kl_js_h(cra + crb, cra + crb)
        return 0.5 * (k1 + k2) / np.maximum(h, 1e-9)

    def boot_binned(phases, stream):
        """Returns (point [bins], acc [n_boot, bins]) — acc kept so group means can be
        formed replicate-by-replicate (shared draws => paired bootstrap of the mean)."""
        use = [i for i in phases if i not in low_h]
        acc = np.zeros((args.n_boot, args.bins))
        point = np.zeros(args.bins)
        for i in use:
            crA, crB = boot_counts[i]["real"]
            cgA, cgB = boot_counts[i][stream]
            point += _sym_norm(crA.sum(0), crB.sum(0), cgA.sum(0), cgB.sum(0))
            for b_i in range(args.n_boot):
                s = draws[b_i]
                acc[b_i] += _sym_norm(crA[s].sum(0), crB[s].sum(0),
                                      cgA[s].sum(0), cgB[s].sum(0))
        point /= max(len(use), 1)
        acc /= max(len(use), 1)
        return point, acc

    def _q(acc):
        return np.quantile(acc, 0.025, 0), np.quantile(acc, 0.975, 0)

    curves = {"headline": {}, "ref_block": {}, "fields": {}, "binned": {}, "zdrift": {},
              "groups": {}}
    curves["headline"]["floor"] = agg(HEADLINE_PHASES, "floor").tolist()
    curves["ref_block"]["floor"] = agg(REF_PHASES, "floor").tolist()
    row_bb = {}
    for r in rows:
        curves["headline"][r] = agg(HEADLINE_PHASES, r).tolist()
        curves["ref_block"][r] = agg(REF_PHASES, r).tolist()
        pt, acc = boot_binned(HEADLINE_PHASES, r)
        row_bb[r] = (pt, acc)
        lo, hi = _q(acc)
        curves["binned"][r] = {"point": pt.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}

    # Seed groups: mean over member rows, replicate-wise on the SHARED draw matrix (the
    # mean's day-bootstrap, correctly paired across members).
    for spec in filter(None, args.groups.split(";")):
        gname, mems = spec.split("=", 1)
        want = [m.strip() for m in mems.split(",") if m.strip()]
        mem = [m for m in want if m in rows]
        if len(mem) != len(want):
            print(f"[analysis] WARN: group '{gname}' missing rows "
                  f"{sorted(set(want) - set(mem))} — averaging over {len(mem)}", flush=True)
        if not mem:
            continue
        pt = np.mean([row_bb[m][0] for m in mem], 0)
        acc = np.mean([row_bb[m][1] for m in mem], 0)
        lo, hi = _q(acc)
        curves["groups"][gname] = {
            "members": mem,
            "headline": np.mean([np.asarray(curves["headline"][m]) for m in mem], 0).tolist(),
            "binned": {"point": pt.tolist(), "lo": lo.tolist(), "hi": hi.tolist()},
        }
        print(f"[analysis] group '{gname}' ({len(mem)} rows): last-bin point {pt[-1]:.4f} "
              f"[{lo[-1]:.4f},{hi[-1]:.4f}]", flush=True)

    # Paired contrasts vs the anchor on the SHARED draw matrix: replicate-wise
    # differences cancel common day-resampling noise AND (to first order) the Jensen
    # inflation of the convex KL estimator, so these CIs — not band overlap — are the
    # correct test for "does this row/group beat the anchor?".
    if "anchor" in row_bb:
        a_pt, a_acc = row_bb["anchor"]
        paired = {}
        for r, (pt_r, acc_r) in row_bb.items():
            if r == "anchor":
                continue
            dlo, dhi = _q(acc_r - a_acc)
            paired[r] = {"point": (pt_r - a_pt).tolist(),
                         "lo": dlo.tolist(), "hi": dhi.tolist()}
        for gname, gv in curves["groups"].items():
            g_pt = np.asarray(gv["binned"]["point"])
            g_acc = np.mean([row_bb[m][1] for m in gv["members"]], 0)
            dlo, dhi = _q(g_acc - a_acc)
            paired[gname] = {"point": (g_pt - a_pt).tolist(),
                             "lo": dlo.tolist(), "hi": dhi.tolist()}
        curves["paired_vs_anchor"] = paired
        np.savez_compressed(
            os.path.join(out_dir, "binned_boot.npz"), draws=draws,
            **{f"pt_{r}": row_bb[r][0] for r in row_bb},
            **{f"acc_{r}": row_bb[r][1] for r in row_bb})
        sep = {r: int(sum(1 for lo_i, hi_i in zip(v["lo"][-8:], v["hi"][-8:])
                          if hi_i < 0 or lo_i > 0)) for r, v in paired.items()}
        print(f"[analysis] paired-vs-anchor sign-separated late-8-bin counts: {sep}",
              flush=True)
    for gname, ph in FIELD_GROUPS.items():
        curves["fields"][gname] = {"floor": agg(ph, "floor").tolist()}
        for r in rows:
            curves["fields"][gname][r] = agg(ph, r).tolist()

    real_f = decode_fields(real)
    for r in rows:
        gf = decode_fields(gens[r])
        curves["zdrift"][r] = {k: zdrift(real_f[k], gf[k]).tolist() for k in real_f}

    res.update(curves=curves, med_entropy={str(i): med_h[i] for i in kept},
               low_entropy_phases=low_h, other_frac={str(i): v for i, v in other_frac.items()},
               raw_kl_low_entropy={str(i): {r: per_phase[i]["kl"][r].tolist() for r in rows}
                                   for i in low_h})
    with open(os.path.join(out_dir, "compounding_error.json"), "w") as f:
        json.dump(res, f)
    print(f"[analysis] wrote compounding_error.json", flush=True)

    # ---- figures (light surface; entity-stable colors; one axis each) --------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {r: SLOT_HEX[k % len(SLOT_HEX)] for k, r in enumerate(rows)}
    mx = np.arange(n_msgs)
    bx = (np.arange(args.bins) + 0.5) * (n_msgs / args.bins)

    def style(ax):
        ax.grid(True, color="#e6e6e2", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("generated message index")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for r in rows:
        ax.plot(mx, rolling(np.asarray(curves["headline"][r]), 10), lw=0.8, alpha=0.35,
                color=color[r])
        b = curves["binned"][r]
        ax.plot(bx, b["point"], lw=1.8, color=color[r], label=r)
        ax.fill_between(bx, b["lo"], b["hi"], color=color[r], alpha=0.15, lw=0)
    ax.plot(mx, rolling(np.asarray(curves["headline"]["floor"]), 10), lw=1.4, ls="--",
            color=FLOOR_HEX, label="real-vs-real floor")
    style(ax)
    ax.set_ylabel("KL(true ‖ gen) / H(true)  —  phases 0–10 mean")
    ax.set_title("Compounding error vs continuation position (day-bootstrap 95% CI)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_compounding_headline.png"))
    plt.close(fig)

    names = list(FIELD_GROUPS) + ["ref_block"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150, sharex=True)
    for k, gname in enumerate(names):
        ax = axes[k // 3][k % 3]
        src = curves["fields"].get(gname, curves["ref_block"])
        for r in rows:
            ax.plot(mx, rolling(np.asarray(src[r]), 25), lw=1.4, color=color[r], label=r)
        ax.plot(mx, rolling(np.asarray(src["floor"]), 25), lw=1.1, ls="--", color=FLOOR_HEX)
        ax.set_title(gname, fontsize=10)
        style(ax)
    axes[0][0].set_ylabel("KL / H(true)")
    axes[1][0].set_ylabel("KL / H(true)")
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Per-field compounding curves (dashed = real-vs-real floor)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_compounding_fields.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150, sharex=True)
    for k, fname in enumerate(["price_rel", "size", "log10_dt"]):
        ax = axes[k]
        for r in rows:
            ax.plot(mx, rolling(np.asarray(curves["zdrift"][r][fname]), 25), lw=1.4,
                    color=color[r], label=r)
        ax.axhline(0, color=FLOOR_HEX, lw=1.0, ls="--")
        ax.set_title(fname, fontsize=10)
        style(ax)
    axes[0].set_ylabel("(mean_gen − mean_true) / std_true")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Literal z-drift per position (z-drift normalization on decoded fields)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_compounding_zdrift.png"))
    plt.close(fig)
    print(f"[analysis] wrote 3 figures -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
