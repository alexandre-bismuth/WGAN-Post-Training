"""S5 — Goodhart-guard eval monitor.

The failure mode of the adversarial loop is the generator reward-hacking a (possibly loosely-Lipschitz,
pretrained-backbone) WGAN critic: critic separation / fake-score RISES while the generation's HELD-OUT
stylized facts stay flat or worsen. With the rank-σ̄ fitness this is the expected trajectory once the
critic is exploitable, not an edge case, so the monitor is CONTROL, not just logging:

  - `stylized_fact_metrics` computes held-out distances of a σ=0 REFERENCE rollout (the current evolving
    head, no perturbation) vs the REAL GOOG continuation replayed through the SAME OrderBook engine, all
    via `lobmamba/lob/evaluation.py` (reuse — do NOT reinvent). It returns a single `composite`
    (lower = closer to real).
  - `normalize_composite` rescales each held-out term by its FIRST-EVAL (pretrained-baseline) value so
    the composite is scale-free (the raw `composite` is swamped by `mid_l1`, which is in raw price units).
  - `goodhart_check` turns the composite history into (a) the BEST checkpoint index (selection = best
    held-out composite, NOT best critic separation — the WS-21 principle) and (b) a `fired` flag (the
    last `patience` evals are ALL worse than the best-so-far by a relative margin `tol` — sustained
    divergence) that the train loop acts on (auto-stop + rollback / escalate λ / retrain D).
  - `population_diversity` is the mode-collapse instrumentation on the free G×Q rollout grid (token
    uniqueness across the G population + per-member event-histogram dispersion).

The σ=0 reference rollout + the engine replay of the real continuation live in the train loop (it holds
the rollout machinery: `gen_fn`, the eval-context batch, `inf._replay_real_msgs_single`); this module is
PURE ARRAYS so the metric wiring + the control logic are fully CPU-testable on a login node.

L2 layout (from `lob/evaluation.py`): a per-step L2 book vector has best-ask price at index 0 and best-bid
price at index 2 (`mid = (l2[0]+l2[2])/2`); `book_loss_l1` reshapes the vector to (price, volume) pairs.
`gen_l2`/`real_l2` are [n_eval, n_gen, W] raw L2 states; event types are [n_eval, n_gen] ints.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def stylized_fact_metrics(gen_l2, real_l2, gen_et, real_et, *, n_levels=10):
    """Held-out stylized-fact distances of the σ=0 reference generation vs the real continuation.

      gen_l2, real_l2 : [n_eval, n_gen, W]  raw L2 book states (gen = reference rollout; real = engine replay)
      gen_et, real_et : [n_eval, n_gen]     decoded event types (1 new / 2 cancel / 3 delete / 4 exec)

    Returns a dict: per-metric distances (lower = closer to real, EXCEPT `ret_corr`/`dir_acc` where
    higher = closer), the return-bench skill pair (`ret_corr` = Pearson, `dir_acc` = direction accuracy),
    and a single `composite` (lower = better) used for checkpoint selection and the Goodhart gate."""
    from lob import evaluation as ev

    ne, ng, W = gen_l2.shape
    gl = gen_l2.reshape(ne * ng, W)
    rl = real_l2.reshape(ne * ng, W)

    # ev.mid_price_loss_l1 is itself @jax.jit @jax.vmap -> it already maps over the leading book axis
    # (the `_batch` alias is a DOUBLE vmap, for [B,N,W]). For our flat [M,W] grid call it directly -> [M].
    mid_l1 = float(jnp.mean(ev.mid_price_loss_l1(gl, rl)))
    book_l1 = float(jnp.mean(ev.book_loss_l1(gl, rl, n_levels)))

    # mid-price returns per eval series over the n_gen steps -> [n_eval, n_gen-1]
    gen_mid = (gen_l2[..., 0] + gen_l2[..., 2]) / 2.0
    real_mid = (real_l2[..., 0] + real_l2[..., 2]) / 2.0
    gen_ret = (gen_mid[:, 1:] - gen_mid[:, :-1]) / (gen_mid[:, :-1] + 1e-8)
    real_ret = (real_mid[:, 1:] - real_mid[:, :-1]) / (real_mid[:, :-1] + 1e-8)
    # return_corr is vmap(in_axes=(1,1)) -> pass [T, series]; mean over the eval series.
    ret_corr = float(jnp.nanmean(ev.return_corr(gen_ret.T, real_ret.T)))

    # return-bench: forecast-skill of the generated path vs the realised path — Pearson
    # coefficient (== ret_corr above) + DIRECTION ACCURACY (sign agreement of the returns).
    # These are paired gen-vs-real metrics for REPORTING; they are NOT the critic's per-sample
    # feature map (a real sample's direction accuracy against itself is degenerate).
    finite_pair = jnp.isfinite(gen_ret) & jnp.isfinite(real_ret)
    dir_acc = float(jnp.sum((jnp.sign(gen_ret) == jnp.sign(real_ret)) * finite_pair)
                    / (jnp.sum(finite_pair) + 1e-8))

    # return-distribution moments (mean / var / skew / kurt) L1 distance.
    # NaN-guard: degenerate eval windows (constant/invalid mid) yield non-finite returns,
    # calc_moments NaNs, and ONE bad window poisons ev_baseline and every later composite
    # (train loop freezes the first eval as the reference). This runs host-side (un-jitted),
    # so boolean filtering is legal: drop non-finite returns before the moments.
    gr = gen_ret.reshape(-1); gr = gr[jnp.isfinite(gr)]
    rr = real_ret.reshape(-1); rr = rr[jnp.isfinite(rr)]
    if gr.size > 1 and rr.size > 1:
        gm = jnp.asarray(ev.calc_moments(gr))
        rm = jnp.asarray(ev.calc_moments(rr))
        moment_l1 = float(jnp.mean(jnp.abs(gm - rm)))
    else:
        moment_l1 = float("nan")    # nothing to measure; normalize_composite guards via baseline

    # normalised event-type histogram L1 distance
    geh = ev.event_type_count(gen_et.reshape(-1)).astype(jnp.float32)
    reh = ev.event_type_count(real_et.reshape(-1)).astype(jnp.float32)
    geh = geh / (jnp.sum(geh) + 1e-8)
    reh = reh / (jnp.sum(reh) + 1e-8)
    event_l1 = float(jnp.sum(jnp.abs(geh - reh)))

    # composite (lower = closer to real): sum of the non-negative distances, ret_corr -> (1 - corr).
    composite = mid_l1 + book_l1 + (1.0 - ret_corr) + moment_l1 + event_l1
    return dict(mid_l1=mid_l1, book_l1=book_l1, ret_corr=ret_corr, dir_acc=dir_acc,
                moment_l1=moment_l1, event_l1=event_l1, composite=float(composite))


def normalize_composite(metrics, baseline, terms=("book_l1", "ret", "moment_l1", "event_l1"), eps=1e-8):
    """Scale-free composite: mean over `terms` of per-term metric/baseline ratios (lower = better).

    WHY: the raw `composite` from `stylized_fact_metrics` SUMS scale-incommensurate terms —
    `mid_price_loss_l1` is in raw price units (observed values in the MILLIONS) and swamps the O(1)
    terms (book_l1, 1-ret_corr, moment_l1, event_l1), so checkpoint selection and the Goodhart gate
    degenerated to a noisy pathwise mid-price L1. Normalizing each term by its FIRST-EVAL
    (pretrained-baseline) value makes the composite a scale-free "relative to the pretrained start"
    number: 1.0 at the baseline, <1 = better. Pathwise `mid_l1` is excluded from the default terms
    because a stochastic rollout's pathwise distance to the ONE realized path is chaos-dominated
    (it is still logged raw by the trainer).

    `metrics` / `baseline` are dicts as returned by `stylized_fact_metrics`. Per-term value: `"ret"` ->
    (1 - metrics["ret_corr"]) / max(1 - baseline["ret_corr"], eps); any other term `t` ->
    metrics[t] / max(baseline[t], eps). Returns the mean over `terms` as a float."""
    vals = []
    for t in terms:
        if t == "ret":
            vals.append((1.0 - metrics["ret_corr"]) / max(1.0 - baseline["ret_corr"], eps))
        else:
            vals.append(metrics[t] / max(baseline[t], eps))
    return float(sum(vals) / len(vals))


def goodhart_check(composites, *, patience=3, tol=0.05):
    """Turn the held-out composite history (eval order, lower=better) into the CONTROL signals:
      best_idx : argmin composite -> the checkpoint to KEEP / roll back to (selection = best held-out
                 stylized-fact composite, NOT best critic separation).
      fired    : True iff the last `patience` evals are ALL worse than the best-so-far by a relative
                 margin `tol`, i.e. len >= patience+1 and all of c[-patience:] > c[best_idx]*(1+tol)
                 (sustained Goodhart divergence) -> the loop should auto-stop + rollback to best_idx.
    MARGIN-VS-BEST, not consecutive-strict-increase: under iid eval noise a strict-increase run of
    length `patience` fires spuriously with prob ~(1/2)^patience per window (12.5% at patience=3)
    even when the composite is FLAT; requiring every recent eval to exceed best*(1+tol) is
    noise-robust — a plateau within `tol` of the best never fires, only genuine divergence does.
    Returns (fired, best_idx); empty history -> (False, -1)."""
    c = [float(x) for x in composites]
    if not c:
        return False, -1
    best_idx = min(range(len(c)), key=lambda i: c[i])
    fired = len(c) >= patience + 1 and all(x > c[best_idx] * (1 + tol) for x in c[-patience:])
    return fired, best_idx


def population_diversity(tokens_grid, et_grid, G, Q, *, token_stride=26):
    """Mode-collapse instrumentation on the G×Q rollout grid: every step the trainer already produces
    G rollouts per shared context (Q contexts) for FREE, so cross-population diversity is observable
    without extra rollouts.

      tokens_grid : [M, T]      generated token ids,    M = Q*G, context-major: rollout r = q*G + g
      et_grid     : [M, n_gen]  decoded event types in {1..4}, same layout

    Returns dict(token_unique_frac, event_hist_disp):
      token_unique_frac : mean over (q, subsampled token position) of #DISTINCT values across the G
                          axis / G (positions subsampled with stride `token_stride` = one per message).
                          Collapsed population (all G rollouts identical) -> exactly 1/G; fully
                          diverse -> approaches min(G, token support)/G.
      event_hist_disp   : per (q, g) the normalized event-type histogram over the n_gen axis,
                          hist[q,g,b] = mean(et == b) for b in 1..4; dispersion = mean over (q, g) of
                          sum_b |hist[q,g,b] - mean_over_g(hist[q,:,b])|. Collapsed -> 0.
    Falling token_unique_frac toward 1/G and event_hist_disp toward 0 across training is the DIRECT
    collapse signal (complements the KL trust region, which only bounds drift from the pretrained
    head, not within-population variety). Pure jnp / jit-friendly: no python loops over Q or G."""
    tok = jnp.asarray(tokens_grid).reshape(Q, G, -1)[..., ::token_stride]     # [Q, G, P]
    s = jnp.sort(tok, axis=1)                                                 # sort along the G axis
    n_unique = 1 + jnp.sum(jnp.diff(s, axis=1) != 0, axis=1)                  # [Q, P] distinct counts
    token_unique_frac = float(jnp.mean(n_unique / G))

    et = jnp.asarray(et_grid).reshape(Q, G, -1)                               # [Q, G, n_gen]
    bins = jnp.arange(1, 5).reshape(1, 1, 1, 4)
    hist = jnp.mean((et[..., None] == bins).astype(jnp.float32), axis=2)      # [Q, G, 4]
    disp = jnp.sum(jnp.abs(hist - jnp.mean(hist, axis=1, keepdims=True)), axis=-1)   # [Q, G]
    return dict(token_unique_frac=token_unique_frac, event_hist_disp=float(jnp.mean(disp)))


# ----------------------------------------------------------------------------------------
# Login-node self-test (no model): metric wiring on synthetic L2 + the Goodhart control logic.
# ----------------------------------------------------------------------------------------
def _cpu_self_test(seed=0):
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("[eval_monitor] CPU self-test — stylized-fact wiring + Goodhart control", flush=True)
    k = jax.random.key(seed)
    ne, ng, n_levels = 6, 8, 5
    W = 2 * n_levels * 2                                   # (price,volume) pairs, 2*n_levels levels
    # plausible positive L2 (prices descending-ish, positive volumes) — only relative structure matters
    base = jnp.abs(jax.random.normal(jax.random.fold_in(k, 1), (ne, ng, W))) + 1.0
    et = jax.random.randint(jax.random.fold_in(k, 2), (ne, ng), 1, 5)

    # identical gen == real -> distances ~0, ret_corr ~1, composite ~0.
    m_id = stylized_fact_metrics(base, base, et, et, n_levels=n_levels)
    chk("(M1) identical -> mid_l1~0", m_id["mid_l1"] < 1e-5, f"mid_l1={m_id['mid_l1']:.2e}")
    chk("(M2) identical -> book_l1~0", m_id["book_l1"] < 1e-5, f"book_l1={m_id['book_l1']:.2e}")
    chk("(M3) identical -> ret_corr~1", abs(m_id["ret_corr"] - 1.0) < 1e-4, f"ret_corr={m_id['ret_corr']:.4f}")
    chk("(M3b) identical -> dir_acc~1 (return-bench)", abs(m_id["dir_acc"] - 1.0) < 1e-4,
        f"dir_acc={m_id['dir_acc']:.4f}")
    chk("(M4) identical -> event_l1~0", m_id["event_l1"] < 1e-6, f"event_l1={m_id['event_l1']:.2e}")
    chk("(M5) identical -> composite~0", m_id["composite"] < 1e-4, f"composite={m_id['composite']:.2e}")

    # perturbed gen -> positive composite, distinct event histogram.
    gen2 = base + 0.5 * jax.random.normal(jax.random.fold_in(k, 3), base.shape)
    et2 = jax.random.randint(jax.random.fold_in(k, 4), (ne, ng), 1, 5)
    m_pt = stylized_fact_metrics(gen2, base, et2, et, n_levels=n_levels)
    chk("(M6) perturbed -> composite > identical", m_pt["composite"] > m_id["composite"] + 1e-3,
        f"comp {m_id['composite']:.3e} -> {m_pt['composite']:.3e}")
    chk("(M7) all metrics finite", all(jnp.isfinite(jnp.asarray(v)) for v in m_pt.values()),
        f"{ {kk: round(vv,3) for kk,vv in m_pt.items()} }")

    # Goodhart control: best_idx = argmin; fired iff the last `patience` evals ALL exceed best*(1+tol).
    fired_dec, best_dec = goodhart_check([1.0, 0.8, 0.6, 0.5], patience=2)
    chk("(M8) improving history -> not fired, best=last", (not fired_dec) and best_dec == 3,
        f"fired={fired_dec} best={best_dec}")
    fired_inc, best_inc = goodhart_check([0.3, 0.4, 0.5, 0.6], patience=2, tol=0.05)
    chk("(M9) clear divergence -> fired, best=first", fired_inc and best_inc == 0,
        f"fired={fired_inc} best={best_inc}")
    fired_mix, best_mix = goodhart_check([0.5, 0.3, 0.4], patience=3)
    chk("(M10) short/mixed history -> not fired, best=min", (not fired_mix) and best_mix == 1,
        f"fired={fired_mix} best={best_mix}")
    fired_pl, best_pl = goodhart_check([0.5, 0.51, 0.52, 0.51], patience=2, tol=0.05)
    chk("(M11) plateau within tol -> not fired (noise-robust)", (not fired_pl) and best_pl == 0,
        f"fired={fired_pl} best={best_pl}")
    fired_e, best_e = goodhart_check([], patience=2)
    chk("(M12) empty history -> (False, -1)", (not fired_e) and best_e == -1,
        f"fired={fired_e} best={best_e}")

    # normalized composite: scale-free vs the first-eval baseline (1.0 at baseline, <1 = better).
    nc_base = normalize_composite(m_pt, m_pt)
    chk("(M13) baseline==metrics -> normalized composite == 1.0", abs(nc_base - 1.0) < 1e-6,
        f"nc={nc_base:.8f}")
    m_half = dict(m_pt, book_l1=m_pt["book_l1"] / 2.0)
    nc_half = normalize_composite(m_half, m_pt)
    chk("(M14) halving book_l1 -> normalized composite < 1.0", nc_half < 1.0 - 1e-6,
        f"nc={nc_half:.6f}")

    # population diversity on the G×Q grid: collapsed -> 1/G & ~0; random -> well above both floors.
    Qd, Gd, Td, ngd = 3, 8, 130, 64
    tok1 = jax.random.randint(jax.random.fold_in(k, 5), (Qd, 1, Td), 0, 600)
    et1 = jax.random.randint(jax.random.fold_in(k, 6), (Qd, 1, ngd), 1, 5)
    d_col = population_diversity(jnp.tile(tok1, (1, Gd, 1)).reshape(Qd * Gd, Td),
                                 jnp.tile(et1, (1, Gd, 1)).reshape(Qd * Gd, ngd), Gd, Qd)
    chk("(M15) collapsed grid -> token_unique_frac == 1/G",
        abs(d_col["token_unique_frac"] - 1.0 / Gd) < 1e-6, f"frac={d_col['token_unique_frac']:.6f}")
    chk("(M16) collapsed grid -> event_hist_disp ~0", d_col["event_hist_disp"] < 1e-6,
        f"disp={d_col['event_hist_disp']:.2e}")
    tok_r = jax.random.randint(jax.random.fold_in(k, 7), (Qd * Gd, Td), 0, 600)
    et_r = jax.random.randint(jax.random.fold_in(k, 8), (Qd * Gd, ngd), 1, 5)
    d_rnd = population_diversity(tok_r, et_r, Gd, Qd)
    chk("(M17) random grid -> token_unique_frac > 2/G", d_rnd["token_unique_frac"] > 2.0 / Gd,
        f"frac={d_rnd['token_unique_frac']:.4f}")
    chk("(M18) random grid -> event_hist_disp > 0.01", d_rnd["event_hist_disp"] > 0.01,
        f"disp={d_rnd['event_hist_disp']:.4f}")

    print("\n[eval_monitor] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _cpu_self_test() else 0)
