"""S4 — fitness for the EGGROLL G-step: a WGAN critic "realness" score (+ optional KL trust region).

The generator wants to MAXIMISE the critic's score on its fakes (fool D). EGGROLL ASCENDS raw
fitness: `EggRoll._do_update` returns `-(grad·√N)` and optax `apply_updates` then adds the solver's
update, i.e. gradient ASCENT on the raw scores fed to `convert_fitnesses` (verified in
HyperscaleES/.../noiser/eggroll.py). So the per-member raw fitness is

    F_i = D(fake_i)  -  kl_coef · KL_i ,

where D(fake_i) is the spectral-normed WGAN critic's scalar on member i's [context ; continuation]
window, and KL_i is an OPTIONAL per-position categorical trust-region penalty between the perturbed
head policy and the reference head policy along the realised rollout. With kl_coef=0 the fitness is
the pure critic score (the S4 smoke default; KL is added only if Goodhart appears).

WHY THIS KL IS COMPUTABLE WITHOUT TOUCHING `generate()`: the discriminator backbone is the SAME frozen
Mamba3 and already produces the pre-decoder hidden state h_t for every continuation position when we
extract critic features. The two policies differ ONLY in the decoder head, so on the realised
trajectory KL_t = KL( softmax(h_t·W_i + b_i) ‖ softmax(h_t·W_ref + b_ref) ) — a teacher-forced
per-position categorical KL evaluated on the SAME hiddens. Time-token positions (TIME_START_I..
TIME_END_I) are deterministic (computed from Δt, not sampled), so the head's output there is never
realised; `build_time_mask` excludes them so the KL is a faithful policy divergence.

CPU-safe: every function here is pure JAX over arrays. The EXPENSIVE backbone pass is inherited from
`data.pooled_features` / the unpooled `cont_hidden` here; run scoring on the GH200 (real backbone
features = compute). The `__main__` self-test exercises the KL + time-mask math on tiny synthetic
arrays and runs on a login node (no model build).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import DEFAULT as _CFG
from ..data import loaders as Ddata
from ..critic.discriminator import PaddedLobPredFeatures

MSG_LEN = _CFG.model.msg_len            # 26
VOCAB = _CFG.model.vocab_size           # 2112
TIME_START_I = _CFG.model.time_start_i  # 11
TIME_END_I = _CFG.model.time_end_i      # 15


# ----------------------------------------------------------------------------------------
# WGAN critic "realness" score on [context ; continuation] windows (the G fitness signal).
# ----------------------------------------------------------------------------------------
def window_features(backbone, bb_params, ctx_tokens, cont_tokens, ctx_book, cont_book,
                    *, pooling="mean", chunk=0, pool_start=0):
    """Frozen-backbone pooled features for [ctx ; cont] windows -> [N, d_model]. Used for BOTH the
    critic's real class and the generator's fake class (identical path -> a fair comparison).
    pool_start: first token position included in 'mean' pooling (n_cond*26 -> continuation-only)."""
    tokens, book = Ddata.assemble_window(ctx_tokens, cont_tokens, ctx_book, cont_book)
    return Ddata.pooled_features(backbone, bb_params, tokens, book, pooling=pooling, chunk=chunk,
                                 pool_start=pool_start)


def critic_scores(head, head_vars, backbone, bb_params, ctx_tokens, cont_tokens, ctx_book, cont_book,
                  *, pooling="mean", chunk=0):
    """Convenience: pooled features -> critic scalar score per window [N]. (The train loop computes
    `window_features` once and reuses it for both D and G, so it does NOT call this — provided for
    eval / external use where only the score is wanted.)"""
    feats = window_features(backbone, bb_params, ctx_tokens, cont_tokens, ctx_book, cont_book,
                            pooling=pooling, chunk=chunk)
    return head.apply(head_vars, feats, train=False)


# ----------------------------------------------------------------------------------------
# KL trust region (optional; kl_coef=0 in the smoke).
# ----------------------------------------------------------------------------------------
def build_time_mask(n_gen, msg_len=MSG_LEN, time_start_i=TIME_START_I, time_end_i=TIME_END_I):
    """Per-continuation-position weight [n_gen*26]: 1.0 at SAMPLED positions, 0.0 at the deterministic
    time block [time_start_i, time_end_i] within each message (those tokens are computed from Δt, not
    sampled, so the head's logits there are never realised -> excluded from the policy KL)."""
    j = jnp.arange(msg_len)
    per_msg = ~((j >= time_start_i) & (j <= time_end_i))     # True = sampled (kept)
    return jnp.tile(per_msg, n_gen).astype(jnp.float32)      # [n_gen*26]


def cont_hidden(backbone, bb_params, ctx_tokens, cont_tokens, ctx_book, cont_book, n_cond, *, chunk=0):
    """Frozen-backbone POLICY hidden states for the continuation tokens -> [N, n_gen*26, d_model].

    Next-token convention: the distribution that SAMPLED continuation token t (absolute position
    start+t, start=n_cond*26) is head(h_{start+t-1}) — the hidden after consuming the PREVIOUS token.
    So the policy hiddens for the T=n_gen*26 realised continuation tokens are feats[start-1 : L-1]
    (NOT feats[start:], which is shifted one position late and includes a hidden whose 'next token'
    lies beyond the window — an off-by-one). This also re-aligns
    build_time_mask: kl_t at index t now matches token t's sampled/deterministic status."""
    tokens, book = Ddata.assemble_window(ctx_tokens, cont_tokens, ctx_book, cont_book)
    x_m, x_b, m_ts, b_ts = Ddata.critic_batch_from_tokens(tokens, book)
    start = n_cond * MSG_LEN

    def one(xm, xb, mt, bt):
        feats = backbone.apply({"params": bb_params}, xm, xb, mt, bt,
                               method=PaddedLobPredFeatures.features)     # [L, d_model]
        return feats[start - 1:-1]                                       # [n_gen*26, d_model] policy hiddens

    N = x_m.shape[0]
    if chunk and chunk < N:
        outs = [jax.vmap(one)(x_m[s:s + chunk], x_b[s:s + chunk], m_ts[s:s + chunk], b_ts[s:s + chunk])
                for s in range(0, N, chunk)]
        return jnp.concatenate(outs, axis=0)
    return jax.vmap(one)(x_m, x_b, m_ts, b_ts)


def head_kl_penalty(hidden_cont, W_ref, b_ref, W_pop, b_pop, *, time_mask=None):
    """Per-member mean per-position KL( perturbed-head policy ‖ reference-head policy ) on the realised
    continuation hiddens.

      hidden_cont : [N, T, d]   (T = n_gen*26 continuation positions)
      W_ref       : [d, V]      reference decoder kernel ;  b_ref : [V]
      W_pop       : [N, d, V]   per-member perturbed decoder kernels ;  b_pop : [N, V]
      time_mask   : [T] in {0,1} (optional) — 0 at deterministic time positions (see build_time_mask)

    Returns [N] (>= 0; == 0 iff W_pop_i==W_ref and b_pop_i==b_ref). KL>0 penalises members whose head
    drifts far from the reference policy -> a trust region around the pretrained generator."""
    logits_ref = jnp.einsum('ntd,dv->ntv', hidden_cont, W_ref) + b_ref            # [N,T,V]
    logits_i = jnp.einsum('ntd,ndv->ntv', hidden_cont, W_pop) + b_pop[:, None, :]  # [N,T,V]
    logp_i = jax.nn.log_softmax(logits_i, axis=-1)
    logp_r = jax.nn.log_softmax(logits_ref, axis=-1)
    kl_t = jnp.sum(jnp.exp(logp_i) * (logp_i - logp_r), axis=-1)                   # [N,T]  KL(i‖ref)
    if time_mask is None:
        return jnp.mean(kl_t, axis=1)
    w = time_mask[None]                                                            # [1,T]
    return jnp.sum(kl_t * w, axis=1) / jnp.maximum(jnp.sum(time_mask), 1.0)        # [N]


def raw_fitness(scores, kl, kl_coef):
    """F_i = D(fake_i) - kl_coef * KL_i (ascended by EggRoll: higher D = more 'real' = better)."""
    return scores - kl_coef * kl


# ----------------------------------------------------------------------------------------
# S5 — centered-rank σ̄ fitness (App-M): per-context rank across G directions, averaged over Q.
# ----------------------------------------------------------------------------------------
def rank_sigma_bar(scores_qg):
    """App-M centered-rank σ̄ fitness. `scores_qg`: [Q, G], context-major (s[q,g] = the critic score of
    direction g on context q). For each context q, ascending-rank the G scores and center to [-1/2, 1/2]
    (higher score -> higher fitness, the direction EGGROLL ascends), then average over the Q contexts ->
    [G]. Uses ONLY the ordering of the scores, so it is robust to the fat-tailed/outlier-driven critic
    scores a z-norm denominator would flatten. Returns the per-direction (G,) vector `do_updates` consumes
    (so `sqrt(fitnesses.size) == sqrt(G)` is consistent — NOT the G·Q matrix)."""
    Q, G = scores_qg.shape
    order = jnp.argsort(scores_qg, axis=1)                       # ascending sort indices, per context
    ranks = jnp.argsort(order, axis=1).astype(jnp.float32)       # ascending rank 0..G-1 of each direction
    centered = ranks / jnp.maximum(G - 1, 1) - 0.5               # [-1/2, 1/2]
    return jnp.mean(centered, axis=0)                            # [G]


def znorm_sigma_bar(scores_qg, *, winsor=0.0):
    """Winsorized z-norm σ̄ FALLBACK (per-context z-norm across G, averaged over Q). winsor=0 is plain
    z-norm (the outlier-sensitive baseline the rank transform replaces); winsor>0 clips each context's
    scores to its [winsor, 1-winsor] quantiles first. Kept for ablation; `rank_sigma_bar` is the live one."""
    s = scores_qg
    if winsor > 0.0:
        lo = jnp.quantile(s, winsor, axis=1, keepdims=True)
        hi = jnp.quantile(s, 1.0 - winsor, axis=1, keepdims=True)
        s = jnp.clip(s, lo, hi)
    z = (s - jnp.mean(s, axis=1, keepdims=True)) / (jnp.std(s, axis=1, keepdims=True) + 1e-8)
    return jnp.mean(z, axis=0)


def _kl_from_hiddens(hidden_cont, W_ref, b_ref, W_pop_dirs, b_pop_dirs, dir_idx, *, time_mask=None):
    """Per-rollout head KL from PRECOMPUTED continuation hiddens [M,T,d], gathering each rollout's
    direction head from `W_pop_dirs[dir_idx]` (so the full tiled [M,d,V] kernel is never built).
    `W_pop_dirs`:[G,d,V], `b_pop_dirs`:[G,V], `dir_idx`:[M] (= r%G). CPU-testable (no backbone)."""
    return head_kl_penalty(hidden_cont, W_ref, b_ref, W_pop_dirs[dir_idx], b_pop_dirs[dir_idx],
                           time_mask=time_mask)


def kl_penalty_chunked(backbone, bb_params, ctx_tokens, cont_tokens, ctx_book, cont_book, n_cond,
                       W_ref, b_ref, W_pop_dirs, b_pop_dirs, dir_idx, *, time_mask=None, chunk=8):
    """Memory-bounded per-rollout head KL over the G×Q grid (M = ctx_tokens.shape[0] rollouts). Slices the
    grid into `chunk` rollouts so neither the [chunk,T,d] hiddens nor the [chunk,T,V=2112] logits ever
    materialise full (at G=512 the un-chunked [M,T,V] tensor is hundreds of GB). Per slice: `cont_hidden`
    -> `_kl_from_hiddens` (gathering the slice's direction heads). Returns [M]. (kl_coef=0 in the sweep
    skips this entirely; it binds only on the full run.)"""
    M = ctx_tokens.shape[0]
    c = chunk if (chunk and chunk < M) else M
    outs = []
    for s in range(0, M, c):
        e = min(s + c, M)
        hid = cont_hidden(backbone, bb_params, ctx_tokens[s:e], cont_tokens[s:e],
                          ctx_book[s:e], cont_book[s:e], n_cond, chunk=0)          # [c, T, d]
        outs.append(_kl_from_hiddens(hid, W_ref, b_ref, W_pop_dirs, b_pop_dirs, dir_idx[s:e],
                                     time_mask=time_mask))
    return jnp.concatenate(outs, axis=0)                                           # [M]


# ----------------------------------------------------------------------------------------
# Login-node self-test (no model): KL + time-mask math on tiny synthetic arrays.
# ----------------------------------------------------------------------------------------
def _cpu_self_test(seed=0):
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("[fitness] CPU self-test — time mask + head KL math", flush=True)
    n_gen, d, V, N = 3, 8, 10, 4
    T = n_gen * MSG_LEN
    k = jax.random.key(seed)
    h = jax.random.normal(jax.random.fold_in(k, 1), (N, T, d))
    W_ref = jax.random.normal(jax.random.fold_in(k, 2), (d, V))
    b_ref = jax.random.normal(jax.random.fold_in(k, 3), (V,))

    # time mask: right shape, exactly the time block zeroed each message.
    tm = build_time_mask(n_gen)
    n_time = (TIME_END_I - TIME_START_I + 1)
    chk("(F1) time_mask shape", tm.shape == (T,), f"shape={tuple(tm.shape)}")
    chk("(F2) time_mask zeros == n_gen*time_block",
        int(jnp.sum(tm == 0.0)) == n_gen * n_time, f"zeros={int(jnp.sum(tm == 0.0))} exp={n_gen * n_time}")
    chk("(F3) time_mask zeros AT the time block (msg 0)",
        bool(jnp.all(tm[TIME_START_I:TIME_END_I + 1] == 0.0))
        and bool(tm[0] == 1.0) and bool(tm[TIME_END_I + 1] == 1.0))

    # KL == 0 when the population head == the reference head (tiled), regardless of time mask.
    W_pop0 = jnp.broadcast_to(W_ref, (N, d, V))
    b_pop0 = jnp.broadcast_to(b_ref, (N, V))
    kl0 = head_kl_penalty(h, W_ref, b_ref, W_pop0, b_pop0)
    kl0m = head_kl_penalty(h, W_ref, b_ref, W_pop0, b_pop0, time_mask=tm)
    chk("(F4) KL==0 at W_pop==W_ref", float(jnp.max(jnp.abs(kl0))) < 1e-6, f"max={float(jnp.max(jnp.abs(kl0))):.2e}")
    chk("(F5) KL==0 also with time mask", float(jnp.max(jnp.abs(kl0m))) < 1e-6)

    # KL > 0 and >= 0 everywhere when the head is perturbed.
    dW = 0.3 * jax.random.normal(jax.random.fold_in(k, 4), (N, d, V))
    db = 0.3 * jax.random.normal(jax.random.fold_in(k, 5), (N, V))
    kl1 = head_kl_penalty(h, W_ref, b_ref, W_ref[None] + dW, b_ref[None] + db, time_mask=tm)
    chk("(F6) KL>=0 always", bool(jnp.all(kl1 >= -1e-7)), f"min={float(jnp.min(kl1)):.3e}")
    chk("(F7) KL>0 under perturbation", float(jnp.min(kl1)) > 1e-6, f"min={float(jnp.min(kl1)):.3e}")

    # raw_fitness is monotone in scores and decreasing in KL.
    scores = jnp.array([0.0, 1.0, 2.0, 3.0])
    rf = raw_fitness(scores, kl1, 0.5)
    chk("(F8) raw_fitness = scores - 0.5*KL", float(jnp.max(jnp.abs(rf - (scores - 0.5 * kl1)))) < 1e-6)
    chk("(F9) kl_coef=0 -> raw_fitness == scores",
        float(jnp.max(jnp.abs(raw_fitness(scores, kl1, 0.0) - scores))) < 1e-6)

    # --- S5 centered-rank σ̄ ---------------------------------------------------------------
    Qc, Gc = 5, 8
    sc = jax.random.normal(jax.random.fold_in(k, 7), (Qc, Gc))
    F = rank_sigma_bar(sc)
    chk("(F10) rank_sigma_bar shape (G,) & range [-1/2,1/2]",
        F.shape == (Gc,) and float(jnp.max(jnp.abs(F))) <= 0.5 + 1e-6, f"shape={tuple(F.shape)}")
    # a direction that is strictly best in EVERY context -> top rank each time -> F == +1/2.
    sc_best = sc.at[:, 3].set(jnp.max(sc, axis=1) + 1.0)
    chk("(F11) strictly-best direction -> F=+1/2", abs(float(rank_sigma_bar(sc_best)[3]) - 0.5) < 1e-6,
        f"F_best={float(rank_sigma_bar(sc_best)[3]):.4f}")
    # OUTLIER ROBUSTNESS: inflating the already-largest score per context preserves ordering ->
    # rank σ̄ IDENTICAL, while z-norm σ̄ shifts (the property motivating the rank transform).
    sc_out = sc.at[jnp.arange(Qc), jnp.argmax(sc, axis=1)].add(1e3)
    rank_same = float(jnp.max(jnp.abs(rank_sigma_bar(sc) - rank_sigma_bar(sc_out)))) < 1e-6
    znorm_moved = float(jnp.max(jnp.abs(znorm_sigma_bar(sc) - znorm_sigma_bar(sc_out)))) > 1e-3
    chk("(F12) rank σ̄ outlier-invariant; z-norm σ̄ is not", rank_same and znorm_moved,
        f"rank_same={rank_same} znorm_moved={znorm_moved}")
    # KL gather (_kl_from_hiddens): per-rollout direction head gathered by dir_idx == explicit tiling.
    Mg = 6
    hidg = jax.random.normal(jax.random.fold_in(k, 8), (Mg, T, d))
    Wpd = W_ref[None] + 0.2 * jax.random.normal(jax.random.fold_in(k, 9), (Gc, d, V))
    bpd = b_ref[None] + 0.2 * jax.random.normal(jax.random.fold_in(k, 10), (Gc, V))
    didx = jnp.arange(Mg, dtype=jnp.int32) % Gc
    kl_gather = _kl_from_hiddens(hidg, W_ref, b_ref, Wpd, bpd, didx, time_mask=tm)
    kl_explicit = head_kl_penalty(hidg, W_ref, b_ref, Wpd[didx], bpd[didx], time_mask=tm)
    chk("(F13) _kl_from_hiddens gather == explicit-tiled head_kl_penalty",
        float(jnp.max(jnp.abs(kl_gather - kl_explicit))) < 1e-6, f"M={Mg}")

    print("\n[fitness] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _cpu_self_test() else 0)
