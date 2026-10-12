"""Learned-encoder WGAN critic (Lever 3).

Unlike the backbone critic (input is the generator's own frozen pretraining hidden, mean-pooled
over the continuation, so it is blind to the temporal drift that is the realism gap) and the
stylized/dynamical critics (a FIXED, hand-designed φ), this critic LEARNS its own representation
of the rollout's dynamics adversarially. Its input is a per-step microstructure descriptor
SEQUENCE [T, F_in] (interpretable channels, kept UNPOOLED); a small neural encoder over time →
pooled vector → the existing spectral-normed CriticHead → scalar score.

LITERATURE GROUNDING (architecture & robustness):
  ENCODER — causal dilated 1-D temporal conv net (TCN):
    * Bai, Kolter & Koltun (2018) "An Empirical Evaluation of Generic Convolutional and
      Recurrent Networks for Sequence Modeling" (arXiv:1803.01271) — the canonical TCN: residual
      blocks of 2 dilated causal convs, kernel k=3, dilation doubling 1,2,4,…; TCN ≥ LSTM/GRU on
      long sequences (longer memory, no vanishing gradient, full parallelism over time).
    * van den Oord et al. (2016) "WaveNet" (arXiv:1609.03499) — dilated causal convolutions.
    * Wiese, Knobloch, Korn & Kretschmer (2020) "Quant GANs" (arXiv:1907.06673) — TCN proven on
      *financial* series, specifically for long-range volatility clustering.
    Receptive field with 2 convs/block over L blocks, kernel k: RF = 1 + 2(k-1)(2^L - 1). For the
    T=500 continuation, k=3, L=7 (dilations 1..64) gives RF = 509 ≥ 500 → full causal coverage.
    POOLING — attention (a learned query over the encoder outputs) dominates last-state for
    sequence-level scoring; mean is the robust, harder-to-hack fallback; never last-state for a
    critic (throws away the sequence, easiest to game).
  ROBUSTNESS (this critic is the MOST hack-prone — capacity discipline matters as much as power):
    * Spectral normalization on EVERY conv and dense layer — Miyato et al. (2018) "Spectral
      Normalization for GANs" (arXiv:1802.05957). Conv rule: reshape the kernel (k, C_in, C_out)
      to the 2-D matrix (k·C_in, C_out), one-step power iteration, divide by σ_max. SN on all
      discriminator layers is the established default (SN-GAN/SAGAN/BigGAN).
    * R1 zero-centered gradient penalty on REAL inputs — Mescheder, Geiger & Nowozin (2018)
      (arXiv:1801.04406); γ≈10 canonical, raise to de-hack. Applied by make_d_step on the [N,T,F_in]
      input tensor (ndim-agnostic), so a learned critic gets R1 for free.
    * TTUR — a faster critic lr than the generator step, Adam β1=0 — Heusel et al. (2017)
      (arXiv:1706.08500). Exposed as enc_lr / enc_adam_b1 and used by make_critic for the learned
      critic only (the EGGROLL ES "lr" is the generator side).
    * Capacity discipline over raw scale (Karras et al. 2020, ADA, arXiv:2006.06567): a learned
      discriminator memorises on limited data; keep width modest, SN+R1 on, and watch the in-loop
      corr(reward,realism) hack gate as the live kill-switch.
  Multi-scale ensemble config — capacity is bounded by robustness, not maximised: a bigger single
  net/transformer/SSM accelerates reward-hacking under frozen-G ES/GRPO (Gao 2023 √KL over-
  optimisation; Karras-ADA 2020 D-overfit; Kim 2021 attention is non-Lipschitz):
    * MultiScaleMember = parallel TCNs over avg-pooled views (scales 1,2,4) — MelGAN (Kumar 2019) /
      HiFi-GAN (Kong 2020) multi-scale discriminator; structural diversity over stacking copies. SN-clean.
    * EnsembleLearnedCritic = `ens_size` seed-diverse members; the generator's reward is the PESSIMISTIC
      mean−β·std (Coste 2024 conservative/uncertainty-penalised — ~eliminates 70% of best-of-n
      over-optimisation), penalising any single member's blind spot. The KL trust region is the PRIMARY
      anti-Goodhart lever (Gao 2023); this ensemble is the second. ens_size=1 + one scale = the bare TCN.

CONTRACT — TWO public surfaces:
  1. Featuriser (the trainer's `LearnedFeats` plumbing binds `_FEAT = this module`):
     `batch_features(l2, msgs, *, n_levels, tick_size) -> [N, T, F_in]` (the per-step descriptor
     SEQUENCE, UNPOOLED), `FEATURE_NAMES` (the F_in channel names), `N_FEATURES = F_in`, and
     SEQUENCE-aware `fit_normalizer`/`standardize` (per-channel stats over (N,T)).
  2. Network: the Flax `LearnedCritic` module + `make_learned_critic(cfg, T, F_in, seed)` /
     `make_learned_tx(cfg)` used by the trainer's `make_critic` learned branch. Its `.apply(vars, x,
     train=...)` signature MATCHES CriticHead, so the shared make_d_step / score path is unchanged.

NOTE on time channels: the descriptor excludes them all, but for two different reasons.
  * Δt (inter-arrival) is a TRAINED + SAMPLED field — post-training genuinely reshapes its
    distribution (it is a LOB-Bench eval feature). It is kept out of THIS channel set only for
    continuity with the sealed production runs; it is a legitimate candidate channel for future
    critics, gated on a real-vs-fake AUC probe of existing rollouts first.
  * ABSOLUTE time (time_s/time_ns) stays excluded on principle: it is nonstationary within the
    trading day, it is loss-masked in pretraining (lobmamba/lob/train_helpers.py zeroes CE on
    TIME_START_I..TIME_END_I) and derived at rollout as t_i = t_{i-1} + Δt_i
    (inference_no_errcorr._add_time_tokens), so fake rollouts satisfy time_i - time_{i-1} == Δt_i
    EXACTLY while real recorded times vs the quantized Δt encoding can disagree in the last
    digits — an exploitable real-vs-fake consistency artifact for a critic shown both Δt and
    absolute time. (An earlier version of this note claimed fake messages carry placeholder
    times; they do not — the rollout clock is a genuine Δt-accumulated timestamp.)
Channels below are all reconstructable identically for real (engine-replayed) and fake
(engine-produced) rollouts.

CPU-safe: pure jnp / flax, no model build. The `__main__` self-test exercises the descriptor map,
the network forward/vmap, spectral-norm activity, real-vs-fake separation under a local d-step, and
that R1 shrinks the input-gradient norm — all on tiny synthetic arrays.
"""
from __future__ import annotations

import math
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from .discriminator import _l2n, CriticHead

_EPS = 1e-8

# Decoded-message field indices (mirror lobmamba/lob/inference_no_errcorr.py:55-68).
EVENT_TYPE_I = 1
DIRECTION_I = 2
SIZE_I = 5


def _ofi_series(bid_p, bid_v, ask_p, ask_v):
    """Cont-Kukanov-Stoikov (2014) per-event order-flow imbalance e_n at the best quotes.

    For each transition (t-1 -> t): the bid-side and ask-side contributions are the best-queue
    size changes CONDITIONED on the best-quote price move (a new/improved level contributes its
    full size; a worsened level subtracts the old size; an unchanged level contributes the size
    delta). OFI = ΔW^bid - ΔW^ask; OFI>0 ⇒ net upward pressure. Length T-1. Verified equal to the
    paper's e_n = 1{Pᵇ↑}qᵇ_n − 1{Pᵇ↓}qᵇ_{n-1} − 1{Pᵃ↓}qᵃ_n + 1{Pᵃ↑}qᵃ_{n-1}."""
    pb_t, pb_p, qb_t, qb_p = bid_p[1:], bid_p[:-1], bid_v[1:], bid_v[:-1]
    dWb = jnp.where(pb_t > pb_p, qb_t, jnp.where(pb_t == pb_p, qb_t - qb_p, -qb_p))
    pa_t, pa_p, qa_t, qa_p = ask_p[1:], ask_p[:-1], ask_v[1:], ask_v[:-1]
    dWa = jnp.where(pa_t < pa_p, qa_t, jnp.where(pa_t == pa_p, qa_t - qa_p, -qa_p))
    return dWb - dWa                                   # [T-1]


# ========================================================================================
# Part 1 — per-step microstructure descriptor sequence  (l2, msgs) -> [T, F_in]
# ========================================================================================
FEATURE_NAMES = (
    "spread",            # best-ask - best-bid, ticks
    "mid_logret",        # Δ log mid (0 at t=0)
    "log_bid_touchvol",  # log1p best-bid volume
    "log_ask_touchvol",  # log1p best-ask volume
    "log_depth",         # log1p total displayed depth (all levels)
    "imbalance",         # touch order-flow imbalance (bid-ask)/(bid+ask)
    "ofi",               # Cont-Kukanov-Stoikov per-event OFI (0 at t=0), scaled
    "signed_dir",        # message direction -> {-1,+1}
    "log_size",          # log1p order size
    "et_new",            # event one-hot: new limit order
    "et_cancel",         # event one-hot: cancel
    "et_delete",         # event one-hot: delete
    "et_exec",           # event one-hot: execution
    "book_valid",        # 1 if the book step is well-formed (uncrossed, both touches present)
)
N_FEATURES = len(FEATURE_NAMES)            # F_in
_OFI_SCALE = 1.0e-2                         # bring per-event OFI (~10^2 shares) toward O(1) pre-standardise


def rollout_descriptor(l2, msgs, *, n_levels: int, tick_size: float):
    """Per-step descriptor of a SINGLE rollout -> [T, F_in]. Pure jnp; vmap externally."""
    l2 = jnp.asarray(l2, jnp.float32)
    msgs = jnp.asarray(msgs, jnp.float32)
    T = l2.shape[0]

    best_ask_p = l2[:, 0]; best_ask_v = l2[:, 1]
    best_bid_p = l2[:, 2]; best_bid_v = l2[:, 3]
    ask_v = l2[:, 1::4]; bid_v = l2[:, 3::4]
    book_valid = ((best_ask_p > 0) & (best_bid_p > 0) & (best_ask_p > best_bid_p)).astype(jnp.float32)

    spread = (best_ask_p - best_bid_p) / tick_size
    mid = (best_ask_p + best_bid_p) / 2.0
    log_mid = jnp.where(mid > 0.0, jnp.log(jnp.maximum(mid, _EPS)), 0.0)
    mid_logret = jnp.concatenate([jnp.zeros(1), log_mid[1:] - log_mid[:-1]])      # 0 at t=0
    log_bid_tv = jnp.log1p(jnp.maximum(best_bid_v, 0.0))
    log_ask_tv = jnp.log1p(jnp.maximum(best_ask_v, 0.0))
    total_depth = jnp.sum(jnp.maximum(ask_v, 0.0) + jnp.maximum(bid_v, 0.0), axis=1)
    log_depth = jnp.log1p(total_depth)
    imb = (best_bid_v - best_ask_v) / (best_bid_v + best_ask_v + _EPS)
    ofi = jnp.concatenate([jnp.zeros(1),
                           _ofi_series(best_bid_p, best_bid_v, best_ask_p, best_ask_v)]) * _OFI_SCALE

    event = msgs[:, EVENT_TYPE_I]
    direction = msgs[:, DIRECTION_I]
    size = msgs[:, SIZE_I]
    signed_dir = direction * 2.0 - 1.0
    log_size = jnp.log1p(jnp.maximum(size, 0.0))
    et_new = (event == 1.0).astype(jnp.float32)
    et_cancel = (event == 2.0).astype(jnp.float32)
    et_delete = (event == 3.0).astype(jnp.float32)
    et_exec = (event == 4.0).astype(jnp.float32)

    desc = jnp.stack([
        spread, mid_logret, log_bid_tv, log_ask_tv, log_depth, imb, ofi,
        signed_dir, log_size, et_new, et_cancel, et_delete, et_exec, book_valid,
    ], axis=1)                                                                    # [T, F_in]
    return jnp.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)


def batch_features(l2, msgs, *, n_levels: int, tick_size: float):
    """Vectorised descriptor. l2: [N, T, W], msgs: [N, T, F] -> [N, T, F_in]."""
    return jax.vmap(lambda a, b: rollout_descriptor(a, b, n_levels=n_levels, tick_size=tick_size))(l2, msgs)


# --- sequence-aware standardiser (per-channel stats over (N, T); shadows the stylized 2-D one) ---
def fit_normalizer(real_seq, *, std_floor: float = 1e-3, robust: bool = False):
    """Per-CHANNEL (center, scale) over the real pool [N, T, F_in], reduced over (sample, time). scale is
    floored so a (near-)constant channel (a rarely-firing one-hot, book_valid≈1) is not divided by ~0.

    robust=False -> (mean, std) (default, bit-identical). robust=True -> (median, 1.4826·MAD): heavy-tail-
    safe standardisation (the OFI channel has fat tails, so a σ-based scale is dominated by a few events);
    the 1.4826 factor makes MAD a consistent σ-estimate under a Gaussian so the two modes are comparable."""
    flat = jnp.asarray(real_seq, jnp.float32).reshape(-1, jnp.shape(real_seq)[-1])
    if robust:
        center = jnp.median(flat, axis=0)
        scale = jnp.maximum(1.4826 * jnp.median(jnp.abs(flat - center), axis=0), std_floor)
    else:
        center = jnp.mean(flat, axis=0)
        scale = jnp.maximum(jnp.std(flat, axis=0), std_floor)
    return center, scale


def standardize(seq, mean, std):
    """(seq - mean) / std broadcast over the last (channel) axis; non-finite mapped to 0."""
    z = (jnp.asarray(seq, jnp.float32) - mean) / std
    return jnp.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


# ========================================================================================
# Part 2 — the learned encoder network
# ========================================================================================
class SNConv1D(nn.Module):
    """Causal dilated 1-D convolution with spectral normalisation (Miyato 2018). The kernel
    (k, C_in, C_out) is reshaped to (k·C_in, C_out) and power-iterated, exactly as SN is applied to
    convolutions in SN-GAN. Causality via left-padding (k-1)·dilation then a VALID conv, so the
    output at t depends only on inputs ≤ t (WaveNet/TCN)."""
    features: int
    kernel_size: int = 3
    dilation: int = 1
    use_spectral_norm: bool = True
    n_power: int = 1

    @nn.compact
    def __call__(self, x, train: bool = False):                 # x: [B, T, C_in]
        c_in = x.shape[-1]
        W = self.param("kernel", nn.initializers.lecun_normal(), (self.kernel_size, c_in, self.features))
        b = self.param("bias", nn.initializers.zeros, (self.features,))
        if self.use_spectral_norm:
            W2 = W.reshape(-1, self.features)                   # [k*C_in, C_out]
            u = self.variable("sn", "u",
                              lambda: jnp.ones((self.features,)) / jnp.sqrt(self.features))
            uu = u.value
            v = None
            for _ in range(self.n_power):
                v = _l2n(W2 @ uu)                               # (k*C_in,)
                uu = _l2n(W2.T @ v)                             # (C_out,)
            sigma = v @ (W2 @ uu)
            if train:
                u.value = jax.lax.stop_gradient(uu)
            W = W / (sigma + 1e-12)
        pad = (self.kernel_size - 1) * self.dilation
        xp = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))            # causal left-pad on the time axis
        y = jax.lax.conv_general_dilated(
            xp, W, window_strides=(1,), padding="VALID", rhs_dilation=(self.dilation,),
            dimension_numbers=("NWC", "WIO", "NWC"))
        return y + b


class TCNBlock(nn.Module):
    """Canonical TCN residual block (Bai 2018): two dilated causal convs (GELU between), residual
    with a 1×1 SN-conv projection when the channel count changes. All convs spectral-normed."""
    features: int
    kernel_size: int = 3
    dilation: int = 1
    use_spectral_norm: bool = True

    @nn.compact
    def __call__(self, x, train: bool = False):
        h = SNConv1D(self.features, self.kernel_size, self.dilation, self.use_spectral_norm)(x, train=train)
        h = nn.gelu(h)
        h = SNConv1D(self.features, self.kernel_size, self.dilation, self.use_spectral_norm)(h, train=train)
        res = x if x.shape[-1] == self.features else \
            SNConv1D(self.features, 1, 1, self.use_spectral_norm)(x, train=train)   # 1×1 channel match
        return nn.gelu(h + res)


def _attn_pool(h, use_spectral_norm, train):
    """Additive attention pooling: a learned scorer over time -> softmax weights -> weighted sum.
    Dominates last-state for sequence scoring; the scorer is spectral-normed like the rest."""
    from .discriminator import SNDense
    e = jnp.tanh(SNDense(h.shape[-1], use_spectral_norm)(h, train=train))          # [B,T,C]
    s = SNDense(1, use_spectral_norm)(e, train=train)                              # [B,T,1]
    a = jax.nn.softmax(s, axis=1)
    return jnp.sum(a * h, axis=1)                                                  # [B,C]


def _pool_time(h, enc_pool: str, use_spectral_norm: bool, train: bool):
    """Time-axis pooling of an encoder output [B,T,C] -> [B,C]: attn (learned, default) | mean | last."""
    if enc_pool == "attn":
        return _attn_pool(h, use_spectral_norm, train)
    if enc_pool == "mean":
        return jnp.mean(h, axis=1)
    if enc_pool == "last":
        return h[:, -1]
    raise ValueError(f"unknown enc_pool {enc_pool!r}")


def _avgpool_time(x, s: int):
    """Non-overlapping average pool over the time axis by factor s (the multi-scale downsample, MelGAN /
    HiFi-GAN). x:[B,T,C] -> [B,T//s,C]; s=1 is identity. VALID windowing drops a partial trailing window."""
    if s == 1:
        return x
    return jax.lax.reduce_window(
        x, 0.0, jax.lax.add,
        window_dimensions=(1, s, 1), window_strides=(1, s, 1), padding="VALID") / float(s)


def _tcn_stack(x, enc_hidden, n_blocks, enc_type, kernel_size, use_spectral_norm, train):
    """A stack of `n_blocks` TCN residual blocks (dilation 2^i for 'tcn', fixed 1 for 'cnn')."""
    h = x
    for i in range(n_blocks):
        d = (2 ** i) if enc_type == "tcn" else 1
        h = TCNBlock(enc_hidden, kernel_size, d, use_spectral_norm)(h, train=train)
    return h


def _maybe_project(x, proj_dim: int, seed: int):
    """FIXED (non-trainable) random projection of the per-step feature axis F_in -> proj_dim (Projected-GAN,
    Sauer et al. 2021 — random projections of a frozen backbone's features + independent discriminators).
    Deterministic in `seed`: the matrix is a pure function of a static int, so XLA constant-folds it — no
    params, no variables, no rng-collection plumbing, and the critic optimiser never touches it. Distinct
    seeds (one per ensemble member) => each member discriminates on a different random channel MIXTURE, so
    the members are projection-diverse and the generator cannot satisfy them all by fixing one channel.
    proj_dim<=0 is identity (bit-identical to the un-projected critic)."""
    if proj_dim <= 0:
        return x
    F_in = x.shape[-1]
    P = jax.random.normal(jax.random.PRNGKey(int(seed)), (F_in, int(proj_dim)), dtype=x.dtype) / jnp.sqrt(F_in)
    return x @ P


class LearnedCritic(nn.Module):
    """Per-step descriptor sequence [B, T, F_in] -> scalar critic score [B].

    Encoder: a stack of `enc_layers` TCN residual blocks with dilation 2^i (enc_type='tcn') or all
    dilation 1 (enc_type='cnn', the non-dilated ablation), width `enc_hidden`. Pool over time
    (attn/mean/last) -> the existing spectral-normed CriticHead -> scalar. The .apply signature
    matches CriticHead so the shared make_d_step / score path is unchanged."""
    enc_hidden: int = 64
    enc_layers: int = 7
    enc_type: str = "tcn"                # 'tcn' (dilated, default) | 'cnn' (dilation fixed 1)
    enc_pool: str = "attn"              # 'attn' (default) | 'mean' | 'last'
    kernel_size: int = 3
    head_hidden: Sequence[int] = (128,)
    use_spectral_norm: bool = True
    proj_dim: int = 0                    # >0: fixed random input projection (Projected-GAN); 0 = off
    proj_seed: int = 0

    @nn.compact
    def __call__(self, x, train: bool = False):        # x: [B, T, F_in]
        x = _maybe_project(x, self.proj_dim, self.proj_seed)
        h = _tcn_stack(x, self.enc_hidden, self.enc_layers, self.enc_type,
                       self.kernel_size, self.use_spectral_norm, train)
        pooled = _pool_time(h, self.enc_pool, self.use_spectral_norm, train)
        return CriticHead(hidden=tuple(self.head_hidden),
                          use_spectral_norm=self.use_spectral_norm)(pooled, train=train)


class MultiScaleMember(nn.Module):
    """ONE ensemble member: a MULTI-SCALE TCN (MelGAN / HiFi-GAN multi-scale discriminator, Kumar 2019 /
    Kong 2020 — structural diversity over stacking copies). The descriptor sequence is processed at
    several temporal resolutions in parallel (avg-pooled by `enc_scales`); each branch is a dilated-causal
    TCN, attention-pooled, and the per-scale vectors are concatenated -> the shared SN CriticHead -> scalar.
    A coarse scale s sees s× the raw span per step, so it uses fewer blocks (`enc_layers - log2(s)`) yet
    still covers the whole window — the fine branch resolves microstructure, the coarse one resolves the
    slow drift that exposure bias produces."""
    enc_hidden: int = 64
    enc_layers: int = 7
    enc_type: str = "tcn"
    enc_pool: str = "attn"
    kernel_size: int = 3
    enc_scales: Sequence[int] = (1, 2, 4)
    head_hidden: Sequence[int] = (128,)
    use_spectral_norm: bool = True
    proj_dim: int = 0                    # >0: fixed random input projection (Projected-GAN); 0 = off
    proj_seed: int = 0                   # distinct per ensemble member -> projection-diverse discriminators

    @nn.compact
    def __call__(self, x, train: bool = False):        # x: [B, T, F_in] -> [B]
        x = _maybe_project(x, self.proj_dim, self.proj_seed)
        pooled = []
        for s in self.enc_scales:
            xs = _avgpool_time(x, int(s))
            n_blocks = max(1, self.enc_layers - int(math.log2(s)))   # coarse scale needs fewer blocks for full RF
            h = _tcn_stack(xs, self.enc_hidden, n_blocks, self.enc_type,
                           self.kernel_size, self.use_spectral_norm, train)
            pooled.append(_pool_time(h, self.enc_pool, self.use_spectral_norm, train))
        feat = jnp.concatenate(pooled, axis=-1)                       # [B, n_scales * enc_hidden]
        return CriticHead(hidden=tuple(self.head_hidden),
                          use_spectral_norm=self.use_spectral_norm)(feat, train=train)


class EnsembleLearnedCritic(nn.Module):
    """The Lever-3 STRONGEST critic: an ensemble of `ens_size` seed-diverse MultiScaleMembers. Each member
    is a distinct Flax scope, so Flax's path-folded init gives each independent (seed-diverse) weights —
    diversity without manual seed plumbing. Two return modes keep the SHARED d-step / score path unchanged:

      * train=True  -> per-member scores [B, K]. The shared `wgan_critic_loss` averages over all axes, which
        equals the MEAN of the per-member WGAN losses; because members have independent params, each
        member's gradient sees only its own score -> the members train INDEPENDENTLY (no diversity collapse).
      * train=False -> the PESSIMISTIC aggregate  mean_k - ens_pessimism·std_k  [B]  (Coste 2024 conservative /
        uncertainty-penalised optimisation). This is the reward the generator climbs: it is penalised for
        exploiting any single member's blind spot (high disagreement => likely a hack), the literature's
        leading anti-Goodhart lever AFTER the KL trust region. corr(reward,realism) is monitored on it.

    ens_size == 1 returns the bare [B] member score (a single multi-scale critic), so the ensemble adds no
    aggregation when disabled."""
    ens_size: int = 3
    ens_pessimism: float = 1.0
    enc_hidden: int = 64
    enc_layers: int = 7
    enc_type: str = "tcn"
    enc_pool: str = "attn"
    kernel_size: int = 3
    enc_scales: Sequence[int] = (1, 2, 4)
    head_hidden: Sequence[int] = (128,)
    use_spectral_norm: bool = True
    proj_dim: int = 0                    # >0: each member gets its OWN fixed random input projection

    @nn.compact
    def __call__(self, x, train: bool = False):        # x: [B, T, F_in]
        scores = [
            MultiScaleMember(enc_hidden=self.enc_hidden, enc_layers=self.enc_layers,
                             enc_type=self.enc_type, enc_pool=self.enc_pool, kernel_size=self.kernel_size,
                             enc_scales=tuple(self.enc_scales), head_hidden=tuple(self.head_hidden),
                             use_spectral_norm=self.use_spectral_norm,
                             proj_dim=self.proj_dim, proj_seed=1000 + 7919 * m,   # distinct random projection/member
                             name=f"member_{m}")(x, train=train)
            for m in range(self.ens_size)
        ]
        S = jnp.stack(scores, axis=-1)                                # [B, K]
        if self.ens_size == 1:
            return S[..., 0]
        if train:
            return S                                                  # per-member -> independent WGAN loss
        return jnp.mean(S, axis=-1) - self.ens_pessimism * jnp.std(S, axis=-1)   # pessimistic reward [B]


def receptive_field(enc_layers: int, kernel_size: int = 3) -> int:
    """RF of the (finest-scale) TCN: 1 + 2(k-1)(2^L - 1) (2 convs/block, dilation doubling). L=7,k=3 -> 509."""
    return 1 + 2 * (kernel_size - 1) * (2 ** enc_layers - 1)


def multiscale_receptive_field(enc_layers: int, kernel_size: int, enc_scales) -> int:
    """Max raw-step coverage across scales: scale s with (enc_layers - log2(s)) blocks covers
    s · RF(enc_layers - log2(s)) raw steps. Used only for the start-up log line."""
    cov = []
    for s in enc_scales:
        n_blocks = max(1, enc_layers - int(math.log2(int(s))))
        cov.append(int(s) * receptive_field(n_blocks, kernel_size))
    return max(cov)


def make_learned_critic(cfg, T: int, F_in: int, seed: int):
    """Build + init the Lever-3 critic from the disc config. Returns (module, params, sn_collection).
    Dispatches to the multi-scale ENSEMBLE (EnsembleLearnedCritic) when `ens_size>1` or more than one
    `enc_scale` is requested (the strongest config), else the bare single-scale LearnedCritic (the
    byte-identical original path). Used by the trainer's make_critic 'learned' branch."""
    d = cfg.disc
    scales = tuple(int(s) for s in getattr(d, "enc_scales", (1,)))
    ens_size = int(getattr(d, "ens_size", 1))
    proj_dim = int(getattr(d, "enc_proj_dim", 0))       # Projected-GAN random-projection heads (0 = off)
    common = dict(enc_hidden=int(d.enc_hidden), enc_layers=int(d.enc_layers),
                  enc_type=str(d.enc_type), enc_pool=str(d.enc_pool),
                  kernel_size=int(getattr(d, "enc_kernel", 3)),
                  head_hidden=tuple(d.head_hidden), use_spectral_norm=bool(d.use_spectral_norm))
    if ens_size > 1 or len(scales) > 1:
        net = EnsembleLearnedCritic(ens_size=ens_size,
                                    ens_pessimism=float(getattr(d, "ens_pessimism", 1.0)),
                                    enc_scales=scales, proj_dim=proj_dim, **common)
    else:
        net = LearnedCritic(proj_dim=proj_dim, proj_seed=1000, **common)
    init_vars = net.init(jax.random.PRNGKey(seed), jnp.zeros((1, T, F_in)), train=False)
    params = init_vars["params"]
    sn = {k: v for k, v in init_vars.items() if k != "params"}
    return net, params, sn


def make_learned_tx(cfg):
    """TTUR optimiser for the learned critic (Heusel 2017): a faster critic lr + Adam β1=0. Falls
    back to the generic disc lr/decay knobs if the enc_* knobs are absent."""
    import optax
    d = cfg.disc
    lr = float(getattr(d, "enc_lr", d.lr))
    b1 = float(getattr(d, "enc_adam_b1", 0.0))
    return optax.adamw(lr, b1=b1, b2=0.9, weight_decay=float(d.weight_decay))


# ========================================================================================
# Login-node self-test (no generator / engine): descriptor + network + d-step + R1.
# ========================================================================================
def _synth_seq(key, N, T, F_in, *, real: bool):
    """Tiny synthetic descriptor batch [N, T, F_in]. 'real' has temporal structure (AR(1) per
    channel); 'fake' is i.i.d. noise with the same per-channel scale — separable only by DYNAMICS."""
    def one(k):
        ks = jax.random.split(k, F_in)
        cols = []
        for j in range(F_in):
            e = jax.random.normal(ks[j], (T,))
            if real:
                def st(p, u):
                    nv = 0.9 * p + u
                    return nv, nv
                _, c = jax.lax.scan(st, 0.0, e)
                cols.append(c)
            else:
                cols.append(e)
        return jnp.stack(cols, axis=1)
    return jax.vmap(one)(jax.random.split(key, N))


def _cpu_self_test(seed=0):
    import optax
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("[learned_critic] CPU self-test — descriptor + learned encoder", flush=True)
    k = jax.random.key(seed)
    T, n_levels = 48, 10          # small: keep XLA/LLVM compile footprint under the login thread limit
    tick = 100.0
    EH, EL = 8, 2                 # tiny encoder for the gate (the prod sizes live in config.disc)

    # --- descriptor (small synthetic random-walk book + message stream) ---
    kk = jax.random.fold_in(k, 1)
    mid = 10000.0 + tick * jnp.cumsum(jax.random.normal(jax.random.fold_in(kk, 0), (T,)))
    half = tick * (1.0 + jax.random.bernoulli(jax.random.fold_in(kk, 1), 0.3, (T,))) / 2.0
    vols = jax.random.exponential(jax.random.fold_in(kk, 2), (T, 2 * n_levels)) * 100.0
    cols = []
    for i in range(n_levels):
        cols += [mid + half + tick * i, vols[:, 2 * i], mid - half - tick * i, vols[:, 2 * i + 1]]
    l2 = jnp.stack(cols, axis=1)                                                # [T, 4*n_levels]
    msgs = jnp.zeros((T, 14), jnp.float32)
    msgs = msgs.at[:, EVENT_TYPE_I].set(
        jax.random.randint(jax.random.fold_in(kk, 3), (T,), 1, 5).astype(jnp.float32))
    msgs = msgs.at[:, DIRECTION_I].set(
        jax.random.bernoulli(jax.random.fold_in(kk, 4), 0.5, (T,)).astype(jnp.float32))
    msgs = msgs.at[:, SIZE_I].set(jax.random.exponential(jax.random.fold_in(kk, 5), (T,)) * 50.0)
    desc = rollout_descriptor(l2, msgs, n_levels=n_levels, tick_size=tick)
    chk("(L1) descriptor shape [T, F_in]", desc.shape == (T, N_FEATURES), f"shape={tuple(desc.shape)}")
    chk("(L2) descriptor finite", bool(jnp.all(jnp.isfinite(desc))))
    db = batch_features(jnp.stack([l2, l2]), jnp.stack([msgs, msgs]), n_levels=n_levels, tick_size=tick)
    chk("(L3) batch_features [N, T, F_in]", db.shape == (2, T, N_FEATURES), f"shape={tuple(db.shape)}")
    mean, std = fit_normalizer(db)
    z = standardize(db, mean, std)
    chk("(L4) sequence standardiser finite + per-channel mean", mean.shape == (N_FEATURES,)
        and bool(jnp.all(jnp.isfinite(z))), f"mean.shape={tuple(mean.shape)}")

    # --- network forward (both encoders × all pools; tiny nets to limit compile threads) ---
    F_in = N_FEATURES
    x = _synth_seq(jax.random.fold_in(k, 2), 6, T, F_in, real=True)
    for et, pl in (("tcn", "attn"), ("tcn", "mean"), ("tcn", "last"), ("cnn", "mean")):
        net = LearnedCritic(enc_hidden=EH, enc_layers=EL, enc_type=et, enc_pool=pl)
        v = net.init(jax.random.PRNGKey(0), jnp.zeros((1, T, F_in)), train=False)
        s = net.apply(v, x, train=False)
        chk(f"(L5:{et}/{pl}) forward -> [B] finite", s.shape == (6,) and bool(jnp.all(jnp.isfinite(s))),
            f"shape={tuple(s.shape)}")

    chk("(L6) RF formula: L=7,k=3 -> 509 ≥ 500", receptive_field(7, 3) == 509, f"RF={receptive_field(7,3)}")

    # --- spectral norm active: the 'sn' collection holds unit-norm power-iteration vectors ---
    net = LearnedCritic(enc_hidden=EH, enc_layers=EL, enc_type="tcn", enc_pool="attn")
    v = net.init(jax.random.PRNGKey(0), jnp.zeros((1, T, F_in)), train=False)
    _, mut = net.apply(v, x, train=True, mutable=["sn"])
    u_leaves = jax.tree_util.tree_leaves(mut["sn"])
    chk("(L7) spectral-norm 'sn' collection populated", len(u_leaves) > 0, f"n_u={len(u_leaves)}")
    chk("(L7b) power-iteration vectors ~ unit norm",
        all(abs(float(jnp.linalg.norm(u)) - 1.0) < 1e-3 for u in u_leaves))

    # --- d-step separates real vs fake (a local WGAN critic update, ndim-agnostic R1) ---
    def dstep(params, sn, opt_state, real_x, fake_x, tx, r1_gamma=0.0):
        n_r = real_x.shape[0]
        feats = jnp.concatenate([real_x, fake_x], 0)

        def loss_fn(p):
            sc, m = net.apply({"params": p, **sn}, feats, train=True, mutable=["sn"])
            sr, sf = sc[:n_r], sc[n_r:]
            loss = jnp.mean(sf) - jnp.mean(sr)
            if r1_gamma > 0.0:
                def ss(xx):
                    return jnp.sum(net.apply({"params": p, **sn}, xx, train=False))
                gx = jax.grad(ss)(real_x)
                loss = loss + 0.5 * r1_gamma * jnp.mean(jnp.sum(gx * gx, axis=tuple(range(1, gx.ndim))))
            return loss, m
        (loss, m), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        upd, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, {"sn": m["sn"]}, opt_state, loss

    real_x = _synth_seq(jax.random.fold_in(k, 3), 12, T, F_in, real=True)
    fake_x = _synth_seq(jax.random.fold_in(k, 4), 12, T, F_in, real=False)
    net = LearnedCritic(enc_hidden=EH, enc_layers=EL, enc_type="tcn", enc_pool="mean")
    v = net.init(jax.random.PRNGKey(1), jnp.zeros((1, T, F_in)), train=False)
    params = v["params"]; sn = {kk: vv for kk, vv in v.items() if kk != "params"}
    tx = optax.adam(3e-3); opt_state = tx.init(params)

    def auc(sr, sf):
        s = jnp.concatenate([sr, sf]); ranks = jnp.argsort(jnp.argsort(s)) + 1.0
        u = jnp.sum(ranks[:sr.shape[0]]) - sr.shape[0] * (sr.shape[0] + 1) / 2.0
        return float(u / (sr.shape[0] * sf.shape[0]))

    auc0 = auc(net.apply({"params": params, **sn}, real_x, train=False),
               net.apply({"params": params, **sn}, fake_x, train=False))
    for _ in range(50):
        params, sn, opt_state, _ = dstep(params, sn, opt_state, real_x, fake_x, tx)
    auc1 = auc(net.apply({"params": params, **sn}, real_x, train=False),
               net.apply({"params": params, **sn}, fake_x, train=False))
    chk("(L8) d-step learns to separate real vs fake (AUC rises toward 1)", auc1 > 0.9,
        f"auc0={auc0:.3f} -> auc1={auc1:.3f}")

    # --- R1 shrinks the critic's input-gradient norm (the anti-hack effect, Mescheder 2018) ---
    def trained_input_gradnorm(r1_gamma):
        net2 = LearnedCritic(enc_hidden=EH, enc_layers=EL, enc_type="tcn", enc_pool="mean")
        v2 = net2.init(jax.random.PRNGKey(2), jnp.zeros((1, T, F_in)), train=False)
        p2 = v2["params"]; s2 = {kk: vv for kk, vv in v2.items() if kk != "params"}
        tx2 = optax.adam(3e-3); os2 = tx2.init(p2)
        for _ in range(50):
            p2, s2, os2, _ = dstep(p2, s2, os2, real_x, fake_x, tx2, r1_gamma=r1_gamma)
        def ss(xx):
            return jnp.sum(net2.apply({"params": p2, **s2}, xx, train=False))
        gx = jax.grad(ss)(real_x)
        return float(jnp.mean(jnp.sum(gx * gx, axis=(1, 2))))
    g0 = trained_input_gradnorm(0.0)
    g1 = trained_input_gradnorm(10.0)
    chk("(L9) R1 shrinks input-gradient norm (g_R1 < g_0)", g1 < g0, f"g0={g0:.3e} g1={g1:.3e}")

    # --- (L10) MULTI-SCALE ENSEMBLE (the strongest config): shapes, two return modes, pessimism, diversity ---
    ens = EnsembleLearnedCritic(ens_size=3, ens_pessimism=1.0, enc_hidden=EH, enc_layers=EL,
                                enc_type="tcn", enc_pool="mean", enc_scales=(1, 2, 4))
    ve = ens.init(jax.random.PRNGKey(5), jnp.zeros((1, T, F_in)), train=False)
    pe = ve["params"]; se = {kk: vv for kk, vv in ve.items() if kk != "params"}
    s_train, _ = ens.apply({"params": pe, **se}, x, train=True, mutable=["sn"])   # per-member [B, K]
    s_eval = ens.apply({"params": pe, **se}, x, train=False)         # pessimistic aggregate [B]
    chk("(L10a) ensemble train mode -> [B, K] per-member", s_train.shape == (6, 3), f"shape={tuple(s_train.shape)}")
    chk("(L10b) ensemble eval mode -> [B] aggregate", s_eval.shape == (6,) and bool(jnp.all(jnp.isfinite(s_eval))),
        f"shape={tuple(s_eval.shape)}")
    agg_ref = jnp.mean(s_train, -1) - 1.0 * jnp.std(s_train, -1)
    chk("(L10c) eval == pessimistic mean-β·std of members", float(jnp.max(jnp.abs(s_eval - agg_ref))) < 1e-4,
        f"max|Δ|={float(jnp.max(jnp.abs(s_eval - agg_ref))):.2e}")
    chk("(L10d) members are seed-diverse (per-member scores differ)",
        float(jnp.mean(jnp.std(s_train, axis=-1))) > 1e-4, f"mean σ_k={float(jnp.mean(jnp.std(s_train, axis=-1))):.3e}")

    # (L10e) ensemble d-step: the SHARED dstep helper (wgan loss on [B,K] = mean of per-member losses)
    #        trains the members; the pessimistic-aggregate AUC rises -> the reward separates real vs fake.
    net = ens                                                       # late-binding: dstep now uses the ensemble
    params = pe; sn = se; tx = optax.adam(3e-3); opt_state = tx.init(params)
    auc0e = auc(ens.apply({"params": params, **sn}, real_x, train=False),
                ens.apply({"params": params, **sn}, fake_x, train=False))
    for _ in range(60):
        params, sn, opt_state, _ = dstep(params, sn, opt_state, real_x, fake_x, tx)
    auc1e = auc(ens.apply({"params": params, **sn}, real_x, train=False),
                ens.apply({"params": params, **sn}, fake_x, train=False))
    chk("(L10e) ensemble d-step separates real vs fake (pessimistic AUC rises)", auc1e > 0.9,
        f"auc0={auc0e:.3f} -> auc1={auc1e:.3f}")

    # (L11) multi-scale RF covers the 500-window; ens_size=1 path returns the bare [B] member score.
    chk("(L11a) multiscale RF >= 500 (scales 1,2,4 @ L=7)", multiscale_receptive_field(7, 3, (1, 2, 4)) >= 500,
        f"RF={multiscale_receptive_field(7, 3, (1, 2, 4))}")
    solo = EnsembleLearnedCritic(ens_size=1, enc_hidden=EH, enc_layers=EL, enc_scales=(1, 2))
    vs_ = solo.init(jax.random.PRNGKey(6), jnp.zeros((1, T, F_in)), train=False)
    chk("(L11b) ens_size=1 returns bare [B]", solo.apply(vs_, x, train=False).shape == (6,))

    print("\n[learned_critic] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _cpu_self_test() else 0)
