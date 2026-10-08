"""WGAN critic discriminator (S1).

  * Real-valued critic (no sigmoid): score = D([context ; continuation], book).
  * Backbone = the SAME pretrained Mamba3-78M, reused as a frozen feature extractor.
    Taps the pre-decoder fused hidden state (no pooling, no decoder, no debug prints)
    via `PaddedLobPredFeatures.features`, mean-pools over the sequence, then a small
    spectral-normed MLP head -> scalar. Lipschitz control is spectral norm (gradient
    penalty is inapplicable: inputs are discrete tokens).

The backbone is frozen and only the critic head trains; LoRA on the backbone (config
`disc.lora_rank`) adds capacity if the pretrained features do not separate real vs
fixed-G samples.

Param tree note: `PaddedLobPredFeatures` subclasses `PaddedLobPredModel`, so its param
tree is IDENTICAL to the checkpoint's (message_encoder/book_encoder/fused_s5[/decoder]);
weights restored by `checkpoint_utils.load_pretrained_generator` graft directly.
"""
from __future__ import annotations

import sys
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from ..config import DEFAULT as _CFG

from lob.lob_seq_model import PaddedLobPredModel  # lobmamba on sys.path via eggroll_gan._paths


# ----------------------------------------------------------------------------------
# Backbone feature extractor (frozen)
# ----------------------------------------------------------------------------------
class PaddedLobPredFeatures(PaddedLobPredModel):
    """Same backbone as PaddedLobPredModel, exposing the pre-decoder hidden state.

    Mirrors `PaddedLobPredModel.__call__` up to (and including) `fused_s5`, but stops
    before pooling/decoder/log_softmax — and drops the leftover `jax.debug.print`s.
    Operates on a SINGLE example (vmap externally for a batch). Returns [L, d_model].
    """

    def features(self, x_m, x_b, message_integration_timesteps, book_integration_timesteps, es=None):
        # es: optional EGGROLL LoRA factor pytree (S5b proj scope) — the perturbed-policy
        # hidden pass for the exact per-member KL. None (the critic-feature path) is the
        # stock frozen-backbone forward.
        from s5 import es_fold
        es_m = es_fold.subtree(es, "message_encoder")
        es_b = es_fold.subtree(es, "book_encoder")
        es_f = es_fold.subtree(es, "fused_s5")
        x_m = (self.message_encoder(x_m, message_integration_timesteps) if es_m is None
               else self.message_encoder(x_m, message_integration_timesteps, es=es_m))
        x_b = (self.book_encoder(x_b, book_integration_timesteps) if es_b is None
               else self.book_encoder(x_b, book_integration_timesteps, es=es_b))
        x = jnp.concatenate([x_m, x_b], axis=1)            # [L, 2*d_model] (book repeated to L in loader)
        x = (self.fused_s5(x, jnp.ones(x.shape[0])) if es_f is None
             else self.fused_s5(x, jnp.ones(x.shape[0]), es=es_f))  # [L, d_model]
        return x


def make_backbone(model_cls) -> PaddedLobPredFeatures:
    """Build a PaddedLobPredFeatures with the SAME kwargs as the loaded model_cls
    (a `partial(BatchPaddedLobPredModel, **kwargs)` from init_train_state). The single
    (non-batched) class is used; we vmap inputs ourselves."""
    kwargs = dict(model_cls.keywords)
    kwargs.setdefault("training", False)   # eval-mode forward (dropout off)
    return PaddedLobPredFeatures(**kwargs)


def _pool(feats: jnp.ndarray, pooling: str, pool_start: int = 0) -> jnp.ndarray:
    """pool_start: first sequence position included in 'mean' pooling. 0 = whole window;
    n_cond*MSG_LEN = continuation-only. Real & fake share an identical context prefix, so
    mean-pooling the context only DILUTES the critic's discriminative signal (at 500:500 the
    continuation is half the pooled mass); 'last' is unaffected (already continuation-sided)."""
    if pooling == "mean":
        return jnp.mean(feats[pool_start:], axis=0)
    if pooling == "last":
        return feats[-1]
    raise ValueError(f"unknown pooling {pooling!r}")


def backbone_pooled_batch(backbone, bb_params, x_m, x_b, m_ts, b_ts, pooling: str = "mean",
                          pool_start: int = 0):
    """Frozen backbone -> pooled features for a BATCH.

    x_m: [B, L] int tokens; x_b: [B, L_b, d_book]; m_ts/b_ts: integration timesteps.
    Returns [B, d_model]. Backbone params are frozen (closed over, stop_gradient'd by
    the caller's optimizer which only touches the head). pool_start: see _pool."""
    def one(xm, xb, mt, bt):
        feats = backbone.apply({"params": bb_params}, xm, xb, mt, bt,
                               method=PaddedLobPredFeatures.features)
        return _pool(feats, pooling, pool_start)
    return jax.vmap(one)(x_m, x_b, m_ts, b_ts)


# ----------------------------------------------------------------------------------
# Spectral-normed Dense (manual power iteration; version-agnostic)
# ----------------------------------------------------------------------------------
def _l2n(v: jnp.ndarray) -> jnp.ndarray:
    return v / (jnp.linalg.norm(v) + 1e-12)


class SNDense(nn.Module):
    """Dense with optional spectral normalisation of the kernel (1-Lipschitz up to the
    bias). Power-iteration vector `u` lives in the mutable 'sn' collection and is
    refreshed when called with train=True (apply mutable=['sn'])."""
    features: int
    use_spectral_norm: bool = True
    n_power: int = 1

    @nn.compact
    def __call__(self, x, train: bool = False):
        in_f = x.shape[-1]
        W = self.param("kernel", nn.initializers.lecun_normal(), (in_f, self.features))
        b = self.param("bias", nn.initializers.zeros, (self.features,))
        if self.use_spectral_norm:
            u = self.variable("sn", "u",
                              lambda: jnp.ones((self.features,)) / jnp.sqrt(self.features))
            uu = u.value
            v = None
            for _ in range(self.n_power):
                v = _l2n(W @ uu)        # (in_f,)
                uu = _l2n(W.T @ v)      # (features,)
            sigma = v @ (W @ uu)        # scalar: top singular value estimate
            if train:
                u.value = jax.lax.stop_gradient(uu)
            W = W / (sigma + 1e-12)
        return x @ W + b


# ----------------------------------------------------------------------------------
# Critic head: pooled features -> scalar score
# ----------------------------------------------------------------------------------
class CriticHead(nn.Module):
    hidden: Sequence[int] = (128,)
    activation: str = "gelu"
    use_spectral_norm: bool = True

    @nn.compact
    def __call__(self, h, train: bool = False):     # h: [B, d_model]
        act = nn.gelu if self.activation == "gelu" else nn.relu
        for hsz in self.hidden:
            h = SNDense(hsz, self.use_spectral_norm)(h, train=train)
            h = act(h)
        score = SNDense(1, self.use_spectral_norm)(h, train=train)   # [B, 1]
        return jnp.squeeze(score, axis=-1)          # [B]


# ----------------------------------------------------------------------------------
# WGAN critic objective + monitoring
# ----------------------------------------------------------------------------------
def wgan_critic_loss(score_real: jnp.ndarray, score_fake: jnp.ndarray) -> jnp.ndarray:
    """Critic minimises E[D(fake)] - E[D(real)] (Wasserstein dual)."""
    return jnp.mean(score_fake) - jnp.mean(score_real)


def critic_separation(score_real: jnp.ndarray, score_fake: jnp.ndarray) -> jnp.ndarray:
    """Monitoring: E[D(real)] - E[D(fake)] (want this to grow positive)."""
    return jnp.mean(score_real) - jnp.mean(score_fake)


def roc_auc(score_real: jnp.ndarray, score_fake: jnp.ndarray) -> jnp.ndarray:
    """Rank-based AUC that real outscores fake (Mann-Whitney U / (n_r*n_f))."""
    s = jnp.concatenate([score_real, score_fake])
    ranks = jnp.argsort(jnp.argsort(s)) + 1.0           # 1..N average-free ranks
    n_r = score_real.shape[0]
    n_f = score_fake.shape[0]
    rank_real = jnp.sum(ranks[:n_r])
    u = rank_real - n_r * (n_r + 1) / 2.0
    return u / (n_r * n_f)


def critic_cross_entropy(score_real: jnp.ndarray, score_fake: jnp.ndarray) -> jnp.ndarray:
    """SIDE DIAGNOSTIC (not a training objective): binary cross-entropy in nats of a fixed-calibration
    1-D logistic separating real (label 1) from fake (label 0) on the critic's scalar scores. Scores are
    pooled-standardised then used directly as logits; orientation-free (min over sign) since a WGAN critic
    is only oriented once trained. Bounded and interpretable — ~ln2≈0.693 = chance (indistinguishable),
    →0 = perfectly separable. Unlike the unbounded WGAN score, this is comparable across steps and runs."""
    sr = jnp.ravel(score_real).astype(jnp.float32)
    sf = jnp.ravel(score_fake).astype(jnp.float32)
    alls = jnp.concatenate([sr, sf])
    z = (lambda mu, sd: ((sr - mu) / sd, (sf - mu) / sd))(jnp.mean(alls), jnp.std(alls) + 1e-6)
    zr, zf = z
    ce_pos = 0.5 * (jnp.mean(jax.nn.softplus(-zr)) + jnp.mean(jax.nn.softplus(zf)))   # real>fake orientation
    ce_neg = 0.5 * (jnp.mean(jax.nn.softplus(zr)) + jnp.mean(jax.nn.softplus(-zf)))   # fake>real orientation
    return jnp.minimum(ce_pos, ce_neg)
