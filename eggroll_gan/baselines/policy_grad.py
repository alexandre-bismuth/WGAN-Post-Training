"""Policy-gradient (GRPO/RLOO) machinery for HEAD-ONLY post-training — the comparison arm.

EGGROLL ascends the WGAN critic gradient-free (≈ a random-projection gradient estimate,
quality ~ sqrt(G/d)); but for the DECODER HEAD the policy gradient is computable EXACTLY given the
samples — the rollout tokens are realised, the frozen backbone already yields the policy hiddens
h_{t-1} for every continuation token t (the same hiddens the critic features and the KL use), and
head-only means the gradient flows ONLY through logits = h·W + b (no backprop through the sampler or
the LOB engine is ever needed: REINFORCE/GRPO needs grad-of-LOG-PROB at realised actions, not
grad-through-actions). So at a matched rollout budget GRPO-vs-EGGROLL isolates the UPDATE RULE.

THE log-probs MUST be of the ACTUAL sampling distribution. `generate()` does NOT sample from
softmax(h·W + b); per token position it applies (lob/validation_helpers.py):
    1. syntactic validity mask:  logits -> where(valid, logits, -1e9)   (filter_valid_pred)
    2. log_softmax                                                       (filter_valid_pred)
    3. top-`sample_top_n` truncation OF THE MASKED LOGITS + renormalise  (sample_pred)
`actual_sampling_logp` reproduces exactly that chain; the top-k SUPPORT is selected under
stop_gradient (the non-differentiable boundary; ties beyond k kept — measure-zero in fp32).
Deterministic time tokens (message positions 11..15, computed from Δt, never sampled) are excluded
via `fitness.build_time_mask`, identically to the KL.

GRPO grouping on the existing G×Q grid: group == context column (G samples per context q), layout
context-major r = q*G + g exactly as the ES trainer, so the critic/eval machinery is shared verbatim.
Advantages are group-relative: `rank_advantages` (per-sample analogue of fitness.rank_sigma_bar —
robust to the fat-tailed critic scores) or `rloo_advantages` (leave-one-out, unbiased).

Memory: the [M, T, V] logits tensor (512 x 13000 x 2112 at production scale) must NEVER materialise.
`make_pg_grad` runs ONE frozen-backbone pass per chunk (lax.scan) with `value_and_grad` INSIDE the
scan body, so reverse-mode stores one chunk's logits at a time — the same discipline as
es_generator.make_feats_kl. KL(current ‖ anchor) uses the plain full-softmax definition of
fitness.head_kl_penalty (NOT the truncated distribution) so the trust region is numerically
comparable across the EGGROLL and GRPO arms.

CPU-safe: every function is pure JAX; the __main__ self-test (PG1..PG9) runs on a login node with a
fake backbone (no model build, no rollout).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import DEFAULT as _CFG
from ..data import loaders as Ddata
from ..es import fitness as F

MSG_LEN = _CFG.model.msg_len            # 26
VOCAB = _CFG.model.vocab_size           # 2112


# ----------------------------------------------------------------------------------------
# The actual sampling distribution (mask -> log_softmax -> top-k -> renormalise).
# ----------------------------------------------------------------------------------------
def _topk_support(masked_logits, top_n):
    """Boolean support of sample_pred's truncation: the top_n highest MASKED logits (all, if top_n
    is <=1 or >= V or negative — generate() then samples the full valid distribution). Selected
    under stop_gradient: the support is data, not a differentiable function of the head."""
    v = masked_logits.shape[-1]
    if top_n is None or top_n <= 1 or top_n >= v:
        return jnp.ones_like(masked_logits, dtype=bool)
    kth = jax.lax.top_k(jax.lax.stop_gradient(masked_logits), top_n)[0][..., -1]
    return jax.lax.stop_gradient(masked_logits) >= kth[..., None]


def actual_sampling_logp(logits, valid_mask, top_n, tok):
    """log pi(tok) under the ACTUAL generate() distribution for one position.
    logits [V] (differentiable in the head), valid_mask [V] bool, tok scalar int32.

    The support is the recomputed top-k UNION the realised token: tok was sampled from the support
    at generation time by construction, but the PG pass RECOMPUTES logits teacher-forced (parallel
    kernels) while sampling ran the incremental scan — TF32-level reordering at the top-k BOUNDARY
    can drop a rank-~k token out of the recomputed set (observed ~18% of positions at -1e9,
    mean_logp -1.8e8). Without the union those positions contribute a
    WRONG-SIGNED gradient (-logsumexp fires, the where'd-out token term is dead). INVALID tokens
    (valid_mask False) still return ~-1e9: they are masked in `ml` itself, not just the support."""
    ml = jnp.where(valid_mask, logits, -1e9)
    keep = _topk_support(ml, top_n) | jax.nn.one_hot(tok, ml.shape[-1], dtype=bool)
    mll = jnp.where(keep, ml, -1e9)
    return mll[tok] - jax.nn.logsumexp(mll)


def sampling_probs(logits, valid_mask, top_n):
    """Full probability vector [V] of the actual sampling distribution (test/baseline parity use:
    e.g. checking that head-scaling == temperature, since top-k support is order-invariant)."""
    ml = jnp.where(valid_mask, logits, -1e9)
    mll = jnp.where(_topk_support(ml, top_n), ml, -1e9)
    return jax.nn.softmax(mll)


def valid_mask_per_position(valid_mask_array, T):
    """[T, V] bool: the syntactic validity mask for each of the T continuation token positions.
    Continuations start on a message boundary, so position t is message position t % 26."""
    vm = jnp.asarray(valid_mask_array, dtype=bool)
    return vm[jnp.arange(T) % vm.shape[0]]


def rollout_logp_mean(W, b, hid, toks, vm_T, top_n, time_mask):
    """Mean log pi over the SAMPLED continuation positions of ONE rollout.
    hid [T, d] (policy hiddens h_{t-1}, stop-gradient), toks [T] realised tokens, vm_T [T, V],
    time_mask [T] in {0,1} (0 at deterministic time positions). Mean (not sum) keeps the gradient
    scale horizon-independent, matching the KL convention; it is a constant lr rescale of REINFORCE."""
    logits = jax.lax.stop_gradient(hid) @ W + b                                  # [T, V]
    lp = jax.vmap(actual_sampling_logp, in_axes=(0, 0, None, 0))(logits, vm_T, top_n, toks)
    return jnp.sum(lp * time_mask) / jnp.maximum(jnp.sum(time_mask), 1.0)


# ----------------------------------------------------------------------------------------
# Group-relative advantages on the [Q, G] score grid (group == context row).
# ----------------------------------------------------------------------------------------
def rank_advantages(scores_qg):
    """Per-sample centered-rank advantage within each context row: A[q,g] = rank_g/(G-1) - 1/2 in
    [-1/2, 1/2]. The per-sample analogue of fitness.rank_sigma_bar (whose ES fitness is exactly
    mean_q of this); uses only the ordering -> robust to fat-tailed critic scores."""
    Q, G = scores_qg.shape
    order = jnp.argsort(scores_qg, axis=1)
    ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
    return ranks / jnp.maximum(G - 1, 1) - 0.5


def rloo_advantages(scores_qg):
    """Leave-one-out advantage (RLOO, unbiased): A[q,g] = s[q,g] - mean_{g' != g} s[q,g']
    = G/(G-1) * (s - row_mean). Keeps score magnitudes (outlier-sensitive — prefer rank for the
    fat-tailed WGAN scores; kept for ablation)."""
    Q, G = scores_qg.shape
    return (scores_qg - jnp.mean(scores_qg, axis=1, keepdims=True)) * (G / max(G - 1, 1))


def grpo_advantages(scores_qg, eps=1e-8):
    """The CANONICAL GRPO advantage (DeepSeek-R1 / DeepSeekMath, arXiv:2501.12948):
        A[q,g] = (s[q,g] - mean_g s[q,·]) / (std_g s[q,·] + eps)
    per-group z-score (outcome supervision: one scalar advantage per sampled output, applied to all
    its tokens via rollout_logp_mean). `eps` guards a degenerate group (all-equal rewards -> std 0 ->
    A 0), matching the reference (r - mean)/(std + eps). The WGAN critic scores are fat-tailed, so a
    single outlier inflates std and SHRINKS the advantages (self-limiting, not a blow-up) — stable;
    `rank_advantages` is the robust drop-in that also matches EGGROLL's rank_sigma_bar for the
    estimator-controlled ablation. Use --adv grpo for paper fidelity, --adv rank for the matched arm."""
    mean = jnp.mean(scores_qg, axis=1, keepdims=True)
    std = jnp.std(scores_qg, axis=1, keepdims=True)
    return (scores_qg - mean) / (std + eps)


# ----------------------------------------------------------------------------------------
# Fused frozen-backbone pass -> chunked PG gradient for the head (the G-step of train_grpo_head).
# ----------------------------------------------------------------------------------------
def make_pg_grad(backbone, bb_params, *, n_cond, n_gen, top_n, valid_mask_array, time_mask,
                 shard="auto", chunk=2, backend=None):
    """Build fn(ctx_tok, cont_tok, ctx_book, cont_book, adv, W, b, W_ref, b_ref, lam) ->
         (loss_sum, gW_sum, gb_sum, logp [M], kl [M])
    where loss_sum = sum_r ( -adv_r * logp_mean_r + lam * KL_r ) and (gW_sum, gb_sum) is its exact
    gradient w.r.t. (W, b). The CALLER divides by the GLOBAL M (correct under sharding, where each
    shard returns partial sums). One frozen-backbone pass per chunk yields the policy hiddens
    hid[start-1 : L-1] (the off-by-one-correct slice, as fitness.cont_hidden) for BOTH the log-probs
    and the anchor KL; value_and_grad runs INSIDE the lax.scan body so only one chunk's [c, T, V]
    logits exist at a time. `chunk` must divide the (per-shard) rollout count."""
    from ..critic.discriminator import PaddedLobPredFeatures

    start = n_cond * MSG_LEN
    T = n_gen * MSG_LEN
    vm_T = valid_mask_per_position(valid_mask_array, T)
    tm = jnp.ones((T,), jnp.float32) if time_mask is None else jnp.asarray(time_mask, jnp.float32)

    def _hid(xm, xb, mt, bt):
        feats = backbone.apply({"params": bb_params}, xm, xb, mt, bt,
                               method=PaddedLobPredFeatures.features)            # [L, d]
        return jax.lax.stop_gradient(feats[start - 1:-1])                        # [T, d]

    def _chunk_obj(W, b, W_ref, b_ref, lam, xm_c, xb_c, mt_c, bt_c, tok_c, adv_c):
        def one(xm, xb, mt, bt, tk):
            hid = _hid(xm, xb, mt, bt)
            lpm = rollout_logp_mean(W, b, hid, tk, vm_T, top_n, tm)
            kl = F.head_kl_penalty(hid[None], W_ref, b_ref, W[None], b[None], time_mask=tm)[0]
            return lpm, kl
        lps, kls = jax.vmap(one)(xm_c, xb_c, mt_c, bt_c, tok_c)
        return jnp.sum(-adv_c * lps + lam * kls), (lps, kls)

    def _run(ctx_tok, cont_tok, ctx_book, cont_book, adv, W, b, W_ref, b_ref, lam):
        tokens, book = Ddata.assemble_window(ctx_tok, cont_tok, ctx_book, cont_book)
        x_m, x_b, m_ts, b_ts = Ddata.critic_batch_from_tokens(tokens, book)
        cont_tok_i = jnp.asarray(cont_tok).astype(jnp.int32)
        M = x_m.shape[0]
        c = chunk if (chunk and 0 < chunk < M) else M
        assert M % c == 0, f"pg chunk {c} must divide the (per-shard) rollout count {M}"

        def _re(x):
            return x.reshape((M // c, c) + x.shape[1:])
        stacked = tuple(_re(x) for x in (x_m, x_b, m_ts, b_ts, cont_tok_i, adv))

        def body(carry, ch):
            loss_s, gW, gb = carry
            (l, aux), (dW, db) = jax.value_and_grad(_chunk_obj, argnums=(0, 1), has_aux=True)(
                W, b, W_ref, b_ref, lam, *ch)
            return (loss_s + l, gW + dW, gb + db), aux
        init = (jnp.zeros(()), jnp.zeros_like(W), jnp.zeros_like(b))
        (loss_sum, gW_sum, gb_sum), (lps, kls) = jax.lax.scan(body, init, stacked)
        return loss_sum, gW_sum, gb_sum, lps.reshape(M), kls.reshape(M)

    from ..es.es_generator import shard_decision
    if shard_decision(shard, jax.device_count()) == "vmap":
        kw = {"backend": backend} if backend else {}
        return jax.jit(_run, **kw)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _run_psum(ctx_tok, cont_tok, ctx_book, cont_book, adv, W, b, W_ref, b_ref, lam):
        loss_s, gW, gb, lps, kls = _run(ctx_tok, cont_tok, ctx_book, cont_book, adv,
                                        W, b, W_ref, b_ref, lam)
        return (jax.lax.psum(loss_s, "pop"), jax.lax.psum(gW, "pop"), jax.lax.psum(gb, "pop"),
                lps, kls)
    mesh = jax.make_mesh((jax.device_count(),), ("pop",))
    # check_rep=False: see es_generator.make_generate_es_sharded — upstream lob code is not
    # VMA-clean under JAX 0.9's static check. The three P() outputs are psum'd above, so they are
    # genuinely replicated; the audit is the only thing disabled.
    return jax.jit(shard_map(_run_psum, mesh=mesh,
                             in_specs=(P("pop"),) * 5 + (P(),) * 5,
                             out_specs=(P(), P(), P(), P("pop"), P("pop")), check_rep=False))


# ----------------------------------------------------------------------------------------
# Proj scope (--scope proj): the REINFORCE gradient flows through the realised-trajectory
# log-probs AND the teacher-forced backbone pass INTO the projection kernels — the decoder head
# stays frozen at the anchor. Unlike head scope, the continuation hiddens are NOT stop-gradient, so
# a remat'd reverse pass per rollout is needed. Still no backprop through the sampler or the LOB
# engine (REINFORCE differentiates the LOG-PROB at realised actions, not through the actions).
# ----------------------------------------------------------------------------------------
def _logp_mean_proj(W0, b0, hid, toks, vm_T, top_n, time_mask):
    """rollout_logp_mean WITHOUT the stop_gradient on `hid`: the proj gradient must flow through the
    continuation hiddens into the backbone. Head (W0, b0) is frozen at the anchor."""
    logits = hid @ W0 + b0                                                        # [T, V]
    lp = jax.vmap(actual_sampling_logp, in_axes=(0, 0, None, 0))(logits, vm_T, top_n, toks)
    return jnp.sum(lp * time_mask) / jnp.maximum(jnp.sum(time_mask), 1.0)


def _proj_kl(W0, b0, hid_i, hid_r, time_mask):
    """KL(pi_i ‖ pi_ref) on the realised continuation, BOTH through the frozen anchor head — pi_i
    from the proj-evolved hiddens hid_i, pi_ref from the anchor hiddens hid_r (stop-gradient). The
    proj analogue of es_generator.make_feats_kl_proj._kl / fitness.head_kl_penalty."""
    logp_i = jax.nn.log_softmax(hid_i @ W0 + b0, axis=-1)                         # [T, V]
    logp_r = jax.nn.log_softmax(jax.lax.stop_gradient(hid_r) @ W0 + b0, axis=-1)
    kl_t = jnp.sum(jnp.exp(logp_i) * (logp_i - logp_r), axis=-1)                  # [T] KL(i‖ref)
    return jnp.sum(kl_t * time_mask) / jnp.maximum(jnp.sum(time_mask), 1.0)


def make_pg_grad_proj(backbone, bb_params, merge_fn, *, n_cond, n_gen, top_n, valid_mask_array,
                      time_mask, shard="auto", chunk=1, remat=True, backend=None):
    """Proj-scope sibling of make_pg_grad. The policy is the proj-kernel-evolved generator; the
    DECODER HEAD stays frozen at the anchor. Build
        fn(ctx_tok, cont_tok, ctx_book, cont_book, adv, tr, lam)
          -> (loss_sum, grad_tr, logp [M], kl [M])
    where `tr` is the flat trainable proj-kernel dict (es_generator.extract_trainable layout),
    `merge_fn(tr)` -> the full generator params (a closure over the anchor + es_map), loss_sum =
    sum_r(-adv_r*logp_mean_r + lam*KL_r), and `grad_tr` is its exact gradient w.r.t. tr (same pytree).
    The CALLER divides by the GLOBAL M (correct under sharding). KL(pi_tr ‖ pi_anchor) — both through
    the frozen anchor head — is the trust region (== make_feats_kl_proj). value_and_grad runs INSIDE
    the lax.scan over chunks so one chunk's reverse tape lives at a time; `remat` wraps the per-rollout
    backbone feature pass (recompute in backward) to bound the activation cache over T = n_gen*26."""
    from ..critic.discriminator import PaddedLobPredFeatures

    start = n_cond * MSG_LEN
    T = n_gen * MSG_LEN
    vm_T = valid_mask_per_position(valid_mask_array, T)
    tm = jnp.ones((T,), jnp.float32) if time_mask is None else jnp.asarray(time_mask, jnp.float32)
    W0 = bb_params["decoder"]["kernel"]
    b0 = bb_params["decoder"]["bias"]

    def _feat(params, xm, xb, mt, bt):
        feats = backbone.apply({"params": params}, xm, xb, mt, bt,
                               method=PaddedLobPredFeatures.features)              # [L, d]
        return feats[start - 1:-1]                                                # [T, d]
    _feat_tr = jax.checkpoint(_feat) if remat else _feat

    def _chunk_obj(tr, lam, xm_c, xb_c, mt_c, bt_c, tok_c, adv_c):
        cur = merge_fn(tr)                                          # shared policy params (full tree)
        def one(xm, xb, mt, bt, tk):
            hid = _feat_tr(cur, xm, xb, mt, bt)                     # [T, d] differentiable in tr
            hid_a = jax.lax.stop_gradient(_feat(bb_params, xm, xb, mt, bt))        # anchor reference
            lpm = _logp_mean_proj(W0, b0, hid, tk, vm_T, top_n, tm)
            kl = _proj_kl(W0, b0, hid, hid_a, tm)
            return lpm, kl
        lps, kls = jax.vmap(one)(xm_c, xb_c, mt_c, bt_c, tok_c)
        return jnp.sum(-adv_c * lps + lam * kls), (lps, kls)

    def _run(ctx_tok, cont_tok, ctx_book, cont_book, adv, tr, lam):
        tokens, book = Ddata.assemble_window(ctx_tok, cont_tok, ctx_book, cont_book)
        x_m, x_b, m_ts, b_ts = Ddata.critic_batch_from_tokens(tokens, book)
        cont_tok_i = jnp.asarray(cont_tok).astype(jnp.int32)
        M = x_m.shape[0]
        c = chunk if (chunk and 0 < chunk < M) else M
        assert M % c == 0, f"pg-proj chunk {c} must divide the (per-shard) rollout count {M}"

        def _re(x):
            return x.reshape((M // c, c) + x.shape[1:])
        stacked = tuple(_re(x) for x in (x_m, x_b, m_ts, b_ts, cont_tok_i, adv))

        def body(carry, ch):
            loss_s, gtr = carry
            (l, aux), g = jax.value_and_grad(_chunk_obj, argnums=0, has_aux=True)(tr, lam, *ch)
            return (loss_s + l, jax.tree_util.tree_map(jnp.add, gtr, g)), aux
        init = (jnp.zeros(()), jax.tree_util.tree_map(jnp.zeros_like, tr))
        (loss_sum, grad_tr), (lps, kls) = jax.lax.scan(body, init, stacked)
        return loss_sum, grad_tr, lps.reshape(M), kls.reshape(M)

    from ..es.es_generator import shard_decision
    if shard_decision(shard, jax.device_count()) == "vmap":
        kw = {"backend": backend} if backend else {}
        return jax.jit(_run, **kw)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _run_psum(ctx_tok, cont_tok, ctx_book, cont_book, adv, tr, lam):
        loss_s, gtr, lps, kls = _run(ctx_tok, cont_tok, ctx_book, cont_book, adv, tr, lam)
        gtr = jax.tree_util.tree_map(lambda g: jax.lax.psum(g, "pop"), gtr)
        return jax.lax.psum(loss_s, "pop"), gtr, lps, kls
    mesh = jax.make_mesh((jax.device_count(),), ("pop",))
    # check_rep=False: same VMA rationale as make_pg_grad. tr is replicated (P() prefix), the grad is
    # psum'd above; the P() out-spec for grad_tr is a prefix over its whole subtree.
    return jax.jit(shard_map(_run_psum, mesh=mesh,
                             in_specs=(P("pop"),) * 5 + (P(), P()),
                             out_specs=(P(), P(), P("pop"), P("pop")), check_rep=False))


# ----------------------------------------------------------------------------------------
# Login-node self-test (no model): PG1..PG9 on tiny synthetic arrays + a fake backbone.
# ----------------------------------------------------------------------------------------
def _cpu_self_test(seed=0):
    import numpy as np

    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("[policy_grad] CPU self-test — actual-distribution log-probs / advantages / chunked grad",
          flush=True)
    k = jax.random.PRNGKey(seed)
    V, top_n = 12, 4
    logits = jax.random.normal(jax.random.fold_in(k, 1), (V,))
    vm = jnp.array([True] * 8 + [False] * 4)

    # (PG1) logp == manual mask -> topk -> renormalise reference.
    ml = np.where(np.asarray(vm), np.asarray(logits), -1e9)
    keep = ml >= np.sort(ml)[-top_n]
    ref = np.where(keep, ml, -1e9)
    ref = np.exp(ref - ref.max()); ref /= ref.sum()
    got = np.asarray(sampling_probs(logits, vm, top_n))
    chk("(PG1) sampling_probs == manual mask+topk+renorm", float(np.max(np.abs(got - ref))) < 1e-6,
        f"max|d|={float(np.max(np.abs(got - ref))):.2e}")
    tok = int(np.argmax(ref))
    lp = float(actual_sampling_logp(logits, vm, top_n, jnp.int32(tok)))
    chk("(PG2) actual_sampling_logp == log(sampling_probs[tok])",
        abs(lp - float(np.log(ref[tok]))) < 1e-6, f"lp={lp:.4f}")

    # (PG3) invalid tokens get ~ -inf; support invariant under temperature scaling.
    lp_inv = float(actual_sampling_logp(logits, vm, top_n, jnp.int32(V - 1)))
    sup1 = np.asarray(sampling_probs(logits, vm, top_n)) > 0
    sup2 = np.asarray(sampling_probs(logits / 0.7, vm, top_n)) > 0
    chk("(PG3) invalid token ~ -inf; top-k support invariant to temperature",
        lp_inv < -1e8 and bool(np.array_equal(sup1, sup2)))

    # (PG3b) union rescue: a VALID token outside the recomputed top-k (the TF32 boundary case) gets
    # a finite logp and a POSITIVE gradient on its own logit (pushes the realised token up).
    tok_out = int(np.argsort(ml)[len(ml) - top_n - 1])        # best valid token NOT in the top-k
    lp_out = float(actual_sampling_logp(logits, vm, top_n, jnp.int32(tok_out)))
    g_out = jax.grad(lambda lg: actual_sampling_logp(lg, vm, top_n, jnp.int32(tok_out)))(logits)
    chk("(PG3b) valid out-of-topk token: finite logp + positive own-logit grad",
        -20.0 < lp_out < 0.0 and float(g_out[tok_out]) > 0.0,
        f"lp={lp_out:.3f} g_tok={float(g_out[tok_out]):.3f}")

    # (PG4) rank_advantages: rows centered, mean over Q == fitness.rank_sigma_bar.
    s = jax.random.normal(jax.random.fold_in(k, 2), (5, 8))
    A = rank_advantages(s)
    chk("(PG4) rank advantages centered + mean_q == rank_sigma_bar",
        float(jnp.max(jnp.abs(jnp.sum(A, axis=1)))) < 1e-5
        and float(jnp.max(jnp.abs(jnp.mean(A, axis=0) - F.rank_sigma_bar(s)))) < 1e-6)

    # (PG5) rloo: zero-mean rows; equals s - loo-mean.
    Ar = rloo_advantages(s)
    G = s.shape[1]
    loo = (jnp.sum(s, axis=1, keepdims=True) - s) / (G - 1)
    chk("(PG5) rloo == s - loo_mean, rows zero-mean",
        float(jnp.max(jnp.abs(Ar - (s - loo)))) < 1e-5
        and float(jnp.max(jnp.abs(jnp.mean(Ar, axis=1)))) < 1e-6)

    # (PG11) canonical GRPO advantage: per-group z-score; rows zero-mean & ~unit-std; eps guards a
    # degenerate (all-equal) group -> exactly 0 (no NaN). The paper's (r - mean)/(std + eps).
    Ag = grpo_advantages(s)
    man = (s - jnp.mean(s, axis=1, keepdims=True)) / (jnp.std(s, axis=1, keepdims=True) + 1e-8)
    s_deg = jnp.ones((2, 4)) * 3.0
    chk("(PG11) grpo adv == (s-mean)/(std+eps), zero-mean rows, degenerate->0",
        float(jnp.max(jnp.abs(Ag - man))) < 1e-6
        and float(jnp.max(jnp.abs(jnp.mean(Ag, axis=1)))) < 1e-6
        and float(jnp.max(jnp.abs(grpo_advantages(s_deg)))) < 1e-3
        and bool(jnp.isfinite(grpo_advantages(s_deg)).all()))

    # (PG6) gradient correctness vs finite differences (tiny dense case). Realised tokens must be
    # IN-SUPPORT (as they always are in real use — they were sampled from the support): an
    # out-of-support token's -1e9 constant swamps the O(1) W-dependent part in fp32 and the finite
    # difference degenerates to 0 while autodiff still differentiates the small part.
    d, Vt, T = 3, 7, 4
    kk = jax.random.fold_in(k, 3)
    W = jax.random.normal(jax.random.fold_in(kk, 0), (d, Vt))
    b = jax.random.normal(jax.random.fold_in(kk, 1), (Vt,))
    hid = jax.random.normal(jax.random.fold_in(kk, 2), (T, d))
    vmt = jnp.ones((T, Vt), bool)
    tmm = jnp.array([1.0, 1.0, 0.0, 1.0])
    logits0 = hid @ W + b
    toks = jnp.argmax(jax.vmap(lambda lg, vv: sampling_probs(lg, vv, 3))(logits0, vmt), axis=-1)

    def f(Wx):
        return rollout_logp_mean(Wx, b, hid, toks, vmt, 3, tmm)
    g = jax.grad(f)(W)
    eps = 3e-3
    i, j = 1, 2
    Wp = W.at[i, j].add(eps); Wm = W.at[i, j].add(-eps)
    fd = (f(Wp) - f(Wm)) / (2 * eps)
    chk("(PG6) d logp / dW matches finite differences",
        abs(float(g[i, j]) - float(fd)) < 2e-2 * max(abs(float(fd)), 1e-2),
        f"ad={float(g[i, j]):.5f} fd={float(fd):.5f}")

    # (PG7) time-masked positions contribute no gradient.
    def f_pos(Wx, t_only):
        logits_ = hid @ Wx + b
        lps = jax.vmap(actual_sampling_logp, in_axes=(0, 0, None, 0))(logits_, vmt, 3, toks)
        return lps[t_only]
    g_masked = jax.grad(lambda Wx: rollout_logp_mean(Wx, b, hid, toks, vmt, 3,
                                                     jnp.array([0.0, 0.0, 1.0, 0.0])))(W)
    g_pos2 = jax.grad(lambda Wx: f_pos(Wx, 2))(W)
    chk("(PG7) time mask isolates positions", float(jnp.max(jnp.abs(g_masked - g_pos2))) < 1e-6)

    # (PG8/PG9) make_pg_grad with a fake backbone: chunked == unchunked; KL matches head_kl_penalty.
    n_cond, n_gen, M, dm = 2, 1, 4, 5
    L = (n_cond + n_gen) * MSG_LEN
    Tc = n_gen * MSG_LEN

    class _FakeBB:
        def apply(self, variables, xm, xb, mt, bt, method=None):
            base = jnp.arange(L, dtype=jnp.float32)[:, None] / L
            return base * (1.0 + 0.1 * jnp.mean(xb)) + 0.01 * xm[:, None].astype(jnp.float32) \
                * jnp.ones((1, dm))
    Wh = jax.random.normal(jax.random.fold_in(k, 5), (dm, VOCAB)) * 0.1
    bh = jnp.zeros((VOCAB,))
    ct = jax.random.randint(jax.random.fold_in(k, 6), (M, n_cond * MSG_LEN), 0, VOCAB)
    cn = jax.random.randint(jax.random.fold_in(k, 7), (M, Tc), 0, VOCAB)
    cb = jax.random.normal(jax.random.fold_in(k, 8), (M, n_cond + 1, 11))
    cnb = jax.random.normal(jax.random.fold_in(k, 9), (M, n_gen, 11))
    adv = jnp.array([0.5, -0.5, 0.25, -0.25])
    vma = jnp.ones((MSG_LEN, VOCAB), bool)
    tmask = F.build_time_mask(n_gen)
    fb = _FakeBB()
    outs = {}
    for c in (1, M):
        fn = make_pg_grad(fb, {}, n_cond=n_cond, n_gen=n_gen, top_n=5, valid_mask_array=vma,
                          time_mask=tmask, shard="off", chunk=c)
        outs[c] = fn(ct, cn, cb, cnb, adv, Wh, bh, Wh, bh, 0.3)
    dmax = max(float(jnp.max(jnp.abs(a - b_))) for a, b_ in zip(outs[1], outs[M]))
    chk("(PG8) make_pg_grad chunked == unchunked", dmax < 1e-5, f"max|d|={dmax:.2e}")
    # ref == current head -> KL must be 0 and the loss reduces to -sum(adv * logp).
    loss_s, gW, gb, lps, kls = outs[M]
    chk("(PG9) KL(current==anchor)=0; loss == -sum(adv*logp); grads finite",
        float(jnp.max(jnp.abs(kls))) < 1e-6
        and abs(float(loss_s) - float(jnp.sum(-adv * lps))) < 1e-5
        and bool(jnp.isfinite(gW).all() and jnp.isfinite(gb).all()))

    print("[policy_grad] " + ("ALL SELF-TESTS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


def _cpu_self_test_proj(seed=0):
    """PG10: proj-scope gradient — KL(anchor)=0 and d loss / d proj-kernel == finite differences,
    through a fake MERGEABLE backbone (features = H0(x) @ proj_kernel). No model build, no rollout."""
    import numpy as np  # noqa: F401  (kept for parity with _cpu_self_test; harmless on a login node)

    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("[policy_grad] CPU self-test (proj) — merge -> teacher-forced backbone grad into the kernels",
          flush=True)
    k = jax.random.PRNGKey(seed)
    n_cond, n_gen, M, dm = 2, 1, 4, 5
    L = (n_cond + n_gen) * MSG_LEN
    Tc = n_gen * MSG_LEN

    th0 = jnp.eye(dm) * 0.9 + 0.01                                    # anchor proj kernel [dm, dm]
    Wh = jax.random.normal(jax.random.fold_in(k, 11), (dm, VOCAB)) * 0.1
    bh = jnp.zeros((VOCAB,))
    bb = {"proj": {"kernel": th0}, "decoder": {"kernel": Wh, "bias": bh}}

    def merge_fn(tr):                                                 # the trainer's closure analogue
        return {"proj": {"kernel": tr["proj/kernel"]}, "decoder": bb["decoder"]}

    class _FakeBBProj:
        def apply(self, variables, xm, xb, mt, bt, *a, method=None):
            th = variables["params"]["proj"]["kernel"]               # [dm, dm] — the merged kernel
            base = (jnp.arange(L, dtype=jnp.float32)[:, None] / L) * jnp.ones((1, dm))
            H0 = base + 0.01 * xm[:, None].astype(jnp.float32) * jnp.ones((1, dm))
            return H0 @ th                                            # [L, dm] depends on the kernel

    ct = jax.random.randint(jax.random.fold_in(k, 6), (M, n_cond * MSG_LEN), 0, VOCAB)
    cn = jax.random.randint(jax.random.fold_in(k, 7), (M, Tc), 0, VOCAB)
    cb = jax.random.normal(jax.random.fold_in(k, 8), (M, n_cond + 1, 11))
    cnb = jax.random.normal(jax.random.fold_in(k, 9), (M, n_gen, 11))
    adv = jnp.array([0.5, -0.5, 0.25, -0.25])
    vma = jnp.ones((MSG_LEN, VOCAB), bool)
    tmask = F.build_time_mask(n_gen)

    # top_n = VOCAB -> full-softmax support, so the (arbitrary) realised tokens are always in support
    # and the finite difference is clean (cf. PG6: an out-of-support -1e9 swamps the O(1) part).
    fn = make_pg_grad_proj(_FakeBBProj(), bb, merge_fn, n_cond=n_cond, n_gen=n_gen, top_n=VOCAB,
                           valid_mask_array=vma, time_mask=tmask, shard="off", chunk=M, remat=False)

    # (PG10a) tr == anchor -> merge == bb -> identical hiddens -> KL == 0.
    _, _, _, kls0 = fn(ct, cn, cb, cnb, adv, {"proj/kernel": th0}, jnp.float32(0.3))
    chk("(PG10a) KL(proj==anchor) == 0", float(jnp.max(jnp.abs(kls0))) < 1e-6,
        f"max|kl|={float(jnp.max(jnp.abs(kls0))):.2e}")

    # (PG10b) d loss / d proj-kernel == finite differences (lam=0 isolates the REINFORCE term).
    def loss_of(thv):
        return float(fn(ct, cn, cb, cnb, adv, {"proj/kernel": thv}, jnp.float32(0.0))[0])
    _, gtr0, _, _ = fn(ct, cn, cb, cnb, adv, {"proj/kernel": th0}, jnp.float32(0.0))
    i, j, eps = 1, 2, 3e-3
    fd = (loss_of(th0.at[i, j].add(eps)) - loss_of(th0.at[i, j].add(-eps))) / (2 * eps)
    ad = float(gtr0["proj/kernel"][i, j])
    chk("(PG10b) d loss / d proj-kernel matches finite differences",
        abs(ad - fd) < 2e-2 * max(abs(fd), 1e-2), f"ad={ad:.5f} fd={fd:.5f}")

    # (PG10c) a non-anchor kernel -> KL > 0 and a finite gradient on the kernel.
    th1 = th0 + 0.05 * jax.random.normal(jax.random.fold_in(k, 12), (dm, dm))
    _, g1, _, kl1 = fn(ct, cn, cb, cnb, adv, {"proj/kernel": th1}, jnp.float32(0.3))
    chk("(PG10c) perturbed kernel: KL>0, grad finite",
        float(jnp.mean(kl1)) > 0.0 and bool(jnp.isfinite(g1["proj/kernel"]).all()),
        f"mean_kl={float(jnp.mean(kl1)):.4f}")

    print("[policy_grad] (proj) " + ("ALL SELF-TESTS PASSED" if not fails else f"FAILED: {fails}"),
          flush=True)
    return fails


if __name__ == "__main__":
    import sys
    all_fails = _cpu_self_test() + _cpu_self_test_proj()
    sys.exit(1 if all_fails else 0)
