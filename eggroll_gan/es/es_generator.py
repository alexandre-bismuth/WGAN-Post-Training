"""S3 — ES generator G-step: AR rollout over an EGGROLL population, perturbing ONLY the decoder head.

`inference_no_errcorr.generate` is run by `generate_batched` with the `train_state` shared across the
batch (`in_axes=None`). For ES only the decoder leaves are per-member: a thin wrapper
(`_generate_es_single`) injects a per-member decoder `{kernel,bias}` into the (otherwise frozen, shared)
train_state and calls the UNCHANGED `generate()`. The `vmap` uses `in_axes=0` on the decoder leaves and
the per-member rollout inputs, `None` on the frozen backbone and all statics — the existing
`generate_batched` in_axes with the train_state slot swapped from `None` to per-member decoder leaves.

The decoder-head scope MATERIALISES the per-member head weight `W_ref + σ·Aᵢ·Bᵢᵀ` (and bias
`b_ref + σ·ηᵢ`) because the model's `decoder` is a Flax `nn.Dense` that computes `x @ W` internally; it
does not fold `do_Tmm` into the rollout (that is the S5 interior-scope optimisation; at N≈1024 the
materialised head is ~8.9 GB, so S5 switches to the low-rank fold). The perturbation is produced by EGGROLL's OWN
`get_lora_update_params` / `get_nonlora_update_params` with the SAME per-leaf keys + iterinfo + σ that
`do_updates` later uses to reconstruct the gradient, so the rollout and the ES update are consistent.

σ=0 ⇒ every member's head == `W_ref` exactly ⇒ the ES rollout reproduces the unperturbed
`generate_batched` bit-for-bit (the S3 gate).

CPU-safe (no model): `build_decoder_population`, `tile_contexts`, `make_decoder_noiser`, and a
`do_updates` round-trip run on the login node. The rollout itself (`make_generate_es_batched`) is GPU
compute; build it and run the equivalence/loop on the GH200. `inference_no_errcorr` is imported LAZILY
(only inside the rollout wrapper) so this module stays import-cheap on a login node.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from .es_plumbing import import_hyperscalees, build_es_tree_key, _key_path_strs  # noqa: F401  (re-exported convenience)


# ----------------------------------------------------------------------------------------
# Noiser for the decoder-head scope.
# ----------------------------------------------------------------------------------------
def make_decoder_noiser(hs, kernel, bias, *, sigma, lr, rank=1, group_size=0, noise_reuse=0,
                        solver=optax.adamw, solver_kwargs=None, seed=0):
    """Build the EggRoll noiser + es_map + per-leaf key tree for the 2-leaf decoder head.

    Returns (frozen_noiser_params, noiser_params, es_map, es_tree_key, leaves) where
    leaves = {"kernel","bias"} is the *current* (reference) head, the thing `do_updates` evolves."""
    leaves = {"kernel": kernel, "bias": bias}
    fnp, npar = hs.EggRoll.init_noiser(
        leaves, sigma, lr, solver=solver, solver_kwargs=(solver_kwargs or {}),
        rank=rank, group_size=group_size, noise_reuse=noise_reuse)
    es_map = {"kernel": hs.MM_PARAM, "bias": hs.PARAM}     # do_Tmm head + full-rank bias
    es_tree_key = build_es_tree_key(leaves, jax.random.key(seed), hs)
    return fnp, npar, es_map, es_tree_key, leaves


def build_decoder_population(hs, frozen_noiser_params, noiser_params, leaves, es_tree_key, iterinfo):
    """Materialise per-member decoder leaves over a population, EXACTLY matching EggRoll's update math.

    iterinfo = (epochs, thread_ids), each shape (N,). thread_id%2 -> ±σ (antithetic), //2 -> shared idx.
    kernel (MM_PARAM / do_Tmm convention): Wᵢ = W + (A·σ_signed) @ Bᵀ, base_sigma = σ/√rank.
    bias   (PARAM):                        bᵢ = b + η·σ_signed,        base_sigma = σ.
    σ=0 ⇒ Wᵢ=W, bᵢ=b bit-exact. Returns {"kernel":(N,in,out), "bias":(N,out)}."""
    epochs, threads = iterinfo
    sig = noiser_params["sigma"]
    rank = frozen_noiser_params["rank"]
    kernel, bias = leaves["kernel"], leaves["bias"]
    kkey, bkey = es_tree_key["kernel"], es_tree_key["bias"]
    base_sig_k = sig / jnp.sqrt(rank)

    def one(ep, th):
        A, B = hs.eggroll.get_lora_update_params(frozen_noiser_params, base_sig_k, (ep, th), kernel, kkey)
        kpert = kernel + A @ B.T                                   # (in,out)
        bupd = hs.eggroll.get_nonlora_update_params(frozen_noiser_params, sig, (ep, th), bias, bkey)
        return kpert, bias + bupd

    kpop, bpop = jax.vmap(one)(epochs, threads)
    return {"kernel": kpop, "bias": bpop}


def population_iterinfo(n_pop, epoch):
    """(epochs, thread_ids) for a population of size n_pop at a given epoch. thread_ids = 0..n_pop-1
    => antithetic ±σ pairs (2k, 2k+1). n_pop should be even."""
    return (jnp.full(n_pop, epoch, dtype=jnp.int32), jnp.arange(n_pop, dtype=jnp.int32))


def tile_contexts(arr, group_size):
    """Tile per-context rows into per-member rows (CRN within a group): each of the n_ctx rows is
    repeated `group_size` times along axis 0 -> n_ctx*group_size members sharing a context.
    Works on any leading-axis array; vmappable over a pytree via jax.tree.map."""
    return jnp.repeat(arr, group_size, axis=0)


# ----------------------------------------------------------------------------------------
# The vmapped rollout (GPU). `inference_no_errcorr` imported lazily so login import stays cheap.
# ----------------------------------------------------------------------------------------
def _generate_es_single(decoder_leaves, sim, frozen_ts, model, batchnorm, encoder,
                        sample_top_n, tick_size, m_seq_cond, b_seq_cond, n_msg_todo,
                        sim_state, rng, init_hidden, conditional, init_time, valid_mask_array,
                        unjit=False):
    """Inject per-member decoder leaves into the frozen train_state, then call the UNCHANGED generate().
    Returns generate()'s tuple: (msgs_decoded, l2_book_states, num_errors, msgs_tokens, b_finals).
    unjit=True (multihost shard_map only): trace generate's PLAIN function instead of its jit wrapper —
    the upstream decorator pins backend='gpu', and a backend-pinned jit nested inside jit(shard_map)
    resolves to the process-local device set, clashing with the global mesh ("Received incompatible
    devices ... jit inside jit with device ids [0]"). Statics become trace-time constants."""
    import lob.inference_no_errcorr as inf
    new_params = {**frozen_ts.params,
                  "decoder": {"kernel": decoder_leaves["kernel"], "bias": decoder_leaves["bias"]}}
    ts = frozen_ts.replace(params=new_params)
    gen_fn = inf.generate.__wrapped__ if unjit else inf.generate
    return gen_fn(
        sim, ts, model, batchnorm, encoder, sample_top_n, tick_size,
        m_seq_cond, b_seq_cond, n_msg_todo, sim_state, rng, init_hidden,
        conditional, init_time, False, None, valid_mask_array)


# arg order of _generate_es_single (17): decoder_leaves, sim, frozen_ts, model, batchnorm, encoder,
#   sample_top_n, tick_size, m_seq_cond, b_seq_cond, n_msg_todo, sim_state, rng, init_hidden,
#   conditional, init_time, valid_mask_array
# This is the existing generate_batched in_axes with the train_state slot replaced by the per-member
# decoder leaves (mapped) + a shared frozen_ts (None). Statics match generate_batched's static set.
_ES_IN_AXES = (0, None, None, None, None, None, None, None, 0, 0, None, 0, 0, 0, None, 0, None)
_ES_STATIC_ARGNUMS = (1, 3, 4, 6, 7, 10, 14)  # sim, model, batchnorm, sample_top_n, tick_size, n_msg_todo, conditional


def make_generate_es_batched(decoder_in_axes=0, backend="gpu"):
    """Return the jitted+vmapped ES rollout. Call ONLY where a device of `backend` exists (GH200).

    decoder_in_axes=0 (default) => per-member (materialised) head: the decoder matmul is BATCHED.
    decoder_in_axes=None        => a single SHARED head broadcast across members: the decoder matmul
                                   is BROADCAST, i.e. structurally identical to stock generate_batched.
                                   Used by the S3 plumbing gate (G1a) to prove the wrapper reproduces
                                   stock bit-for-bit when the head is not batched (so the only source of
                                   non-bit-exactness in the materialised path is the batched matmul kernel
                                   — numerically ≡ stock per S2 — amplified by discrete top-n sampling)."""
    in_axes = (decoder_in_axes,) + _ES_IN_AXES[1:]
    return jax.jit(jax.vmap(_generate_es_single, in_axes=in_axes),
                   static_argnums=_ES_STATIC_ARGNUMS, backend=backend)


# ========================================================================================
# S5 — the G×Q rollout grid + multi-host population sharding.
#
# Layout (context-major): rollout r = q*G + g  ->  direction g = r % G, context q = r // G.
#   - decoder heads: per-DIRECTION [G,...]  -> tile_dirs_over_Q -> [Q*G,...]  (rollout r uses head g)
#   - contexts/sim/init: per-CONTEXT [Q,...] -> grid_repeat_contexts -> [Q*G,...] (rollout r uses ctx q)
#   - sampling rng: INDEPENDENT per (g,q) (grid_rngs) — shared context tokens down a column, but NOT the
#     sampling noise, else σ̄'s per-context centering measures rng collisions, not perturbation quality.
# Scores come back [Q*G] in this order -> reshape (Q, G) -> fitness.rank_sigma_bar -> (G,).
# ========================================================================================
def gq_grid_indices(G, Q):
    """(dir_idx[G*Q] = r%G, ctx_idx[G*Q] = r//G) for the context-major grid r = q*G + g."""
    dir_idx = jnp.tile(jnp.arange(G, dtype=jnp.int32), Q)        # g, cycling fastest
    ctx_idx = jnp.repeat(jnp.arange(Q, dtype=jnp.int32), G)      # q, held for G in a row
    return dir_idx, ctx_idx


def grid_rngs(step_key, G, Q):
    """Per-(g,q) sampling keys [Q*G] (context-major): rng[q*G+g] = fold_in(fold_in(step_key, q), g).
    Independent across BOTH g and q — see the σ̄ RNG-axis note above."""
    gs = jnp.arange(G)

    def per_q(q):
        kq = jax.random.fold_in(step_key, q)
        return jax.vmap(lambda g: jax.random.fold_in(kq, g))(gs)  # [G] keys
    keys_qg = jax.vmap(per_q)(jnp.arange(Q))                      # [Q, G(, 2)] keys
    return keys_qg.reshape((Q * G,) + keys_qg.shape[2:])          # [Q*G(, 2)], index q*G+g


def tile_dirs_over_Q(pop, Q):
    """Per-DIRECTION leaves [G,...] -> grid [Q*G,...] by TILING (rollout r=q*G+g uses pop[g]). For the
    head-only materialised population; tree-maps over {kernel,bias}."""
    return jax.tree_util.tree_map(lambda x: jnp.tile(x, (Q,) + (1,) * (x.ndim - 1)), pop)


def grid_repeat_contexts(tree, G):
    """Per-CONTEXT leaves [Q,...] -> grid [Q*G,...] by REPEATING each context G times (rollout r=q*G+g
    uses context q). Tree-maps over the per-context rollout inputs (m_seq, b_seq, sim_state, init_*)."""
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x, G, axis=0), tree)


def shard_decision(shard, n_dev):
    """'vmap' (single device / shard off) vs 'shard_map' (multi-device). CPU-testable selector."""
    return "vmap" if (shard == "off" or (shard == "auto" and n_dev <= 1)) else "shard_map"


def _setup_shard_escapes():
    """The GPU-validated escape recipe for tracing `generate` inside jit(shard_map):
    ALL of generate's internal device/backend pins must go. `__wrapped__` strips generate's own
    decorator (see _generate_es_single); setting the upstream escape hatch `valh._TP_MESH` makes
    inference_no_errcorr skip its inner jit(_partial_msg, device=jax.devices()[0]) / jit(_partial)
    wrappers (inference_no_errcorr.py:839,1030) — plain functions trace inline.
    preproc.transform_L2_state_gpu is ALSO pinned (backend='gpu', preproc.py:65) and is called inside
    the generation scan (inference_no_errcorr.py:1003) — upstream's own TP path lists it as the one
    transitive jit target to unwrap (inference_no_errcorr.py:1857). Re-jit WITHOUT the pin (instead
    of bare __wrapped__) so host-side callers in this process keep compiled performance.
    Returns the 1-axis 'pop' mesh. Idempotent."""
    import lob.validation_helpers as valh
    import preproc
    mesh = jax.make_mesh((jax.device_count(),), ("pop",))
    valh._TP_MESH = mesh
    if hasattr(preproc.transform_L2_state_gpu, "__wrapped__"):
        preproc.transform_L2_state_gpu = jax.jit(
            preproc.transform_L2_state_gpu.__wrapped__, static_argnums=(1, 2))
    return mesh


def make_generate_es_sharded(model, batchnorm, encoder, sample_top_n, tick_size, n_msg_todo,
                             sim_init, valid_mask_array, *, conditional=True, shard="auto", backend="gpu"):
    """Multi-host population-sharded ES rollout. Closes over the (non-array) statics `shard_map` can't take
    and returns `gen(pop, train_state, m_seq_cond, b_seq_cond, sim_state, rng, init_hidden, init_time)` ->
    generate()'s 5-tuple, mapped over the G·Q grid (axis 0). `train_state` (the frozen backbone + the
    swapped decoder slot) is REPLICATED `P()`; every per-rollout array is sharded on a 'pop' mesh axis
    `P('pop')`. Auto-falls back to `jit(vmap)` on a single device (the CPU/1-GPU path). Mirrors
    HyperscaleES/llm_experiments/general_do_evolution.py's `shard_map(vmap(_gen, ...))`. The caller asserts
    `G*Q % jax.device_count() == 0`."""
    def _gen_thread(pop_i, ts, m_i, b_i, sim_i, rng_i, ih_i, it_i, unjit=False):
        return _generate_es_single(pop_i, sim_init, ts, model, batchnorm, encoder,
                                   sample_top_n, tick_size, m_i, b_i, n_msg_todo,
                                   sim_i, rng_i, ih_i, conditional, it_i, valid_mask_array,
                                   unjit=unjit)

    vmapped = jax.vmap(_gen_thread, in_axes=(0, None, 0, 0, 0, 0, 0, 0))
    if shard_decision(shard, jax.device_count()) == "vmap":
        return jax.jit(vmapped, backend=backend)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    mesh = _setup_shard_escapes()
    vmapped = jax.vmap(partial(_gen_thread, unjit=True),
                       in_axes=(0, None, 0, 0, 0, 0, 0, 0))
    in_specs = (P("pop"), P(), P("pop"), P("pop"), P("pop"), P("pop"), P("pop"), P("pop"))
    # check_rep=False: upstream lob code mixes unvarying constants and sharded data inside lax.cond
    # (e.g. encoding.combine_field returns NA_VAL vs combine_int(x)), which JAX 0.9's static
    # varying-manual-axes check rejects at trace time. The check is a static
    # replication audit, not a runtime semantic: every output here is per-rollout P("pop"), so
    # nothing is mis-declared by disabling it.
    return jax.jit(shard_map(vmapped, mesh=mesh, in_specs=in_specs, out_specs=P("pop"),
                             check_rep=False))


def make_feats_kl(backbone, bb_params, *, pooling, pool_start=0, n_cond=None, time_mask=None,
                  shard="auto", chunk=2, with_kl=False, backend="gpu"):
    """ONE frozen-backbone pass per [ctx ; cont] window -> pooled critic features, and (optionally) the
    per-rollout head-policy KL vs a reference head, from the SAME hiddens.

    Replaces the old `make_features_sharded` + separate `fitness.kl_penalty_chunked` pass, fusing the
    feature and KL passes to avoid two costs:
      * the old chunking was a PYTHON loop — under the sharded jit it UNROLLED into hundreds of backbone
        copies (catastrophic compile at scale); here chunks go through `jax.lax.map` (one traced copy);
      * the KL recomputed the full-window hiddens the feature pass had just produced — at n_gen=500 the
        window is 26k tokens, so that DOUBLED the dominant backbone cost; here the hiddens yield the
        pooled feature AND the KL, then are discarded chunk by chunk (never materialised for the grid).

    pool_start: first token position included in 'mean' pooling (0 = whole window; n_cond*26 =
    continuation-only — the context prefix is identical for real & fake, so pooling it only dilutes D).
    KL positions: the policy that sampled continuation token t reads the hidden at position t-1, so the
    KL consumes hid[start-1 : L-1] with start = n_cond*26 (the off-by-one fix; matches fitness.cont_hidden).
    `time_mask` [n_gen*26] zeroes the deterministic time-token positions (fitness.build_time_mask).

    Returns:
      with_kl=False: fn(ctx_tok, cont_tok, ctx_book, cont_book) -> feats [M, d_model]
      with_kl=True : fn(ctx_tok, cont_tok, ctx_book, cont_book, dir_idx, W_dirs, b_dirs, W_ref, b_ref)
                       -> (feats [M, d_model], kl [M])
    where `W_dirs` [G,d,V] / `b_dirs` [G,V] are the per-DIRECTION heads and `dir_idx` [M] (= r%G) gathers
    each rollout's head INSIDE the chunk (the full tiled [M,d,V] kernel is never built). Sharded mode:
    window inputs + dir_idx are `P('pop')`, heads/ref replicated `P()`; single-device falls back to a
    plain jit. `chunk` must divide the (per-shard) row count."""
    from ..data import loaders as Ddata
    from ..critic.discriminator import PaddedLobPredFeatures

    start = (n_cond or 0) * Ddata.MSG_LEN
    if with_kl:
        assert n_cond, "with_kl=True requires n_cond (KL hidden slice start)"

    def _hid(xm, xb, mt, bt):
        return backbone.apply({"params": bb_params}, xm, xb, mt, bt,
                              method=PaddedLobPredFeatures.features)            # [L, d_model]

    def _pooled(hid):
        return jnp.mean(hid[pool_start:], axis=0) if pooling == "mean" else hid[-1]

    def _kl(hid, Wd, bd, Wr, br):
        hc = hid[start - 1:-1]                                                  # [T, d] policy hiddens
        logp_r = jax.nn.log_softmax(hc @ Wr + br, axis=-1)
        logp_i = jax.nn.log_softmax(hc @ Wd + bd, axis=-1)
        kl_t = jnp.sum(jnp.exp(logp_i) * (logp_i - logp_r), axis=-1)            # [T]  KL(perturbed‖ref)
        if time_mask is None:
            return jnp.mean(kl_t)
        return jnp.sum(kl_t * time_mask) / jnp.maximum(jnp.sum(time_mask), 1.0)

    def _chunked(x_m, x_b, m_ts, b_ts, extra_xs, body):
        """lax.map `body` over [M/chunk, chunk, ...] reshapes of the window arrays (+ extras)."""
        M = x_m.shape[0]
        c = chunk if (chunk and 0 < chunk < M) else M
        assert M % c == 0, (f"feat/kl chunk {c} must divide the (per-shard) row count {M}; "
                            "adjust --feat_chunk/--kl_chunk or the G·Q/devices split")

        def _re(x):
            return x.reshape((M // c, c) + x.shape[1:])
        out = jax.lax.map(body, tuple(_re(x) for x in (x_m, x_b, m_ts, b_ts) + extra_xs))
        return jax.tree_util.tree_map(lambda o: o.reshape((M,) + o.shape[2:]), out)

    def _run_feats(ctx_tok, cont_tok, ctx_book, cont_book):
        tokens, book = Ddata.assemble_window(ctx_tok, cont_tok, ctx_book, cont_book)
        x_m, x_b, m_ts, b_ts = Ddata.critic_batch_from_tokens(tokens, book)

        def body(args):
            return jax.vmap(lambda xm, xb, mt, bt: _pooled(_hid(xm, xb, mt, bt)))(*args)
        return _chunked(x_m, x_b, m_ts, b_ts, (), body)                          # [M, d]

    def _run_feats_kl(ctx_tok, cont_tok, ctx_book, cont_book, dir_idx, W_dirs, b_dirs, W_ref, b_ref):
        tokens, book = Ddata.assemble_window(ctx_tok, cont_tok, ctx_book, cont_book)
        x_m, x_b, m_ts, b_ts = Ddata.critic_batch_from_tokens(tokens, book)

        def body(args):
            xm_c, xb_c, mt_c, bt_c, di_c = args

            def one(xm, xb, mt, bt, di):
                hid = _hid(xm, xb, mt, bt)
                return _pooled(hid), _kl(hid, W_dirs[di], b_dirs[di], W_ref, b_ref)
            return jax.vmap(one)(xm_c, xb_c, mt_c, bt_c, di_c)
        return _chunked(x_m, x_b, m_ts, b_ts, (dir_idx,), body)                  # ([M, d], [M])

    if shard_decision(shard, jax.device_count()) == "vmap":
        return jax.jit(_run_feats_kl if with_kl else _run_feats, backend=backend)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    mesh = jax.make_mesh((jax.device_count(),), ("pop",))
    # check_rep=False for the same reason as make_generate_es_sharded: the backbone is upstream lob
    # code that is not VMA-clean; all outputs are per-rollout P("pop") so the audit loses nothing.
    if with_kl:
        return jax.jit(shard_map(_run_feats_kl, mesh=mesh,
                                 in_specs=(P("pop"),) * 5 + (P(), P(), P(), P()),
                                 out_specs=(P("pop"), P("pop")), check_rep=False))
    return jax.jit(shard_map(_run_feats, mesh=mesh, in_specs=(P("pop"),) * 4, out_specs=P("pop"),
                             check_rep=False))


# ========================================================================================
# S5b — in/out projection LoRA scope (the do_Tmm fold; no materialised weights).
#
# The per-direction perturbation is a HOISTED LoRA factor tree: `build_proj_factors`
# pre-samples (A_g, B_g) for every MM_PARAM projection
# kernel with EggRoll's OWN `get_lora_update_params` — the SAME per-leaf key + iterinfo +
# sigma/sqrt(rank) that `EggRoll.do_updates` later uses to reconstruct the gradient, and with
# the antithetic sigma sign already folded into A (exactly what `do_Tmm` does internally).
# The model-side fold (s5/es_fold.fold_kernel) then computes `x @ W + x @ A @ B.T` == do_Tmm,
# with `x @ W` the UNCHANGED stock op — so es=None is byte-identical and sigma=0 is bit-exact.
# Rollout/update consistency is therefore the same mechanism S3 GPU-validated for the head.
# ========================================================================================
def make_proj_noiser(hs, params, es_map, *, sigma, lr, rank, group_size=0, noise_reuse=1,
                     solver=optax.adamw, solver_kwargs=None, seed=0):
    """EggRoll noiser over the FULL generator params for the proj scope.

    The solver is `optax.masked(solver, mask = es_map==MM_PARAM)`: without the mask, weight decay
    (or any param-dependent term) would silently move the frozen
    EXCLUDED leaves every step even though their ES gradient is exactly zero. Masked-out leaves
    get `optax.MaskedNode` in the opt state (no moment memory for the frozen 78M backbone) and a
    pass-through update (the zero gradient), so they stay bit-identical forever.
    `use_batched_update=True` buckets the ~identical projection kernels per (shape, class) so the
    78M-leaf update compiles fast. Returns (fnp, npar, es_tree_key)."""
    mask = jax.tree_util.tree_map(lambda m: int(m) == int(hs.MM_PARAM), es_map)

    def masked_solver(lr_, **kw):
        return optax.masked(solver(lr_, **kw), mask)

    fnp, npar = hs.EggRoll.init_noiser(
        params, sigma, lr, solver=masked_solver, solver_kwargs=(solver_kwargs or {}),
        rank=rank, group_size=group_size, noise_reuse=noise_reuse, use_batched_update=True)
    esk = build_es_tree_key(params, jax.random.key(seed), hs)
    return fnp, npar, esk


def build_proj_factors(hs, fnp, npar, params, es_map, esk, iterinfo):
    """Per-member LoRA factor tree for every MM_PARAM kernel: a nested dict mirroring `params`
    at the perturbed leaves only, each kernel leaf -> {"A": [N, in, r], "B": [N, out, r]} with
    A already scaled by the antithetic ±sigma/sqrt(rank) (get_lora_update_params semantics).
    iterinfo = (epochs[N], thread_ids[N]) as from `population_iterinfo`. sigma=0 -> A == 0 exactly."""
    epochs, threads = iterinfo
    base_sig = npar["sigma"] / jnp.sqrt(fnp["rank"])

    leaves_p = jax.tree_util.tree_flatten_with_path(params)[0]
    flat_m = jax.tree_util.tree_flatten(es_map)[0]
    flat_k = jax.tree_util.tree_flatten(esk)[0]
    assert len(leaves_p) == len(flat_m) == len(flat_k), "params/es_map/esk must share a treedef"

    out = {}
    for (path, leaf), m, key in zip(leaves_p, flat_m, flat_k):
        if int(m) != int(hs.MM_PARAM):
            continue

        def one(ep, th, _leaf=leaf, _key=key):
            return hs.eggroll.get_lora_update_params(fnp, base_sig, (ep, th), _leaf, _key)

        A, B = jax.vmap(one)(epochs, threads)                       # [N, in, r], [N, out, r]
        node = out
        names = _key_path_strs(path)
        for nm in names[:-1]:
            node = node.setdefault(nm, {})
        node[names[-1]] = {"A": A, "B": B}
    assert out, "es_map marked no MM_PARAM leaves — nothing to perturb"
    return out


def zero_proj_factors(factors, n=None):
    """Zeros-like factor tree: the exact no-op population (x @ 0 @ B.T == 0 bit-exact). With `n`,
    the leading member axis is resized to n (reference/eval rollouts at a different breadth)."""
    if n is None:
        return jax.tree_util.tree_map(jnp.zeros_like, factors)
    return jax.tree_util.tree_map(lambda x: jnp.zeros((n,) + x.shape[1:], x.dtype), factors)


def extract_trainable(hs, params, es_map):
    """Flat {'a/b/c/kernel': leaf} dict of the MM_PARAM (trainable) leaves — the proj-scope
    checkpoint payload (the EXCLUDED leaves never move under the masked solver, so the anchor
    checkpoint + this subtree reconstructs the full model)."""
    leaves_p = jax.tree_util.tree_flatten_with_path(params)[0]
    flat_m = jax.tree_util.tree_flatten(es_map)[0]
    return {"/".join(_key_path_strs(path)): leaf
            for (path, leaf), m in zip(leaves_p, flat_m) if int(m) == int(hs.MM_PARAM)}


def merge_trainable(hs, params, es_map, flat):
    """Inverse of extract_trainable: overwrite the MM_PARAM leaves of `params` from `flat`.
    A missing key fails loudly (structural mismatch = wrong checkpoint for this scope/toggles)."""
    def repl(path, leaf, m):
        if int(m) == int(hs.MM_PARAM):
            return flat["/".join(_key_path_strs(path))]
        return leaf
    return jax.tree_util.tree_map_with_path(repl, params, es_map)


def _generate_es_proj_single(es_factors, sim, ts, model_es, batchnorm, encoder,
                             sample_top_n, tick_size, m_seq_cond, b_seq_cond, n_msg_todo,
                             sim_state, rng, init_hidden, conditional, init_time,
                             valid_mask_array, unjit=False):
    """One member's proj-scope rollout: the UNCHANGED generate() with this member's LoRA factor
    tree threaded down the model (es kwarg chain). `ts` carries the CURRENT evolved params
    (shared across members); `model_es` is the BatchPaddedLobPredModelES wrapper. ZERO factors
    reproduce the stock rollout bit-exact; es_factors=None is NOT supported here — the ES
    wrapper's nn.vmap in_axes always expects the es arg (arity mismatch otherwise);
    use `zero_proj_factors` for the no-op population. unjit: see _generate_es_single."""
    assert es_factors is not None, "proj rollout needs a factor tree (zero_proj_factors for no-op)"
    import lob.inference_no_errcorr as inf
    gen_fn = inf.generate.__wrapped__ if unjit else inf.generate
    return gen_fn(
        sim, ts, model_es, batchnorm, encoder, sample_top_n, tick_size,
        m_seq_cond, b_seq_cond, n_msg_todo, sim_state, rng, init_hidden,
        conditional, init_time, False, None, valid_mask_array, es_factors)


def make_generate_es_proj_sharded(model_es, batchnorm, encoder, sample_top_n, tick_size,
                                  n_msg_todo, sim_init, valid_mask_array, *, conditional=True,
                                  shard="auto", backend="gpu"):
    """Proj-scope sibling of `make_generate_es_sharded` (kept separate so the W3-validated head
    path is untouched): gen(factors, train_state, m, b, sim, rng, ih, it) -> generate()'s 5-tuple.
    `factors` = the per-member LoRA tree (axis 0 = the G·Q grid, like the head pop); `train_state`
    is replicated. Same single-device jit(vmap) fallback and the same shard escapes."""
    def _gen_thread(fac_i, ts, m_i, b_i, sim_i, rng_i, ih_i, it_i, unjit=False):
        return _generate_es_proj_single(fac_i, sim_init, ts, model_es, batchnorm, encoder,
                                        sample_top_n, tick_size, m_i, b_i, n_msg_todo,
                                        sim_i, rng_i, ih_i, conditional, it_i, valid_mask_array,
                                        unjit=unjit)

    vmapped = jax.vmap(_gen_thread, in_axes=(0, None, 0, 0, 0, 0, 0, 0))
    if shard_decision(shard, jax.device_count()) == "vmap":
        return jax.jit(vmapped, backend=backend)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    mesh = _setup_shard_escapes()
    vmapped = jax.vmap(partial(_gen_thread, unjit=True),
                       in_axes=(0, None, 0, 0, 0, 0, 0, 0))
    in_specs = (P("pop"), P(), P("pop"), P("pop"), P("pop"), P("pop"), P("pop"), P("pop"))
    # check_rep=False: same VMA rationale as make_generate_es_sharded (upstream lax.cond mixing).
    return jax.jit(shard_map(vmapped, mesh=mesh, in_specs=in_specs, out_specs=P("pop"),
                             check_rep=False))


def make_feats_kl_proj(backbone, bb_params, *, pooling, pool_start=0, n_cond=None,
                       time_mask=None, shard="auto", chunk=1, backend="gpu"):
    """Proj-scope fused critic-features + EXACT per-member policy KL.

    Head scope got the KL for free from the shared anchor hiddens; in proj scope the PERTURBED
    BACKBONE defines the policy, so each window needs a second, ES-folded backbone pass:
      * anchor pass (bb_params, es=None)        -> pooled critic features  AND  pi_ref logits
        (the critic featurizer stays the FROZEN anchor — D must score samples in a fixed feature
        space, not chase a moving featurizer);
      * perturbed pass (cur_params, es=factors[dir]) -> pi_i logits.
    Both sets of logits use the FROZEN anchor decoder head (proj scope excludes the head), closed
    over from bb_params. KL positions/time-mask semantics match make_feats_kl exactly.

    Returns fn(ctx_tok, cont_tok, ctx_book, cont_book, dir_idx, cur_params, factors_G)
      -> (feats [M, d_model], kl [M])
    with `factors_G` the per-DIRECTION [G,...] LoRA tree (replicated; dir_idx gathers inside the
    chunk so the tiled [M,...] tree is never built) and `cur_params` the evolving full params
    (replicated; changes every step so it must be an argument, not a closure)."""
    from ..data import loaders as Ddata
    from ..critic.discriminator import PaddedLobPredFeatures

    assert n_cond, "make_feats_kl_proj requires n_cond (KL hidden slice start)"
    start = n_cond * Ddata.MSG_LEN
    W0 = bb_params["decoder"]["kernel"]
    b0 = bb_params["decoder"]["bias"]

    def _hid(params, xm, xb, mt, bt, es=None):
        return backbone.apply({"params": params}, xm, xb, mt, bt, es,
                              method=PaddedLobPredFeatures.features)              # [L, d_model]

    def _pooled(hid):
        return jnp.mean(hid[pool_start:], axis=0) if pooling == "mean" else hid[-1]

    def _kl(hid_i, hid_r):
        hci = hid_i[start - 1:-1]                                                 # [T, d] pi_i hiddens
        hcr = hid_r[start - 1:-1]                                                 # [T, d] pi_ref hiddens
        logp_i = jax.nn.log_softmax(hci @ W0 + b0, axis=-1)
        logp_r = jax.nn.log_softmax(hcr @ W0 + b0, axis=-1)
        kl_t = jnp.sum(jnp.exp(logp_i) * (logp_i - logp_r), axis=-1)              # [T] KL(pi_i || pi_ref)
        if time_mask is None:
            return jnp.mean(kl_t)
        return jnp.sum(kl_t * time_mask) / jnp.maximum(jnp.sum(time_mask), 1.0)

    def _run(ctx_tok, cont_tok, ctx_book, cont_book, dir_idx, cur_params, factors_G):
        tokens, book = Ddata.assemble_window(ctx_tok, cont_tok, ctx_book, cont_book)
        x_m, x_b, m_ts, b_ts = Ddata.critic_batch_from_tokens(tokens, book)
        M = x_m.shape[0]
        c = chunk if (chunk and 0 < chunk < M) else M
        assert M % c == 0, (f"feats_kl_proj chunk {c} must divide the (per-shard) row count {M}; "
                            "adjust --kl_chunk or the G·Q/devices split")

        def _re(x):
            return x.reshape((M // c, c) + x.shape[1:])

        def body(args):
            xm_c, xb_c, mt_c, bt_c, di_c = args

            def one(xm, xb, mt, bt, di):
                hid_r = _hid(bb_params, xm, xb, mt, bt)                           # anchor pass
                fac_i = jax.tree_util.tree_map(lambda a: a[di], factors_G)
                hid_i = _hid(cur_params, xm, xb, mt, bt, es=fac_i)                # perturbed pass
                return _pooled(hid_r), _kl(hid_i, hid_r)
            return jax.vmap(one)(xm_c, xb_c, mt_c, bt_c, di_c)
        out = jax.lax.map(body, tuple(_re(x) for x in (x_m, x_b, m_ts, b_ts, dir_idx)))
        return jax.tree_util.tree_map(lambda o: o.reshape((M,) + o.shape[2:]), out)

    if shard_decision(shard, jax.device_count()) == "vmap":
        return jax.jit(_run, backend=backend)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    mesh = jax.make_mesh((jax.device_count(),), ("pop",))
    # check_rep=False: same VMA rationale as make_feats_kl.
    return jax.jit(shard_map(_run, mesh=mesh,
                             in_specs=(P("pop"),) * 5 + (P(), P()),
                             out_specs=(P("pop"), P("pop")), check_rep=False))
