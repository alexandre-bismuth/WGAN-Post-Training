"""S5 — scaled EGGROLL-GAN with App-M centered-rank σ̄ fitness, the G×Q rollout grid, multi-host
population sharding, a monitored Goodhart guard (as CONTROL), and ≥500-step ckpt/resume.

FORK of `train_eggroll_gan.py` (S4): reuses the GH200-validated `rollout_to_cont` / `make_critic` /
`make_d_step` UNCHANGED so the proven critic path can't regress. `--scope head` (default) materialises
the per-direction decoder head and runs the bit-validated rollout (S5a — validates all the NEW machinery
with no model surgery). `--scope proj` (S5b) = LoRA r=4 EGGROLL on the interior in/out
projection kernels via the hoisted-factor do_Tmm fold (es_generator.build_proj_factors + the es kwarg
chain through generate/the model; masked-adamw moves ONLY the MM_PARAM leaves; full-tree do_updates with
use_batched_update; trainable-subtree checkpoints; exact per-member KL via make_feats_kl_proj's second,
ES-folded backbone pass — the critic featurizer stays the frozen anchor).

Design decisions baked into this build:
  * KL trust region anchors to the FROZEN PRETRAINED head (`--kl_ref anchor`) — passing the EVOLVING
    `leaves` as reference (`--kl_ref current`, ablation only) measures only the per-step σ-perturbation
    and never penalises cumulative drift.
  * The in-loop composite + `<out_dir>/best` selection is REMOVED — held-out evals showed it
    mis-selects. The σ=0 reference rollout stays for per-family ev_* diagnostics + invalidity +
    cross_entropy in the history; model selection is held-out LOB-Bench WS-21 over
    `--keep_step_ckpts` step dirs.
  * ONE backbone pass yields critic features AND the head KL (`es_generator.make_feats_kl`, lax.map
    chunking — no python-loop unrolling under jit, no duplicated 26k-token hidden pass at n_gen=500).
  * Critic pooling defaults to CONTINUATION-ONLY (`--pool_scope cont`): real & fake share the context
    prefix, so whole-window mean-pooling diluted D's signal.
  * Mode-collapse instrumentation: per-context across-direction diversity (token unique-fraction +
    event-histogram dispersion from the FREE G×Q grid) logged every eval.
  * Real-context pool REFRESH (`--pool_refresh_every`) against critic memorisation of a fixed pool.
  * Multi-host correctness layer (`dist_utils`): global-array wrapping for `jit(shard_map)` inputs and
    explicit all-gather before any host-side use — eager ops on cross-host shards raise. NOTE: gate on
    a 2-process single-node test before real multi-node use.
  * Generator noiser adamw gets weight_decay=0 (optax default 1e-4 silently decayed the head).
  * `--diagnose` noise estimator isolates sampling-RNG noise (same direction, SAME contexts, fresh rng)
    and subtracts the noise floor from the signal spread — the old gate passed ~50% under zero signal.

DEFAULT HORIZON IS THE THESIS TARGET: n_cond=500, n_gen=500. A 500-msg rollout is ~31x the 16-msg
smoke — run the TIMING GATE (N_G=2 at target G·Q) before committing 1000 steps. The materialised-head
grid costs G·Q·8.7MB (≈142 GB at G=512×Q=32) — head scope is for MACHINERY validation at moderate G·Q;
the full-scale run is `--scope proj` (S5b, do_Tmm fold, no materialisation). A hard guard
(`--max_head_grid_gb`) refuses configs that would OOM.

ONE STEP (G directions × Q shared contexts = G·Q rollouts; layout r = q*G + g, dir g = r%G, ctx q = r//G):
  1. draw Q fresh contexts from the staged pool; build the grid (head tiled over Q, contexts repeated over
     G, sampling rng INDEPENDENT per (g,q)); roll out -> G·Q fakes; ONE backbone pass -> critic features
     (+ KL vs the anchor head when kl_coef>0).
  2. D-step(s) (backprop): WGAN critic on the Q drawn reals (tiled to G·Q) vs the G·Q fakes.
  3. G-step (EGGROLL): score the SAME fakes; raw = D(fake) - λ·KL; reshape [Q,G];
     `fitness.rank_sigma_bar` -> per-direction (G,) (so sqrt(fitnesses.size)=sqrt(G) is consistent);
     `do_updates` ascends -> the head moves.
Goodhart guard (every eval_every): σ=0 reference rollout on a DISJOINT held-out eval set vs the real
continuation (engine replay) -> normalised stylized-fact composite. Checkpoint SELECTION = best composite
(kept under out_dir/best); sustained margin-worsening -> auto-stop (roll back = use out_dir/best).

CLUSTER: `--run` does heavy compute (G·Q AR rollouts + backbone passes) — GH200 ONLY (single-node via
`_run_s5.sh`, multi-node via `_run_s5_multinode.sbatch` with `jax.distributed`). Outputs -> $TMPDIR, rsync
to Lustre each ckpt. Without `--run` it executes ONLY the CPU-safe glue checks (login-node safe).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import jax

# Multi-host MUST initialise the coordination service BEFORE any JAX call that initialises the XLA
# backend — and the eggroll_gan import chain below does exactly that at IMPORT time (data ->
# discriminator -> lob modules build jnp tables on import). So peek at
# argv here, before those imports. main() then only logs; `import jax` alone does not init.
if "--distributed" in sys.argv:
    # Slingshot: the bare SLURM auto-detect resolves the coordinator to a non-routable
    # interface IP (-> gRPC connection refused). Use the EXPLICIT recipe from
    # lobmamba/run_train.py + train_full_autoreg.batch: coordinator = first node's HOSTNAME:port
    # (set as JAX_COORDINATOR_ADDRESS by the launcher), with num_processes/process_id from SLURM and
    # local_device_ids spanning the node's GPUs. Fall back to auto-detect only if the addr is unset.
    _coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if _coord:
        jax.distributed.initialize(
            coordinator_address=_coord,
            num_processes=int(os.environ.get("SLURM_NTASKS", os.environ.get("SLURM_NNODES", "1"))),
            process_id=int(os.environ.get("SLURM_PROCID", "0")),
            local_device_ids=list(range(int(os.environ.get("GPUS_PER_NODE", "4")))),
        )
    else:
        jax.distributed.initialize()

import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from ..config import DEFAULT as CFG
from ..es.es_plumbing import import_hyperscalees, build_es_map_proj
from ..es.es_generator import (make_decoder_noiser, build_decoder_population, population_iterinfo,
                           make_generate_es_sharded, make_feats_kl, gq_grid_indices,
                           grid_rngs, tile_dirs_over_Q, grid_repeat_contexts, shard_decision,
                           make_proj_noiser, build_proj_factors, zero_proj_factors,
                           extract_trainable, merge_trainable,
                           make_generate_es_proj_sharded, make_feats_kl_proj)
from .train_eggroll_gan import rollout_to_cont, make_critic, make_d_step, _SOLVERS, _solver_kwargs
from ..critic import discriminator as D
from ..critic import learned_critic as LC               # Lever 3 (learned TCN encoder critic)
from ..critic import raw_message_features as RMF        # raw decoded-message channels (same TCN)
from . import dist_utils as DU
from ..es import fitness as F
from ..eval import eval_monitor as EM
from ..eval import invalidity_audit as IA


# ----------------------------------------------------------------------------------------
# KL trust-region reference schedule (fixed ref + annealed λ; refresh is a toggle).
# ----------------------------------------------------------------------------------------
def kl_lambda(step, n_g_steps, kl_coef, *, anneal="cosine"):
    """λ schedule for the fixed-π_ref trust region. `cosine` anneals kl_coef -> 0 over the run (loosens the
    hard pretrained anchor as the generator legitimately moves); `const` holds kl_coef. kl_coef=0 -> 0."""
    if kl_coef <= 0.0:
        return 0.0
    if anneal == "const":
        return kl_coef
    frac = min(max(step / max(n_g_steps, 1), 0.0), 1.0)
    return kl_coef * 0.5 * (1.0 + math.cos(math.pi * frac))     # cosine kl_coef -> 0


# ----------------------------------------------------------------------------------------
# Diagnostic estimators (pure arrays; CPU-testable). The non-stationarity gate is in `diagnose`.
# ----------------------------------------------------------------------------------------
def fitness_noise_std(scores_draws):
    """One fixed direction's score across several draws [n] -> std (an ES measurement-noise probe)."""
    return float(jnp.std(jnp.asarray(scores_draws)))


def signal_spread(scores_dirs):
    """Several directions on ONE fixed Q-batch [G] -> std across directions (the between-direction signal)."""
    return float(jnp.std(jnp.asarray(scores_dirs)))


def _rank_of(x):
    return jnp.argsort(jnp.argsort(x))


def ranking_stability(fitness_steps):
    """[n_steps, G] per-direction fitness across steps -> mean consecutive-step rank correlation in [-1,1].
    ~1 = stable rankings (ES is tracking); low/churning = either ES variance (D frozen) or D non-stationarity
    (D moving) — `diagnose` distinguishes the two by running this with D frozen vs D live."""
    fs = jnp.asarray(fitness_steps)
    n = fs.shape[0]
    if n < 2:
        return 1.0
    corrs = []
    for i in range(n - 1):
        ra = _rank_of(fs[i]).astype(jnp.float32)
        rb = _rank_of(fs[i + 1]).astype(jnp.float32)
        corrs.append(jnp.corrcoef(ra, rb)[0, 1])
    return float(jnp.mean(jnp.asarray(corrs)))


# ----------------------------------------------------------------------------------------
# Checkpoint / resume (atomic breadcrumb; ≥500-step cadence). CPU-testable round-trip.
# ----------------------------------------------------------------------------------------
def save_ckpt(out_dir, step, gen_payload, head_vars, npar, history, composites, meta, scope="head"):
    """gen_payload: head scope = the {kernel,bias} decoder leaves (legacy format, byte-compatible
    with every prior S5a/W3 checkpoint); proj scope = the FLAT trainable subtree from
    `extract_trainable` (the EXCLUDED 78M backbone never moves under the masked solver, so the
    anchor checkpoint + this subtree reconstructs the full model — ~10x smaller ckpt)."""
    os.makedirs(out_dir, exist_ok=True)
    gen_key, gen_file = (("generator_head", "s5_generator_head.msgpack") if scope == "head"
                         else ("generator_proj", "s5_generator_proj.msgpack"))
    paths = {gen_key: gen_file,
             "critic_head": "s5_critic_head.msgpack",
             "noiser_state": "s5_noiser_state.msgpack"}
    payload = ({"kernel": gen_payload["kernel"], "bias": gen_payload["bias"]} if scope == "head"
               else gen_payload)
    with open(os.path.join(out_dir, gen_file), "wb") as f:
        f.write(serialization.to_bytes(payload))
    with open(os.path.join(out_dir, paths["critic_head"]), "wb") as f:
        f.write(serialization.to_bytes(head_vars))
    with open(os.path.join(out_dir, paths["noiser_state"]), "wb") as f:
        f.write(serialization.to_bytes(npar))
    bc = dict(stage="S5", step=int(step), history=history, composites=composites, **paths, **meta)
    tmp = os.path.join(out_dir, "latest_checkpoint.json.tmp")
    with open(tmp, "w") as f:
        json.dump(bc, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, "latest_checkpoint.json"))   # atomic: a partial write is never read
    return bc


def load_ckpt(resume_dir, gen_tmpl, head_vars_tmpl, npar_tmpl):
    """Restore from the breadcrumb ONLY (never `ls` — Lustre rule). Returns (gen_payload, head_vars,
    npar, bc). The generator file key is read from the breadcrumb itself (generator_head |
    generator_proj), so a head checkpoint can never silently restore into a proj run or vice versa
    (the template/treedef mismatch fails loudly)."""
    with open(os.path.join(resume_dir, "latest_checkpoint.json")) as f:
        bc = json.load(f)
    gen_file = bc.get("generator_proj") or bc["generator_head"]
    with open(os.path.join(resume_dir, gen_file), "rb") as f:
        gen_payload = serialization.from_bytes(gen_tmpl, f.read())
    with open(os.path.join(resume_dir, bc["critic_head"]), "rb") as f:
        head_vars = serialization.from_bytes(head_vars_tmpl, f.read())
    with open(os.path.join(resume_dir, bc["noiser_state"]), "rb") as f:
        npar = serialization.from_bytes(npar_tmpl, f.read())
    return gen_payload, head_vars, npar, bc


# ----------------------------------------------------------------------------------------
# Pool slicing helper.
# ----------------------------------------------------------------------------------------
def _take(tree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def _pop_size(pop):
    """Leading (member) axis of a population pytree — works for head leaves AND LoRA factor trees."""
    return int(jax.tree_util.tree_leaves(pop)[0].shape[0])


# ----------------------------------------------------------------------------------------
# The scaled loop (GPU; --run on the GH200 / multi-node).
# ----------------------------------------------------------------------------------------
def train(hs, args):
    from ..tests.s3_es_rollout import _prep_real_batch, _sample_windows

    cfg = CFG
    G, Q = args.G, args.Q
    assert G % 2 == 0, f"G must be even for antithetic ±σ pairs (got {G})"
    assert args.n_d_steps >= 1
    n_dev = jax.device_count()
    sharded = shard_decision(args.shard, n_dev) == "shard_map"
    mesh = DU.make_pop_mesh() if sharded else None
    mh = DU.is_multihost()
    p0 = jax.process_index() == 0

    def log(msg):
        if p0:
            print(msg, flush=True)

    n_pool = args.n_ctx_pool + args.n_eval_ctx
    if sharded:
        assert (G * Q) % n_dev == 0, f"G*Q={G*Q} must divide device_count={n_dev}"
        assert args.n_eval_ctx % n_dev == 0, f"n_eval_ctx={args.n_eval_ctx} must divide device_count={n_dev}"
        assert args.n_ctx_pool % n_dev == 0 and n_pool % n_dev == 0, \
            f"pool sizes ({args.n_ctx_pool}, {n_pool}) must divide device_count={n_dev}"
    solver = _SOLVERS[args.solver]
    # optax.adamw default weight_decay=1e-4 would silently decay the evolving head every step.
    solver_kwargs = _solver_kwargs(args.solver)
    log(f"[S5] scope={args.scope} G={G} Q={Q} (G*Q={G*Q} rollouts/step) rank={args.rank} sigma={args.sigma} "
        f"lr={args.lr:.3e} solver={args.solver} fitness_control={args.fitness_control} "
        f"kl_coef={args.kl_coef} kl_ref={args.kl_ref} "
        f"n_d={args.n_d_steps} n_g={args.n_g_steps} n_cond={args.n_cond} n_gen={args.n_gen} "
        f"pool_scope={args.pool_scope} devices={n_dev} shard={'shard_map' if sharded else 'vmap'} "
        f"multihost={mh}")
    proj = args.scope == "proj"
    if proj and args.kl_ref != "anchor":
        log("[S5] --scope proj: the KL reference is the ANCHOR backbone by construction "
            "(kl_ref=current is a head-scope-only ablation); proceeding with anchor.")

    # --- staged pool: train contexts + a DISJOINT held-out eval set (one dataset-load path) ---
    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, n_pool,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels, wide_book_dir=args.wide_book_dir)
    inf = P["inf"]
    model, batchnorm, encoder = P["model"], P["batchnorm"], P["encoder"]
    train_state = P["train_state"]
    backbone = D.make_backbone(P["model_cls"])
    bb_params = P["bb_params"]
    pooling = cfg.disc.pooling
    pool_start = args.n_cond * F.MSG_LEN if args.pool_scope == "cont" else 0
    learned = args.critic_input in ("learned", "raw")   # sequence critic [N, T, F_in] vs backbone hiddens
    if learned:
        _sync_enc_cfg(cfg, args)                    # push --enc_* CLI knobs into cfg.disc for make_critic
    log(f"[S5] critic_input={args.critic_input}"
        + (f" ({args.critic_input} {cfg.disc.enc_type} L={cfg.disc.enc_layers} h={cfg.disc.enc_hidden} "
           f"pool={cfg.disc.enc_pool} scales={tuple(cfg.disc.enc_scales)} ens={cfg.disc.ens_size}"
           f"(β={cfg.disc.ens_pessimism}), RF="
           f"{LC.multiscale_receptive_field(cfg.disc.enc_layers, cfg.disc.enc_kernel, cfg.disc.enc_scales)})"
           if learned else " (frozen-backbone pooled hidden)"))

    # Materialised-head memory guard (head scope only exists for machinery validation; S5b folds via do_Tmm).
    if not proj:
        head_bytes = G * Q * (P["kernel"].size + P["bias"].size) * 4
        per_dev_gb = head_bytes / (n_dev if sharded else 1) / 1e9
        log(f"[S5] materialised head grid: {head_bytes / 1e9:.1f} GB total, {per_dev_gb:.1f} GB/device")
        if per_dev_gb > args.max_head_grid_gb:
            raise RuntimeError(
                f"head-only scope at G*Q={G*Q} materialises {per_dev_gb:.1f} GB/device of decoder copies "
                f"(> --max_head_grid_gb {args.max_head_grid_gb}). Reduce G*Q for S5a machinery tests; the "
                "full-scale run is --scope proj (S5b do_Tmm fold, no materialisation).")

    tr = slice(0, args.n_ctx_pool)
    ev = slice(args.n_ctx_pool, n_pool)
    eval_idx = list(P["idx"][args.n_ctx_pool:])     # held-out window ids — pool refresh stays disjoint
    pool = dict(m=P["m_seq_inp"][tr], b=P["b_seq_inp"][tr], sim=_take(P["sim_states_init"], tr),
                ih=_take(P["init_hidden_batched"], tr), it=P["init_time_batched"][tr],
                ctx_tok=P["ctx_tokens"][tr], ctx_book=P["b_seq_inp"][tr],
                real_tok=P["real_cont_tokens"][tr], real_book=P["real_cont_book"][tr],
                raw_cont=P["m_seq_raw_cont"][tr])   # raw decoded real continuation msgs (learned critic)
    eval_in = dict(m=P["m_seq_inp"][ev], b=P["b_seq_inp"][ev], sim=_take(P["sim_states_init"], ev),
                   ih=_take(P["init_hidden_batched"], ev), it=P["init_time_batched"][ev],
                   rng=P["rngs"][ev])
    eval_raw_cont = P["m_seq_raw_cont"][ev]                       # host-side (engine replay input)
    eval_dev = {k: DU.shard_pop(v, mesh) for k, v in eval_in.items()}

    # ONE backbone pass per window: pooled critic features (+ KL when kl_coef>0). lax.map-chunked.
    # head scope: KL is free off the shared anchor hiddens (per-direction heads on hid_ref).
    # proj scope: the PERTURBED BACKBONE is the policy -> make_feats_kl_proj adds one ES-folded
    # backbone pass per window for the exact per-member KL (the critic featurizer stays the anchor).
    time_mask = F.build_time_mask(args.n_gen) if args.kl_coef > 0 else None
    feats_fn = make_feats_kl(backbone, bb_params, pooling=pooling, pool_start=pool_start,
                             n_cond=args.n_cond, shard=args.shard, chunk=args.feat_chunk, with_kl=False)
    featskl_fn = None
    if args.kl_coef > 0:
        if proj:
            featskl_fn = make_feats_kl_proj(backbone, bb_params, pooling=pooling,
                                            pool_start=pool_start, n_cond=args.n_cond,
                                            time_mask=time_mask, shard=args.shard,
                                            chunk=args.kl_chunk)
        else:
            featskl_fn = make_feats_kl(backbone, bb_params, pooling=pooling, pool_start=pool_start,
                                       n_cond=args.n_cond, time_mask=time_mask, shard=args.shard,
                                       chunk=args.kl_chunk, with_kl=True)

    def _pool_feats(ct, cn, cb, cnb):
        out = feats_fn(DU.shard_pop(ct, mesh), DU.shard_pop(cn, mesh),
                       DU.shard_pop(cb, mesh), DU.shard_pop(cnb, mesh))
        return np.asarray(DU.gather_host(out, mesh))

    # --- REAL critic features for the train pool (recomputed on every pool refresh) ---
    # learned: descriptor sequence of the real continuation replayed through the engine (LearnedFeats);
    #   standardiser fit once on the first real pool. backbone (legacy): frozen-Mamba3 pooled hidden.
    sf = ((RawMsgFeats if args.critic_input == "raw" else LearnedFeats)(
        inf, P["sim_init"], P["tick_size"],
        robust=(cfg.disc.enc_whiten == "robust")).fit(pool) if learned else None)

    def _real_feats(pool_):
        if learned:
            return sf.real(pool_)
        return _pool_feats(pool_["ctx_tok"], pool_["real_tok"], pool_["ctx_book"], pool_["real_book"])

    real_feats_pool = _real_feats(pool)
    if not bool(np.all(np.isfinite(real_feats_pool))):
        raise RuntimeError(f"non-finite real critic features (critic_input={args.critic_input})")

    # noiser around the pretrained start (THE FROZEN ANCHOR); rollout fn (sharded or vmap).
    # `leaves` = the thing do_updates evolves: head scope -> the 2 decoder leaves;
    # proj scope -> the FULL params tree (masked solver moves only the MM_PARAM projections).
    kernel0, bias0 = P["kernel"], P["bias"]
    if proj:
        params0 = train_state.params                              # frozen pretrained anchor (full tree)
        es_map = build_es_map_proj(params0, hs,
                                   perturb_glu=bool(args.perturb_glu),
                                   perturb_book_proj=bool(args.perturb_book_proj),
                                   perturb_fused_encoder=bool(args.perturb_fused_encoder))
        fnp, npar, esk = make_proj_noiser(
            hs, params0, es_map, sigma=args.sigma, lr=args.lr, rank=args.rank,
            group_size=0, noise_reuse=cfg.eggroll.noise_reuse, solver=solver,
            solver_kwargs=solver_kwargs, seed=args.seed)
        leaves = params0
        tr_anchor = extract_trainable(hs, params0, es_map)        # frozen reference for ||Δ|| / finite
        n_mm = len(tr_anchor)
        mm_params = int(sum(v.size for v in tr_anchor.values()))
        fac_dims = int(sum(args.rank * (v.shape[0] + v.shape[1]) for v in tr_anchor.values()))
        fac_gb = G * Q * fac_dims * 4 / (n_dev if sharded else 1) / 1e9
        log(f"[S5] proj scope: {n_mm} MM_PARAM projection kernels, {mm_params/1e6:.1f}M trainable "
            f"params; LoRA factor grid ~{fac_gb:.2f} GB/device at G*Q={G*Q} (r={args.rank})")
        from lob.lob_seq_model import BatchPaddedLobPredModelES
        model_es = BatchPaddedLobPredModelES(**dict(P["model_cls"].keywords),
                                             training=False, step_rescale=1.0)
        gen = make_generate_es_proj_sharded(model_es, batchnorm, encoder, P["sample_top_n"],
                                            P["tick_size"], args.n_gen, P["sim_init"],
                                            P["valid_mask_array"], conditional=True, shard=args.shard)
    else:
        fnp, npar, es_map, esk, leaves = make_decoder_noiser(
            hs, kernel0, bias0, sigma=args.sigma, lr=args.lr, rank=args.rank,
            group_size=0, noise_reuse=cfg.eggroll.noise_reuse, solver=solver,
            solver_kwargs=solver_kwargs, seed=args.seed)
        gen = make_generate_es_sharded(model, batchnorm, encoder, P["sample_top_n"], P["tick_size"],
                                       args.n_gen, P["sim_init"], P["valid_mask_array"],
                                       conditional=True, shard=args.shard)
    ts_in = DU.replicate_tree(train_state, mesh)

    # learned critic: feed make_critic the (T, F_in) SEQUENCE shape (a tuple selects the TCN critic);
    # vector critics pass the int feature width exactly as before.
    feat_shape = tuple(real_feats_pool.shape[1:]) if learned else int(real_feats_pool.shape[-1])
    head, hparams, sn, tx, opt_state = make_critic(cfg, feat_shape, args.seed)
    d_step = make_d_step(head, tx, r1_gamma=args.r1_gamma)
    dir_idx, _ = gq_grid_indices(G, Q)
    dir_idx_dev = DU.shard_pop(dir_idx, mesh)
    ETi = int(inf.EVENT_TYPE_i)
    _trim = jax.jit(lambda mt, bf: rollout_to_cont(mt, bf, args.n_gen))
    _et_slice = jax.jit(lambda md: md[..., ETi].astype(jnp.int32))
    _tok_sub = jax.jit(lambda t: t[:, ::args.token_div_stride])

    # Grid builders: device-side (validated jnp path) single-process; HOST-side under multi-host so the
    # full grid is never materialised on one device before sharding (DU.shard_pop slices per shard).
    def _grid_dirs(pop_):
        if mh:
            t = jax.tree_util.tree_map(
                lambda x: np.tile(np.asarray(x), (Q,) + (1,) * (x.ndim - 1)), pop_)
        else:
            t = tile_dirs_over_Q(pop_, Q)
        return DU.shard_pop(t, mesh)

    def _grid_ctx(tree_):
        if mh:
            t = jax.tree_util.tree_map(lambda x: np.repeat(np.asarray(x), G, axis=0), tree_)
        else:
            t = grid_repeat_contexts(tree_, G)
        return DU.shard_pop(t, mesh)

    history, composites, diverged = [], [], False
    ev_baseline, best_comp = None, float("inf")
    step0, last_step = 0, 0
    if args.resume_dir and os.path.exists(os.path.join(args.resume_dir, "latest_checkpoint.json")):
        gen_tmpl = extract_trainable(hs, leaves, es_map) if proj else leaves
        payload, head_vars0, npar, bc = load_ckpt(args.resume_dir, gen_tmpl, {"params": hparams, **sn}, npar)
        leaves = merge_trainable(hs, leaves, es_map, payload) if proj else payload
        hparams = head_vars0["params"]; sn = {k: v for k, v in head_vars0.items() if k != "params"}
        step0, history, composites = bc["step"], bc.get("history", []), bc.get("composites", [])
        ev_baseline = bc.get("ev_baseline") or None
        best_comp = bc.get("best_composite", min(composites) if composites else float("inf"))
        last_step = step0
        log(f"[S5] resumed from {args.resume_dir} @ step {step0} (baseline={'set' if ev_baseline else 'unset'})")

    base_key = jax.random.PRNGKey(args.seed)

    def _gen_payload():
        """Checkpoint payload for the current scope (reads `leaves` at call time)."""
        return extract_trainable(hs, leaves, es_map) if proj else leaves

    def _refresh_pool(k):
        """k-th pool refresh: redraw n_ctx_pool train windows (disjoint from eval) + recompute real feats.
        Keyed by the refresh ordinal -> deterministic across resume and across processes."""
        nonlocal pool, real_feats_pool
        rk = jax.random.fold_in(base_key, 777_000_000 + k)
        rk_i, rk_r = jax.random.split(rk)
        W = _sample_windows(inf, P["ds"], args.n_cond, args.n_gen, args.n_ctx_pool, rk_i, rk_r,
                            sim_init=P["sim_init"], tick_size=P["tick_size"], exclude_idx=eval_idx,
                            include_idx=P["include_idx"])
        pool = dict(m=W["m_seq_inp"], b=W["b_seq_inp"], sim=W["sim_states_init"],
                    ih=pool["ih"], it=W["init_time_batched"],
                    ctx_tok=W["ctx_tokens"], ctx_book=W["b_seq_inp"],
                    real_tok=W["real_cont_tokens"], real_book=W["real_cont_book"],
                    raw_cont=W["m_seq_raw_cont"])
        real_feats_pool = _real_feats(pool)        # featuriser standardiser stays fixed across refreshes
        log(f"[S5] pool refresh #{k}: {args.n_ctx_pool} fresh train contexts (eval set untouched)")

    if args.pool_refresh_every and step0 > args.pool_refresh_every:
        k_last = (step0 - 1) // args.pool_refresh_every
        if k_last > 0:
            _refresh_pool(k_last)                                  # restore the resumed run's pool state

    for step in range(step0 + 1, args.n_g_steps + 1):
        if args.pool_refresh_every and step > 1 and (step - 1) % args.pool_refresh_every == 0:
            _refresh_pool((step - 1) // args.pool_refresh_every)
        step_key = jax.random.fold_in(base_key, step)
        drawn = jax.random.choice(step_key, args.n_ctx_pool, shape=(Q,), replace=False)
        drawn_np = np.asarray(drawn)

        # (1) build the G×Q grid and roll out.
        it_G = population_iterinfo(G, step - 1)
        if proj:
            # the evolving thing is INSIDE the params -> rebuild the rollout train_state each step.
            ts_in = DU.replicate_tree(train_state.replace(params=leaves), mesh)
            pop = build_proj_factors(hs, fnp, npar, leaves, es_map, esk, it_G)  # per-direction LoRA [G,...]
        else:
            pop = build_decoder_population(hs, fnp, npar, leaves, esk, it_G)    # per-direction heads [G,...]
        pop_grid = _grid_dirs(pop)                                              # [Q*G,...] (rollout r uses dir g)
        m_grid = _grid_ctx(pool["m"][drawn])
        b_grid = _grid_ctx(pool["b"][drawn])
        sim_grid = _grid_ctx(_take(pool["sim"], drawn))
        ih_grid = _grid_ctx(_take(pool["ih"], drawn))
        itime_grid = _grid_ctx(pool["it"][drawn])
        rng_grid = DU.shard_pop(grid_rngs(step_key, G, Q), mesh)
        ctx_tok_grid = _grid_ctx(pool["ctx_tok"][drawn])
        ctx_book_grid = _grid_ctx(pool["ctx_book"][drawn])

        g_out = gen(pop_grid, ts_in, m_grid, b_grid, sim_grid, rng_grid, ih_grid, itime_grid)
        fct, fcb = _trim(g_out[3], g_out[4])                                    # [Q*G, ...] (stay sharded)
        num_errors = np.asarray(DU.gather_host(g_out[2], mesh)).astype(np.float32)

        # critic FAKE features + per-rollout KL trust region. KL (when on) needs the backbone
        # continuation hiddens regardless of critic_input; its pooled `feats_dev` is the critic input
        # ONLY in backbone mode (ignored under learned, which featurises the decoded order book).
        if args.kl_coef > 0:
            if proj:
                fac_rep = DU.replicate_tree(pop, mesh)                          # per-direction [G,...]
                feats_dev, kl_dev = featskl_fn(ctx_tok_grid, fct, ctx_book_grid, fcb, dir_idx_dev,
                                               ts_in.params, fac_rep)
            else:
                W_ref = kernel0 if args.kl_ref == "anchor" else leaves["kernel"]
                b_ref = bias0 if args.kl_ref == "anchor" else leaves["bias"]
                heads_rep = DU.replicate_tree(dict(Wd=pop["kernel"], bd=pop["bias"],
                                                   Wr=W_ref, br=b_ref), mesh)
                feats_dev, kl_dev = featskl_fn(ctx_tok_grid, fct, ctx_book_grid, fcb, dir_idx_dev,
                                               heads_rep["Wd"], heads_rep["bd"],
                                               heads_rep["Wr"], heads_rep["br"])
            kl_grid = np.asarray(DU.gather_host(kl_dev, mesh))
        else:
            feats_dev = None if learned else feats_fn(ctx_tok_grid, fct, ctx_book_grid, fcb)
            kl_grid = np.zeros(G * Q, dtype=np.float32)
        if learned:
            # descriptors on the generated rollout: the engine already produced its per-message L2
            # (g_out[1]) and decoded messages (g_out[0]); featurise with the SAME map + standardiser
            # as the real class.
            gl2_h = np.asarray(DU.gather_host(g_out[1], mesh))                  # [Q*G, n_gen, W]
            gmsg_h = np.asarray(DU.gather_host(g_out[0], mesh))                 # [Q*G, n_gen, n_fields]
            fake_feats = sf.fake(gl2_h, gmsg_h)                                 # [Q*G, N_FEATURES]
        else:
            fake_feats = np.asarray(DU.gather_host(feats_dev, mesh))           # [Q*G, d]
        real_feats_grid = np.repeat(real_feats_pool[drawn_np], G, axis=0)      # [Q*G, k] (balanced vs fakes)

        # (2) D-step(s): sharpen the critic on real vs the current fakes. The critic is NOT sharded
        # (only the rollout/KL are), so for the learned (sequence) critic the d-step forward+backward
        # over [2*Q*G, T, F_in] OOMs on one device at large Q. --crit_batch caps the critic's minibatch
        # to a strided subset spanning contexts, so critic memory is independent of Q: the ES gets more
        # samples/perturbation (big Q -> low-noise σ̄) while the critic trains on a fixed minibatch and
        # EVERY fake is still scored below. crit_batch<=0 (default) = full batch -> the proven
        # backbone path is bit-identical.
        CB = int(args.crit_batch); n_all = fake_feats.shape[0]
        if 0 < CB < n_all:
            d_idx = np.linspace(0, n_all - 1, CB).astype(np.int64)             # strided, spans contexts
            d_real, d_fake = real_feats_grid[d_idx], fake_feats[d_idx]
        else:
            d_real, d_fake = real_feats_grid, fake_feats
        for _ in range(args.n_d_steps):
            hparams, sn, opt_state, dloss, s_real, s_fake = d_step(hparams, sn, opt_state,
                                                                   d_real, d_fake)
        head_vars = {"params": hparams, **sn}

        # (3) G-step (EGGROLL): score the SAME fakes; centered-rank σ̄ -> per-direction (G,) -> ascend.
        # Per-rollout-independent, so chunking the train=False forward over Q*G is EXACT (no approximation)
        # and bounds the critic activation memory regardless of Q.
        if 0 < CB < n_all:
            scores = np.concatenate([np.asarray(head.apply(head_vars, fake_feats[s:s + CB], train=False))
                                     for s in range(0, n_all, CB)], axis=0)    # [Q*G]
        else:
            scores = np.asarray(head.apply(head_vars, fake_feats, train=False))    # [Q*G]
        lam = kl_lambda(step, args.n_g_steps, args.kl_coef, anneal=args.kl_anneal)
        # SCALE-INVARIANT KL trust region: z-score the critic scores PER CONTEXT (across G) before the
        # λ·KL subtract, so λ means "fraction of a score-std" regardless of the critic's output scale.
        # The learned TCN critic's WGAN score can reach ~1e4 (residual-compounded, unbounded), which
        # would otherwise swamp λ·kl and silently disable the KL anchor (the primary anti-Goodhart lever);
        # standardising restores it and makes λ comparable across ALL critics. NO-OP when λ=0 (a per-context
        # affine map leaves the σ̄ ranks unchanged), so the kl_coef=0 path stays bit-identical.
        scores_qg = jnp.asarray(scores).reshape(Q, G)
        kl_qg = jnp.asarray(kl_grid).reshape(Q, G)
        s_sd = jnp.std(scores_qg, axis=1, keepdims=True) + 1e-6
        raw = (scores_qg - jnp.mean(scores_qg, axis=1, keepdims=True)) / s_sd - lam * kl_qg   # [Q, G]
        fit = F.rank_sigma_bar(raw)                                            # [G]
        if args.fitness_control == "shuffle":
            # Drift control (attribution null): permute the per-direction fitness so do_updates
            # consumes ZERO information from D (and from the KL term) while the fitness MULTISET —
            # hence the update-magnitude statistics, adamw moments, weight decay, and σ noise —
            # stays identical to a live run. Whatever this lane improves is optimizer drift, not
            # critic signal. Key stream 888M+step is disjoint from pool refresh (777M+k) and
            # step_key (fold_in(base_key, step)); deterministic across hosts and resume.
            perm = jax.random.permutation(jax.random.fold_in(base_key, 888_000_000 + step), G)
            fit = fit[perm]
        npar, leaves = hs.EggRoll.do_updates(fnp, npar, leaves, esk, fit, it_G, es_map)
        last_step = step

        # --- monitor / divergence guard ---
        if step % args.eval_every == 0 or step == 1 or step == args.n_g_steps:
            sep = float(D.critic_separation(s_real, s_fake)); auc = float(D.roc_auc(s_real, s_fake))
            ce = float(D.critic_cross_entropy(s_real, s_fake))   # side diagnostic (nats): ~0.693 chance, →0 separable
            mean_score = float(np.mean(scores)); mean_kl = float(np.mean(kl_grid))
            if proj:
                tr_now = extract_trainable(hs, leaves, es_map)
                dk = max(float(jnp.max(jnp.abs(tr_now[k] - tr_anchor[k]))) for k in tr_anchor)
                gen_finite = all(bool(jnp.isfinite(v).all()) for v in tr_now.values())
            else:
                dk = float(jnp.max(jnp.abs(leaves["kernel"] - kernel0)))
                gen_finite = bool(jnp.isfinite(leaves["kernel"]).all())
            finite = bool(gen_finite and np.isfinite(scores).all() and np.isfinite(float(dloss)))
            # mode-collapse instrumentation: per-context across-G diversity from the free G×Q grid.
            et_h = np.asarray(DU.gather_host(_et_slice(g_out[0]), mesh))
            tok_h = np.asarray(DU.gather_host(_tok_sub(fct), mesh))
            div = EM.population_diversity(tok_h, et_h, G, Q, token_stride=1)   # pre-strided by _tok_sub
            rec = dict(step=step, d_loss=float(dloss), separation=sep, auc=auc, cross_entropy=ce,
                       g_mean_score=mean_score,
                       mean_kl=mean_kl, lam=lam, r1_gamma=float(args.r1_gamma),
                       mean_num_errors=float(np.mean(num_errors)),
                       dkernel=dk, finite=finite,
                       token_unique_frac=float(div["token_unique_frac"]),
                       event_hist_disp=float(div["event_hist_disp"]))
            # Goodhart guard: σ=0 reference rollout on the held-out eval set vs the real continuation.
            if proj:
                # ts_in was built BEFORE this step's do_updates — rebuild so the composite scores
                # the POST-update params (the exact thing save_ckpt writes / best-ckpt selects).
                ts_eval = DU.replicate_tree(train_state.replace(params=leaves), mesh)
                gl2, get_, gmsg = _reference_rollout_proj(gen, pop, ts_eval, eval_dev, mesh,
                                                          G_eval=args.n_eval_ctx, et_slice=_et_slice)
            else:
                gl2, get_, gmsg = _reference_rollout(gen, leaves, ts_in, eval_dev, mesh,
                                                     G_eval=args.n_eval_ctx, et_slice=_et_slice)
            rl2 = _replay_real(inf, P["sim_init"], eval_in["sim"], eval_raw_cont)
            ev_metrics = EM.stylized_fact_metrics(
                gl2, rl2, get_, eval_raw_cont[..., ETi].astype(jnp.int32), n_levels=inf.l2_state_n)
            # Reference-rollout stylized metrics are per-family DIAGNOSTICS only. The in-loop
            # composite + best/-ckpt selection is removed (held-out evals showed it mis-selects).
            # Selection is held-out LOB-Bench WS-21 over the step-keyed ckpts (--keep_step_ckpts).
            if ev_baseline is None and all(math.isfinite(float(v)) for v in ev_metrics.values()):
                ev_baseline = {k: float(v) for k, v in ev_metrics.items()}     # run-start reference (meta only)
            rec.update({f"ev_{k}": float(v) for k, v in ev_metrics.items()})
            # Engine-truthful LOB-validity of the σ=0 reference rollout (the REAL invalidity, not the
            # num_errors no-op rate). try/except so a verdict hiccup never aborts a training run.
            try:
                inv = IA.rollout_invalidity_stats(P["sim_init"], eval_in["sim"], gmsg, inf)
                rec.update(inv_valid=inv["frac_valid"], inv_partial=inv["frac_partial"],
                           inv_hard=inv["frac_hard_invalid"],
                           inv_phantom=inv["by_code"]["phantom_cancel"],
                           inv_no_cparty=inv["by_code"]["no_counterparty_exec"])
                log(IA.invalidity_line(inv))
            except Exception as _e:
                log(f"[invalidity] skipped: {_e}")
            history.append(rec)
            log(f"[S5] step {step:5d}  D_loss {float(dloss):+.4f}  sep {sep:+.4f}  auc {auc:.3f}  "
                f"ce {ce:.3f} | G score {mean_score:+.4f}  KL {mean_kl:.4f}(λ{lam:.3f}) | "
                f"corr {ev_metrics['ret_corr']:.3f} | "
                f"div tok {div['token_unique_frac']:.3f} ev {div['event_hist_disp']:.3f} | "
                f"||Δ|| {dk:.2e} finite={finite}")
            if not finite:
                log("[S5] *** NON-FINITE — divergence guard tripped (lower lr/σ) ***")
                diverged = True; break
        if step % args.ckpt_every == 0 or step == args.n_g_steps:
            if p0:
                save_ckpt(args.out_dir, step, _gen_payload(), {"params": hparams, **sn}, npar, history,
                          composites, meta=_meta(args, diverged, ev_baseline, best_comp),
                          scope=args.scope)
                log(f"[S5] checkpoint @ step {step} -> {args.out_dir}")
                if args.keep_step_ckpts:
                    sdir = os.path.join(args.out_dir, f"step{step:04d}")
                    save_ckpt(sdir, step, _gen_payload(), {"params": hparams, **sn}, npar, history,
                              composites, meta=_meta(args, diverged, ev_baseline, best_comp),
                              scope=args.scope)
                    log(f"[S5] step-keyed checkpoint @ step {step} -> {sdir}")

    if p0:
        save_ckpt(args.out_dir, last_step, _gen_payload(), {"params": hparams, **sn}, npar, history,
                  composites, meta=_meta(args, diverged, ev_baseline, best_comp), scope=args.scope)
    log(f"[S5] {'DIVERGED' if diverged else 'done'} @ step {last_step} — {len(history)} logged evals -> "
        f"{args.out_dir} (selection: held-out LOB-Bench WS-21 over the step ckpts)")
    return 1 if diverged else 0


def _meta(args, diverged, ev_baseline, best_comp, **extra):
    return dict(scope=args.scope, G=args.G, Q=args.Q, rank=args.rank, sigma=args.sigma, lr=args.lr,
                solver=args.solver, fitness_control=args.fitness_control,
                enc_proj_dim=int(getattr(args, "enc_proj_dim", 0)),
                enc_whiten=str(getattr(args, "enc_whiten", "std")),
                kl_coef=args.kl_coef, kl_ref=args.kl_ref,
                n_cond=args.n_cond, n_gen=args.n_gen, pool_scope=args.pool_scope,
                pool_refresh_every=args.pool_refresh_every, diverged=diverged,
                perturb_glu=int(args.perturb_glu), perturb_book_proj=int(args.perturb_book_proj),
                perturb_fused_encoder=int(args.perturb_fused_encoder),
                ev_baseline=ev_baseline, best_composite=best_comp, **extra)


def _reference_rollout(gen, leaves, ts_in, eval_dev, mesh, *, G_eval, et_slice):
    """σ=0 reference generation (current head, no perturbation) on the eval contexts -> (gen_l2, gen_et)."""
    bcast = np.broadcast_to if DU.is_multihost() else jnp.broadcast_to
    pop_ref = {"kernel": bcast(np.asarray(leaves["kernel"]) if DU.is_multihost() else leaves["kernel"],
                               (G_eval,) + leaves["kernel"].shape),
               "bias": bcast(np.asarray(leaves["bias"]) if DU.is_multihost() else leaves["bias"],
                             (G_eval,) + leaves["bias"].shape)}
    out = gen(DU.shard_pop(pop_ref, mesh), ts_in, eval_dev["m"], eval_dev["b"], eval_dev["sim"],
              eval_dev["rng"], eval_dev["ih"], eval_dev["it"])
    gen_l2 = np.asarray(DU.gather_host(out[1], mesh))                          # [G_eval, n_gen, W]
    gen_et = np.asarray(DU.gather_host(et_slice(out[0]), mesh))                # [G_eval, n_gen]
    gen_msgs = np.asarray(DU.gather_host(out[0], mesh))                        # [G_eval, n_gen, n_fields]
    return jnp.asarray(gen_l2), jnp.asarray(gen_et), jnp.asarray(gen_msgs)


def _reference_rollout_proj(gen, factors_tmpl, ts_in, eval_dev, mesh, *, G_eval, et_slice):
    """Proj-scope σ=0 reference: the CURRENT params ride ts_in; the perturbation population is a
    ZERO LoRA factor tree (x @ 0 @ B.T == 0 — an exact no-op), resized to G_eval members."""
    pop_ref = zero_proj_factors(factors_tmpl, G_eval)
    if DU.is_multihost():
        pop_ref = jax.tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), pop_ref)
    out = gen(DU.shard_pop(pop_ref, mesh), ts_in, eval_dev["m"], eval_dev["b"], eval_dev["sim"],
              eval_dev["rng"], eval_dev["ih"], eval_dev["it"])
    gen_l2 = np.asarray(DU.gather_host(out[1], mesh))                          # [G_eval, n_gen, W]
    gen_et = np.asarray(DU.gather_host(et_slice(out[0]), mesh))                # [G_eval, n_gen]
    gen_msgs = np.asarray(DU.gather_host(out[0], mesh))                        # [G_eval, n_gen, n_fields]
    return jnp.asarray(gen_l2), jnp.asarray(gen_et), jnp.asarray(gen_msgs)


def _replay_real(inf, sim_init, eval_sims, eval_raw_cont):
    """Replay the REAL eval continuation through the SAME engine from the SAME init states (as noop_audit).
    Host-local jit: identical replicated computation on every process (small, eval-only)."""
    from functools import partial
    replay = jax.jit(jax.vmap(partial(inf._replay_real_msgs_single, sim_init, n_levels=inf.l2_state_n),
                              in_axes=(0, 0)))
    return replay(eval_sims, eval_raw_cont)                                    # [G_eval, n_gen, W]


class LearnedFeats:
    """Learned-critic featuriser (critic/learned_critic.py), shared by the EGGROLL and GRPO loops:
    emits the per-step microstructure descriptor SEQUENCE [N, T, F_in] (UNPOOLED).

    REAL class: replay the real continuation messages through the SAME order-book engine the generator
    rolls out from (per-context sim state) -> L2 ladder -> descriptor. FAKE class: descriptor on the
    engine-produced L2 (g_out[1]) + decoded msgs (g_out[0]). Both use the SAME map
    (`learned_critic.batch_features`) and a standardiser FIT ONCE on the first real pool (`fit`) and
    reused, so the critic always sees consistently-scaled real-vs-fake inputs; the standardiser is the
    sequence-aware one (per-channel stats over (N, T)). make_critic builds the SN-regularised TCN (a
    (T, F_in) shape selects it); the shared d-step/score path is unchanged. Returns host numpy (the
    loops keep critic feats on host).

    `robust` (from cfg.disc.enc_whiten=='robust') swaps the per-channel (mean,std) standardiser for a
    heavy-tail-safe (median, MAD) one — the OFI channel is fat-tailed, so a σ scale is event-dominated.

    The `_FEAT` indirection is the plug point for alternative featurisers (a subclass overriding
    `_FEAT` with any module exposing the same batch_features/fit_normalizer/standardize contract).
    The retired fixed-φ featurisers (stylized / dynamical / signature summary vectors) lived here as
    such subclasses — recover them from git history if ever needed."""

    _FEAT = LC

    def __init__(self, inf, sim_init, tick_size, robust: bool = False):
        self.inf, self.sim_init = inf, sim_init
        self.n_levels, self.tick = int(inf.l2_state_n), float(tick_size)
        self.mean = self.std = None
        self._robust = bool(robust)

    def _raw_real(self, sims, raw_cont):
        rl2 = _replay_real(self.inf, self.sim_init, sims, jnp.asarray(raw_cont))           # [n, n_gen, W]
        return self._FEAT.batch_features(rl2, jnp.asarray(raw_cont, jnp.float32),
                                         n_levels=self.n_levels, tick_size=self.tick)

    def fit(self, pool):
        """Fit the standardiser on the real pool's descriptors. Call once, on the initial pool."""
        self.mean, self.std = self._FEAT.fit_normalizer(
            self._raw_real(pool["sim"], pool["raw_cont"]), robust=self._robust)
        return self

    def real(self, pool):
        """Standardised real-class features [n, T, F_in] for a pool (recomputed on pool refresh)."""
        return np.asarray(self._FEAT.standardize(self._raw_real(pool["sim"], pool["raw_cont"]), self.mean, self.std))

    def fake(self, gen_l2, gen_msgs):
        """Standardised fake-class features [M, T, F_in] from a generated rollout's L2 + decoded msgs."""
        raw = self._FEAT.batch_features(jnp.asarray(gen_l2), jnp.asarray(gen_msgs, jnp.float32),
                                        n_levels=self.n_levels, tick_size=self.tick)
        return np.asarray(self._FEAT.standardize(raw, self.mean, self.std))


class RawMsgFeats(LearnedFeats):
    """--critic_input raw: the featuriser is critic/raw_message_features — the decoded-message FIELD
    sequence itself under lossless transforms (no book-derived channels, no hand-designed statistics).
    Same engine-replay real class, same fit-once standardiser, same TCN critic; only _FEAT changes."""
    _FEAT = RMF


def _parse_scales(s):
    """Parse the --enc_scales CLI value '1,2,4' -> (1, 2, 4) (multi-scale critic downsample factors).
    Accepts a tuple/list verbatim so the CFG.disc.enc_scales default round-trips."""
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(",") if str(x).strip())


def _sync_enc_cfg(cfg, args):
    """Push the --enc_* / --ens_* CLI knobs (Lever-3 learned critic) into cfg.disc so make_critic /
    make_learned_critic see them (make_critic only receives cfg, not args). No-op for any knob the
    parser didn't define."""
    for k in ("enc_type", "enc_layers", "enc_hidden", "enc_pool", "enc_kernel", "enc_lr", "enc_adam_b1",
              "enc_scales", "ens_size", "ens_pessimism", "enc_proj_dim", "enc_whiten"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg.disc, k, v)


# ----------------------------------------------------------------------------------------
# Diagnostic mode (GPU; --diagnose): noise vs signal, and the D-frozen-vs-live ranking-stability gate.
# ----------------------------------------------------------------------------------------
def diagnose(hs, args, *, k_rounds=4, n_noise_reps=4):
    """Pre-scaling GPU diagnostic. Builds the SAME rollout/critic machinery as train(), then reports
    two gates:
      (A/B) noise vs signal — (A) ONE fixed direction on ONE fixed Q-batch, re-rolled `n_noise_reps`
            times with FRESH SAMPLING RNG only -> per-context score std across repeats = the sampling
            noise that actually corrupts within-context ranks (the old estimator used a fixed direction
            ACROSS contexts, which bundles the context main effect that per-context ranking cancels).
            (B) G directions on one shared Q-batch -> spread of Q-averaged direction means. Since that
            spread itself contains a noise floor of var/Q, the SIGNAL is estimated as
            sqrt(max(spread^2 - noise^2/Q, 0)) — the old gate (raw spread > noise/sqrt(Q)) passed ~50%
            of the time under ZERO signal. Gate: signal_est > noise/sqrt(Q). FAIL ⇒ raise Q (drop G), NOT G.
      (stationarity) hold D FROZEN over k context-redraws → rank-stability of the per-direction σ̄ fitness
            (churn here = pure ES variance ⇒ raise Q); then let D MOVE between redraws → if stability drops
            relative to frozen, that extra churn is D non-stationarity ⇒ throttle D (lower disc.lr / fewer
            n_d / KL up). Distinguishing the two IS the point."""
    from ..tests.s3_es_rollout import _prep_real_batch
    cfg = CFG
    G, Q = args.G, args.Q
    proj = args.scope == "proj"
    assert G % 2 == 0 and Q >= 2 and args.n_ctx_pool >= Q, "need even G, Q>=2, n_ctx_pool>=Q"
    assert not DU.is_multihost(), "--diagnose is a single-process (1-node) tool"
    solver = _SOLVERS[args.solver]
    solver_kwargs = _solver_kwargs(args.solver)
    print(f"[S5] --diagnose: scope={args.scope} G={G} Q={Q} rank={args.rank} sigma={args.sigma} "
          f"n_d={args.n_d_steps} k_rounds={k_rounds} noise_reps={n_noise_reps} pool={args.n_ctx_pool} "
          f"devices={jax.device_count()}", flush=True)

    # --- machinery (mirror train()) ---
    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, args.n_ctx_pool,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels, wide_book_dir=args.wide_book_dir)
    model, batchnorm, encoder = P["model"], P["batchnorm"], P["encoder"]
    train_state = P["train_state"]
    backbone = D.make_backbone(P["model_cls"]); bb_params = P["bb_params"]
    pooling = cfg.disc.pooling
    pool_start = args.n_cond * F.MSG_LEN if args.pool_scope == "cont" else 0
    pool = dict(m=P["m_seq_inp"], b=P["b_seq_inp"], sim=P["sim_states_init"],
                ih=P["init_hidden_batched"], it=P["init_time_batched"],
                ctx_tok=P["ctx_tokens"], ctx_book=P["b_seq_inp"],
                real_tok=P["real_cont_tokens"], real_book=P["real_cont_book"])
    feats_fn = make_feats_kl(backbone, bb_params, pooling=pooling, pool_start=pool_start,
                             n_cond=args.n_cond, shard=args.shard, chunk=args.feat_chunk, with_kl=False)
    real_feats_pool = np.asarray(feats_fn(pool["ctx_tok"], pool["real_tok"],
                                          pool["ctx_book"], pool["real_book"]))
    if proj:
        params0 = train_state.params
        es_map = build_es_map_proj(params0, hs,
                                   perturb_glu=bool(args.perturb_glu),
                                   perturb_book_proj=bool(args.perturb_book_proj),
                                   perturb_fused_encoder=bool(args.perturb_fused_encoder))
        fnp, npar, esk = make_proj_noiser(
            hs, params0, es_map, sigma=args.sigma, lr=args.lr, rank=args.rank,
            group_size=0, noise_reuse=cfg.eggroll.noise_reuse, solver=solver,
            solver_kwargs=solver_kwargs, seed=args.seed)
        leaves = params0
        from lob.lob_seq_model import BatchPaddedLobPredModelES
        model_es = BatchPaddedLobPredModelES(**dict(P["model_cls"].keywords),
                                             training=False, step_rescale=1.0)
        gen = make_generate_es_proj_sharded(model_es, batchnorm, encoder, P["sample_top_n"],
                                            P["tick_size"], args.n_gen, P["sim_init"],
                                            P["valid_mask_array"], conditional=True, shard=args.shard)
        pop_diag = build_proj_factors(hs, fnp, npar, leaves, es_map, esk,
                                      population_iterinfo(G, 0))                # FIXED G dirs
    else:
        fnp, npar, es_map, esk, leaves = make_decoder_noiser(
            hs, P["kernel"], P["bias"], sigma=args.sigma, lr=args.lr, rank=args.rank,
            group_size=0, noise_reuse=cfg.eggroll.noise_reuse, solver=solver,
            solver_kwargs=solver_kwargs, seed=args.seed)
        gen = make_generate_es_sharded(model, batchnorm, encoder, P["sample_top_n"], P["tick_size"],
                                       args.n_gen, P["sim_init"], P["valid_mask_array"],
                                       conditional=True, shard=args.shard)
        pop_diag = build_decoder_population(hs, fnp, npar, leaves, esk, population_iterinfo(G, 0))  # FIXED G dirs
    head, hparams, sn, tx, opt_state = make_critic(cfg, int(real_feats_pool.shape[-1]), args.seed)
    d_step = make_d_step(head, tx, r1_gamma=args.r1_gamma)

    def roll_feats(pop, ctx_idx, key):
        """[Gl dirs] × [Ql contexts] -> fake_feats [Ql*Gl, d] (grid r=q*Gl+g, mirrors train())."""
        Gl = _pop_size(pop)
        pg = tile_dirs_over_Q(pop, ctx_idx.shape[0])
        mg = grid_repeat_contexts(pool["m"][ctx_idx], Gl); bg = grid_repeat_contexts(pool["b"][ctx_idx], Gl)
        sg = grid_repeat_contexts(_take(pool["sim"], ctx_idx), Gl); ihg = grid_repeat_contexts(_take(pool["ih"], ctx_idx), Gl)
        itg = grid_repeat_contexts(pool["it"][ctx_idx], Gl); rg = grid_rngs(key, Gl, ctx_idx.shape[0])
        out = gen(pg, train_state, mg, bg, sg, rg, ihg, itg)
        fct, fcb = rollout_to_cont(out[3], out[4], args.n_gen)
        ctg = grid_repeat_contexts(pool["ctx_tok"][ctx_idx], Gl); cbg = grid_repeat_contexts(pool["ctx_book"][ctx_idx], Gl)
        return feats_fn(ctg, fct, cbg, fcb)

    kd = jax.random.PRNGKey(args.seed + 7)
    # warm the critic so D(fake) is a non-degenerate scorer (else fitness is pure noise).
    drawn0 = jax.random.choice(jax.random.fold_in(kd, 0), args.n_ctx_pool, (Q,), replace=False)
    ff0 = roll_feats(pop_diag, drawn0, jax.random.fold_in(kd, 1))
    real0 = grid_repeat_contexts(jnp.asarray(real_feats_pool)[drawn0], G)
    for _ in range(max(20, args.n_d_steps)):
        hparams, sn, opt_state, _, _, _ = d_step(hparams, sn, opt_state, real0, ff0)
    hv = {"params": hparams, **sn}

    # (A) NOISE: ONE fixed direction (g=0) on ONE fixed Q-batch, re-rolled with fresh sampling rng only.
    pop1 = _take(pop_diag, jnp.array([0]))                                   # single direction, shape [1,...]
    drawnA = jax.random.choice(jax.random.fold_in(kd, 2), args.n_ctx_pool, (Q,), replace=False)
    s_reps = []
    for r in range(n_noise_reps):
        ffA = roll_feats(pop1, drawnA, jax.random.fold_in(kd, 30 + r))       # [Q, d] (Gl=1)
        s_reps.append(np.asarray(head.apply(hv, ffA, train=False)).reshape(-1))   # [Q]
    per_ctx_noise = float(np.mean(np.std(np.stack(s_reps), axis=0, ddof=1)))
    q_avg_noise = per_ctx_noise / math.sqrt(Q)

    # (B) SIGNAL: G directions on ONE shared Q-batch -> noise-floor-corrected spread of direction means.
    drawnB = jax.random.choice(jax.random.fold_in(kd, 3), args.n_ctx_pool, (Q,), replace=False)
    ff_B = roll_feats(pop_diag, drawnB, jax.random.fold_in(kd, 4))           # [Q*G, d]
    dir_means = jnp.mean(head.apply(hv, ff_B, train=False).reshape(Q, G), axis=0)  # [G]
    spread = signal_spread(dir_means)
    signal_est = math.sqrt(max(spread ** 2 - per_ctx_noise ** 2 / Q, 0.0))
    snr = signal_est / (q_avg_noise + 1e-12)
    gateAB = snr > 1.0

    # (stationarity) per-direction σ̄ fitness across k context-redraws: D frozen vs D moving.
    def fit_round(r, hv_r):
        dr = jax.random.choice(jax.random.fold_in(kd, 100 + r), args.n_ctx_pool, (Q,), replace=False)
        ff = roll_feats(pop_diag, dr, jax.random.fold_in(kd, 200 + r))
        return F.rank_sigma_bar(head.apply(hv_r, ff, train=False).reshape(Q, G)), ff, dr

    fz = [fit_round(r, hv)[0] for r in range(k_rounds)]                      # D frozen across rounds
    stab_frozen = ranking_stability(jnp.stack(fz))
    hp2, sn2, os2 = hparams, sn, opt_state; fl = []
    for r in range(k_rounds):
        fit_r, ff_r, dr = fit_round(r, {"params": hp2, **sn2}); fl.append(fit_r)
        real_r = grid_repeat_contexts(jnp.asarray(real_feats_pool)[dr], G)
        for _ in range(args.n_d_steps):                                     # let D move between rounds
            hp2, sn2, os2, _, _, _ = d_step(hp2, sn2, os2, real_r, ff_r)
    stab_live = ranking_stability(jnp.stack(fl))

    es_variance_ok = stab_frozen > 0.5                                      # rankings track across redraws
    stationarity_ok = (stab_frozen - stab_live) < 0.25                     # D-move doesn't add big churn
    print("\n[S5] ===== DIAGNOSTIC REPORT =====", flush=True)
    print(f"[S5] (A) sampling-rng noise std (fixed dir, fixed ctxs, {n_noise_reps} reps) = {per_ctx_noise:.4f}"
          f"  -> Q-avg noise (/√{Q}) = {q_avg_noise:.4f}", flush=True)
    print(f"[S5] (B) raw direction-mean spread = {spread:.4f}  -> noise-floor-corrected signal = "
          f"{signal_est:.4f}", flush=True)
    print(f"[S5] SNR = signal_est / (noise/√Q) = {snr:.3f}  -> {'PASS' if gateAB else 'FAIL: raise Q (drop G to pay), NOT G'}", flush=True)
    print(f"[S5] ranking_stability  D-frozen = {stab_frozen:+.3f}   D-live = {stab_live:+.3f}   Δ = {stab_frozen - stab_live:+.3f}", flush=True)
    print(f"[S5]   ES-variance gate (frozen>0.5): {'PASS' if es_variance_ok else 'FAIL: pure ES variance -> raise Q'}", flush=True)
    print(f"[S5]   D-stationarity gate (Δ<0.25): {'PASS' if stationarity_ok else 'FAIL: non-stationarity -> throttle D (lower disc.lr / fewer n_d / KL up)'}", flush=True)
    all_ok = gateAB and es_variance_ok and stationarity_ok
    print(f"[S5] ===== {'DIAGNOSTIC GATES PASS' if all_ok else 'DIAGNOSTIC GATES: see FAILs above'} =====\n", flush=True)
    return 0 if all_ok else 3


# ----------------------------------------------------------------------------------------
# CPU-safe glue checks (login node; no model, no rollout).
# ----------------------------------------------------------------------------------------
def cpu_checks(hs, seed=0):
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("\n[S5] CPU checks — grid layout / rank-σ̄ G-step / shard / ckpt / diagnostics", flush=True)
    k = jax.random.key(seed)
    G, Q = 6, 4

    # (C1) grid layout: rollout r=q*G+g -> head g (tiled over Q), context q (repeated over G).
    di, ci = gq_grid_indices(G, Q)
    pop = {"kernel": jnp.arange(G * 3).reshape(G, 3).astype(jnp.float32), "bias": jnp.arange(G).astype(jnp.float32)}
    pg = tile_dirs_over_Q(pop, Q)
    ctx = {"m": jnp.arange(Q * 5).reshape(Q, 5)}
    cg = grid_repeat_contexts(ctx, G)
    layout_ok = all(int(di[q * G + g]) == g and int(ci[q * G + g]) == q
                    and float(pg["bias"][q * G + g]) == g
                    and bool(jnp.array_equal(cg["m"][q * G + g], ctx["m"][q]))
                    for q in range(Q) for g in range(G))
    rk = grid_rngs(jax.random.PRNGKey(seed), G, Q)
    rng_distinct = len({tuple(np.array(rk[i]).ravel().tolist()) for i in range(G * Q)}) == G * Q
    chk("(C1) G×Q grid layout (head/ctx/rng)", layout_ok and rk.shape[0] == G * Q and rng_distinct,
        f"layout={layout_ok} rng_distinct={rng_distinct}")

    # (C2) rank-σ̄ G-step round-trip: scores [Q,G] -> rank_sigma_bar (G,) -> do_updates moves head & finite.
    kernel = jax.random.normal(jax.random.fold_in(k, 1), (12, 10)); bias = jax.random.normal(jax.random.fold_in(k, 2), (10,))
    fnp, npar, es_map, esk, leaves = make_decoder_noiser(
        hs, kernel, bias, sigma=1e-2, lr=1e-3, rank=CFG.eggroll.rank, group_size=0, solver=optax.sgd, seed=seed)
    it_G = population_iterinfo(G, 0)
    _ = build_decoder_population(hs, fnp, npar, leaves, esk, it_G)
    raw_qg = jax.random.normal(jax.random.fold_in(k, 3), (Q, G))
    fit = F.rank_sigma_bar(raw_qg)
    npar2, leaves2 = hs.EggRoll.do_updates(fnp, npar, leaves, esk, fit, it_G, es_map)
    dk = float(jnp.max(jnp.abs(leaves2["kernel"] - kernel)))
    chk("(C2) rank-σ̄ -> do_updates moves head & finite",
        fit.shape == (G,) and dk > 0.0 and bool(jnp.isfinite(leaves2["kernel"]).all()), f"||Δ||={dk:.3e}")

    # (C3) shard decision + divisibility.
    chk("(C3) shard_decision auto@1=vmap, auto@8=shard_map, off@8=vmap",
        shard_decision("auto", 1) == "vmap" and shard_decision("auto", 8) == "shard_map"
        and shard_decision("off", 8) == "vmap" and (512 * 32) % 8 == 0)

    # (C4) checkpoint round-trip (msgpack + atomic breadcrumb).
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"s5_ckpt_test_{os.getpid()}")
    head, hparams, sn, tx, opt_state = make_critic(CFG, 16, seed)
    hv = {"params": hparams, **sn}
    bc = save_ckpt(out, 7, leaves2, hv, npar2, history=[{"step": 7}], composites=[0.5],
                   meta={"sigma": 1e-2, "ev_baseline": {"mid_l1": 1.0}, "best_composite": 0.5})
    leaves_r, hv_r, npar_r, bc_r = load_ckpt(out, leaves2, hv, npar2)
    rt_ok = (bc_r["step"] == 7
             and bc_r["ev_baseline"] == {"mid_l1": 1.0}
             and float(jnp.max(jnp.abs(leaves_r["kernel"] - leaves2["kernel"]))) == 0.0
             and float(jnp.max(jnp.abs(hv_r["params"]["SNDense_0"]["kernel"] - hparams["SNDense_0"]["kernel"]))) == 0.0
             and os.path.exists(os.path.join(out, "latest_checkpoint.json")))
    chk("(C4) ckpt save/load round-trip + atomic breadcrumb (+ev_baseline)", rt_ok, f"step={bc_r['step']}")
    import shutil; shutil.rmtree(out, ignore_errors=True)

    # (C5) WGAN critic d_step separates two offset clouds & stays finite (reused S4 path).
    d = 16
    real = jax.random.normal(jax.random.fold_in(k, 4), (16, d)) + 1.5
    fake = jax.random.normal(jax.random.fold_in(k, 5), (16, d)) - 1.5
    head, hp, sn2, tx2, os2 = make_critic(CFG, d, seed); ds = make_d_step(head, tx2)
    sep0 = None
    for i in range(120):
        hp, sn2, os2, dl, sr, sf = ds(hp, sn2, os2, real, fake)
        if i == 0:
            sep0 = float(D.critic_separation(sr, sf))
    chk("(C5) critic d_step separates & finite",
        float(D.critic_separation(sr, sf)) > sep0 and np.isfinite(float(dl)),
        f"sep {sep0:+.3f} -> {float(D.critic_separation(sr, sf)):+.3f}")

    # (C6) diagnostic estimators: noise std, signal spread, ranking stability.
    chk("(C6a) fitness_noise_std: const->0, varying->>0",
        fitness_noise_std(jnp.ones(5)) == 0.0 and signal_spread(jnp.arange(6.0)) > 0.0)
    stable = ranking_stability(jnp.stack([jnp.arange(8.0), jnp.arange(8.0) + 0.01 * jnp.arange(8.0)]))
    churned = ranking_stability(jnp.stack([jnp.arange(8.0), jnp.arange(8.0)[::-1]]))
    chk("(C6b) ranking_stability: aligned~1, reversed~-1", stable > 0.9 and churned < -0.9,
        f"stable={stable:.3f} reversed={churned:.3f}")
    # noise-floor-corrected SNR: zero signal (spread == noise/sqrt(Q)) must NOT pass the gate.
    noise_, Q_ = 0.4, 16
    spread_null = noise_ / math.sqrt(Q_)
    sig_null = math.sqrt(max(spread_null ** 2 - noise_ ** 2 / Q_, 0.0))
    spread_real = math.sqrt((3.0 * noise_ / math.sqrt(Q_)) ** 2 + noise_ ** 2 / Q_)
    sig_real = math.sqrt(max(spread_real ** 2 - noise_ ** 2 / Q_, 0.0))
    chk("(C6c) SNR noise-floor correction: null->0 (no spurious pass), real signal recovered",
        sig_null == 0.0 and abs(sig_real - 3.0 * noise_ / math.sqrt(Q_)) < 1e-9,
        f"null={sig_null:.3e} real={sig_real:.4f}")

    # (C7) KL λ schedule: cosine anneals to ~0 at the end; const holds; kl_coef=0 -> 0.
    chk("(C7) kl_lambda schedule",
        abs(kl_lambda(0, 100, 0.1) - 0.1) < 1e-6 and kl_lambda(100, 100, 0.1) < 1e-6
        and abs(kl_lambda(50, 100, 0.1, anneal="const") - 0.1) < 1e-6 and kl_lambda(10, 100, 0.0) == 0.0)

    # (C8) normalised composite + margin Goodhart (full coverage in eval_monitor's own self-test).
    base = dict(mid_l1=2.9e6, book_l1=2.0, ret_corr=0.5, moment_l1=0.1, event_l1=0.2)
    better = dict(base, book_l1=1.0)
    fired_n, _ = EM.goodhart_check([1.0, 1.004, 1.008, 1.004], patience=2, tol=0.05)   # plateau: no fire
    fired_y, best_y = EM.goodhart_check([0.5, 0.8, 0.9, 1.0], patience=2, tol=0.05)
    chk("(C8) normalize_composite + margin goodhart",
        abs(EM.normalize_composite(base, base) - 1.0) < 1e-6
        and EM.normalize_composite(better, base) < 1.0
        and (not fired_n) and fired_y and best_y == 0,
        f"id={EM.normalize_composite(base, base):.4f} better={EM.normalize_composite(better, base):.4f}")

    # (C9) population diversity: collapsed grid -> token_unique_frac == 1/G, event disp ~ 0.
    Gd, Qd, Td = 8, 3, 12
    toks = np.tile(np.arange(Td)[None, None, :], (Qd, Gd, 1)).reshape(Qd * Gd, Td)
    ets = np.tile(np.array([1, 2, 3, 4] * 3)[None, None, :], (Qd, Gd, 1)).reshape(Qd * Gd, -1)
    div = EM.population_diversity(toks, ets, Gd, Qd, token_stride=1)
    chk("(C9) diversity collapse signal: unique_frac==1/G, disp~0",
        abs(div["token_unique_frac"] - 1.0 / Gd) < 1e-6 and div["event_hist_disp"] < 1e-6,
        f"frac={div['token_unique_frac']:.4f} disp={div['event_hist_disp']:.2e}")

    # (C10) dist_utils: no-op fast paths + forced global-array round-trip on the 1-device mesh.
    mesh1 = DU.make_pop_mesh()
    tr_ = {"a": np.arange(8.0, dtype=np.float32).reshape(4, 2)}
    back = DU.gather_host(DU.shard_pop(tr_, mesh1, force=True), mesh1)
    noop = DU.shard_pop(tr_, None)
    chk("(C10) dist_utils round-trip + no-op",
        np.array_equal(back["a"], tr_["a"]) and np.array_equal(np.asarray(noop["a"]), tr_["a"])
        and not DU.is_multihost())

    # ===================== S5b proj-scope glue (C11-C14) =====================
    from s5 import es_fold as EF

    # (C11) hoisted fold == EggRoll.do_Tmm BIT-exact; sigma=0 factors vanish; antithetic mirror.
    fnp_min = {"rank": 4, "noise_reuse": 1}
    npar_min = {"sigma": 0.05}
    Wm = jax.random.normal(jax.random.fold_in(k, 10), (12, 8))
    xm = jax.random.normal(jax.random.fold_in(k, 11), (5, 12))
    kleaf = jax.random.key(seed + 99)
    it35 = (jnp.int32(3), jnp.int32(5))
    ref_tmm = hs.EggRoll.do_Tmm(fnp_min, npar_min, Wm, kleaf, it35, xm)
    A35, B35 = hs.eggroll.get_lora_update_params(fnp_min, npar_min["sigma"] / jnp.sqrt(4.0),
                                                 it35, Wm, kleaf)
    got_tmm = EF.fold_kernel({"p": {"kernel": {"A": A35, "B": B35}}}, "p", xm, xm @ Wm)
    A0, _ = hs.eggroll.get_lora_update_params(fnp_min, 0.0, it35, Wm, kleaf)
    Ae, Be = hs.eggroll.get_lora_update_params(fnp_min, 0.1, (jnp.int32(0), jnp.int32(2)), Wm, kleaf)
    Ao, Bo = hs.eggroll.get_lora_update_params(fnp_min, 0.1, (jnp.int32(0), jnp.int32(3)), Wm, kleaf)
    chk("(C11) fold==do_Tmm bit-exact; sigma=0 -> A==0; antithetic A mirror, B shared",
        bool(jnp.array_equal(ref_tmm, got_tmm))
        and float(jnp.max(jnp.abs(A0))) == 0.0
        and bool(jnp.array_equal(Ae, -Ao)) and bool(jnp.array_equal(Be, Bo)),
        f"max|fold-do_Tmm|={float(jnp.max(jnp.abs(ref_tmm - got_tmm))):.3e}")

    # mock proj params tree (values random so movement/freezing is observable).
    def rnd(i, *s):
        return jax.random.normal(jax.random.fold_in(k, 1000 + i), s)
    mock = {"message_encoder": {"encoder": {"embedding": rnd(0, 7, 4)},
                                "layers_0": {"seq": {"in_proj": {"kernel": rnd(1, 4, 10)},
                                                     "out_proj": {"kernel": rnd(2, 6, 4)},
                                                     "dt_bias": rnd(3, 2)},
                                             "out2": {"kernel": rnd(4, 4, 4), "bias": rnd(5, 4)},
                                             "norm": {"scale": rnd(6, 4), "bias": rnd(7, 4)}}},
            "book_encoder": {"projection": {"kernel": rnd(8, 5, 4), "bias": rnd(9, 4)}},
            "fused_s5": {"encoder": {"kernel": rnd(10, 4, 4), "bias": rnd(11, 4)},
                         "layers_0": {"seq": {"in_proj": {"kernel": rnd(12, 4, 10)},
                                              "out_proj": {"kernel": rnd(13, 6, 4)}}}},
            "decoder": {"kernel": rnd(14, 4, 7), "bias": rnd(15, 7)}}
    from ..es.es_plumbing import build_es_map_proj as _bemp
    es_map_m = _bemp(mock, hs)                       # defaults: glu ON, book proj OFF, fused enc ON
    fnp_m, npar_m, esk_m = make_proj_noiser(hs, mock, es_map_m, sigma=0.02, lr=1e-3, rank=2,
                                            solver=optax.adamw,
                                            solver_kwargs={"weight_decay": 1e-2}, seed=seed + 3)
    Gm = 6
    it6 = population_iterinfo(Gm, 0)
    fac = build_proj_factors(hs, fnp_m, npar_m, mock, es_map_m, esk_m, it6)

    # (C12) factor tree structure: exactly the MM_PARAM paths, member axis leading, zeros resize.
    in_proj_ok = fac["message_encoder"]["layers_0"]["seq"]["in_proj"]["kernel"]["A"].shape == (Gm, 4, 2)
    fenc_ok = fac["fused_s5"]["encoder"]["kernel"]["B"].shape == (Gm, 4, 2)
    book_off = "book_encoder" not in fac             # book proj toggle defaults OFF -> whole subtree absent
    dec_off = "decoder" not in fac
    zf = zero_proj_factors(fac, n=3)
    z_ok = (zf["message_encoder"]["layers_0"]["seq"]["out_proj"]["kernel"]["A"].shape[0] == 3
            and float(jnp.max(jnp.abs(jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(zf)])))) == 0.0)
    chk("(C12) build_proj_factors structure (MM-only, [G,...] axis) + zero_proj_factors",
        in_proj_ok and fenc_ok and book_off and dec_off and z_ok)

    # (C13) masked-adamw do_updates (use_batched_update=True): MM leaves MOVE, EXCLUDED leaves stay
    # BIT-identical even with weight_decay=1e-2 — the unmasked-adamw hazard this exists to prevent.
    cur = mock
    npar_c = npar_m
    for s_ in range(3):
        it_s = population_iterinfo(Gm, s_)
        fit6 = F.rank_sigma_bar(jax.random.normal(jax.random.fold_in(k, 2000 + s_), (3, Gm)))
        npar_c, cur = hs.EggRoll.do_updates(fnp_m, npar_c, cur, esk_m, fit6, it_s, es_map_m)
    flat0 = jax.tree_util.tree_flatten_with_path(mock)[0]
    flatc = jax.tree_util.tree_leaves(cur)
    flatm = jax.tree_util.tree_leaves(es_map_m)
    moved_mm, frozen_ok, mm_seen = 0.0, True, 0
    for (path, l0), lc, m in zip(flat0, flatc, flatm):
        if int(m) == int(hs.MM_PARAM):
            mm_seen += 1
            moved_mm = max(moved_mm, float(jnp.max(jnp.abs(lc - l0))))
        else:
            frozen_ok = frozen_ok and bool(jnp.array_equal(lc, l0))
    # 6 MM kernels in the mock: msg in/out_proj + msg out2 + fused encoder + fused in/out_proj.
    chk("(C13) masked-adamw(wd=1e-2)+batched do_updates: MM move, EXCLUDED bit-frozen",
        mm_seen == 6 and moved_mm > 0.0 and frozen_ok
        and bool(jnp.isfinite(jnp.concatenate([x.ravel() for x in flatc])).all()),
        f"mm={mm_seen} max|dMM|={moved_mm:.3e} frozen_bitexact={frozen_ok}")

    # (C14) trainable subtree extract/merge + proj ckpt round-trip (incl. masked opt state).
    tr_c = extract_trainable(hs, cur, es_map_m)
    merged = merge_trainable(hs, mock, es_map_m, tr_c)
    merge_ok = all(bool(jnp.array_equal(a, b)) for a, b in
                   zip(jax.tree_util.tree_leaves(merged), jax.tree_util.tree_leaves(cur)))
    outp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"s5b_ckpt_test_{os.getpid()}")
    headp, hp_p, sn_p, _, _ = make_critic(CFG, 16, seed)
    hv_p = {"params": hp_p, **sn_p}
    save_ckpt(outp, 11, tr_c, hv_p, npar_c, history=[{"step": 11}], composites=[0.7],
              meta={"scope": "proj", "sigma": 0.02}, scope="proj")
    tr_r, hv_r, npar_r, bc_r = load_ckpt(outp, tr_c, hv_p, npar_c)
    rt_ok = (bc_r["step"] == 11 and "generator_proj" in bc_r
             and all(bool(jnp.array_equal(tr_r[kk], tr_c[kk])) for kk in tr_c)
             and len(jax.tree_util.tree_leaves(npar_r)) == len(jax.tree_util.tree_leaves(npar_c))
             and all(bool(jnp.array_equal(a, b)) for a, b in
                     zip(jax.tree_util.tree_leaves(npar_r), jax.tree_util.tree_leaves(npar_c))))
    chk("(C14) proj subtree merge round-trip + proj ckpt save/load (masked opt state)",
        merge_ok and rt_ok, f"keys={len(tr_c)}")
    import shutil as _sh
    _sh.rmtree(outp, ignore_errors=True)

    # (C15) drift-control shuffle: same multiset / order destroyed / fresh perm per step /
    # deterministic for a given (seed, step) — the multihost + resume requirement.
    bk = jax.random.PRNGKey(seed)
    fitv = jnp.arange(16.0)
    p1 = jax.random.permutation(jax.random.fold_in(bk, 888_000_001), 16)
    p1b = jax.random.permutation(jax.random.fold_in(bk, 888_000_001), 16)
    p2 = jax.random.permutation(jax.random.fold_in(bk, 888_000_002), 16)
    shuf = fitv[p1]
    chk("(C15) fitness_control=shuffle: permutation semantics",
        bool(jnp.array_equal(jnp.sort(shuf), fitv))            # multiset preserved
        and not bool(jnp.array_equal(shuf, fitv))              # information destroyed
        and bool(jnp.array_equal(p1, p1b))                     # deterministic per (seed, step)
        and not bool(jnp.array_equal(p1, p2)),                 # fresh perm each step
        f"perm1[:4]={np.asarray(p1[:4]).tolist()}")

    print("\n[S5] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


def main():
    ap = argparse.ArgumentParser(description="S5: scaled EGGROLL-GAN (rank-σ̄, G×Q grid, sharded) + checks")
    ap.add_argument("--run", action="store_true", help="run the scaled loop (GH200); else CPU glue checks only")
    ap.add_argument("--diagnose", action="store_true", help="run the noise/signal + D-stationarity diagnostic (GH200)")
    ap.add_argument("--scope", choices=["head", "proj"], default="head")
    ap.add_argument("--critic_input",
                    choices=["learned", "raw", "backbone"],
                    default=CFG.disc.critic_input,
                    help="what the WGAN critic scores: 'learned' (DEFAULT) = SN-regularised causal TCN "
                         "over the per-step microstructure descriptor sequence; 'raw' = the SAME TCN "
                         "over the decoded-message FIELD sequence itself (lossless transforms, "
                         "critic/raw_message_features.py — pair with --enc_whiten robust); 'backbone' "
                         "= LEGACY frozen-Mamba3 pooled hidden (weak/hackable). Retired fixed-φ modes "
                         "(stylized/dynamical/signature) were removed — recover from git history.")
    # Lever-3 learned-critic knobs (inert unless --critic_input learned). Defaults from CFG.disc.
    ap.add_argument("--enc_type", choices=["tcn", "cnn"], default=CFG.disc.enc_type,
                    help="learned encoder: 'tcn' (dilated causal, default) | 'cnn' (dilation 1 ablation)")
    ap.add_argument("--enc_layers", type=int, default=CFG.disc.enc_layers,
                    help="learned encoder TCN blocks (L=7,k=3 -> receptive field 509 >= 500-step window)")
    ap.add_argument("--enc_hidden", type=int, default=CFG.disc.enc_hidden, help="learned encoder channel width")
    ap.add_argument("--enc_pool", choices=["attn", "mean", "last"], default=CFG.disc.enc_pool,
                    help="learned encoder time pooling (attn default; never last for a critic)")
    ap.add_argument("--enc_kernel", type=int, default=CFG.disc.enc_kernel, help="learned encoder conv kernel size")
    ap.add_argument("--enc_lr", type=float, default=CFG.disc.enc_lr, help="learned critic TTUR lr (Heusel 2017)")
    ap.add_argument("--enc_adam_b1", type=float, default=CFG.disc.enc_adam_b1, help="learned critic Adam β1")
    # Lever-3 STRONGEST config: multi-scale conv trunk + seed-diverse critic ensemble (pessimistic reward).
    ap.add_argument("--enc_scales", type=_parse_scales, default=CFG.disc.enc_scales,
                    help="multi-scale downsample factors '1,2,4' (MelGAN/HiFi-GAN; one TCN branch each)")
    ap.add_argument("--ens_size", type=int, default=CFG.disc.ens_size,
                    help="seed-diverse critic ensemble members (1 disables; Coste 2024 conservative)")
    ap.add_argument("--ens_pessimism", type=float, default=CFG.disc.ens_pessimism,
                    help="β in the reward mean_k − β·std_k (member-disagreement / anti-hack penalty)")
    ap.add_argument("--enc_proj_dim", type=int, default=CFG.disc.enc_proj_dim,
                    help="Phase-1 broad critic: fixed random input projection F_in->enc_proj_dim per "
                         "ensemble member (Projected-GAN, Sauer 2021); 0 = off (bit-identical)")
    ap.add_argument("--enc_whiten", choices=["std", "robust"], default=CFG.disc.enc_whiten,
                    help="learned-critic input standardiser: 'std' (mean/σ) | 'robust' (median/MAD, heavy-tail-safe)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "s5_out"))
    ap.add_argument("--resume_dir", default=None)
    ap.add_argument("--G", type=int, default=512, help="perturbation directions (ES population)")
    ap.add_argument("--Q", type=int, default=32, help="shared contexts/step (σ̄ averaging breadth)")
    ap.add_argument("--crit_batch", type=int, default=0,
                    help="cap the critic d-step minibatch + chunk the score forward over Q*G (0=full "
                         "batch). Bounds the learned (sequence) critic's per-device memory so Q can scale "
                         "with --shard while the ES still scores ALL Q*G rollouts. No-op (bit-identical) "
                         "for vector critics / when >= Q*G.")
    ap.add_argument("--rank", type=int, default=CFG.eggroll.rank)
    ap.add_argument("--perturb_glu", type=int, default=1,
                    help="proj scope: LoRA on the GLU out2 kernels (1=on, plan default)")
    ap.add_argument("--perturb_book_proj", type=int, default=0,
                    help="proj scope: LoRA on book_encoder/projection (DEFAULT OFF per review Part 4 — "
                         "it is the book-INPUT interface; switch on as the first ablation if proj-scope "
                         "separation stalls)")
    ap.add_argument("--perturb_fused_encoder", type=int, default=1,
                    help="proj scope: LoRA on fused_s5/encoder (1=on, plan default)")
    ap.add_argument("--n_cond", type=int, default=500,
                    help="context messages (500 = the thesis target & the pretraining msg_seq_len)")
    ap.add_argument("--n_gen", type=int, default=500,
                    help="messages generated per rollout (500 = the thesis target horizon; "
                         "run the N_G=2 timing gate before committing 1000 steps)")
    ap.add_argument("--wide_levels", type=int, default=10)
    ap.add_argument("--wide_book_dir", default=None,
                    help="dir of deep-init wide-book npz snapshots (e.g. data/wide_L500_<month>/GOOG); "
                         "training rollouts start from these books instead of padded-L10. MUST match "
                         "--wide_levels and the eval init regime (else train/eval mismatch).")
    ap.add_argument("--n_g_steps", type=int, default=1000)
    ap.add_argument("--n_d_steps", type=int, default=5)
    ap.add_argument("--r1_gamma", type=float, default=CFG.disc.r1_gamma,
                    help="R1 gradient penalty on the critic's continuous input (γ·½·E_real||∇_h D||²); "
                         "curbs critic reward-hacking (backbone-hack diagnostic). 0 = off (bit-identical).")
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-3, help="η direct (360M HFT anchor); NOT σ²√N")
    ap.add_argument("--solver", choices=list(_SOLVERS), default="adamw")
    ap.add_argument("--fitness_control", choices=["none", "shuffle"], default="none",
                    help="'shuffle' = drift-control lane: per-step permutation of the per-direction "
                         "fitness severs the D->G information channel; isolates adamw drift + sigma "
                         "noise from critic signal for gain attribution")
    ap.add_argument("--kl_coef", type=float, default=0.0)
    ap.add_argument("--kl_ref", choices=["anchor", "current"], default="anchor",
                    help="KL trust-region reference: 'anchor' = FROZEN pretrained head (the plan's locked "
                         "decision; penalises cumulative drift); 'current' = evolving head (ablation only — "
                         "measures just the per-step σ-perturbation)")
    ap.add_argument("--kl_anneal", choices=["cosine", "const"], default="cosine")
    ap.add_argument("--kl_chunk", type=int, default=2,
                    help="windows per fused feats+KL chunk (logits are [chunk, n_gen*26, 2112])")
    ap.add_argument("--n_ctx_pool", type=int, default=1024)
    ap.add_argument("--n_eval_ctx", type=int, default=128)
    ap.add_argument("--pool_scope", choices=["cont", "window"], default="cont",
                    help="critic pooling range: 'cont' = continuation positions only (context prefix is "
                         "identical for real & fake -> pooling it dilutes D); 'window' = legacy whole-window")
    ap.add_argument("--pool_refresh_every", type=int, default=0,
                    help=">0: redraw the real-context train pool every k steps (anti-memorisation; "
                         "eval set stays fixed & disjoint). 0 = off (legacy fixed pool)")
    ap.add_argument("--token_div_stride", type=int, default=26,
                    help="token-position stride for the per-eval diversity probe (26 = 1 token/message)")
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--keep_step_ckpts", action="store_true",
                    help="also write each ckpt_every checkpoint to a step-keyed subdir step<NNNN>/ (in "
                         "addition to overwriting out_dir with the latest), so post-hoc selection can load "
                         "ANY evaluated step — not just best/ (raw-composite-min) and latest. The raw "
                         "composite is moment_l1-noise-dominated, so best/ tends to be a noise-lucky step; "
                         "step-keyed ckpts let posthoc_select pick by the de-noised+smoothed composite. "
                         "proj ckpts are small (~0.05 GB factors) so the extra disk is negligible.")
    ap.add_argument("--goodhart_patience", type=int, default=3)
    ap.add_argument("--goodhart_tol", type=float, default=0.05,
                    help="relative margin over the best composite for the Goodhart gate")
    ap.add_argument("--no_goodhart_stop", action="store_true",
                    help="compute/log the Goodhart composite and keep per-eval best/ bookkeeping, but do "
                         "NOT early-stop on it; train the full --n_g_steps so checkpoint selection is done "
                         "post-hoc over the saved ckpts (the in-training composite at small n_eval_ctx is "
                         "moment_l1-noise-dominated and trips the margin gate on noise). The NaN/non-finite "
                         "divergence guard stays active.")
    ap.add_argument("--max_head_grid_gb", type=float, default=30.0,
                    help="hard cap on materialised-head grid GB/device (head scope only)")
    ap.add_argument("--shard", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--distributed", action="store_true",
                    help="multi-host: jax.distributed.initialize() before any JAX op (SLURM auto-detect)")
    ap.add_argument("--feat_chunk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # jax.distributed.initialize() already ran at module top (argv peek, pre-import — see header);
    # SLURM env (srun) is auto-detected; SPMD processes skip the login-only cpu_checks.
    if args.distributed:
        print(f"[S5] jax.distributed: process {jax.process_index()}/{jax.process_count()}; "
              f"global devices={jax.device_count()}", flush=True)

    hs = import_hyperscalees()
    if not args.distributed:
        fails = cpu_checks(hs, seed=args.seed)
        if fails:
            print(f"[S5] CPU checks FAILED: {fails}"); sys.exit(1)

    if args.diagnose:
        if not args.data_dir:
            print("[S5] --diagnose requires --data_dir (node-local GOOG dir)"); sys.exit(2)
        sys.exit(diagnose(hs, args))
    if args.run:
        if not args.data_dir:
            print("[S5] --run requires --data_dir (node-local GOOG dir)"); sys.exit(2)
        sys.exit(train(hs, args))
    print("\n[S5] (skipped GPU run — pass --run on the GH200; CPU glue checks PASSED)")


if __name__ == "__main__":
    main()
