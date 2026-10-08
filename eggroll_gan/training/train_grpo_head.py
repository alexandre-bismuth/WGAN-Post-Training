"""S5-PG — GRPO/RLOO HEAD-ONLY alternating GAN loop: the policy-gradient comparison arm (task #3).

RQ2: at a MATCHED rollout budget, does an exact head policy gradient beat EGGROLL's gradient-free
estimate? Everything except the G-step update rule is shared with train_eggroll_gan_s5 (imported,
not forked): same WGAN critic + d_step, same frozen-backbone fused feature pass, same G×Q rollout
grid and context-major layout (GROUP == the G samples sharing context q), same KL-to-anchor trust
region (plain-softmax KL, fitness.head_kl_penalty definition), same held-out stylized-fact
composite / Goodhart-as-control / best-checkpoint selection, same breadcrumb ckpt/resume.

The G-step (policy_grad.make_pg_grad):
    A[q, g]   = centered-rank (default) or RLOO advantage of the CRITIC SCORE within context row q
                (the KL is NOT in the advantage; it enters the LOSS as an explicit differentiable
                penalty, so it does not need the score-function trick).
    loss      = -(1/M) sum_r A_r * mean-logp_r  +  lam * (1/M) sum_r KL_r
    logp      = log-probs of the ACTUAL sampling distribution (validity mask -> log_softmax ->
                top-`sample_top_n` truncation -> renormalise), deterministic time tokens excluded.
ONE on-policy gradient step per rollout batch (no PPO epochs/clipping: theta == theta_old at
sampling, so the importance ratio is identically 1; clipping only matters for multi-epoch reuse).

Allocation guidance (matched TOTAL budget G*Q): PG averages gradient samples across ALL rollouts —
the group only stabilises the baseline (residual noise ~ 1/(G-1)) — while ES needs large G for
ranking reliability. So the PG arm wants SMALL groups x MANY contexts (e.g. 16 x 1024) where ES
runs 512 x 32. Both axes are CLI-free; the budget-matched comparison fixes G*Q, not the split.
NOTE the rollout still materialises per-rollout head copies [G*Q, d, V] (the gen machinery's pop
axis), so the same --max_head_grid_gb guard applies until S5b's do_Tmm fold.

lr is NOT comparable to the ES eta: rank advantages are in [-1/2, 1/2] on a mean-logp objective —
start at --lr 1e-4 (adamw, wd=0) and treat it as THE knob, like sigma for ES.

CPU-safe without --run (glue checks GR1..GR4); --run needs the GH200 + node-local GOOG data.
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

# Multi-host init BEFORE the eggroll_gan imports below — they initialise the XLA backend at IMPORT
# time (data -> discriminator -> lob jnp tables), same as train_eggroll_gan_s5.
_dist = "--distributed" in sys.argv
if _dist:
    # Slingshot: bare SLURM auto-detect resolves the coordinator to a non-routable interface
    # IP (-> gRPC "connection refused"). Use the EXPLICIT recipe (coordinator =
    # first node HOSTNAME:port via JAX_COORDINATOR_ADDRESS from the launcher; num_processes/process_id
    # from SLURM; local_device_ids over the node GPUs). Auto-detect only if the addr is unset.
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
    # We just initialised the coordination service. HIDE --distributed from sys.argv so the
    # `from .train_eggroll_gan_s5 import ...` below does NOT re-run ITS OWN module-top
    # jax.distributed.initialize() — a second init crashes ("must be called before any JAX calls").
    # Restored right after the imports so argparse still sees --distributed.
    sys.argv = [a for a in sys.argv if a != "--distributed"]

import numpy as np
import jax.numpy as jnp
import optax

from ..config import DEFAULT as CFG
from ..critic import discriminator as D
from ..es import fitness as F
from ..eval import eval_monitor as EM
from ..eval import invalidity_audit as IA
from . import dist_utils as DU
from ..baselines import policy_grad as PG
from ..es.es_generator import (make_generate_es_sharded, make_generate_es_proj_sharded, make_feats_kl,
                           gq_grid_indices, grid_rngs, grid_repeat_contexts, shard_decision,
                           extract_trainable, merge_trainable, zero_proj_factors, tile_dirs_over_Q)
from ..es.es_plumbing import import_hyperscalees, build_es_map_proj, _key_path_strs
from .train_eggroll_gan_s5 import (save_ckpt, load_ckpt, make_critic, make_d_step, rollout_to_cont, RawMsgFeats,
                                   kl_lambda, _take, _reference_rollout, _reference_rollout_proj,
                                   _replay_real, LearnedFeats, _sync_enc_cfg, _parse_scales)
from ..critic import learned_critic as LC

# Restore --distributed (scrubbed above to prevent train_eggroll_gan_s5's re-init) so argparse sees it.
if _dist and "--distributed" not in sys.argv:
    sys.argv.append("--distributed")

# adv transforms: rank (== EGGROLL rank_sigma_bar, matched-estimator arm) | rloo | grpo (paper z-score).
_ADV = {"rank": PG.rank_advantages, "rloo": PG.rloo_advantages, "grpo": PG.grpo_advantages}


def _lr_or_warmup(args):
    """--lr_warmup W > 0: linear ramp lr*min(1,(t+1)/W) over the first W updates (cold Adam's
    bias-corrected first step is full-magnitude — the step-2 damage signature at lr 1e-4), then
    constant. W=0 returns the plain float (opt_state pytree unchanged; a schedule adds a count
    leaf, so resume must keep --lr_warmup consistent with the run being resumed)."""
    w = int(getattr(args, "lr_warmup", 0) or 0)
    if w > 0:
        return lambda count: args.lr * jnp.minimum(1.0, (count + 1.0) / w)
    return args.lr


def _meta(args, diverged, ev_baseline, best_comp, **extra):
    # 'algo', not 'stage': save_ckpt stamps its own stage="S5" into the breadcrumb.
    return dict(algo="S5-PG", adv=args.adv, G=args.G, Q=args.Q, lr=args.lr,
                top_n=int(CFG.rollout.sample_top_n), kl_coef=args.kl_coef, kl_ref="anchor",
                n_cond=args.n_cond, n_gen=args.n_gen, pool_scope=args.pool_scope,
                pool_refresh_every=args.pool_refresh_every, diverged=diverged,
                ev_baseline=ev_baseline, best_composite=best_comp, **extra)


# ----------------------------------------------------------------------------------------
# The GRPO loop (GPU; --run). Mirrors train_eggroll_gan_s5.train step-for-step except the G-step.
# ----------------------------------------------------------------------------------------
def train(args):
    from ..tests.s3_es_rollout import _prep_real_batch, _sample_windows

    cfg = CFG
    G, Q = args.G, args.Q
    assert G >= 2, f"GRPO group needs >= 2 samples per context for a baseline (got G={G})"
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
        assert (G * Q) % n_dev == 0 and args.n_eval_ctx % n_dev == 0
        assert args.n_ctx_pool % n_dev == 0 and n_pool % n_dev == 0
    log(f"[PG] adv={args.adv} G={G} (group) x Q={Q} (contexts) = {G*Q} rollouts/step lr={args.lr:.1e} "
        f"top_n={CFG.rollout.sample_top_n} kl_coef={args.kl_coef} n_d={args.n_d_steps} "
        f"n_g={args.n_g_steps} n_cond={args.n_cond} n_gen={args.n_gen} devices={n_dev} "
        f"shard={'shard_map' if sharded else 'vmap'} multihost={mh}")

    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, n_pool,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels)
    inf = P["inf"]
    backbone = D.make_backbone(P["model_cls"])
    bb_params = P["bb_params"]
    pooling = cfg.disc.pooling
    pool_start = args.n_cond * F.MSG_LEN if args.pool_scope == "cont" else 0
    learned = args.critic_input in ("learned", "raw")   # sequence critic [N, T, F_in] vs backbone hiddens
    if learned:
        _sync_enc_cfg(cfg, args)
    log(f"[PG] critic_input={args.critic_input}")

    head_bytes = G * Q * (P["kernel"].size + P["bias"].size) * 4
    per_dev_gb = head_bytes / (n_dev if sharded else 1) / 1e9
    log(f"[PG] broadcast head grid: {head_bytes / 1e9:.1f} GB total, {per_dev_gb:.1f} GB/device")
    if per_dev_gb > args.max_head_grid_gb:
        raise RuntimeError(f"head grid {per_dev_gb:.1f} GB/device > --max_head_grid_gb "
                           f"{args.max_head_grid_gb}; reduce G*Q (S5b do_Tmm removes this).")

    tr = slice(0, args.n_ctx_pool)
    ev = slice(args.n_ctx_pool, n_pool)
    eval_idx = list(P["idx"][args.n_ctx_pool:])
    pool = dict(m=P["m_seq_inp"][tr], b=P["b_seq_inp"][tr], sim=_take(P["sim_states_init"], tr),
                ih=_take(P["init_hidden_batched"], tr), it=P["init_time_batched"][tr],
                ctx_tok=P["ctx_tokens"][tr], ctx_book=P["b_seq_inp"][tr],
                real_tok=P["real_cont_tokens"][tr], real_book=P["real_cont_book"][tr],
                raw_cont=P["m_seq_raw_cont"][tr])   # raw decoded real continuation msgs (learned critic)
    eval_in = dict(m=P["m_seq_inp"][ev], b=P["b_seq_inp"][ev], sim=_take(P["sim_states_init"], ev),
                   ih=_take(P["init_hidden_batched"], ev), it=P["init_time_batched"][ev],
                   rng=P["rngs"][ev])
    eval_raw_cont = P["m_seq_raw_cont"][ev]
    eval_dev = {k: DU.shard_pop(v, mesh) for k, v in eval_in.items()}

    feats_fn = make_feats_kl(backbone, bb_params, pooling=pooling, pool_start=pool_start,
                             n_cond=args.n_cond, shard=args.shard, chunk=args.feat_chunk,
                             with_kl=False)

    def _pool_feats(ct, cn, cb, cnb):
        out = feats_fn(DU.shard_pop(ct, mesh), DU.shard_pop(cn, mesh),
                       DU.shard_pop(cb, mesh), DU.shard_pop(cnb, mesh))
        return np.asarray(DU.gather_host(out, mesh))

    # learned: descriptor sequence of the real continuation replayed through the engine (LearnedFeats),
    # standardiser fit once; backbone (legacy): frozen-Mamba3 pooled hidden of the real token window.
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

    # current head (the policy) + FROZEN anchor reference; plain optax on the two leaves.
    kernel0, bias0 = P["kernel"], P["bias"]
    leaves = {"kernel": jnp.asarray(kernel0), "bias": jnp.asarray(bias0)}
    lr = _lr_or_warmup(args)
    tx_g = (optax.adamw(lr, weight_decay=0.0) if args.solver == "adamw"
            else optax.sgd(lr))
    opt_state = tx_g.init(leaves)

    gen = make_generate_es_sharded(P["model"], P["batchnorm"], P["encoder"], P["sample_top_n"],
                                   P["tick_size"], args.n_gen, P["sim_init"],
                                   P["valid_mask_array"], conditional=True, shard=args.shard)
    ts_in = DU.replicate_tree(P["train_state"], mesh)

    time_mask = F.build_time_mask(args.n_gen)
    pg_fn = PG.make_pg_grad(backbone, bb_params, n_cond=args.n_cond, n_gen=args.n_gen,
                            top_n=int(P["sample_top_n"]), valid_mask_array=P["valid_mask_array"],
                            time_mask=time_mask, shard=args.shard, chunk=args.pg_chunk)
    adv_fn = _ADV[args.adv]

    feat_shape = tuple(real_feats_pool.shape[1:]) if learned else int(real_feats_pool.shape[-1])
    head, hparams, sn, tx, opt_state_d = make_critic(cfg, feat_shape, args.seed)
    d_step = make_d_step(head, tx)
    ETi = int(inf.EVENT_TYPE_i)
    _trim = jax.jit(lambda mt, bf: rollout_to_cont(mt, bf, args.n_gen))
    _et_slice = jax.jit(lambda md: md[..., ETi].astype(jnp.int32))
    _tok_sub = jax.jit(lambda t: t[:, ::args.token_div_stride])

    def _grid_head():
        """All G*Q rollouts use the SAME current head (the policy); broadcast, host-side under mh."""
        if mh:
            t = jax.tree_util.tree_map(
                lambda x: np.broadcast_to(np.asarray(x), (G * Q,) + x.shape), leaves)
        else:
            t = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (G * Q,) + x.shape), leaves)
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
        leaves, head_vars0, opt_state, bc = load_ckpt(args.resume_dir, leaves,
                                                      {"params": hparams, **sn}, opt_state)
        hparams = head_vars0["params"]; sn = {k: v for k, v in head_vars0.items() if k != "params"}
        step0, history, composites = bc["step"], bc.get("history", []), bc.get("composites", [])
        ev_baseline = bc.get("ev_baseline") or None
        best_comp = bc.get("best_composite", min(composites) if composites else float("inf"))
        last_step = step0
        log(f"[PG] resumed from {args.resume_dir} @ step {step0}")

    base_key = jax.random.PRNGKey(args.seed)

    def _refresh_pool(k):
        nonlocal pool, real_feats_pool
        rk = jax.random.fold_in(base_key, 777_000_000 + k)
        rk_i, rk_r = jax.random.split(rk)
        W = _sample_windows(inf, P["ds"], args.n_cond, args.n_gen, args.n_ctx_pool, rk_i, rk_r,
                            sim_init=P["sim_init"], tick_size=P["tick_size"], exclude_idx=eval_idx)
        pool = dict(m=W["m_seq_inp"], b=W["b_seq_inp"], sim=W["sim_states_init"],
                    ih=pool["ih"], it=W["init_time_batched"],
                    ctx_tok=W["ctx_tokens"], ctx_book=W["b_seq_inp"],
                    real_tok=W["real_cont_tokens"], real_book=W["real_cont_book"],
                    raw_cont=W["m_seq_raw_cont"])
        real_feats_pool = _real_feats(pool)
        log(f"[PG] pool refresh #{k}")

    if args.pool_refresh_every and step0 > args.pool_refresh_every:
        k_last = (step0 - 1) // args.pool_refresh_every
        if k_last > 0:
            _refresh_pool(k_last)

    M = G * Q
    W_ref_dev = DU.replicate_tree({"k": jnp.asarray(kernel0), "b": jnp.asarray(bias0)}, mesh)
    for step in range(step0 + 1, args.n_g_steps + 1):
        if args.pool_refresh_every and step > 1 and (step - 1) % args.pool_refresh_every == 0:
            _refresh_pool((step - 1) // args.pool_refresh_every)
        step_key = jax.random.fold_in(base_key, step)
        drawn = jax.random.choice(step_key, args.n_ctx_pool, shape=(Q,), replace=False)
        drawn_np = np.asarray(drawn)

        # (1) roll out the G×Q grid under the CURRENT policy (independent sampling rng per (g, q)).
        pop_grid = _grid_head()
        m_grid = _grid_ctx(pool["m"][drawn]); b_grid = _grid_ctx(pool["b"][drawn])
        sim_grid = _grid_ctx(_take(pool["sim"], drawn)); ih_grid = _grid_ctx(_take(pool["ih"], drawn))
        itime_grid = _grid_ctx(pool["it"][drawn])
        rng_grid = DU.shard_pop(grid_rngs(step_key, G, Q), mesh)
        ctx_tok_grid = _grid_ctx(pool["ctx_tok"][drawn])
        ctx_book_grid = _grid_ctx(pool["ctx_book"][drawn])
        g_out = gen(pop_grid, ts_in, m_grid, b_grid, sim_grid, rng_grid, ih_grid, itime_grid)
        fct, fcb = _trim(g_out[3], g_out[4])
        num_errors = np.asarray(DU.gather_host(g_out[2], mesh)).astype(np.float32)

        # (2) critic features on the fakes; D-step(s) vs the matched real windows.
        if learned:
            gl2_h = np.asarray(DU.gather_host(g_out[1], mesh))                  # [Q*G, n_gen, W]
            gmsg_h = np.asarray(DU.gather_host(g_out[0], mesh))                 # [Q*G, n_gen, n_fields]
            fake_feats = sf.fake(gl2_h, gmsg_h)                                 # [Q*G, N_FEATURES]
        else:
            feats_dev = feats_fn(ctx_tok_grid, fct, ctx_book_grid, fcb)
            fake_feats = np.asarray(DU.gather_host(feats_dev, mesh))
        real_feats_grid = np.repeat(real_feats_pool[drawn_np], G, axis=0)
        # --crit_batch caps the (unsharded, sequence) critic's d-step to a strided subset spanning
        # contexts, so d-step memory is independent of G*Q (needed for the learned critic at large Q).
        # crit_batch<=0 = full batch -> bit-identical. EVERY fake is still scored for the advantage
        # below (chunked). Mirrors train_eggroll_gan_s5.py:523-541.
        CB = int(args.crit_batch); n_all = fake_feats.shape[0]
        if 0 < CB < n_all:
            d_idx = np.linspace(0, n_all - 1, CB).astype(np.int64)
            d_real, d_fake = real_feats_grid[d_idx], fake_feats[d_idx]
        else:
            d_real, d_fake = real_feats_grid, fake_feats
        for _ in range(args.n_d_steps):
            hparams, sn, opt_state_d, dloss, s_real, s_fake = d_step(hparams, sn, opt_state_d,
                                                                     d_real, d_fake)
        head_vars = {"params": hparams, **sn}

        # (3) G-step: group-relative advantages on the critic scores -> ONE on-policy PG step.
        if 0 < CB < n_all:   # chunk the train=False forward (exact; advantage is stop-gradient)
            scores = np.concatenate([np.asarray(head.apply(head_vars, fake_feats[s:s + CB], train=False))
                                     for s in range(0, n_all, CB)], axis=0)   # [Q*G]
        else:
            scores = np.asarray(head.apply(head_vars, fake_feats, train=False))    # [Q*G]
        adv = adv_fn(jnp.asarray(scores).reshape(Q, G)).reshape(M)             # context-major flat
        lam = kl_lambda(step, args.n_g_steps, args.kl_coef, anneal=args.kl_anneal)
        # lam as a DEVICE scalar: a fresh python float every step would retrace the jitted PG fn.
        loss_sum, gW_sum, gb_sum, lps, kls = pg_fn(
            ctx_tok_grid, fct, ctx_book_grid, fcb, DU.shard_pop(adv, mesh),
            leaves["kernel"], leaves["bias"], W_ref_dev["k"], W_ref_dev["b"], jnp.float32(lam))
        grads = {"kernel": gW_sum / M, "bias": gb_sum / M}
        updates, opt_state = tx_g.update(grads, opt_state, leaves)
        leaves = optax.apply_updates(leaves, updates)
        kl_grid = np.asarray(DU.gather_host(kls, mesh))
        last_step = step

        # --- monitor / eval / Goodhart: shared machinery, identical to the ES trainer ---
        if step % args.eval_every == 0 or step == 1 or step == args.n_g_steps:
            sep = float(D.critic_separation(s_real, s_fake)); auc = float(D.roc_auc(s_real, s_fake))
            mean_score = float(np.mean(scores)); mean_kl = float(np.mean(kl_grid))
            dk = float(jnp.max(jnp.abs(leaves["kernel"] - jnp.asarray(kernel0))))
            gnorm = float(jnp.sqrt(jnp.sum(grads["kernel"] ** 2) + jnp.sum(grads["bias"] ** 2)))
            finite = bool(jnp.isfinite(leaves["kernel"]).all() and np.isfinite(scores).all()
                          and np.isfinite(float(dloss)) and np.isfinite(gnorm))
            et_h = np.asarray(DU.gather_host(_et_slice(g_out[0]), mesh))
            tok_h = np.asarray(DU.gather_host(_tok_sub(fct), mesh))
            div = EM.population_diversity(tok_h, et_h, G, Q, token_stride=1)
            rec = dict(step=step, d_loss=float(dloss), separation=sep, auc=auc,
                       g_mean_score=mean_score, mean_kl=mean_kl, lam=lam,
                       pg_loss=float(loss_sum) / M, grad_norm=gnorm,
                       mean_logp=float(np.mean(np.asarray(DU.gather_host(lps, mesh)))),
                       mean_num_errors=float(np.mean(num_errors)), dkernel=dk, finite=finite,
                       token_unique_frac=float(div["token_unique_frac"]),
                       event_hist_disp=float(div["event_hist_disp"]))
            gl2, get_, gmsg = _reference_rollout(gen, leaves, ts_in, eval_dev, mesh,
                                                 G_eval=args.n_eval_ctx, et_slice=_et_slice)
            rl2 = _replay_real(inf, P["sim_init"], eval_in["sim"], eval_raw_cont)
            ev_metrics = EM.stylized_fact_metrics(
                gl2, rl2, get_, eval_raw_cont[..., ETi].astype(jnp.int32), n_levels=inf.l2_state_n)
            if ev_baseline is None:
                ev_baseline = {k: float(v) for k, v in ev_metrics.items()}
            comp = EM.normalize_composite(ev_metrics, ev_baseline)
            composites.append(comp)
            fired, best_idx = EM.goodhart_check(composites, patience=args.goodhart_patience,
                                                tol=args.goodhart_tol)
            rec.update({f"ev_{k}": float(v) for k, v in ev_metrics.items()})
            rec["composite_norm"] = comp
            rec["goodhart_fired"] = bool(fired)
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
            if comp < best_comp:
                best_comp = comp
                if p0:
                    save_ckpt(os.path.join(args.out_dir, "best"), step, leaves,
                              {"params": hparams, **sn}, opt_state, history, composites,
                              meta=_meta(args, diverged, ev_baseline, best_comp, best_at_step=step))
                    log(f"[PG] new BEST composite {comp:.4f} @ step {step} -> {args.out_dir}/best")
            log(f"[PG] step {step:5d}  D_loss {float(dloss):+.4f}  sep {sep:+.4f}  auc {auc:.3f} | "
                f"G score {mean_score:+.4f}  logp {rec['mean_logp']:+.3f}  ||g|| {gnorm:.2e}  "
                f"KL {mean_kl:.4f}(λ{lam:.3f}) | comp {comp:.4f} best@{best_idx} | "
                f"div tok {div['token_unique_frac']:.3f} ev {div['event_hist_disp']:.3f} | "
                f"||Δ|| {dk:.2e} finite={finite}")
            if not finite:
                log("[PG] *** NON-FINITE — divergence guard tripped (lower lr) ***")
                diverged = True; break
            if fired:
                log(f"[PG] *** GOODHART fired — best=eval#{best_idx}; rollback ckpt is "
                    f"{args.out_dir}/best. Stopping. ***")
                break
        if step % args.ckpt_every == 0 or step == args.n_g_steps:
            if p0:
                save_ckpt(args.out_dir, step, leaves, {"params": hparams, **sn}, opt_state,
                          history, composites, meta=_meta(args, diverged, ev_baseline, best_comp))
                log(f"[PG] checkpoint @ step {step} -> {args.out_dir}")
                if args.keep_step_ckpts:
                    sdir = os.path.join(args.out_dir, f"step{step:04d}")
                    save_ckpt(sdir, step, leaves, {"params": hparams, **sn}, opt_state,
                              history, composites, meta=_meta(args, diverged, ev_baseline, best_comp))
                    log(f"[PG] step-keyed checkpoint @ step {step} -> {sdir}")

    if p0:
        save_ckpt(args.out_dir, last_step, leaves, {"params": hparams, **sn}, opt_state,
                  history, composites, meta=_meta(args, diverged, ev_baseline, best_comp))
    log(f"[PG] {'DIVERGED' if diverged else 'done'} @ step {last_step} — {len(history)} evals -> "
        f"{args.out_dir} (best composite {best_comp:.4f} -> {args.out_dir}/best)")
    return 1 if diverged else 0


# ----------------------------------------------------------------------------------------
# --scope proj: the matched EGGROLL-proj comparison arm. Identical loop to train() except the policy
# is the proj-kernel-evolved generator (head FROZEN at the anchor) and the G-step backprops the
# REINFORCE objective through the (remat'd) teacher-forced backbone into the projection kernels
# (PG.make_pg_grad_proj). The head train() is left byte-identical.
# ----------------------------------------------------------------------------------------
def train_proj(args):
    from ..tests.s3_es_rollout import _prep_real_batch, _sample_windows
    from lob.lob_seq_model import BatchPaddedLobPredModelES

    cfg = CFG
    G, Q = args.G, args.Q
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
        assert (G * Q) % n_dev == 0 and args.n_eval_ctx % n_dev == 0
        assert args.n_ctx_pool % n_dev == 0 and n_pool % n_dev == 0
    log(f"[PG-proj] adv={args.adv} G={G} (group) x Q={Q} (contexts) = {G*Q} rollouts/step "
        f"lr={args.lr:.1e} top_n={CFG.rollout.sample_top_n} kl_coef={args.kl_coef} "
        f"n_d={args.n_d_steps} n_g={args.n_g_steps} n_cond={args.n_cond} n_gen={args.n_gen} "
        f"devices={n_dev} shard={'shard_map' if sharded else 'vmap'} multihost={mh}")

    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, n_pool,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels, wide_book_dir=args.wide_book_dir)
    inf = P["inf"]
    backbone = D.make_backbone(P["model_cls"])
    bb_params = P["bb_params"]
    pooling = cfg.disc.pooling
    pool_start = args.n_cond * F.MSG_LEN if args.pool_scope == "cont" else 0
    learned = args.critic_input in ("learned", "raw")   # sequence critic [N, T, F_in] vs backbone hiddens
    if learned:
        _sync_enc_cfg(cfg, args)
    log(f"[PG-proj] critic_input={args.critic_input}")
    train_state = P["train_state"]

    # --- proj scope: trainable = the MM_PARAM projection kernels (same set EGGROLL evolves); the
    #     decoder head stays frozen at the anchor. The policy params ride ts; es=zero no-op rollout. ---
    hs = import_hyperscalees()
    params0 = train_state.params                                  # frozen pretrained anchor (full tree)
    es_map = build_es_map_proj(params0, hs, perturb_glu=bool(args.perturb_glu),
                               perturb_book_proj=bool(args.perturb_book_proj),
                               perturb_fused_encoder=bool(args.perturb_fused_encoder))
    tr = extract_trainable(hs, params0, es_map)                   # policy params, INIT AT the anchor
    tr_anchor = {k: jnp.asarray(v) for k, v in tr.items()}       # frozen reference for ||Δ||
    n_mm = len(tr); mm_params = int(sum(v.size for v in tr.values()))
    log(f"[PG-proj] {n_mm} MM_PARAM projection kernels, {mm_params/1e6:.1f}M trainable params "
        f"(head frozen at anchor)")

    def merge_fn(t):
        return merge_trainable(hs, params0, es_map, t)

    def _zero_factor_template(rank=1):
        """Rank-1 ZERO LoRA tree (x @ 0 @ B.T == 0 — exact no-op); the policy rides ts, not a grid."""
        leaves_p = jax.tree_util.tree_flatten_with_path(params0)[0]
        flat_m = jax.tree_util.tree_flatten(es_map)[0]
        out = {}
        for (path, leaf), m in zip(leaves_p, flat_m):
            if int(m) != int(hs.MM_PARAM):
                continue
            inn, outn = leaf.shape
            names = _key_path_strs(path)
            node = out
            for nm in names[:-1]:
                node = node.setdefault(nm, {})
            node[names[-1]] = {"A": jnp.zeros((1, inn, rank), leaf.dtype),
                               "B": jnp.zeros((1, outn, rank), leaf.dtype)}
        return out
    fac_tmpl = _zero_factor_template()

    model_es = BatchPaddedLobPredModelES(**dict(P["model_cls"].keywords), training=False, step_rescale=1.0)
    gen = make_generate_es_proj_sharded(model_es, P["batchnorm"], P["encoder"], P["sample_top_n"],
                                        P["tick_size"], args.n_gen, P["sim_init"],
                                        P["valid_mask_array"], conditional=True, shard=args.shard)

    trn = slice(0, args.n_ctx_pool)
    ev = slice(args.n_ctx_pool, n_pool)
    eval_idx = list(P["idx"][args.n_ctx_pool:])
    pool = dict(m=P["m_seq_inp"][trn], b=P["b_seq_inp"][trn], sim=_take(P["sim_states_init"], trn),
                ih=_take(P["init_hidden_batched"], trn), it=P["init_time_batched"][trn],
                ctx_tok=P["ctx_tokens"][trn], ctx_book=P["b_seq_inp"][trn],
                real_tok=P["real_cont_tokens"][trn], real_book=P["real_cont_book"][trn],
                raw_cont=P["m_seq_raw_cont"][trn])   # raw decoded real continuation msgs (learned critic)
    eval_in = dict(m=P["m_seq_inp"][ev], b=P["b_seq_inp"][ev], sim=_take(P["sim_states_init"], ev),
                   ih=_take(P["init_hidden_batched"], ev), it=P["init_time_batched"][ev],
                   rng=P["rngs"][ev])
    eval_raw_cont = P["m_seq_raw_cont"][ev]
    eval_dev = {k: DU.shard_pop(v, mesh) for k, v in eval_in.items()}

    feats_fn = make_feats_kl(backbone, bb_params, pooling=pooling, pool_start=pool_start,
                             n_cond=args.n_cond, shard=args.shard, chunk=args.feat_chunk, with_kl=False)

    def _pool_feats(ct, cn, cb, cnb):
        out = feats_fn(DU.shard_pop(ct, mesh), DU.shard_pop(cn, mesh),
                       DU.shard_pop(cb, mesh), DU.shard_pop(cnb, mesh))
        return np.asarray(DU.gather_host(out, mesh))

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

    lr = _lr_or_warmup(args)
    tx_g = (optax.adamw(lr, weight_decay=0.0) if args.solver == "adamw" else optax.sgd(lr))
    opt_state = tx_g.init(tr)

    time_mask = F.build_time_mask(args.n_gen)
    pg_fn = PG.make_pg_grad_proj(backbone, bb_params, merge_fn, n_cond=args.n_cond, n_gen=args.n_gen,
                                 top_n=int(P["sample_top_n"]), valid_mask_array=P["valid_mask_array"],
                                 time_mask=time_mask, shard=args.shard, chunk=args.pg_chunk)
    adv_fn = _ADV[args.adv]

    feat_shape = tuple(real_feats_pool.shape[1:]) if learned else int(real_feats_pool.shape[-1])
    head, hparams, sn, tx, opt_state_d = make_critic(cfg, feat_shape, args.seed)
    d_step = make_d_step(head, tx)
    ETi = int(inf.EVENT_TYPE_i)
    _trim = jax.jit(lambda mt, bf: rollout_to_cont(mt, bf, args.n_gen))
    _et_slice = jax.jit(lambda md: md[..., ETi].astype(jnp.int32))
    _tok_sub = jax.jit(lambda t: t[:, ::args.token_div_stride])

    def _grid_ctx(tree_):
        if mh:
            t = jax.tree_util.tree_map(lambda x: np.repeat(np.asarray(x), G, axis=0), tree_)
        else:
            t = grid_repeat_contexts(tree_, G)
        return DU.shard_pop(t, mesh)

    def _grid_dirs(pop_):
        if mh:
            t = jax.tree_util.tree_map(lambda x: np.tile(np.asarray(x), (Q,) + (1,) * (x.ndim - 1)), pop_)
        else:
            t = tile_dirs_over_Q(pop_, Q)
        return DU.shard_pop(t, mesh)

    pop_zero_grid = _grid_dirs(zero_proj_factors(fac_tmpl, G))     # constant [Q*G] no-op factor grid

    # --gen_chunk (single-device only): the [Q*G] grid's resident-rollout memory caps one GH200 at
    # ~512; slicing the CONTEXT axis into GC-context sub-grids (each [GC*G]) lifts the per-update Q
    # ceiling (Q=128 context-parity with the EGGROLL arm) with ZERO estimator change — the rng rows
    # are the SAME grid_rngs rows (context-major r=q*G+g), rollouts are element-independent under
    # vmap, and every downstream consumer reads the host concat. The PG backward already microbatches
    # (--pg_microbatch) and the d-step already strides (--crit_batch), so generation was the only
    # stage whose memory scaled with G*Q.
    GC = int(getattr(args, "gen_chunk", 0))
    use_gc = (mesh is None) and (0 < GC < Q)
    if use_gc:
        assert Q % GC == 0, f"--gen_chunk {GC} must divide Q={Q}"
        assert learned, "--gen_chunk supports the sequence critics (learned/raw) only"
        pop_zero_gc = DU.shard_pop(tile_dirs_over_Q(zero_proj_factors(fac_tmpl, G), GC), mesh)

    history, composites, diverged = [], [], False
    ev_baseline, best_comp = None, float("inf")
    step0, last_step = 0, 0
    if args.resume_dir and os.path.exists(os.path.join(args.resume_dir, "latest_checkpoint.json")):
        tr, head_vars0, opt_state, bc = load_ckpt(args.resume_dir, extract_trainable(hs, params0, es_map),
                                                  {"params": hparams, **sn}, opt_state)
        hparams = head_vars0["params"]; sn = {k: v for k, v in head_vars0.items() if k != "params"}
        step0, history, composites = bc["step"], bc.get("history", []), bc.get("composites", [])
        ev_baseline = bc.get("ev_baseline") or None
        best_comp = bc.get("best_composite", min(composites) if composites else float("inf"))
        last_step = step0
        log(f"[PG-proj] resumed from {args.resume_dir} @ step {step0}")

    base_key = jax.random.PRNGKey(args.seed)

    def _refresh_pool(k):
        nonlocal pool, real_feats_pool
        rk = jax.random.fold_in(base_key, 777_000_000 + k)
        rk_i, rk_r = jax.random.split(rk)
        W = _sample_windows(inf, P["ds"], args.n_cond, args.n_gen, args.n_ctx_pool, rk_i, rk_r,
                            sim_init=P["sim_init"], tick_size=P["tick_size"], exclude_idx=eval_idx)
        pool = dict(m=W["m_seq_inp"], b=W["b_seq_inp"], sim=W["sim_states_init"],
                    ih=pool["ih"], it=W["init_time_batched"],
                    ctx_tok=W["ctx_tokens"], ctx_book=W["b_seq_inp"],
                    real_tok=W["real_cont_tokens"], real_book=W["real_cont_book"],
                    raw_cont=W["m_seq_raw_cont"])
        real_feats_pool = _real_feats(pool)
        log(f"[PG-proj] pool refresh #{k}")

    if args.pool_refresh_every and step0 > args.pool_refresh_every:
        k_last = (step0 - 1) // args.pool_refresh_every
        if k_last > 0:
            _refresh_pool(k_last)

    M = G * Q
    for step in range(step0 + 1, args.n_g_steps + 1):
        if args.pool_refresh_every and step > 1 and (step - 1) % args.pool_refresh_every == 0:
            _refresh_pool((step - 1) // args.pool_refresh_every)
        step_key = jax.random.fold_in(base_key, step)
        drawn = jax.random.choice(step_key, args.n_ctx_pool, shape=(Q,), replace=False)
        drawn_np = np.asarray(drawn)

        # (1) roll out the G×Q grid under the CURRENT policy: evolved proj kernels ride ts; es=zero no-op.
        ts_in = DU.replicate_tree(train_state.replace(params=merge_fn(tr)), mesh)
        rng_full = grid_rngs(step_key, G, Q)
        if use_gc:
            gm_l, gl2_l, ne_l, fct_l, fcb_l = [], [], [], [], []
            for q0 in range(0, Q, GC):
                slq = drawn[q0:q0 + GC]
                out = gen(pop_zero_gc, ts_in,
                          _grid_ctx(pool["m"][slq]), _grid_ctx(pool["b"][slq]),
                          _grid_ctx(_take(pool["sim"], slq)),
                          DU.shard_pop(rng_full[q0 * G:(q0 + GC) * G], mesh),
                          _grid_ctx(_take(pool["ih"], slq)), _grid_ctx(pool["it"][slq]))
                fct_i, fcb_i = _trim(out[3], out[4])
                gm_l.append(np.asarray(out[0])); gl2_l.append(np.asarray(out[1]))
                ne_l.append(np.asarray(out[2]))
                fct_l.append(np.asarray(fct_i)); fcb_l.append(np.asarray(fcb_i))
                del out, fct_i, fcb_i               # free the chunk's device buffers before the next
            gmsg_h = np.concatenate(gm_l); gl2_h = np.concatenate(gl2_l)       # host [Q*G, ...]
            num_errors = np.concatenate(ne_l).astype(np.float32)
            fct = np.concatenate(fct_l); fcb = np.concatenate(fcb_l)           # host [Q*G, ...]
            # host ctx grids: pg_fn below slices [s0:s0+MB] and jit-converts per micro-slice
            ctx_tok_grid = np.repeat(np.asarray(pool["ctx_tok"][drawn]), G, axis=0)
            ctx_book_grid = np.repeat(np.asarray(pool["ctx_book"][drawn]), G, axis=0)
        else:
            m_grid = _grid_ctx(pool["m"][drawn]); b_grid = _grid_ctx(pool["b"][drawn])
            sim_grid = _grid_ctx(_take(pool["sim"], drawn)); ih_grid = _grid_ctx(_take(pool["ih"], drawn))
            itime_grid = _grid_ctx(pool["it"][drawn])
            rng_grid = DU.shard_pop(rng_full, mesh)
            ctx_tok_grid = _grid_ctx(pool["ctx_tok"][drawn]); ctx_book_grid = _grid_ctx(pool["ctx_book"][drawn])
            g_out = gen(pop_zero_grid, ts_in, m_grid, b_grid, sim_grid, rng_grid, ih_grid, itime_grid)
            fct, fcb = _trim(g_out[3], g_out[4])
            num_errors = np.asarray(DU.gather_host(g_out[2], mesh)).astype(np.float32)

        # (2) critic features (FROZEN anchor featurizer) on the fakes; D-step(s) vs matched real windows.
        if learned:
            if not use_gc:
                gl2_h = np.asarray(DU.gather_host(g_out[1], mesh))              # [Q*G, n_gen, W]
                gmsg_h = np.asarray(DU.gather_host(g_out[0], mesh))             # [Q*G, n_gen, n_fields]
            fake_feats = sf.fake(gl2_h, gmsg_h)                                 # [Q*G, N_FEATURES]
        else:
            feats_dev = feats_fn(ctx_tok_grid, fct, ctx_book_grid, fcb)
            fake_feats = np.asarray(DU.gather_host(feats_dev, mesh))
        real_feats_grid = np.repeat(real_feats_pool[drawn_np], G, axis=0)
        # --crit_batch caps the (unsharded, sequence) critic's d-step to a strided subset spanning
        # contexts, so d-step memory is independent of G*Q (needed for the learned critic at large Q).
        # crit_batch<=0 = full batch -> bit-identical. EVERY fake is still scored for the advantage
        # below (chunked). Mirrors train_eggroll_gan_s5.py:523-541.
        CB = int(args.crit_batch); n_all = fake_feats.shape[0]
        if 0 < CB < n_all:
            d_idx = np.linspace(0, n_all - 1, CB).astype(np.int64)
            d_real, d_fake = real_feats_grid[d_idx], fake_feats[d_idx]
        else:
            d_real, d_fake = real_feats_grid, fake_feats
        for _ in range(args.n_d_steps):
            hparams, sn, opt_state_d, dloss, s_real, s_fake = d_step(hparams, sn, opt_state_d,
                                                                     d_real, d_fake)
        head_vars = {"params": hparams, **sn}

        # (3) G-step: group-relative advantages -> ONE on-policy PG step INTO the projection kernels.
        if 0 < CB < n_all:   # chunk the train=False forward (exact; advantage is stop-gradient)
            scores = np.concatenate([np.asarray(head.apply(head_vars, fake_feats[s:s + CB], train=False))
                                     for s in range(0, n_all, CB)], axis=0)   # [Q*G]
        else:
            scores = np.asarray(head.apply(head_vars, fake_feats, train=False))    # [Q*G]
        adv = adv_fn(jnp.asarray(scores).reshape(Q, G)).reshape(M)             # context-major flat
        if args.fitness_control == "shuffle":
            # Attribution null: permute the advantages so the PG step ascends RANDOM advantages —
            # zero information from D — while the gradient-magnitude statistics (||g||, adamw moments)
            # stay those of a live run. Key 888M+step is disjoint from step_key and the pool refresh
            # (777M+k); deterministic across resume. Same pattern as the ES shuffle control.
            adv = adv[jax.random.permutation(jax.random.fold_in(base_key, 888_000_000 + step), M)]
        lam = kl_lambda(step, args.n_g_steps, args.kl_coef, anneal=args.kl_anneal)  # python float (-> rec)
        lam_dev = jnp.float32(lam)   # device scalar for the jitted pg_fn (a fresh python float retraces)
        # The teacher-forced proj backward materialises [mb, n_gen*26, d_in_proj] activations; at the
        # full M=G*Q it OOMs one GPU (73.9 GB at 512; 80.5 GiB single alloc at just 16 rollouts/device).
        # MICROBATCH the PG gradient — accumulate over MB-sized slices, each pg_fn
        # call fully executing/freeing before the next (the internal chunk-scan does not free under
        # reverse-mode). EXACT: the grad is a linear sum over rollouts (advantages are group-normalised
        # over the FULL group BEFORE slicing, line ~641). BOTH the 1-device and the sharded path now
        # microbatch — the sharded path shards EACH micro-slice across the pop devices (else branch).
        if mesh is None:
            MB = args.pg_microbatch if (0 < args.pg_microbatch < M) else M
            assert M % MB == 0, f"--pg_microbatch {MB} must divide G*Q={M}"
            loss_sum = jnp.zeros(()); grad_tr = jax.tree_util.tree_map(jnp.zeros_like, tr)
            lp_parts, kl_parts = [], []
            for s0 in range(0, M, MB):
                sl = slice(s0, s0 + MB)
                l_i, g_i, lp_i, kl_i = pg_fn(ctx_tok_grid[sl], fct[sl], ctx_book_grid[sl], fcb[sl],
                                            adv[sl], tr, lam_dev)
                loss_sum = loss_sum + l_i
                grad_tr = jax.tree_util.tree_map(jnp.add, grad_tr, g_i)
                lp_parts.append(lp_i); kl_parts.append(kl_i)
            lps = jnp.concatenate(lp_parts); kls = jnp.concatenate(kl_parts)
        else:
            # SHARDED microbatch — same accumulation as the mesh=None path, but each micro-slice is
            # ITSELF sharded across the n_dev pop devices, so MB must be BOTH a device multiple and an
            # M divisor. shard_pop is CONTIGUOUS per device (dist_utils.shard_pop), so we CANNOT slice
            # the already-sharded device grids on the pop axis; instead we slice HOST (un-sharded)
            # [M, ...] copies and shard_pop each micro-slice. The context grids' host pre-shard form is
            # exactly _grid_ctx's np.repeat(..., G) (== grid_repeat_contexts, context-major r=q*G+g);
            # fct/fcb are gathered from the sharded generate. All five inputs share the SAME [s0:s0+MB]
            # slice so rollouts stay aligned. OUTPUTS: pg_fn returns loss/grad REPLICATED (psum over the
            # pop axis -> already summed over the slice's rollouts) and lp/kl SHARDED (P("pop")); we
            # gather each to host and accumulate there. lps/kls therefore end up as HOST numpy [M] —
            # DU.gather_host at lines 675/693 then passes them straight through (a numpy array is not a
            # jax.Array, so gather_host's per-leaf replicate is skipped), which keeps 675/693 correct
            # WITHOUT trying to re-shard a non-global host concat back onto the mesh.
            MB = args.pg_microbatch if (0 < args.pg_microbatch < M) else M
            assert MB % n_dev == 0 and M % MB == 0, (
                f"--pg_microbatch {MB} must be a multiple of n_dev={n_dev} AND divide G*Q={M} "
                f"(each sharded micro-slice spans all {n_dev} pop devices)")
            ctx_tok_h = np.repeat(np.asarray(pool["ctx_tok"][drawn]), G, axis=0)     # host [M, ...]
            ctx_book_h = np.repeat(np.asarray(pool["ctx_book"][drawn]), G, axis=0)   # (do NOT pre-shard)
            fct_h = np.asarray(DU.gather_host(fct, mesh))                            # gather sharded gen
            fcb_h = np.asarray(DU.gather_host(fcb, mesh))
            loss_sum = 0.0
            grad_tr = jax.tree_util.tree_map(jnp.zeros_like, tr)
            lp_parts, kl_parts = [], []
            for s0 in range(0, M, MB):
                sl = slice(s0, s0 + MB)
                l_i, g_i, lp_i, kl_i = pg_fn(
                    DU.shard_pop(ctx_tok_h[sl], mesh), DU.shard_pop(fct_h[sl], mesh),
                    DU.shard_pop(ctx_book_h[sl], mesh), DU.shard_pop(fcb_h[sl], mesh),
                    DU.shard_pop(adv[sl], mesh), tr, lam_dev)
                loss_sum = loss_sum + float(DU.gather_host(l_i, mesh))
                g_i_h = DU.gather_host(g_i, mesh)                                    # replicated -> host
                grad_tr = jax.tree_util.tree_map(lambda a, b: a + jnp.asarray(b), grad_tr, g_i_h)
                lp_parts.append(np.asarray(DU.gather_host(lp_i, mesh)))
                kl_parts.append(np.asarray(DU.gather_host(kl_i, mesh)))
            lps = np.concatenate(lp_parts); kls = np.concatenate(kl_parts)          # host [M]
        grads = jax.tree_util.tree_map(lambda g: g / M, grad_tr)
        updates, opt_state = tx_g.update(grads, opt_state, tr)
        tr = optax.apply_updates(tr, updates)
        kl_grid = np.asarray(DU.gather_host(kls, mesh))
        last_step = step

        # --- monitor / eval / Goodhart: shared machinery, identical to the head trainer ---
        if step % args.eval_every == 0 or step == 1 or step == args.n_g_steps:
            sep = float(D.critic_separation(s_real, s_fake)); auc = float(D.roc_auc(s_real, s_fake))
            mean_score = float(np.mean(scores)); mean_kl = float(np.mean(kl_grid))
            dk = max(float(jnp.max(jnp.abs(tr[kk] - tr_anchor[kk]))) for kk in tr_anchor)
            gnorm = float(jnp.sqrt(sum(float(jnp.sum(g ** 2)) for g in jax.tree_util.tree_leaves(grads))))
            gen_finite = all(bool(jnp.isfinite(v).all()) for v in tr.values())
            finite = bool(gen_finite and np.isfinite(scores).all() and np.isfinite(float(dloss))
                          and np.isfinite(gnorm))
            et_h = (gmsg_h[..., ETi].astype(np.int32) if use_gc     # gc: msgs already host-gathered
                    else np.asarray(DU.gather_host(_et_slice(g_out[0]), mesh)))
            tok_h = np.asarray(DU.gather_host(_tok_sub(fct), mesh))
            div = EM.population_diversity(tok_h, et_h, G, Q, token_stride=1)
            rec = dict(step=step, d_loss=float(dloss), separation=sep, auc=auc,
                       g_mean_score=mean_score, mean_kl=mean_kl, lam=lam,
                       pg_loss=float(loss_sum) / M, grad_norm=gnorm,
                       mean_logp=float(np.mean(np.asarray(DU.gather_host(lps, mesh)))),
                       mean_num_errors=float(np.mean(num_errors)), dkernel=dk, finite=finite,
                       token_unique_frac=float(div["token_unique_frac"]),
                       event_hist_disp=float(div["event_hist_disp"]))
            # Goodhart guard: σ=0 reference rollout (zero factors) on the POST-update params.
            ts_eval = DU.replicate_tree(train_state.replace(params=merge_fn(tr)), mesh)
            gl2, get_, gmsg = _reference_rollout_proj(gen, fac_tmpl, ts_eval, eval_dev, mesh,
                                                      G_eval=args.n_eval_ctx, et_slice=_et_slice)
            rl2 = _replay_real(inf, P["sim_init"], eval_in["sim"], eval_raw_cont)
            ev_metrics = EM.stylized_fact_metrics(
                gl2, rl2, get_, eval_raw_cont[..., ETi].astype(jnp.int32), n_levels=inf.l2_state_n)
            if ev_baseline is None:
                ev_baseline = {k: float(v) for k, v in ev_metrics.items()}
            comp = EM.normalize_composite(ev_metrics, ev_baseline)
            composites.append(comp)
            fired, best_idx = EM.goodhart_check(composites, patience=args.goodhart_patience,
                                                tol=args.goodhart_tol)
            rec.update({f"ev_{k}": float(v) for k, v in ev_metrics.items()})
            rec["composite_norm"] = comp
            rec["goodhart_fired"] = bool(fired)
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
            if comp < best_comp:
                best_comp = comp
                if p0:
                    save_ckpt(os.path.join(args.out_dir, "best"), step, tr, {"params": hparams, **sn},
                              opt_state, history, composites,
                              meta=_meta(args, diverged, ev_baseline, best_comp, best_at_step=step,
                                         scope="proj"), scope="proj")
                    log(f"[PG-proj] new BEST composite {comp:.4f} @ step {step} -> {args.out_dir}/best")
            log(f"[PG-proj] step {step:5d}  D_loss {float(dloss):+.4f}  sep {sep:+.4f}  auc {auc:.3f} | "
                f"G score {mean_score:+.4f}  logp {rec['mean_logp']:+.3f}  ||g|| {gnorm:.2e}  "
                f"KL {mean_kl:.4f}(λ{lam:.3f}) | comp {comp:.4f} best@{best_idx} | "
                f"div tok {div['token_unique_frac']:.3f} | ||Δ|| {dk:.2e} finite={finite}")
            if not finite:
                log("[PG-proj] *** NON-FINITE — divergence guard tripped (lower lr) ***")
                diverged = True; break
            if fired:
                log(f"[PG-proj] *** GOODHART fired — best=eval#{best_idx}; rollback ckpt is "
                    f"{args.out_dir}/best. Stopping. ***")
                break
        if step % args.ckpt_every == 0 or step == args.n_g_steps:
            if p0:
                save_ckpt(args.out_dir, step, tr, {"params": hparams, **sn}, opt_state,
                          history, composites,
                          meta=_meta(args, diverged, ev_baseline, best_comp, scope="proj"), scope="proj")
                log(f"[PG-proj] checkpoint @ step {step} -> {args.out_dir}")
                if args.keep_step_ckpts:
                    sdir = os.path.join(args.out_dir, f"step{step:04d}")
                    save_ckpt(sdir, step, tr, {"params": hparams, **sn}, opt_state,
                              history, composites,
                              meta=_meta(args, diverged, ev_baseline, best_comp, scope="proj"), scope="proj")
                    log(f"[PG-proj] step-keyed checkpoint @ step {step} -> {sdir}")

    if p0:
        save_ckpt(args.out_dir, last_step, tr, {"params": hparams, **sn}, opt_state,
                  history, composites, meta=_meta(args, diverged, ev_baseline, best_comp, scope="proj"),
                  scope="proj")
    log(f"[PG-proj] {'DIVERGED' if diverged else 'done'} @ step {last_step} — {len(history)} evals -> "
        f"{args.out_dir} (best composite {best_comp:.4f} -> {args.out_dir}/best)")
    return 1 if diverged else 0


# ----------------------------------------------------------------------------------------
# CPU-safe glue checks (login node; policy_grad's own PG1..PG9 cover the math).
# ----------------------------------------------------------------------------------------
def cpu_checks(seed=0):
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("\n[PG] CPU checks — advantage/grid layout / optax step / ckpt slot", flush=True)
    k = jax.random.PRNGKey(seed)
    G, Q = 5, 3

    # (GR1) flat advantage ordering matches the context-major rollout grid (r = q*G + g).
    s = jax.random.normal(jax.random.fold_in(k, 1), (Q, G))
    A = PG.rank_advantages(s)
    flat = A.reshape(Q * G)
    di, ci = gq_grid_indices(G, Q)
    ok = all(float(flat[r]) == float(A[int(ci[r]), int(di[r])]) for r in range(G * Q))
    chk("(GR1) adv.reshape(Q*G) aligns with grid (dir=r%G, ctx=r//G)", ok)

    # (GR2) optax G-step moves the head, stays finite, anchor untouched.
    leaves = {"kernel": jax.random.normal(jax.random.fold_in(k, 2), (6, 9)),
              "bias": jnp.zeros((9,))}
    anchor = leaves["kernel"].copy()
    tx = optax.adamw(1e-3, weight_decay=0.0)
    st = tx.init(leaves)
    grads = {"kernel": jnp.ones_like(leaves["kernel"]), "bias": jnp.ones_like(leaves["bias"])}
    up, st = tx.update(grads, st, leaves)
    new = optax.apply_updates(leaves, up)
    chk("(GR2) optax step moves head & finite; anchor frozen",
        float(jnp.max(jnp.abs(new["kernel"] - leaves["kernel"]))) > 0
        and bool(jnp.isfinite(new["kernel"]).all())
        and float(jnp.max(jnp.abs(anchor - leaves["kernel"]))) == 0.0)

    # (GR3) ckpt round-trip with the optax state in the noiser_state slot.
    head, hparams, sn, _tx, _os = make_critic(CFG, 8, seed)
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"pg_ckpt_test_{os.getpid()}")
    save_ckpt(out, 5, new, {"params": hparams, **sn}, st, history=[{"step": 5}], composites=[1.0],
              meta=_meta(argparse.Namespace(adv="rank", G=G, Q=Q, lr=1e-3, kl_coef=0.0,
                                            n_cond=4, n_gen=2, pool_scope="cont",
                                            pool_refresh_every=0),
                         False, {"book_l1": 1.0}, 1.0))
    l_r, hv_r, st_r, bc = load_ckpt(out, new, {"params": hparams, **sn}, st)
    same_st = jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda a, b: bool(jnp.array_equal(jnp.asarray(a), jnp.asarray(b))), st, st_r))
    chk("(GR3) ckpt round-trip (head + optax state + meta)",
        bc["step"] == 5 and bc["algo"] == "S5-PG" and same_st
        and float(jnp.max(jnp.abs(l_r["kernel"] - new["kernel"]))) == 0.0)
    import shutil; shutil.rmtree(out, ignore_errors=True)

    # (GR4) sampling top_n consistency: rollout machinery and log-probs use the SAME config value.
    chk("(GR4) CFG.rollout.sample_top_n > 1 (PG needs a stochastic policy)",
        int(CFG.rollout.sample_top_n) > 1, f"top_n={CFG.rollout.sample_top_n}")

    # (GR5) proj-scope ckpt round-trip: a FLAT trainable dict saves/loads as generator_proj (the
    # test_eval / soup-compatible breadcrumb), distinct from the head's {kernel,bias} payload + carries
    # the scope tag. Also exercises grpo_advantages wiring via _ADV.
    chk("(GR5a) grpo advantage registered in _ADV", "grpo" in _ADV and _ADV["grpo"] is PG.grpo_advantages)
    tr_fake = {"message_encoder/layers_0/seq/in_proj/kernel": jax.random.normal(jax.random.fold_in(k, 7), (4, 6)),
               "fused_s5/layers_0/seq/out_proj/kernel": jax.random.normal(jax.random.fold_in(k, 8), (6, 4))}
    out2 = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"pg_proj_ckpt_{os.getpid()}")
    save_ckpt(out2, 7, tr_fake, {"params": hparams, **sn}, st, history=[{"step": 7}], composites=[0.9],
              meta=_meta(argparse.Namespace(adv="grpo", G=G, Q=Q, lr=1e-4, kl_coef=0.05, n_cond=4,
                                            n_gen=2, pool_scope="cont", pool_refresh_every=0),
                         False, {"book_l1": 1.0}, 0.9, scope="proj"), scope="proj")
    payload, _, _, bc2 = load_ckpt(out2, {kk: jnp.zeros_like(vv) for kk, vv in tr_fake.items()},
                                   {"params": hparams, **sn}, st)
    chk("(GR5b) proj ckpt round-trips as generator_proj (flat dict + scope tag)",
        bool(bc2.get("generator_proj")) and bc2.get("scope") == "proj"
        and all(float(jnp.max(jnp.abs(payload[kk] - tr_fake[kk]))) == 0.0 for kk in tr_fake))
    shutil.rmtree(out2, ignore_errors=True)

    print("[PG] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


def main():
    ap = argparse.ArgumentParser(description="S5-PG: GRPO/RLOO head-only GAN loop (+ checks)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--adv", choices=list(_ADV), default="rank",
                    help="group-relative advantage: grpo (paper z-score (r-mean)/std) | rank "
                         "(fat-tail-robust, matches EGGROLL) | rloo")
    ap.add_argument("--scope", choices=["head", "proj"], default="head",
                    help="head = evolve the decoder head; proj = evolve the in/out projection kernels "
                         "(the matched EGGROLL-proj comparison arm), head frozen at the anchor")
    ap.add_argument("--critic_input",
                    choices=["learned", "raw", "backbone"],
                    default=CFG.disc.critic_input,
                    help="what the WGAN critic scores (matches the EGGROLL arm for a fair ES-vs-PG "
                         "comparison): 'learned' (DEFAULT) SN-regularised causal TCN over the descriptor "
                         "sequence; 'raw' the SAME TCN over the decoded-message field sequence "
                         "(critic/raw_message_features.py); 'backbone' LEGACY pooled hidden. Retired "
                         "fixed-φ modes were removed — recover from git history.")
    # Lever-3 learned-critic knobs (inert unless --critic_input learned).
    ap.add_argument("--enc_type", choices=["tcn", "cnn"], default=CFG.disc.enc_type)
    ap.add_argument("--enc_layers", type=int, default=CFG.disc.enc_layers)
    ap.add_argument("--enc_hidden", type=int, default=CFG.disc.enc_hidden)
    ap.add_argument("--enc_pool", choices=["attn", "mean", "last"], default=CFG.disc.enc_pool)
    ap.add_argument("--enc_kernel", type=int, default=CFG.disc.enc_kernel)
    ap.add_argument("--enc_lr", type=float, default=CFG.disc.enc_lr)
    ap.add_argument("--enc_adam_b1", type=float, default=CFG.disc.enc_adam_b1)
    # Lever-3 STRONGEST config: multi-scale conv trunk + seed-diverse critic ensemble (pessimistic reward).
    ap.add_argument("--enc_scales", type=_parse_scales, default=CFG.disc.enc_scales,
                    help="multi-scale downsample factors '1,2,4' (one TCN branch each)")
    ap.add_argument("--ens_size", type=int, default=CFG.disc.ens_size,
                    help="seed-diverse critic ensemble members (1 disables)")
    ap.add_argument("--ens_pessimism", type=float, default=CFG.disc.ens_pessimism,
                    help="β in the reward mean_k − β·std_k (anti-hack disagreement penalty)")
    ap.add_argument("--enc_proj_dim", type=int, default=CFG.disc.enc_proj_dim,
                    help="per-member fixed random input projection F_in->dim (Projected-GAN); 0 = off")
    ap.add_argument("--enc_whiten", choices=["std", "robust"], default=CFG.disc.enc_whiten,
                    help="critic input standardiser: 'std' (mean/σ) | 'robust' (median/MAD, heavy-tail-safe)")
    ap.add_argument("--perturb_glu", type=int, default=1, help="proj: include GLU out2 kernels")
    ap.add_argument("--perturb_book_proj", type=int, default=0, help="proj: include book input projection")
    ap.add_argument("--perturb_fused_encoder", type=int, default=1, help="proj: include fused_s5 encoder")
    ap.add_argument("--wide_book_dir", default=None, help="wide-L500 book dir (held-out eval parity)")
    ap.add_argument("--fitness_control", choices=["none", "shuffle"], default="none",
                    help="attribution null (paired with the seed-matched lane): 'shuffle' permutes the "
                         "per-step advantages so the PG step ascends RANDOM advantages (zero info from "
                         "D, identical update-magnitude statistics)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "pg_out"))
    ap.add_argument("--resume_dir", default=None)
    ap.add_argument("--G", type=int, default=16, help="GROUP size (samples per context)")
    ap.add_argument("--Q", type=int, default=32, help="contexts per step (PG wants this large)")
    ap.add_argument("--n_cond", type=int, default=500)
    ap.add_argument("--n_gen", type=int, default=500)
    ap.add_argument("--wide_levels", type=int, default=10)
    ap.add_argument("--n_g_steps", type=int, default=1000)
    ap.add_argument("--n_d_steps", type=int, default=5)
    ap.add_argument("--crit_batch", type=int, default=0,
                    help="cap the (unsharded) critic d-step minibatch to CB strided samples spanning "
                         "contexts; 0=full batch (bit-identical). Needed for the learned/sequence critic "
                         "at large G*Q. Mirrors train_eggroll_gan_s5.py --crit_batch.")
    ap.add_argument("--lr", type=float, default=1e-4, help="THE PG knob (not comparable to ES eta)")
    ap.add_argument("--lr_warmup", type=int, default=0,
                    help="linear LR warmup over the first W updates (0 = constant lr; "
                         "changes the opt_state pytree, keep consistent across --resume_dir)")
    ap.add_argument("--solver", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--kl_coef", type=float, default=0.0)
    ap.add_argument("--kl_anneal", choices=["cosine", "const"], default="cosine")
    ap.add_argument("--pg_chunk", type=int, default=2,
                    help="rollouts per PG-grad chunk (logits are [chunk, n_gen*26, 2112])")
    ap.add_argument("--pg_microbatch", type=int, default=32,
                    help="proj: rollouts per pg_fn CALL; the proj backward is accumulated over "
                         "M/pg_microbatch slices to bound the teacher-forced activation memory (the "
                         "full M=G*Q OOMs one GPU). Exact (the grad is a linear sum over rollouts)")
    ap.add_argument("--gen_chunk", type=int, default=0,
                    help="proj, single-device (--shard off), sequence critics only: generate the "
                         "G*Q grid in slices of this many CONTEXTS (each slice [chunk*G] rollouts), "
                         "lifting the ~512 resident-rollout/GPU ceiling on Q. Exact (same grid_rngs "
                         "rows, element-independent rollouts, host concat). 0 = single call; must "
                         "divide Q")
    ap.add_argument("--n_ctx_pool", type=int, default=1024)
    ap.add_argument("--n_eval_ctx", type=int, default=128)
    ap.add_argument("--pool_scope", choices=["cont", "window"], default="cont")
    ap.add_argument("--pool_refresh_every", type=int, default=0)
    ap.add_argument("--token_div_stride", type=int, default=26)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--goodhart_patience", type=int, default=3)
    ap.add_argument("--keep_step_ckpts", action="store_true",
                    help="ALSO save an immutable step-keyed copy (out_dir/stepNNNN) at every ckpt_every "
                         "— the grid the chain SEL eval scores (matches the EGGROLL trainer)")
    ap.add_argument("--goodhart_tol", type=float, default=0.05)
    ap.add_argument("--max_head_grid_gb", type=float, default=30.0)
    ap.add_argument("--shard", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--feat_chunk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # initialize() already ran at module top (argv peek, pre-import).
    if args.distributed:
        print(f"[PG] jax.distributed: process {jax.process_index()}/{jax.process_count()}; "
              f"global devices={jax.device_count()}", flush=True)
    if not args.distributed:
        fails = cpu_checks(seed=args.seed)
        if fails:
            print(f"[PG] CPU checks FAILED: {fails}"); sys.exit(1)
    if args.run:
        if not args.data_dir:
            print("[PG] --run requires --data_dir (node-local GOOG dir)"); sys.exit(2)
        sys.exit(train_proj(args) if args.scope == "proj" else train(args))
    print("\n[PG] (skipped GPU run — pass --run on the GH200; CPU glue checks PASSED)")


if __name__ == "__main__":
    main()
