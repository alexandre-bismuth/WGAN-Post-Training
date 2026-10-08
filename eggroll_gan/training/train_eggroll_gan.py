"""S4 — the full alternating EGGROLL-GAN loop (+ a short 1-node smoke).

One step = ONE population rollout, reused for BOTH players (no extra rollout for D):

    1. pop  = build_decoder_population(leaves ± σ·noise)           # antithetic low-rank head perturb
       fakes = gen_es(pop)                                         # N AR rollouts (S3 wrapper, GH200)
       fake_feats = frozen-backbone pooled features of [ctx ; fake_cont]   # the expensive pass, ONCE
    2. D-step (backprop):  WGAN critic head trained n_d steps on  real_feats  vs  fake_feats
                           (spectral-normed, Lipschitz; reuses the S1 critic exactly).
    3. G-step (EGGROLL):   score the SAME fakes with the freshly-updated critic
                           F_i = D(fake_i) - kl_coef·KL_i ; convert_fitnesses -> do_updates -> leaves moves.

The evolving reference head `leaves` (decoder {kernel,bias}) is the SINGLE source of truth for the
generator: D rolls out the current `leaves`; G perturbs around it; `do_updates` moves it. The frozen
Mamba3 backbone (message/book/fused encoders) never changes and is shared by the generator rollout and
the discriminator feature extractor — so `bb_params = train_state.params` stays valid throughout (the
backbone's `features` method never touches the decoder).

Smoke success (verification): critic separation MOVES, G fitness (mean fake score) trends
up under the sharpening critic, KL bounded, and EVERYTHING stays finite (the Phase-3 divergence guard).

Scope = head-only (whole-interior-minus-I/O is S5). group_size=0 (global rank-norm, the S3
config); CRN context-tiling + GRPO grouping (tile_contexts + group_size>0) is an S5 variance-reduction
step, not enabled here so the smoke stays on the bit-for-bit-validated S3 path.

CLUSTER: `--run` does full AR rollouts + backbone passes (heavy compute) and reads npy via SquashFS —
RUN ONLY ON THE GH200 (see _run_s4_local.sh). Outputs go to $TMPDIR; rsync to Lustre on success only.
Without `--run` it executes ONLY the CPU-safe glue checks (login-node safe; no model, no rollout).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from ..config import DEFAULT as CFG
from ..es.es_plumbing import import_hyperscalees
from ..es.es_generator import (make_decoder_noiser, build_decoder_population, population_iterinfo,
                           make_generate_es_batched)
from ..critic import discriminator as D
from ..es import fitness as F

_SOLVERS = {"adamw": optax.adamw, "sgd": optax.sgd, "muon": optax.contrib.muon}


def _solver_kwargs(name):
    """Per-solver constructor kwargs for the ES noiser. weight_decay is forced to 0 for EVERY
    solver: the ES update evolves a low-rank delta on the FROZEN anchor, so any decay-toward-zero
    would corrupt the anchor's learned weights (this is the optax.adamw default-wd=1e-4 hazard the
    masked solver already guards against). For muon BOTH the matrix-branch `weight_decay` and the
    adam-fallback `adam_weight_decay` are zeroed — and because the masked wrapper restricts muon to
    the 2D MM_PARAM kernels, the adam branch is never reached at all (proven by tests/muon_gate.py,
    which asserts every kernel receives a Newton-Schulz update, never a silent AdamW fallback)."""
    if name == "adamw":
        return {"weight_decay": 0.0}
    if name == "muon":
        return {"weight_decay": 0.0, "adam_weight_decay": 0.0}
    return {}


# ----------------------------------------------------------------------------------------
# Rollout output -> critic continuation arrays (identical trim to generate_fakes.generate_pairs).
# ----------------------------------------------------------------------------------------
def rollout_to_cont(msgs_tokens, b_finals, n_gen, msg_len=F.MSG_LEN):
    """generate()'s msgs_tokens (out[3]) + b_finals (out[4]) -> (fake_cont_tokens [N, n_gen*26],
    fake_cont_book [N, n_gen, 503]). b_finals is the in-loop 503-wide transform the model consumed,
    so it matches ctx_book / real_cont_book exactly."""
    fct = jnp.reshape(msgs_tokens, (msgs_tokens.shape[0], -1))[:, : n_gen * msg_len]
    fcb = b_finals[:, : n_gen]
    return fct, fcb


# ----------------------------------------------------------------------------------------
# WGAN critic head (S1) — init + one jitted train step. Mirrors train_s1.train_critic_head.
# ----------------------------------------------------------------------------------------
def make_critic(cfg, feat_dim, seed):
    # `feat_dim` as a (T, F_in) TUPLE selects the Lever-3 learned SEQUENCE critic (critic/learned_critic);
    # an INT keeps the vector head (backbone/stylized/dynamical/signature) byte-identical to the original
    # path. Inferring the kind from feat_dim's type (not cfg.disc.critic_input) avoids any dependence on
    # whether the CLI choice was synced back into the global config.
    if isinstance(feat_dim, (tuple, list)):
        from ..critic.learned_critic import make_learned_critic, make_learned_tx
        T_seq, F_in = int(feat_dim[0]), int(feat_dim[1])
        head, params, sn = make_learned_critic(cfg, T_seq, F_in, seed)
        tx = make_learned_tx(cfg)                                  # TTUR (faster critic lr, Adam β1=0)
        return head, params, sn, tx, tx.init(params)
    head = D.CriticHead(hidden=tuple(cfg.disc.head_hidden), use_spectral_norm=cfg.disc.use_spectral_norm)
    init_vars = head.init(jax.random.PRNGKey(seed), jnp.zeros((1, feat_dim)), train=False)
    params = init_vars["params"]
    sn = {k: v for k, v in init_vars.items() if k != "params"}     # 'sn' collection (empty if SN off)
    tx = optax.adamw(cfg.disc.lr, weight_decay=cfg.disc.weight_decay)
    opt_state = tx.init(params)
    return head, params, sn, tx, opt_state


def make_d_step(head, tx, r1_gamma: float = 0.0):
    """Critic (WGAN) update. With r1_gamma>0, adds the R1 gradient penalty
    `γ·½·E_real||∇_h D(h)||²` on the critic's CONTINUOUS feature input — the standard fix for an
    unstable / hackable critic (smooths D so the generator can't climb sharp non-realistic
    directions in feature space). r1_gamma=0.0 is bit-identical to the original path."""
    @jax.jit
    def d_step(params, sn, opt_state, real_feats, fake_feats):
        n_r = real_feats.shape[0]
        feats = jnp.concatenate([real_feats, fake_feats], axis=0)

        def loss_fn(p):
            variables = {"params": p, **sn}
            if sn:
                scores, mutated = head.apply(variables, feats, train=True, mutable=["sn"])
            else:
                scores, mutated = head.apply(variables, feats, train=True), {}
            s_real, s_fake = scores[:n_r], scores[n_r:]
            loss = D.wgan_critic_loss(s_real, s_fake)
            if r1_gamma > 0.0:
                # R1: penalise ||∇_input D(real)||^2. Input is continuous (pooled hidden / features),
                # so this is well-defined regardless of the discrete token sampler upstream.
                def _score_sum(x):
                    return jnp.sum(head.apply({"params": p, **sn}, x, train=False))
                gx = jax.grad(_score_sum)(real_feats)          # [n_r, feat_dim] or [n_r, T, F_in]
                # sum ||∇_x D||^2 over ALL input dims per example, mean over the batch (ndim-agnostic:
                # identical to the 2-D vector-critic path; correct for the 3-D learned sequence critic).
                r1 = jnp.mean(jnp.sum(gx * gx, axis=tuple(range(1, gx.ndim))))
                loss = loss + 0.5 * r1_gamma * r1
            return loss, (mutated, s_real, s_fake)

        (loss, (mutated, s_real, s_fake)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # Keep the 'sn' collection keyed so the next apply still finds the power-iteration var u.
        new_sn = {"sn": mutated["sn"]} if (isinstance(mutated, dict) and "sn" in mutated) else sn
        # The Lever-3 ENSEMBLE critic returns per-member scores [n, K] under train=True (so the shared
        # WGAN loss = mean of independent per-member losses); collapse the member axis to [n] for the
        # monitor (roc_auc / critic_separation are 1-D). No-op for vector & single-critic paths ([n]).
        if s_real.ndim > 1:
            s_real = jnp.mean(s_real, axis=-1)
            s_fake = jnp.mean(s_fake, axis=-1)
        return params, new_sn, opt_state, loss, s_real, s_fake
    return d_step


# ----------------------------------------------------------------------------------------
# The alternating loop (GPU; --run on the GH200).
# ----------------------------------------------------------------------------------------
def train(hs, args):
    from ..tests.s3_es_rollout import _prep_real_batch        # lazy: imports lob.inference_no_errcorr (compute)

    cfg = CFG
    N = args.n_pop
    assert N % 2 == 0, f"n_pop must be even for antithetic ±σ pairs (got {N})"
    assert args.n_d_steps >= 1, "n_d_steps must be >= 1 (the monitor reads the D-step's critic scores)"
    sigma = args.sigma
    # EGGROLL lr rule lr = lr_scale·σ²·√N is the setting for large N (S5). For the short
    # smoke at small N that step is ~0 (no visible movement), so `--lr` overrides it; the smoke wrapper
    # passes lr=1e-3 (finite, visibly moves the head) with sgd.
    lr = args.lr if args.lr is not None else cfg.eggroll.lr_scale * sigma ** 2 * math.sqrt(N)
    solver = _SOLVERS[args.solver]
    print(f"[S4] config: n_pop={N} n_cond={args.n_cond} n_gen={args.n_gen} sigma={sigma} lr={lr:.3e} "
          f"solver={args.solver} kl_coef={args.kl_coef} n_d_steps={args.n_d_steps} "
          f"n_g_steps={args.n_g_steps}", flush=True)

    # --- one real batch: rollout scaffolding + critic windows (single dataset-load path) ---
    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, N,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed,
                         wide_levels=args.wide_levels)
    inf = P["inf"]
    model, batchnorm, encoder = P["model"], P["batchnorm"], P["encoder"]
    train_state = P["train_state"]                    # FROZEN backbone; decoder slot swapped per call
    backbone = D.make_backbone(P["model_cls"])
    bb_params = P["bb_params"]                         # frozen encoders (decoder leaf unused by features)
    ctx_tokens, ctx_book = P["ctx_tokens"], P["b_seq_inp"]
    pooling, chunk = cfg.disc.pooling, args.feat_chunk

    # --- generator head = evolving reference leaves; noiser built once around the start head ---
    kernel0, bias0 = P["kernel"], P["bias"]
    fnp, npar, es_map, esk, leaves = make_decoder_noiser(
        hs, kernel0, bias0, sigma=sigma, lr=lr, rank=cfg.eggroll.rank,
        group_size=0, solver=solver, seed=args.seed)
    gen_es = make_generate_es_batched()               # decoder BATCHED (the real ES path)

    def es_args(pop):
        return (pop, P["sim_init"], train_state, model, batchnorm, encoder,
                P["sample_top_n"], P["tick_size"], P["m_seq_inp"], P["b_seq_inp"], args.n_gen,
                P["sim_states_init"], P["rngs"], P["init_hidden_batched"], True,
                P["init_time_batched"], P["valid_mask_array"])

    # --- real critic features (constant across the smoke; one fixed context batch) ---
    real_feats = F.window_features(backbone, bb_params, ctx_tokens, P["real_cont_tokens"],
                                   ctx_book, P["real_cont_book"], pooling=pooling, chunk=chunk)
    if not bool(jnp.all(jnp.isfinite(real_feats))):
        raise RuntimeError("non-finite real backbone features — check the chunked feature path")
    head, hparams, sn, tx, opt_state = make_critic(cfg, int(real_feats.shape[-1]), args.seed)
    d_step = make_d_step(head, tx)
    time_mask = F.build_time_mask(args.n_gen) if args.kl_coef > 0 else None

    history, diverged = [], False
    for step in range(1, args.n_g_steps + 1):
        # (1) ONE population rollout (perturbed heads around the current reference).
        it = population_iterinfo(N, step - 1)
        pop = build_decoder_population(hs, fnp, npar, leaves, esk, it)
        g_out = gen_es(*es_args(pop))
        fct, fcb = rollout_to_cont(g_out[3], g_out[4], args.n_gen)
        num_errors = g_out[2].astype(jnp.float32)                          # [N] (top-l2 unchanged; monitor only)
        fake_feats = F.window_features(backbone, bb_params, ctx_tokens, fct, ctx_book, fcb,
                                       pooling=pooling, chunk=chunk)        # [N, d_model]

        # (2) D-step(s): sharpen the critic on real vs the current fakes (backprop, head only).
        for _ in range(args.n_d_steps):
            hparams, sn, opt_state, dloss, s_real, s_fake = d_step(hparams, sn, opt_state,
                                                                   real_feats, fake_feats)
        head_vars = {"params": hparams, **sn}

        # (3) G-step (EGGROLL): score the SAME fakes with the freshly-updated critic; ascend fitness.
        scores = head.apply(head_vars, fake_feats, train=False)            # [N]
        if args.kl_coef > 0:
            hid = F.cont_hidden(backbone, bb_params, ctx_tokens, fct, ctx_book, fcb, args.n_cond, chunk=chunk)
            kl = F.head_kl_penalty(hid, leaves["kernel"], leaves["bias"], pop["kernel"], pop["bias"],
                                   time_mask=time_mask)                     # [N]
        else:
            kl = jnp.zeros_like(scores)
        raw_fit = F.raw_fitness(scores, kl, args.kl_coef)
        fit = hs.EggRoll.convert_fitnesses(fnp, npar, raw_fit)
        npar, leaves = hs.EggRoll.do_updates(fnp, npar, leaves, esk, fit, it, es_map)

        # --- monitor / divergence guard ---
        if step % args.eval_every == 0 or step == 1 or step == args.n_g_steps:
            sep = float(D.critic_separation(s_real, s_fake))
            auc = float(D.roc_auc(s_real, s_fake))
            mean_score = float(jnp.mean(scores))
            mean_kl = float(jnp.mean(kl))
            mean_ne = float(jnp.mean(num_errors))
            dk = float(jnp.max(jnp.abs(leaves["kernel"] - kernel0)))
            finite = bool(jnp.isfinite(leaves["kernel"]).all() and jnp.isfinite(scores).all()
                          and np.isfinite(float(dloss)))
            history.append(dict(step=step, d_loss=float(dloss), separation=sep, auc=auc,
                                g_mean_score=mean_score, mean_kl=mean_kl, mean_num_errors=mean_ne,
                                dkernel=dk, finite=finite))
            print(f"[S4] step {step:4d}  D_loss {float(dloss):+.4f}  sep {sep:+.4f}  auc {auc:.3f}  | "
                  f"G mean_score {mean_score:+.4f}  KL {mean_kl:.4f}  num_err {mean_ne:.1f}  "
                  f"||Δhead|| {dk:.2e}  finite={finite}", flush=True)
            if not finite:
                print("[S4] *** NON-FINITE state — divergence guard tripped, stopping "
                      "(lower lr/σ per the Phase-3 lesson) ***", flush=True)
                diverged = True
                break

    # --- save (TMPDIR; rsync to Lustre handled by the wrapper on success) ---
    os.makedirs(args.out_dir, exist_ok=True)
    head_vars = {"params": hparams, **sn}
    gen_path = os.path.join(args.out_dir, "s4_generator_head.msgpack")
    critic_path = os.path.join(args.out_dir, "s4_critic_head.msgpack")
    with open(gen_path, "wb") as f:
        f.write(serialization.to_bytes({"kernel": leaves["kernel"], "bias": leaves["bias"]}))
    with open(critic_path, "wb") as f:
        f.write(serialization.to_bytes(head_vars))
    with open(os.path.join(args.out_dir, "latest_checkpoint.json"), "w") as f:
        json.dump(dict(stage="S4", diverged=diverged, n_g_steps=args.n_g_steps,
                       generator_head=os.path.basename(gen_path),
                       critic_head=os.path.basename(critic_path),
                       sigma=sigma, lr=lr, solver=args.solver, kl_coef=args.kl_coef,
                       history=history), f, indent=2)
    print(f"[S4] {'DIVERGED' if diverged else 'done'} — {len(history)} logged steps. "
          f"heads -> {args.out_dir}", flush=True)
    return 1 if diverged else 0


# ----------------------------------------------------------------------------------------
# CPU-safe glue checks (login node; no model, no rollout).
# ----------------------------------------------------------------------------------------
def cpu_checks(hs, seed=0):
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("\n[S4] CPU checks — rollout trim / critic step / fitness round-trip", flush=True)
    k = jax.random.key(seed)

    # (C1) rollout_to_cont trims msgs_tokens + b_finals to the critic continuation shapes.
    N, n_gen, V503 = 4, 3, 503
    msgs_tokens = jnp.arange(N * (n_gen + 2) * F.MSG_LEN).reshape(N, (n_gen + 2) * F.MSG_LEN)
    b_finals = jnp.zeros((N, n_gen + 2, V503))
    fct, fcb = rollout_to_cont(msgs_tokens, b_finals, n_gen)
    chk("(C1) rollout_to_cont shapes", fct.shape == (N, n_gen * F.MSG_LEN) and fcb.shape == (N, n_gen, V503),
        f"fct={tuple(fct.shape)} fcb={tuple(fcb.shape)}")

    # (C2) WGAN critic d_step on synthetic features: separates two offset clouds + stays finite.
    d = 16
    real = jax.random.normal(jax.random.fold_in(k, 1), (N, d)) + 1.5
    fake = jax.random.normal(jax.random.fold_in(k, 2), (N, d)) - 1.5
    head, hparams, sn, tx, opt_state = make_critic(CFG, d, seed)
    d_step = make_d_step(head, tx)
    sep0 = None
    for i in range(150):
        hparams, sn, opt_state, dloss, sr, sf = d_step(hparams, sn, opt_state, real, fake)
        if i == 0:
            sep0 = float(D.critic_separation(sr, sf))
    sepN = float(D.critic_separation(sr, sf))
    finite = np.isfinite(float(dloss)) and bool(jnp.isfinite(hparams["SNDense_0"]["kernel"]).all())
    chk("(C2) critic separation grows & finite", sepN > sep0 and finite,
        f"sep {sep0:+.3f} -> {sepN:+.3f} finite={finite}")

    # (C3) fitness round-trip: build σ>0 population, critic-score-shaped raw fitness ->
    #      convert_fitnesses -> do_updates moves the head & stays finite (the exact G-step calls).
    kernel = jax.random.normal(jax.random.fold_in(k, 4), (12, 10))    # tiny (in,out) stand-in head
    bias = jax.random.normal(jax.random.fold_in(k, 5), (10,))
    fnp, npar, es_map, esk, leaves = make_decoder_noiser(
        hs, kernel, bias, sigma=1e-2, lr=1e-3, rank=CFG.eggroll.rank, group_size=0, solver=optax.sgd, seed=seed)
    it = population_iterinfo(N, 0)
    pop = build_decoder_population(hs, fnp, npar, leaves, esk, it)
    chk("(C3a) population head shapes", pop["kernel"].shape == (N, 12, 10) and pop["bias"].shape == (N, 10),
        f"k={tuple(pop['kernel'].shape)} b={tuple(pop['bias'].shape)}")
    raw_fit = jax.random.normal(jax.random.fold_in(k, 6), (N,))    # stands in for critic scores
    fit = hs.EggRoll.convert_fitnesses(fnp, npar, raw_fit)
    chk("(C3b) convert_fitnesses zero-mean unit-ish", abs(float(jnp.mean(fit))) < 1e-4,
        f"mean={float(jnp.mean(fit)):.2e}")
    npar2, leaves2 = hs.EggRoll.do_updates(fnp, npar, leaves, esk, fit, it, es_map)
    dk = float(jnp.max(jnp.abs(leaves2["kernel"] - kernel)))
    chk("(C3c) do_updates moves head & finite", dk > 0.0 and bool(jnp.isfinite(leaves2["kernel"]).all()),
        f"||Δ||={dk:.3e}")

    print("\n[S4] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


def main():
    ap = argparse.ArgumentParser(description="S4: full alternating EGGROLL-GAN loop + smoke")
    ap.add_argument("--run", action="store_true", help="run the GPU smoke (GH200); else CPU glue checks only")
    ap.add_argument("--data_dir", default=None, help="node-local GOOG dir (SquashFS-staged)")
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "s4_out"))
    ap.add_argument("--n_pop", type=int, default=8)
    ap.add_argument("--n_cond", type=int, default=64)
    ap.add_argument("--n_gen", type=int, default=16)
    ap.add_argument("--wide_levels", type=int, default=10)
    ap.add_argument("--n_g_steps", type=int, default=20)
    ap.add_argument("--n_d_steps", type=int, default=5, help="critic-head gradient steps per G-step")
    ap.add_argument("--sigma", type=float, default=CFG.eggroll.sigma)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="ES lr; None -> principled lr_scale·σ²·√N (S5). 1e-3 (S3-proven) for the smoke.")
    ap.add_argument("--solver", choices=list(_SOLVERS), default="sgd")
    ap.add_argument("--kl_coef", type=float, default=0.0, help="λ for the KL trust region (0 = pure critic)")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--feat_chunk", type=int, default=4, help="rows per backbone forward slice")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hs = import_hyperscalees()
    fails = cpu_checks(hs, seed=args.seed)
    if fails:
        print(f"[S4] CPU checks FAILED: {fails}")
        sys.exit(1)

    if args.run:
        if not args.data_dir:
            print("[S4] --run requires --data_dir (node-local GOOG dir)"); sys.exit(2)
        sys.exit(train(hs, args))
    else:
        print("\n[S4] (skipped GPU smoke — pass --run on the GH200 node)")
        print("[S4] CPU glue checks PASSED (GPU smoke pending).")


if __name__ == "__main__":
    main()
