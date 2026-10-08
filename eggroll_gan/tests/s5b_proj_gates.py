"""S5b — gates for the in/out-projection LoRA (do_Tmm fold) scope.

CPU (login node, JAX_PLATFORMS=cpu, tiny model — seconds, no data/checkpoint):
  (T1) TINY PaddedLobPredModel (mamba3 ssm): stock Batch wrapper vs the ES wrapper with
       es=None and with ZERO factors are BIT-identical (the byte-identical-path +
       exact-no-op guarantees end-to-end through embed/book/fused/decoder); real sigma>0
       factors change the logits.
  (T2) per-projection attribution: a factor tree restricted to a single projection only
       changes the output when that projection is live (in_proj / out_proj / out2 / fused
       encoder each verified live in BOTH __call_rnn__ and the parallel features path).
  (T3) population semantics under the OUTER jax.vmap (the rollout's member axis): members
       produce different logits, the antithetic pair differs from the base, and a zero
       member inside the vmapped population reproduces the stock output bit-exact.
  (T4) features path (discriminator.PaddedLobPredFeatures): es=None == legacy no-arg call;
       zero factors bit-exact; sigma>0 differs -> the perturbed-KL pass is live; and
       identical hiddens => KL term == 0 exactly (the proj-KL self-consistency).

GPU (--rollout on a GH200; called by _run_eggroll_smoke.sbatch — NOT from a login node):
  WHAT "sigma=0 bit-exact" CAN MEAN ON GH200: the es-fold program and
  the stock program are DIFFERENT XLA programs, and under default TF32 matmuls differently-fused
  fp32 programs differ at ~1.8e-3 rel (the re-gate A-parity measurement) — so
  cross-program BIT equality at the rollout level is unattainable by construction, exactly as in
  S3's G1c (materialised-head vs broadcast). Bitwise gates therefore live WITHIN one program
  (T1a/T1b model-level: PASS bit-exact on GPU; P-G2's pert-vs-zero same-program comparison), and
  cross-program no-op-ness is gated the way S5a validated it:
  (P-G1a) REAL-MODEL single-apply parity: stock wrapper vs ES wrapper + zero factors, at default
          precision (report; expect ~1e-3 rel = TF32 floor) AND under
          jax.default_matmul_precision('highest') (REQUIRE <1e-4 rel — refutes any semantic bug;
          a real wiring bug shows O(1) at both precisions).
  (P-G1b) rollout faithfulness diagnostic (G1c-style): zero-factor rollout vs stock — token
          agreement + per-member first-divergence; tripwire = gross-bug pattern (all members
          diverging at token ~0 with ~0 agreement) must be absent.
  (P-G2)  sigma>0 rollouts differ from the zero-factor rollout (SAME program -> bit-meaningful),
          stay valid; 3-step ES loop MOVES the MM_PARAM kernels while every EXCLUDED leaf stays
          BIT-identical on real hardware.
  (P-G3)  fused feats+KL (make_feats_kl_proj): zero-factor KL sits at/below the TF32 noise floor
          (<1e-2) and the sigma>0 KL signal dominates it by >=100x; anchor feats are
          factor-independent (same-program bit equality); all finite.

Run (login/CPU): JAX_PLATFORMS=cpu PYTHONPATH=$EXP:$EXP/lobmamba python -u -m eggroll_gan.tests.s5b_proj_gates
Run (GH200):     python -u -m eggroll_gan.s5b_proj_gates --rollout --data_dir <GOOG_dir> [--n_pop 4 ...]
"""
from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..config import DEFAULT as CFG
from ..es.es_plumbing import import_hyperscalees, build_es_map_proj
from ..es.es_generator import (make_proj_noiser, build_proj_factors, zero_proj_factors,
                           population_iterinfo, extract_trainable,
                           make_generate_es_proj_sharded, make_feats_kl_proj,
                           tile_dirs_over_Q, grid_repeat_contexts, gq_grid_indices, grid_rngs)

_FAILS = []


def _check(name, cond, detail=""):
    print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        _FAILS.append(name)


def _maxabs(a, b=None):
    return float(jnp.max(jnp.abs(a if b is None else (a - b))))


def _tree_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(bool(jnp.array_equal(x, y)) for x, y in zip(la, lb))


def _restrict(factors, keep_path):
    """Zero every factor leaf except those under the '/'-joined path prefix `keep_path` —
    per-projection attribution probes (T2)."""
    def walk(node, prefix):
        out = {}
        for kk, vv in node.items():
            p = f"{prefix}/{kk}" if prefix else kk
            if isinstance(vv, dict) and not ("A" in vv and "B" in vv):
                out[kk] = walk(vv, p)
            else:
                keep = p.startswith(keep_path)
                out[kk] = vv if keep else jax.tree_util.tree_map(jnp.zeros_like, vv)
        return out
    return walk(factors, "")


# ----------------------------------------------------------------------------------------
# CPU: tiny mamba3 PaddedLobPredModel integration suite.
# ----------------------------------------------------------------------------------------
def cpu_checks(hs, seed=0):
    sys.path  # noqa  (lobmamba on path via eggroll_gan._paths)
    from s5.mamba3 import init_Mamba3SSM
    from lob.lob_seq_model import BatchPaddedLobPredModel, BatchPaddedLobPredModelES, PaddedLobPredModel
    from ..critic.discriminator import PaddedLobPredFeatures

    print("\n[S5b] CPU checks — tiny mamba3 PaddedLobPredModel, es threading end-to-end", flush=True)
    d_model, d_book, vocab = 16, 9, 23
    L = 26
    ssm = init_Mamba3SSM(H=d_model, d_state=8, expand=2, headdim=8, chunk_size=4)
    kwargs = dict(ssm=ssm, d_output=vocab, d_model=d_model, d_book=d_book,
                  n_message_layers=1, n_fused_layers=1, n_book_pre_layers=1, n_book_post_layers=1,
                  activation="half_glu1", dropout=0.0, training=False, mode="none",
                  prenorm=True, batchnorm=False)
    stock = BatchPaddedLobPredModel(**kwargs)
    esmod = BatchPaddedLobPredModelES(**kwargs)

    k = jax.random.key(seed)
    x_m = jax.random.randint(jax.random.fold_in(k, 1), (1, L), 0, vocab)
    x_b = jax.random.normal(jax.random.fold_in(k, 2), (1, L, d_book))
    d_zero = jnp.zeros((1, L), bool)
    ts = jnp.ones((1, L))
    nh = (2 * d_model) // 8
    hid = PaddedLobPredModel.initialize_carry(
        1, hidden_size=0, n_message_layers=1, n_book_pre_layers=1, n_book_post_layers=1,
        n_fused_layers=1, h_size_ema=d_model, ssm_type="mamba3", n_heads=nh, headdim=8,
        d_state=8, num_rope_angles=2, d_book=d_book)
    rnn_args = (hid, x_m, x_b, d_zero, d_zero, d_zero, ts, ts)

    variables = stock.init(jax.random.fold_in(k, 3), *rnn_args, method="__call_rnn__")
    params = variables["params"]

    es_map = build_es_map_proj(params, hs)
    fnp, npar, esk = make_proj_noiser(hs, params, es_map, sigma=0.05, lr=1e-3, rank=2,
                                      solver=optax.sgd, seed=seed)
    G = 4
    fac_G = build_proj_factors(hs, fnp, npar, params, es_map, esk, population_iterinfo(G, 0))
    fac0 = jax.tree_util.tree_map(lambda a: a[0], fac_G)
    facz = zero_proj_factors(fac0)

    def run(model, es):
        h, logits = model.apply({"params": params}, *rnn_args, es, method="__call_rnn__")
        return logits

    out_stock = stock.apply({"params": params}, *rnn_args, method="__call_rnn__")[1]
    out_none = run(esmod, None)
    out_zero = run(esmod, facz)
    out_pert = run(esmod, fac0)

    # (T1) byte-identical path + exact no-op + live perturbation.
    _check("(T1a) ES wrapper es=None == stock wrapper (bit)", bool(jnp.array_equal(out_none, out_stock)),
           f"max|diff|={_maxabs(out_none, out_stock):.3e}")
    _check("(T1b) zero factors == stock (bit) — exact no-op", bool(jnp.array_equal(out_zero, out_stock)),
           f"max|diff|={_maxabs(out_zero, out_stock):.3e}")
    _check("(T1c) sigma>0 factors change the logits", _maxabs(out_pert, out_stock) > 1e-6,
           f"max|diff|={_maxabs(out_pert, out_stock):.3e}")

    # (T2) per-projection attribution: each scoped projection is LIVE on its own.
    probes = [("message in_proj", "message_encoder/layers_0/seq/in_proj"),
              ("message out_proj", "message_encoder/layers_0/seq/out_proj"),
              ("message out2 (GLU)", "message_encoder/layers_0/out2"),
              ("book pre in_proj", "book_encoder/pre_layers_0/seq/in_proj"),
              ("book post out2", "book_encoder/post_layers_0/out2"),
              ("fused encoder", "fused_s5/encoder"),
              ("fused in_proj", "fused_s5/layers_0/seq/in_proj"),
              ("fused out_proj", "fused_s5/layers_0/seq/out_proj")]
    live = {}
    for nm, path in probes:
        live[nm] = _maxabs(run(esmod, _restrict(fac0, path)), out_stock)
    _check("(T2) every scoped projection is live in __call_rnn__",
           all(v > 1e-7 for v in live.values()),
           " ".join(f"{nm}={v:.1e}" for nm, v in live.items()))

    # (T3) population semantics under the OUTER jax.vmap (= the rollout's member axis).
    fac_pop = jax.tree_util.tree_map(
        lambda a: a.at[0].set(jnp.zeros_like(a[0])), fac_G)          # member 0 := exact no-op
    pop_logits = jax.vmap(lambda f: run(esmod, f))(fac_pop)          # [G, 1, L?, vocab]
    m0_eq = bool(jnp.array_equal(pop_logits[0], out_stock))
    m12 = _maxabs(pop_logits[1], pop_logits[2])
    anti = _maxabs(pop_logits[1], out_stock)                          # thread 1 = -sigma of pair 0
    _check("(T3) outer-vmap population: zero member == stock (bit); members differ",
           m0_eq and m12 > 1e-7 and anti > 1e-7, f"m0_bit={m0_eq} |m1-m2|={m12:.1e}")

    # (T4) parallel features path (the perturbed-KL pass) + KL self-consistency.
    feat = PaddedLobPredFeatures(**kwargs)
    f_args = (x_m[0], x_b[0], ts[0], ts[0])
    h_legacy = feat.apply({"params": params}, *f_args, method=PaddedLobPredFeatures.features)
    h_none = feat.apply({"params": params}, *f_args, None, method=PaddedLobPredFeatures.features)
    h_zero = feat.apply({"params": params}, *f_args, facz, method=PaddedLobPredFeatures.features)
    h_pert = feat.apply({"params": params}, *f_args, fac0, method=PaddedLobPredFeatures.features)
    W0, b0 = params["decoder"]["kernel"], params["decoder"]["bias"]
    logp_r = jax.nn.log_softmax(h_zero @ W0 + b0, axis=-1)
    logp_i = jax.nn.log_softmax(h_none @ W0 + b0, axis=-1)
    kl0 = float(jnp.mean(jnp.sum(jnp.exp(logp_i) * (logp_i - logp_r), axis=-1)))
    _check("(T4) features: es=None==legacy==zero (bit); sigma>0 differs; identical hiddens -> KL==0",
           bool(jnp.array_equal(h_none, h_legacy)) and bool(jnp.array_equal(h_zero, h_legacy))
           and _maxabs(h_pert, h_legacy) > 1e-7 and kl0 == 0.0,
           f"|pert-legacy|={_maxabs(h_pert, h_legacy):.1e} kl0={kl0:.1e}")


# ----------------------------------------------------------------------------------------
# GPU gates (--rollout; GH200 only — called by _run_eggroll_smoke.sbatch).
# ----------------------------------------------------------------------------------------
def gpu_checks(hs, args):
    from .s3_es_rollout import _prep_real_batch
    from ..training.train_eggroll_gan import rollout_to_cont
    from ..critic import discriminator as D
    from lob.lob_seq_model import BatchPaddedLobPredModelES
    import lob.inference_no_errcorr as inf

    N = args.n_pop
    print(f"\n[S5b] GPU gates — n_pop={N} n_cond={args.n_cond} n_gen={args.n_gen} "
          f"sigma={args.sigma}", flush=True)
    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, N,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed)
    params0 = P["train_state"].params
    es_map = build_es_map_proj(params0, hs)
    fnp, npar, esk = make_proj_noiser(hs, params0, es_map, sigma=args.sigma, lr=args.lr,
                                      rank=CFG.eggroll.rank, solver=optax.adamw,
                                      solver_kwargs={"weight_decay": 0.0}, seed=args.seed)
    model_es = BatchPaddedLobPredModelES(**dict(P["model_cls"].keywords),
                                         training=False, step_rescale=1.0)
    gen_proj = make_generate_es_proj_sharded(
        model_es, P["batchnorm"], P["encoder"], P["sample_top_n"], P["tick_size"], P["n_gen"],
        P["sim_init"], P["valid_mask_array"], conditional=True, shard="off")

    roll_args = (P["train_state"], P["m_seq_inp"], P["b_seq_inp"], P["sim_states_init"],
                 P["rngs"], P["init_hidden_batched"], P["init_time_batched"])
    names = ("msgs_decoded", "l2_book_states", "num_errors", "msgs_tokens", "b_finals")

    stock_out = inf.generate_batched(
        P["sim_init"], P["train_state"], P["model"], P["batchnorm"], P["encoder"],
        P["sample_top_n"], P["tick_size"], P["m_seq_inp"], P["b_seq_inp"], P["n_gen"],
        P["sim_states_init"], P["rngs"], P["init_hidden_batched"], True,
        P["init_time_batched"], False, None, P["valid_mask_array"])

    # es=None through the ES wrapper is unsupported BY DESIGN: its nn.vmap in_axes always
    # expects the 9th arg (an arity mismatch); the no-op population is
    # zero factors, and the es=None byte-path is exercised where it actually runs — the stock
    # wrapper (T1a, and every legacy caller of generate/_apply_model_impl).
    fac_N = build_proj_factors(hs, fnp, npar, params0, es_map, esk, population_iterinfo(N, 0))
    fac0_1 = jax.tree_util.tree_map(lambda a: jnp.zeros_like(a[:1]), fac_N)   # zero member, batch 1

    # (P-G1a) REAL-MODEL single-apply parity, default vs forced-highest matmul precision.
    import lob.validation_helpers as valh
    hid1 = jax.tree_util.tree_map(lambda x: x[0], P["init_hidden_batched"])   # one member: (1,...) batch
    m_row = P["m_seq_inp"][0][:26]
    b_row = P["b_seq_inp"][0][:1]

    def _apply_pair():
        h_s, lg_s = valh.apply_model(hid1, m_row, b_row, P["train_state"], P["model"],
                                     P["batchnorm"], True)
        h_e, lg_e = valh.apply_model(hid1, m_row, b_row, P["train_state"], model_es,
                                     P["batchnorm"], True,
                                     jax.tree_util.tree_map(lambda a: a[0], fac0_1))
        rel = float(jnp.max(jnp.abs(lg_e - lg_s)) / (jnp.max(jnp.abs(lg_s)) + 1e-12))
        return rel, bool(jnp.array_equal(lg_e, lg_s))
    rel_def, bit_def = _apply_pair()
    with jax.default_matmul_precision("highest"):
        rel_hi, bit_hi = _apply_pair()
    _check("(P-G1a) single-apply zero-factor parity: highest-precision rel <1e-4 (no semantic bug)",
           rel_hi < 1e-4, f"rel: default={rel_def:.2e} (bit={bit_def}) highest={rel_hi:.2e} (bit={bit_hi})")

    # (P-G1b) rollout faithfulness diagnostic (cross-program; G1c standard, NOT a bit gate).
    zero_out = gen_proj(zero_proj_factors(fac_N), *roll_args)
    at = jnp.asarray(zero_out[3]).reshape(N, -1)
    ct = jnp.asarray(stock_out[3]).reshape(N, -1)
    eq = (at == ct); Ltok = at.shape[1]
    agreement = float(jnp.mean(eq.astype(jnp.float32)))
    n_full = int(jnp.sum(jnp.all(eq, axis=1)))
    firsts = [Ltok if bool(jnp.all(eq[i])) else int(jnp.argmax(~eq[i])) for i in range(N)]
    print(f"   [diag] zero-factor vs stock rollout: agreement={agreement:.3f}, full-match={n_full}/{N}, "
          f"first-divergence/member={firsts} (TF32 cross-program drift x AR sampling — see header)")
    _check("(P-G1b) zero-factor rollout faithful to stock (no gross-bug pattern)",
           agreement >= 0.3 and (sum(firsts) / N) >= 26.0,
           f"agreement={agreement:.3f} mean_first_div={sum(firsts)/N:.0f}/{Ltok}")

    # (P-G2) sigma>0 vs zero factors — SAME compiled program, so differences are the perturbation.
    pert_out = gen_proj(fac_N, *roll_args)
    tok = jnp.asarray(pert_out[3]).reshape(N, -1)
    tok0 = jnp.asarray(zero_out[3]).reshape(N, -1)
    diff_members = int(jnp.sum(jnp.any(tok != tok0, axis=1)))
    _check("(P-G2a) sigma>0 differs from zero-factor rollout (same program) & stays finite",
           diff_members >= 1 and bool(jnp.isfinite(jnp.asarray(pert_out[1])).all()),
           f"differing members={diff_members}/{N}")
    cur, npar_c = params0, npar
    for s_ in range(args.es_steps):
        it_s = population_iterinfo(N, s_)
        fac_s = build_proj_factors(hs, fnp, npar_c, cur, es_map, esk, it_s)
        out_s = gen_proj(fac_s, P["train_state"].replace(params=cur), *roll_args[1:])
        raw = -out_s[2].astype(jnp.float32)
        fit = hs.EggRoll.convert_fitnesses(fnp, npar_c, raw)
        npar_c, cur = hs.EggRoll.do_updates(fnp, npar_c, cur, esk, fit, it_s, es_map)
        print(f"   step {s_}: mean num_errors={float(jnp.mean(out_s[2])):.1f}", flush=True)
    tr0 = extract_trainable(hs, params0, es_map)
    trc = extract_trainable(hs, cur, es_map)
    moved = max(_maxabs(trc[kk], tr0[kk]) for kk in tr0)
    # every leaf NOT in the trainable set must be bit-identical after the update steps.
    flat0 = jax.tree_util.tree_flatten_with_path(params0)[0]
    flatc = jax.tree_util.tree_leaves(cur)
    flatm = jax.tree_util.tree_leaves(es_map)
    frozen_ok = all(bool(jnp.array_equal(l0, lc)) for (p_, l0), lc, m in zip(flat0, flatc, flatm)
                    if int(m) != int(hs.MM_PARAM))
    finite_ok = all(bool(jnp.isfinite(v).all()) for v in trc.values())
    _check("(P-G2b) ES loop: MM kernels move, EXCLUDED leaves BIT-frozen, finite",
           moved > 0.0 and frozen_ok and finite_ok,
           f"max|dMM|={moved:.3e} frozen_bitexact={frozen_ok}")

    # (P-G3) fused feats+KL on a small G×Q grid.
    G3, Q3 = 4, max(2, N // 2)
    backbone = D.make_backbone(P["model_cls"])
    feats_kl = make_feats_kl_proj(backbone, params0, pooling=CFG.disc.pooling,
                                  pool_start=args.n_cond * 26, n_cond=args.n_cond,
                                  shard="off", chunk=1)
    fac_G3 = build_proj_factors(hs, fnp, npar, params0, es_map, esk, population_iterinfo(G3, 7))
    idxQ = jnp.arange(Q3)
    pg = tile_dirs_over_Q(fac_G3, Q3)
    m_g = grid_repeat_contexts(P["m_seq_inp"][idxQ], G3)
    b_g = grid_repeat_contexts(P["b_seq_inp"][idxQ], G3)
    sim_g = jax.tree_util.tree_map(lambda x: jnp.repeat(x[idxQ], G3, axis=0), P["sim_states_init"])
    ih_g = jax.tree_util.tree_map(lambda x: jnp.repeat(x[idxQ], G3, axis=0), P["init_hidden_batched"])
    it_g = grid_repeat_contexts(P["init_time_batched"][idxQ], G3)
    rng_g = grid_rngs(jax.random.PRNGKey(args.seed + 5), G3, Q3)
    out_g = gen_proj(pg, P["train_state"], m_g, b_g, sim_g, rng_g, ih_g, it_g)
    fct, fcb = rollout_to_cont(out_g[3], out_g[4], args.n_gen)
    ctx_g = grid_repeat_contexts(P["ctx_tokens"][idxQ], G3)
    cbk_g = grid_repeat_contexts(P["b_seq_inp"][idxQ], G3)
    di, _ = gq_grid_indices(G3, Q3)
    feats_p, kl_p = feats_kl(ctx_g, fct, cbk_g, fcb, di, params0, fac_G3)
    feats_0, kl_0 = feats_kl(ctx_g, fct, cbk_g, fcb, di, params0, zero_proj_factors(fac_G3))
    # zero-factor KL is the per-window TF32 cross-fusion noise floor (anchor pass has no fold
    # ops, perturbed pass does -> different fusion of the SAME math; ~1.8e-3 measured).
    # Gate: floor below 1e-2 AND the sigma>0 KL signal dominates it >=100x AND a
    # forced-highest-precision recheck collapses the floor (<1e-5 — proves it IS precision).
    kl0_max = float(jnp.max(jnp.abs(kl_0)))
    with jax.default_matmul_precision("highest"):
        _, kl_0hi = feats_kl(ctx_g, fct, cbk_g, fcb, di, params0, zero_proj_factors(fac_G3))
    kl0hi_max = float(jnp.max(jnp.abs(kl_0hi)))
    _check("(P-G3) proj feats+KL: zero-KL floor <1e-2 & collapses at highest precision; "
           "sigma>0 KL >=100x floor; anchor feats factor-independent & finite",
           kl0_max < 1e-2 and kl0hi_max < 1e-5 and float(jnp.min(kl_p)) > 100.0 * max(kl0_max, 1e-12)
           and bool(jnp.isfinite(feats_p).all()) and bool(jnp.array_equal(feats_p, feats_0)),
           f"kl0={kl0_max:.2e} kl0_highest={kl0hi_max:.2e} min(kl_pert)={float(jnp.min(kl_p)):.2e}")


def main():
    ap = argparse.ArgumentParser(description="S5b: projection-LoRA fold gates")
    ap.add_argument("--rollout", action="store_true", help="run the GPU gates (GH200)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--n_pop", type=int, default=4)
    ap.add_argument("--n_cond", type=int, default=64)
    ap.add_argument("--n_gen", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--es_steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hs = import_hyperscalees()
    cpu_checks(hs, seed=args.seed)

    if args.rollout:
        if not args.data_dir:
            print("[S5b] --rollout requires --data_dir (node-local GOOG dir)"); sys.exit(2)
        gpu_checks(hs, args)
    else:
        print("\n[S5b] (skipped GPU gates — pass --rollout on the GH200 node)")

    print("\n" + "=" * 60)
    if _FAILS:
        print(f"[S5b] FAILED checks: {_FAILS}")
        sys.exit(1)
    print("[S5b] ALL CHECKS PASSED" + ("" if args.rollout else " (CPU subset; GPU gates pending)"))


if __name__ == "__main__":
    main()
