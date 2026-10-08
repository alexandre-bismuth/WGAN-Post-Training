"""S3 — validate the ES generator G-step.

CPU-safe checks (no model; run on login node):
  (E1/E2) σ=0 population materialises the head bit-exact (Wᵢ=W_ref, bᵢ=b_ref).
  (E3)    σ>0 members differ; (E4/E5) antithetic kernel & bias perturbations are exact mirrors.
  (E6)    tile_contexts replicates each context group_size times (CRN layout).
  (E7/E8) a do_updates round-trip (consistent with the materialised population) moves the decoder leaves.

GPU checks (`--rollout`; GH200 — real rollout = compute):
  (G1a) shared-head wrapper == stock `generate_batched` BIT-FOR-BIT — proves the `ts.replace(params=…)` +
        `generate()` plumbing reproduces stock exactly when the head is broadcast (not batched).
  (G1b) σ=0 ES rollout == W_ref-tiled rollout BIT-FOR-BIT — the EGGROLL materialisation is a true no-op at σ=0.
  (G1c) faithfulness diagnostic: the materialised (BATCHED) decoder matmul ≡ stock's BROADCAST matmul to
        ~1e-6 (S2), but discrete top-n AR sampling amplifies that into LATE, PARTIAL divergence for a
        minority of members — so it is NOT bit-for-bit vs the broadcast stock. We report token agreement +
        per-member first-divergence and assert a gross-bug tripwire (a real backbone/wiring bug would
        diverge ALL members at token ~0 with ~0 agreement). [Why full-rollout bit-for-bit vs stock is the
        WRONG metric here: a chaotic discrete sampler + a batched-vs-broadcast head differ by sampling, not math.]
  (G2)  a tiny end-to-end ES loop: population rollout -> fitness (-num_errors) -> convert_fitnesses ->
        do_updates -> the decoder head moves, rollouts stay finite/valid across steps.

The GPU batch prep mirrors the proven `generate_fakes.generate_pairs` setup (restore generator,
inference_no_errcorr.get_dataset on a node-local GOOG dir, OrderBook sim, one batch of N sequences).

Run (login/CPU):  JAX_PLATFORMS=cpu PYTHONPATH=<exp_root> python -u -m eggroll_gan.s3_es_rollout
Run (GH200):      PYTHONPATH=<exp_root>:<mamba_root> python -u -m eggroll_gan.s3_es_rollout \
                      --rollout --data_dir <GOOG_dir> --n_pop 8 --n_cond 64 --n_gen 16
"""
from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import optax

from ..config import DEFAULT as CFG
from ..es.es_plumbing import import_hyperscalees
from ..es.es_generator import (make_decoder_noiser, build_decoder_population, population_iterinfo,
                           tile_contexts, make_generate_es_batched)

_FAILS = []


def _check(name, cond, detail=""):
    print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        _FAILS.append(name)


def _maxabs(a, b=None):
    return float(jnp.max(jnp.abs(a if b is None else (a - b))))


# ----------------------------------------------------------------------------------------
# CPU-safe checks (no model).
# ----------------------------------------------------------------------------------------
def cpu_checks(hs, seed=0, n_pop=8):
    print("\n[S3] CPU checks — population materialisation / antithetic / tiling / do_updates", flush=True)
    assert n_pop % 2 == 0, f"n_pop must be even for antithetic pairs (got {n_pop})"
    k = jax.random.key(seed)
    kernel = jax.random.normal(jax.random.fold_in(k, 1), (32, 40))   # small stand-in for (1024,2112)
    bias = jax.random.normal(jax.random.fold_in(k, 2), (40,))
    it = population_iterinfo(n_pop, 0)

    # σ=0 -> head materialises to the reference, bit-exact.
    fnp0, np0, es_map, esk0, leaves0 = make_decoder_noiser(
        hs, kernel, bias, sigma=0.0, lr=1e-3, rank=CFG.eggroll.rank, solver=optax.sgd, seed=seed)
    pop0 = build_decoder_population(hs, fnp0, np0, leaves0, esk0, it)
    _check("(E1) sigma=0 kernel pop == W_ref", _maxabs(pop0["kernel"], kernel[None]) == 0.0,
           f"max|diff|={_maxabs(pop0['kernel'], kernel[None]):.3e}")
    _check("(E2) sigma=0 bias pop == b_ref", _maxabs(pop0["bias"], bias[None]) == 0.0,
           f"max|diff|={_maxabs(pop0['bias'], bias[None]):.3e}")

    # σ>0 -> real, antithetically-mirrored perturbation.
    fnp1, np1, _, esk1, leaves1 = make_decoder_noiser(
        hs, kernel, bias, sigma=0.1, lr=1e-3, rank=CFG.eggroll.rank, solver=optax.sgd, seed=seed)
    pop1 = build_decoder_population(hs, fnp1, np1, leaves1, esk1, it)
    dW = pop1["kernel"] - kernel[None]                                 # (N,in,out)
    per_member = jnp.max(jnp.abs(dW), axis=(1, 2))
    _check("(E3) sigma>0 members differ", float(jnp.min(per_member)) > 1e-6,
           f"min max|dW|={float(jnp.min(per_member)):.3e}")
    _check("(E4) antithetic kernel mirror", _maxabs(dW[0::2] + dW[1::2]) < 1e-5,
           f"max|W[2k]+W[2k+1]-2W_ref|={_maxabs(dW[0::2] + dW[1::2]):.3e}")
    db = pop1["bias"] - bias[None]
    _check("(E5) antithetic bias mirror", _maxabs(db[0::2] + db[1::2]) < 1e-5,
           f"max|b[2k]+b[2k+1]-2b_ref|={_maxabs(db[0::2] + db[1::2]):.3e}")

    # tile_contexts: each of 3 contexts repeated group_size=2 -> 6 members.
    ctx = jnp.arange(3 * 4).reshape(3, 4)
    tiled = tile_contexts(ctx, 2)
    ok = (tiled.shape == (6, 4)
          and bool(jnp.array_equal(tiled[0], tiled[1]))
          and bool(jnp.array_equal(tiled[2], ctx[1])))
    _check("(E6) tile_contexts CRN layout", ok, f"shape={tuple(tiled.shape)}")

    # do_updates round-trip (consistent with the materialised population) moves the head.
    fit = jnp.arange(n_pop, dtype=jnp.float32) - (n_pop - 1) / 2.0      # distinct -> nonzero grad
    _, new_leaves = hs.EggRoll.do_updates(fnp1, np1, leaves1, esk1, fit, it, es_map)
    dk = _maxabs(new_leaves["kernel"], kernel)
    dbi = _maxabs(new_leaves["bias"], bias)
    _check("(E7) do_updates moves kernel", dk > 0.0 and bool(jnp.isfinite(new_leaves["kernel"]).all()),
           f"max|delta|={dk:.3e}")
    _check("(E8) do_updates moves bias", dbi > 0.0 and bool(jnp.isfinite(new_leaves["bias"]).all()),
           f"max|delta|={dbi:.3e}")


# ----------------------------------------------------------------------------------------
# GPU rollout checks (--rollout). Batch prep mirrors generate_fakes.generate_pairs.
# ----------------------------------------------------------------------------------------
def _sample_windows(inf, ds, n_cond, n_gen, n, rk_idx, rk_rng, *, sim_init, tick_size,
                    n_vol_series=500, exclude_idx=(), include_idx=None):
    """Draw `n` windows from `ds` and build the rollout scaffolding + critic window arrays.

    Extracted from `_prep_real_batch` so the S5 trainer can REFRESH its real-context pool mid-run
    (anti-memorisation: a fixed pool of real windows lets the critic memorise specific examples and
    turns the fitness into "differ from these windows"). Key usage is delegated to the caller
    (`rk_idx` draws the window indices, `rk_rng` the per-window rollout keys), so `_prep_real_batch`'s
    startup draw stays BIT-IDENTICAL to the S3/S4-validated flow; refresh calls pass fresh fold_in
    keys. `exclude_idx` keeps refreshed TRAIN draws disjoint from the held-out eval windows."""
    from lob.encoding import Message_Tokenizer
    MSG_LEN = Message_Tokenizer.MSG_LEN
    seq_len_cond = n_cond * MSG_LEN

    # include_idx restricts the draw (e.g. to wide-book-snapshot-covered windows: the npz
    # loader silently falls back to a padded L10 init for uncovered seq indices).
    if include_idx is not None:
        import numpy as onp
        base = onp.asarray(include_idx, dtype=onp.int32)
    else:
        import numpy as onp
        base = onp.arange(len(ds), dtype=onp.int32)
    if len(exclude_idx):
        base = onp.setdiff1d(base, onp.asarray(list(exclude_idx), dtype=onp.int32))
    cand = jnp.asarray(base)
    idx = jax.random.choice(rk_idx, cand, shape=(n,), replace=False).tolist()
    m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[idx]
    m_seq = jnp.array(m_seq); b_seq_pv = jnp.array(b_seq_pv)
    msg_seq_raw = jnp.array(msg_seq_raw); book_l2_init = jnp.array(book_l2_init)

    b_seq = inf.transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)
    init_time_batched = b_seq_pv[:, 0, 1:3]
    m_seq_inp = m_seq[:, : seq_len_cond + 1]
    b_seq_inp = b_seq[:, : n_cond + 1]
    m_seq_raw_inp = msg_seq_raw[:, : n_cond]
    m_seq_raw_cont = msg_seq_raw[:, n_cond:n_cond + n_gen]   # REAL continuation (for the no-op baseline)
    sim_states_init = inf.get_sims_vmap(book_l2_init, m_seq_raw_inp, init_time_batched, sim_init)
    rngs = jax.random.split(rk_rng, n)

    # Critic windows (token-level), mirroring generate_fakes.generate_pairs EXACTLY so the GAN loop
    # scores real & fake on identically-shaped [context ; continuation] windows. ctx drops the +1 init
    # token; the real continuation starts at seq_len_cond+1 (the +1 init-token offset the rollout
    # conditions on); book is the 503-wide transform (b_seq), ctx incl. its initial state (b_seq_inp),
    # real cont = the next n_gen states.
    ctx_tokens = m_seq[:, : seq_len_cond]                                              # [n, n_cond*26]
    real_cont_tokens = m_seq[:, seq_len_cond + 1: seq_len_cond + 1 + n_gen * MSG_LEN]  # [n, n_gen*26]
    real_cont_book = b_seq[:, n_cond + 1: n_cond + 1 + n_gen]                          # [n, n_gen, 503]

    return dict(idx=idx, m_seq_inp=m_seq_inp, b_seq_inp=b_seq_inp, sim_states_init=sim_states_init,
                rngs=rngs, init_time_batched=init_time_batched, m_seq_raw_cont=m_seq_raw_cont,
                ctx_tokens=ctx_tokens, real_cont_tokens=real_cont_tokens, real_cont_book=real_cont_book)


def _apply_unseen_mask(ds, include_idx, npz_path, *, complement=False):
    """Restrict the window draw to slots that pretraining NEVER consumed.

    Manifest: unseen_manifest_v1.npz from Mamba3_GOOG_pretraining/tools/
    replay_unseen_manifest.py (see docs/reference/unseen_manifest.md). Grid
    contract verified here: ds windows are stride-(n_cond+n_gen) from message 0
    (randomize_offset off), so ds slot k of day d spans messages
    [k*L, (k+1)*L). Slot eligible iff that span intersects NO seen pretraining
    window [off_d + 500w, off_d + 500w + 500) with seen_bit[w]=1. Days missing
    from the manifest are a FATAL error (never silently train on seen data).
    Composes with the wide-book coverage restriction via intersection.

    complement=True inverts the selection: keep only slots overlapping >=1
    pretraining-SEEN window. Those slots are disjoint BY CONSTRUCTION from every
    window the post-training loader can draw (its draws require zero seen
    overlap), so an eval universe built this way never scores post-training
    data. Val-holdout days (role 1) are fully unseen and contribute no slots.
    """
    import os as _os
    import re as _re
    import numpy as onp

    man = onp.load(npz_path, allow_pickle=False)
    m_dates = [str(x) for x in man["dates"]]
    lut = {d: i for i, d in enumerate(m_dates)}
    roles, m_rows = man["roles"], man["rows"]
    offs, ws_, wc_ = man["offsets"], man["win_start"], man["win_count"]
    bits = onp.unpackbits(man["seen_bits_packed"])[: int(man["n_seen_bits"])]

    L = int(ds.n_messages)                      # GAN slot length (1000)
    assert int(onp.asarray(ds.seq_offsets).max(initial=0)) == 0, \
        "unseen mask requires randomize_offset=False (slot grid from msg 0)"

    keep, report = [], []
    for f_i, mf in enumerate(ds.message_files):
        m = _re.search(r"(\d{4}-\d{2}-\d{2})", _os.path.basename(mf))
        assert m, f"no date in {mf}"
        d = m.group(1)
        assert d in lut, (f"day {d} missing from unseen manifest {npz_path} — "
                          "refusing to train on possibly-seen data")
        j = lut[d]
        n_slots = int(ds._seqs_per_file[f_i])
        assert int(ds._num_rows_per_file[f_i]) == int(m_rows[j]), (
            f"{d}: ds rows {int(ds._num_rows_per_file[f_i])} != manifest rows "
            f"{int(m_rows[j])} — different file content, manifest invalid here")
        if n_slots == 0:
            report.append((d, 0, 0))
            continue
        if int(roles[j]) == 1:                  # val-holdout day: fully unseen
            elig_k = (onp.zeros(0, dtype=onp.int64) if complement
                      else onp.arange(n_slots, dtype=onp.int64))
        else:
            off = int(offs[j]); wc = int(wc_[j])
            seen = bits[int(ws_[j]): int(ws_[j]) + wc].astype(onp.int64)
            pre = onp.concatenate(([0], onp.cumsum(seen)))
            a = onp.arange(n_slots, dtype=onp.int64) * L
            b = a + L
            # pretraining windows overlapping [a,b): off+500w < b AND off+500w+500 > a
            w_lo = onp.maximum(0, -(-(a - off - 499) // 500))
            w_hi = onp.minimum(wc - 1, (b - 1 - off) // 500)
            n_seen_overlap = onp.where(
                w_hi >= w_lo, pre[onp.minimum(w_hi + 1, wc)] - pre[onp.maximum(w_lo, 0)], 0)
            keep_mask = (n_seen_overlap > 0) if complement else (n_seen_overlap == 0)
            elig_k = onp.flatnonzero(keep_mask).astype(onp.int64)
        keep.append(int(ds._seqs_cumsum[f_i]) + elig_k)
        report.append((d, n_slots, len(elig_k)))

    tag = "seen-compl" if complement else "unseen"
    unseen_ids = onp.concatenate(keep).astype(onp.int32) if keep else \
        onp.zeros(0, dtype=onp.int32)
    for d, tot, el in report:
        print(f"[{tag}] {d}: {el}/{tot} slots eligible", flush=True)
    if include_idx is not None:
        out = onp.intersect1d(onp.asarray(include_idx, dtype=onp.int32), unseen_ids)
    else:
        out = unseen_ids
    print(f"[{tag}] manifest {_os.path.basename(npz_path)}: "
          f"{len(unseen_ids)} {tag} slots; after coverage intersect: {len(out)} "
          f"(over {len(ds.message_files)} days)", flush=True)
    assert len(out) > 0, f"{tag} mask left zero eligible windows"
    return out


def _prep_real_batch(data_dir, n_cond, n_gen, n_pop, *, ckpt_dir, ckpt_step, seed=0,
                     wide_levels=10, n_vol_series=500, wide_book_dir=None):
    from ..data import checkpoint_utils as ck
    import lob.inference_no_errcorr as inf
    from lob import validation_helpers as valh
    from lob.encoding import Message_Tokenizer

    MSG_LEN = Message_Tokenizer.MSG_LEN
    sample_top_n = CFG.rollout.sample_top_n
    tick_size = CFG.rollout.tick_size
    assert n_pop % 1 == 0

    loaded = ck.load_pretrained_generator(ckpt_dir, ckpt_step, build_loaders=False)
    args = loaded["args"]
    train_state = loaded["train_state"]
    model = loaded["model_cls"](training=False, step_rescale=1.0)
    batchnorm = bool(getattr(args, "batchnorm", False))
    encoder = inf.Vocab().ENCODING
    seq_len_cond = n_cond * MSG_LEN

    m3_expand = getattr(args, "mamba3_expand", 2)
    m3_headdim = getattr(args, "mamba3_headdim", 64)
    m3_d_state = getattr(args, "mamba3_d_state", 128)
    d_inner = m3_expand * args.d_model
    m3_nh = max(1, d_inner // m3_headdim)
    num_rope_angles = int(m3_d_state * getattr(args, "mamba3_rope_fraction", 0.5)) // 2
    init_hidden = model.initialize_carry(
        1, hidden_size=0, ssm_type="mamba3",
        n_message_layers=args.n_message_layers, n_book_pre_layers=args.n_book_pre_layers,
        n_book_post_layers=args.n_book_post_layers, n_fused_layers=args.n_layers,
        h_size_ema=args.d_model, n_heads=m3_nh, headdim=m3_headdim, d_state=m3_d_state,
        num_rope_angles=num_rope_angles, d_book=getattr(args, "d_book", 503))
    init_hidden_batched = jax.tree_util.tree_map(lambda x: jnp.resize(x, (n_pop,) + x.shape), init_hidden)

    sim_nOrders = 2 * wide_levels + 50 if wide_levels <= 100 else wide_levels + 500
    sim_init = inf.OrderBook(cfg=inf.JAXLOB_Configuration(
        nOrders=sim_nOrders, book_depth=wide_levels,
        cancel_mode=inf.cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))
    valid_mask_array = valh.syntax_validation_matrix(block_start_tok=False)

    ds = inf.get_dataset(data_dir, n_cond, n_gen, test_split=0.0, wide_book_dir=wide_book_dir)
    # Sparse wide-book npz snapshots cover only specific seq indices (fixed eval windows);
    # uncovered indices silently fall back to padded-L10 init in the loader, so restrict
    # the draw to covered windows. target_market_rows == local_seq_indices*1000 matches
    # our windowing (n_messages = n_cond+n_gen = 1000, randomize_offset off).
    include_idx = None
    wbf = getattr(ds, "wide_book_files", None)
    if wide_book_dir is not None and wbf and str(wbf[0]).endswith(".npz"):
        import numpy as onp
        cov = []
        for f_i, wb in enumerate(wbf):
            li = onp.load(wb)["local_seq_indices"]
            li = li[li < int(ds._seqs_per_file[f_i])]
            cov.append(int(ds._seqs_cumsum[f_i]) + li.astype(onp.int64))
        include_idx = onp.concatenate(cov).astype(onp.int32)
        assert n_pop <= len(include_idx), (
            f"n_pop={n_pop} > {len(include_idx)} snapshot-covered windows")
        print(f"[prep] wide-book npz coverage: {len(include_idx)} windows over "
              f"{len(wbf)} files (n_pop={n_pop})", flush=True)
    # UNSEEN-half restriction (headline arm): env UNSEEN_MANIFEST -> only slots the
    # anchor's pretraining never consumed are drawable (initial pool AND refreshes,
    # since include_idx flows into every _sample_windows call).
    import os as _os
    _unseen = _os.environ.get("UNSEEN_MANIFEST", "")
    if _unseen:
        include_idx = _apply_unseen_mask(ds, include_idx, _unseen)
        assert n_pop <= len(include_idx), (
            f"n_pop={n_pop} > {len(include_idx)} unseen-eligible windows")
    # SEEN-complement restriction (selection universes): env PRETRAIN_SEEN_MANIFEST ->
    # only slots overlapping >=1 pretraining-SEEN window are drawable. Disjoint by
    # construction from every window post-training can draw (those need zero seen
    # overlap), so a selection universe built this way never scores post-training data.
    _seen = _os.environ.get("PRETRAIN_SEEN_MANIFEST", "")
    if _seen:
        assert not _unseen, \
            "UNSEEN_MANIFEST and PRETRAIN_SEEN_MANIFEST are mutually exclusive"
        include_idx = _apply_unseen_mask(ds, include_idx, _seen, complement=True)
        assert n_pop <= len(include_idx), (
            f"n_pop={n_pop} > {len(include_idx)} seen-complement windows")
    # Key flow preserved bit-exactly vs the pre-refactor code: split #1 -> idx choice, split #2 -> rngs.
    rng = jax.random.PRNGKey(seed)
    rng, rk_idx = jax.random.split(rng)
    rng, rk_rng = jax.random.split(rng)
    W = _sample_windows(inf, ds, n_cond, n_gen, n_pop, rk_idx, rk_rng,
                        sim_init=sim_init, tick_size=tick_size, n_vol_series=n_vol_series,
                        include_idx=include_idx)

    return dict(
        inf=inf, train_state=train_state, model=model, model_cls=loaded["model_cls"],
        bb_params=train_state.params, batchnorm=batchnorm, encoder=encoder,
        sample_top_n=sample_top_n, tick_size=tick_size, sim_init=sim_init,
        valid_mask_array=valid_mask_array, m_seq_inp=W["m_seq_inp"], b_seq_inp=W["b_seq_inp"],
        sim_states_init=W["sim_states_init"], rngs=W["rngs"], init_hidden_batched=init_hidden_batched,
        init_time_batched=W["init_time_batched"], n_gen=n_gen, m_seq_raw_cont=W["m_seq_raw_cont"],
        ctx_tokens=W["ctx_tokens"], real_cont_tokens=W["real_cont_tokens"],
        real_cont_book=W["real_cont_book"], ds=ds, idx=W["idx"], include_idx=include_idx,
        kernel=train_state.params["decoder"]["kernel"], bias=train_state.params["decoder"]["bias"])


def gpu_checks(hs, args):
    print(f"\n[S3] GPU checks — n_pop={args.n_pop} n_cond={args.n_cond} n_gen={args.n_gen} "
          f"sigma={args.sigma} es_steps={args.es_steps}", flush=True)
    P = _prep_real_batch(args.data_dir, args.n_cond, args.n_gen, args.n_pop,
                         ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step, seed=args.seed)
    inf = P["inf"]

    def _es_args(pop):
        return (pop, P["sim_init"], P["train_state"], P["model"], P["batchnorm"], P["encoder"],
                P["sample_top_n"], P["tick_size"], P["m_seq_inp"], P["b_seq_inp"], P["n_gen"],
                P["sim_states_init"], P["rngs"], P["init_hidden_batched"], True,
                P["init_time_batched"], P["valid_mask_array"])

    N = args.n_pop
    gen_es = make_generate_es_batched()                              # decoder BATCHED (the real ES path)
    gen_es_shared = make_generate_es_batched(decoder_in_axes=None)   # decoder BROADCAST (plumbing gate)
    names = ("msgs_decoded", "l2_book_states", "num_errors", "msgs_tokens", "b_finals")

    stock_out = inf.generate_batched(
        P["sim_init"], P["train_state"], P["model"], P["batchnorm"], P["encoder"],
        P["sample_top_n"], P["tick_size"], P["m_seq_inp"], P["b_seq_inp"], P["n_gen"],
        P["sim_states_init"], P["rngs"], P["init_hidden_batched"], True,
        P["init_time_batched"], False, None, P["valid_mask_array"])

    # (G1a) WRAPPER PLUMBING: shared-decoder wrapper == stock generate_batched, BIT-FOR-BIT.
    # With the head shared (in_axes=None) the decoder matmul is BROADCAST (== stock), so this isolates
    # `ts.replace(params={**params, decoder})` + generate() and proves it reproduces stock exactly.
    W_ref = {"kernel": P["kernel"], "bias": P["bias"]}
    shared_out = gen_es_shared(*_es_args(W_ref))
    for nm, a, b in zip(names, shared_out, stock_out):
        _check(f"(G1a) shared-head wrapper == generate_batched [{nm}]",
               bool(jnp.array_equal(a, b)), f"max|diff|={_maxabs(jnp.asarray(a), jnp.asarray(b)):.3e}")

    # (G1b) σ=0 PERTURBATION VANISHES: gen_es(build_decoder_population(σ=0)) == gen_es(W_ref-tiled),
    # BIT-FOR-BIT (same batched program; pop0 values == W_ref). Proves the EGGROLL materialisation is a
    # true no-op at σ=0 through the real rollout.
    fnp0, np0, es_map, esk0, leaves = make_decoder_noiser(
        hs, P["kernel"], P["bias"], sigma=0.0, lr=1e-3, rank=CFG.eggroll.rank, solver=optax.sgd)
    pop0 = build_decoder_population(hs, fnp0, np0, leaves, esk0, population_iterinfo(N, 0))
    pop_tiled = {"kernel": jnp.broadcast_to(P["kernel"], (N,) + P["kernel"].shape),
                 "bias": jnp.broadcast_to(P["bias"], (N,) + P["bias"].shape)}
    dk0 = _maxabs(pop0["kernel"], P["kernel"][None])
    _check("(G1b-pop) build_decoder_population(sigma=0) == W_ref", dk0 == 0.0, f"max|diff|={dk0:.3e}")
    es0_out = gen_es(*_es_args(pop0))
    tiled_out = gen_es(*_es_args(pop_tiled))
    for nm, a, b in zip(names, es0_out, tiled_out):
        _check(f"(G1b) sigma=0 ES rollout == W_ref-tiled rollout [{nm}]",
               bool(jnp.array_equal(a, b)), f"max|diff|={_maxabs(jnp.asarray(a), jnp.asarray(b)):.3e}")

    # Faithfulness diagnostic (NOT bit-for-bit vs stock): the materialised (batched) decoder matmul ≡
    # stock's BROADCAST matmul to ~1e-6 (S2), but discrete top-n AR sampling amplifies that into divergent
    # token sequences for a minority of members. Report agreement/onset; assert a gross-bug tripwire (a
    # real backbone/wiring bug diverges ALL members at token ~0 with ~0 agreement).
    at = jnp.asarray(es0_out[3]).reshape(N, -1); ct = jnp.asarray(stock_out[3]).reshape(N, -1)
    eq = (at == ct); L = at.shape[1]
    agreement = float(jnp.mean(eq.astype(jnp.float32)))
    n_full = int(jnp.sum(jnp.all(eq, axis=1)))
    firsts = [L if bool(jnp.all(eq[i])) else int(jnp.argmax(~eq[i])) for i in range(N)]
    print(f"   [diag] materialised-head σ=0 vs stock: token agreement={agreement:.3f}, "
          f"members matching FULL rollout={n_full}/{N}, first-divergence/member={firsts}")
    _check("(G1c) materialised σ=0 faithful to stock (chaos-only divergence, no gross bug)",
           agreement >= 0.5 and n_full >= 1, f"agreement={agreement:.3f} n_full={n_full}")

    # (G2) tiny end-to-end ES loop.
    fnp, npar, es_map, esk, leaves = make_decoder_noiser(
        hs, P["kernel"], P["bias"], sigma=args.sigma, lr=args.lr, rank=CFG.eggroll.rank, solver=optax.sgd)
    k0 = leaves["kernel"]
    moved, all_finite = 0.0, True
    for step in range(args.es_steps):
        it = population_iterinfo(args.n_pop, step)
        pop = build_decoder_population(hs, fnp, npar, leaves, esk, it)
        out = gen_es(*_es_args(pop))
        num_errors = out[2].astype(jnp.float32)                       # (N,)
        raw = -num_errors                                              # placeholder fitness: fewer no-ops better
        fit = hs.EggRoll.convert_fitnesses(fnp, npar, raw)
        npar, leaves = hs.EggRoll.do_updates(fnp, npar, leaves, esk, fit, it, es_map)
        all_finite = all_finite and bool(jnp.isfinite(leaves["kernel"]).all())
        moved = _maxabs(leaves["kernel"], k0)
        print(f"   step {step}: mean num_errors={float(jnp.mean(num_errors)):.1f} "
              f"||Δkernel||_max={moved:.3e}", flush=True)
    _check("(G2) ES loop moves the decoder head & stays finite", moved > 0.0 and all_finite,
           f"max|Δkernel|={moved:.3e} finite={all_finite}")


def main():
    ap = argparse.ArgumentParser(description="S3: ES generator G-step validation")
    ap.add_argument("--rollout", action="store_true", help="run the GPU rollout checks (GH200)")
    ap.add_argument("--data_dir", default=None, help="node-local GOOG dir (for get_dataset)")
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--n_pop", type=int, default=8)
    ap.add_argument("--n_cond", type=int, default=64)
    ap.add_argument("--n_gen", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=CFG.eggroll.sigma)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--es_steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hs = import_hyperscalees()
    cpu_checks(hs, seed=args.seed, n_pop=args.n_pop)

    if args.rollout:
        if not args.data_dir:
            print("[S3] --rollout requires --data_dir (node-local GOOG dir)"); sys.exit(2)
        gpu_checks(hs, args)
    else:
        print("\n[S3] (skipped GPU rollout checks — pass --rollout on the GH200 node)")

    print("\n" + "=" * 60)
    if _FAILS:
        print(f"[S3] FAILED checks: {_FAILS}")
        sys.exit(1)
    print("[S3] ALL CHECKS PASSED" + ("" if args.rollout else " (CPU subset; GPU rollout pending)"))


if __name__ == "__main__":
    main()
