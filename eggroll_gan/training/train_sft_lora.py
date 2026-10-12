"""SFT comparison arm — matched-budget SUPERVISED fine-tune of the EGGROLL anchor (task: "does
adversarial post-training beat plain supervised specialization?").

WHAT THIS TRAINS. Plain next-token cross-entropy on real GOOG 2026-01 sequences (the SAME
node-staged corpus the EGGROLL GOOG arm trains on), teacher-forced through the SAME frozen-anchor
architecture, updating EXACTLY the trainable scope EGGROLL evolves: the MM_PARAM in/out projection
kernels of `es_plumbing.build_es_map_proj` (31 leaves at the production toggles glu=1 book_proj=0
fused_enc=1). Everything else — decoder head, embeddings, SSM core, norms, biases — stays
bit-frozen at the anchor, enforced STRUCTURALLY: gradients are taken w.r.t. the flat trainable
dict only (`extract_trainable` -> `merge_trainable`, the exact mechanism train_grpo_head.train_proj
GPU-validated), so no optimizer hazard can ever move an excluded leaf.

Uses direct 31-kernel fine-tuning rather than LoRA-rank-4 adapters. EGGROLL's per-step
perturbations are rank-4, but `EggRoll.do_updates` accumulates them through the optimizer into
FULL-RANK kernel updates (the proj checkpoints store full evolved kernels — see soup.py's header).
The update class that matches what EGGROLL can express in weight space is therefore full-kernel
updates on those 31 kernels; a rank-4 adapter would add a rank constraint EGGROLL does not have,
and would need merging to full kernels at save time anyway. Direct FT also emits the eval's
required checkpoint payload natively. The trust region matching EGGROLL's λ=0.1 anchor-KL is the
--kl_coef term below (default 0.1, `const` anneal — the production setting).

OBJECTIVE (per window, msg_seq_len=500 messages = 13,000 tokens, pretraining-faithful:
ignore_times=False, p_dropout=0.0, use_book_data=True):
    loss = CE + λ·KL,
    CE = -mean_t log p_θ(y_t | x_<=t, book)          (all 26 token positions, like pretraining)
    KL = mean_t KL(π_θ(·|t) || π_anchor(·|t))         (same positions; both through the frozen head)
Data windows come from the SAME lob loader family as pretraining/EGGROLL (`inference_no_errcorr.
get_dataset` -> LOBSTER_Dataset with the START-token shift == LOBSTER_Dataset.no_mask alignment:
x = [START; tok_0..tok_{L-2}], y = [tok_0..tok_{L-1}], book row k (state BEFORE message k)
repeated over message k's 26 token positions — verified against train_helpers.repeat_book).

MATCHED BUDGET (one EGGROLL seed = 2 nodes x 4 GH200 x ~2 h ≈ 16 GPU-h; 50 steps x 2048 rollouts
x 500 gen msgs = 51.2M generated messages ≈ 1.33e9 sampled tokens): one SFT seed = 1 node x 4
GH200 x ≤4 h wall (≤16 GPU-h), enforced by --max_hours inside the loop (budget-matched wall-clock
is the primary criterion). Tokens-supervised are logged per step and written to the breadcrumb
(`tokens_seen`) so both budgets are reported; at --batch 16 one step supervises 208k tokens
(~250 steps ≈ the 51M-token yardstick).

CHECKPOINTS — the HARD requirement: every --ckpt_every steps (plus step-keyed `stepNNNN/` dirs)
this writes a proj-checkpoint directory {latest_checkpoint.json breadcrumb + s5_generator_proj.
msgpack holding the full evolved kernel dict, keys = extract_trainable's 'a/b/c/kernel' paths} —
byte-compatible with `eggroll_gan/eval/soup.py:load_proj_flat` and `test_eval.load_proj_payload`,
so `scripts/eval/_run_test_eval.sbatch --eggroll sft_s0=<dir>` consumes SFT checkpoints with ZERO
eval-side changes. The trainer self-verifies the round-trip through soup.load_proj_flat after the
first save. The same A-day/B-day selection protocol as the ES arm applies over the step dirs.

CLUSTER: `--run` is GPU compute (teacher-forced fwd+bwd over 13k-token windows) — launch via
scripts/train/_run_sft_goog.sbatch (TMPDIR=/tmp staging, breadcrumb-only discovery, ckpts ->
/lus post_training_GAN, logs -> home). Without `--run` it executes ONLY the CPU-safe glue checks
(tiny fake backbone; login-node safe under JAX_PLATFORMS=cpu).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from ..config import DEFAULT as CFG
from ..es.es_plumbing import import_hyperscalees, build_es_map_proj
from ..es.es_generator import extract_trainable, merge_trainable, shard_decision
from . import dist_utils as DU

MSG_LEN = CFG.model.msg_len            # 26
VOCAB = CFG.model.vocab_size           # 2112


# ----------------------------------------------------------------------------------------
# KL λ schedule — local copy of train_eggroll_gan_s5.kl_lambda (importing that module drags the
# whole lob import chain onto a login node; this file must stay import-cheap).
# ----------------------------------------------------------------------------------------
def kl_lambda(step, n_steps, kl_coef, *, anneal="const"):
    if kl_coef <= 0.0:
        return 0.0
    if anneal == "const":
        return kl_coef
    frac = min(max(step / max(n_steps, 1), 0.0), 1.0)
    return kl_coef * 0.5 * (1.0 + math.cos(math.pi * frac))


# ----------------------------------------------------------------------------------------
# Batch assembly: the pretraining (no_mask) alignment from an inference_mask window.
# ----------------------------------------------------------------------------------------
def split_xy(m_seq):
    """m_seq [B, n_msg*26 + 1] = [START ; tok_0..tok_{L-1}] (inference_mask) ->
    (x [B, L] = [START ; tok_0..tok_{L-2}], y [B, L] = [tok_0..tok_{L-1}]) — EXACTLY
    LOBSTER_Dataset.no_mask's (seq=[START;seq[:-1]], y=seq), i.e. the pretraining alignment."""
    m = jnp.asarray(m_seq)
    return m[:, :-1].astype(jnp.int32), m[:, 1:].astype(jnp.int32)


def book_to_token_rate(book):
    """book [.., n_msg+1, d_book] -> [.., n_msg*26, d_book]: drop the trailing post-last-message
    state (book[k] = state BEFORE message k — the training loader's causal alignment), repeat each
    state over its message's 26 token positions (train_helpers.repeat_book's np.repeat)."""
    b = jnp.asarray(book, jnp.float32)
    return jnp.repeat(b[..., :-1, :], MSG_LEN, axis=-2)


# ----------------------------------------------------------------------------------------
# The SFT grad/eval factories — modeled line-for-line on the GPU-validated
# baselines/policy_grad.make_pg_grad_proj (merge -> remat'd teacher-forced backbone pass ->
# frozen anchor head -> chunked lax.scan with value_and_grad inside; sharded via shard_map).
# ----------------------------------------------------------------------------------------
def _sft_parts(backbone, bb_params, merge_fn, *, with_kl, remat):
    """Shared per-sample closures for make_sft_grad / make_sft_eval."""
    from ..critic.discriminator import PaddedLobPredFeatures

    W0 = bb_params["decoder"]["kernel"]
    b0 = bb_params["decoder"]["bias"]

    def _feat(params, xm, xb, mt, bt):
        return backbone.apply({"params": params}, xm, xb, mt, bt,
                              method=PaddedLobPredFeatures.features)             # [L, d_model]
    _feat_tr = jax.checkpoint(_feat) if remat else _feat

    def _logp(hid):
        # matches __call_ar__: decoder then log_softmax in fp32.
        return jax.nn.log_softmax((hid @ W0 + b0).astype(jnp.float32), axis=-1)  # [L, V]

    def one(cur, xm, xb, y):
        """One window: (ce, kl) — CE over ALL L positions (pretraining ignore_times=False);
        KL(π_θ || π_anchor) over the same positions, both through the frozen anchor head."""
        L = xm.shape[0]
        mt = jnp.ones((L,), jnp.float32)
        lp = _logp(_feat_tr(cur, xm, xb, mt, mt))
        ce = -jnp.mean(jnp.take_along_axis(lp, y[:, None], axis=-1)[:, 0])
        if with_kl:
            lp_a = _logp(jax.lax.stop_gradient(_feat(bb_params, xm, xb, mt, mt)))
            klv = jnp.mean(jnp.sum(jnp.exp(lp) * (lp - lp_a), axis=-1))
        else:
            klv = jnp.zeros(())
        return ce, klv

    return one


def make_sft_grad(backbone, bb_params, merge_fn, *, with_kl=True, shard="auto", chunk=1,
                  remat=True, backend=None):
    """Build fn(x_m [M, L] int, book [M, n_msg+1, d_book] f32, y [M, L] int, tr, lam) ->
         (loss_sum, grad_tr, ce [M], kl [M])
    with loss_sum = sum_r(ce_r + lam*kl_r) and grad_tr its exact gradient w.r.t. the flat
    trainable dict `tr` (extract_trainable layout; merge_fn(tr) -> full params). The CALLER
    divides by the GLOBAL M (correct under sharding: each shard psums partial sums). One chunk's
    reverse tape lives at a time (value_and_grad inside lax.scan); `remat` recomputes the backbone
    features in the backward pass. `chunk` must divide the (per-shard) row count."""
    one = _sft_parts(backbone, bb_params, merge_fn, with_kl=with_kl, remat=remat)

    def _chunk_obj(tr, lam, xm_c, bk_c, y_c):
        cur = merge_fn(tr)
        xb_c = book_to_token_rate(bk_c)
        ces, kls = jax.vmap(lambda xm, xb, y: one(cur, xm, xb, y))(xm_c, xb_c, y_c)
        return jnp.sum(ces + lam * kls), (ces, kls)

    def _run(x_m, book, y, tr, lam):
        x_m = jnp.asarray(x_m).astype(jnp.int32)
        y = jnp.asarray(y).astype(jnp.int32)
        book = jnp.asarray(book, jnp.float32)
        M = x_m.shape[0]
        c = chunk if (chunk and 0 < chunk < M) else M
        assert M % c == 0, f"sft chunk {c} must divide the (per-shard) row count {M}"

        def _re(x):
            return x.reshape((M // c, c) + x.shape[1:])
        stacked = (_re(x_m), _re(book), _re(y))

        def body(carry, ch):
            loss_s, gtr = carry
            (l, aux), g = jax.value_and_grad(_chunk_obj, argnums=0, has_aux=True)(tr, lam, *ch)
            return (loss_s + l, jax.tree_util.tree_map(jnp.add, gtr, g)), aux
        init = (jnp.zeros(()), jax.tree_util.tree_map(jnp.zeros_like, tr))
        (loss_sum, grad_tr), (ces, kls) = jax.lax.scan(body, init, stacked)
        return loss_sum, grad_tr, ces.reshape(M), kls.reshape(M)

    if shard_decision(shard, jax.device_count()) == "vmap":
        kw = {"backend": backend} if backend else {}
        return jax.jit(_run, **kw)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _run_psum(x_m, book, y, tr, lam):
        loss_s, gtr, ces, kls = _run(x_m, book, y, tr, lam)
        gtr = jax.tree_util.tree_map(lambda g: jax.lax.psum(g, "pop"), gtr)
        return jax.lax.psum(loss_s, "pop"), gtr, ces, kls
    mesh = jax.make_mesh((jax.device_count(),), ("pop",))
    # check_rep=False: same rationale as make_pg_grad_proj — the upstream lob backbone is not
    # VMA-clean under JAX's static replication audit; loss/grad are psum'd (genuinely replicated).
    return jax.jit(shard_map(_run_psum, mesh=mesh,
                             in_specs=(P("pop"), P("pop"), P("pop"), P(), P()),
                             out_specs=(P(), P(), P("pop"), P("pop")), check_rep=False))


def make_sft_eval(backbone, bb_params, merge_fn, *, with_kl=True, shard="auto", chunk=1,
                  backend=None):
    """Forward-only sibling of make_sft_grad: fn(x_m, book, y, tr) -> (ce [M], kl [M]).
    Used for the held-out val-CE monitor (no reverse tape; remat pointless so it is off)."""
    one = _sft_parts(backbone, bb_params, merge_fn, with_kl=with_kl, remat=False)

    def _run(x_m, book, y, tr):
        x_m = jnp.asarray(x_m).astype(jnp.int32)
        y = jnp.asarray(y).astype(jnp.int32)
        book = jnp.asarray(book, jnp.float32)
        cur = merge_fn(tr)
        M = x_m.shape[0]
        c = chunk if (chunk and 0 < chunk < M) else M
        assert M % c == 0, f"sft eval chunk {c} must divide the (per-shard) row count {M}"

        def _re(x):
            return x.reshape((M // c, c) + x.shape[1:])

        def body(ch):
            xm_c, bk_c, y_c = ch
            xb_c = book_to_token_rate(bk_c)
            return jax.vmap(lambda xm, xb, yy: one(cur, xm, xb, yy))(xm_c, xb_c, y_c)
        ces, kls = jax.lax.map(body, (_re(x_m), _re(book), _re(y)))
        return ces.reshape(M), kls.reshape(M)

    if shard_decision(shard, jax.device_count()) == "vmap":
        kw = {"backend": backend} if backend else {}
        return jax.jit(_run, **kw)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    mesh = jax.make_mesh((jax.device_count(),), ("pop",))
    return jax.jit(shard_map(_run, mesh=mesh, in_specs=(P("pop"), P("pop"), P("pop"), P()),
                             out_specs=(P("pop"), P("pop")), check_rep=False))


# ----------------------------------------------------------------------------------------
# Optimizer + checkpointing (proj-checkpoint format: the eval's hard-required payload).
# ----------------------------------------------------------------------------------------
def make_solver(name, lr, grad_clip=1.0):
    """adamw with weight_decay=0 (decay-to-zero would fight the KL-to-ANCHOR trust region; matches
    the ES arm's noiser adamw). Global-norm clip first (standard SFT hygiene; 0 disables)."""
    base = optax.adamw(lr, weight_decay=0.0) if name == "adamw" else optax.sgd(lr)
    if grad_clip and grad_clip > 0:
        return optax.chain(optax.clip_by_global_norm(grad_clip), base)
    return base


def save_sft_ckpt(out_dir, step, tr, opt_state, history, meta):
    """Write a proj-scope checkpoint dir: s5_generator_proj.msgpack = the FULL evolved kernel
    dict (flat 'a/b/c/kernel' keys — extract_trainable layout, the exact payload
    soup.load_proj_flat / test_eval.load_proj_payload consume) + the optimizer state + an ATOMIC
    latest_checkpoint.json breadcrumb (a partial write is never read; breadcrumb-only discovery)."""
    os.makedirs(out_dir, exist_ok=True)
    gen_file = "s5_generator_proj.msgpack"
    opt_file = "sft_opt_state.msgpack"
    with open(os.path.join(out_dir, gen_file), "wb") as f:
        f.write(serialization.to_bytes({k: np.asarray(v) for k, v in tr.items()}))
    with open(os.path.join(out_dir, opt_file), "wb") as f:
        f.write(serialization.to_bytes(opt_state))
    bc = dict(stage="SFT", step=int(step), generator_proj=gen_file, opt_state=opt_file,
              history=history, **meta)
    tmp = os.path.join(out_dir, "latest_checkpoint.json.tmp")
    with open(tmp, "w") as f:
        json.dump(bc, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, "latest_checkpoint.json"))
    for fn in (gen_file, opt_file, "latest_checkpoint.json"):
        try:
            os.chmod(os.path.join(out_dir, fn), 0o664)
        except OSError:
            pass
    return bc


def load_sft_ckpt(resume_dir, tr_tmpl, opt_tmpl):
    """Breadcrumb-only restore (never ls — Lustre rule). -> (tr, opt_state, bc)."""
    with open(os.path.join(resume_dir, "latest_checkpoint.json")) as f:
        bc = json.load(f)
    with open(os.path.join(resume_dir, bc["generator_proj"]), "rb") as f:
        tr = serialization.from_bytes(tr_tmpl, f.read())
    with open(os.path.join(resume_dir, bc["opt_state"]), "rb") as f:
        opt_state = serialization.from_bytes(opt_tmpl, f.read())
    return tr, opt_state, bc


def _verify_ckpt_roundtrip(out_dir, tr):
    """Prove the saved payload round-trips BIT-exactly through the eval's own loader
    (eggroll_gan.eval.soup.load_proj_flat — the format contract). Raises on any mismatch."""
    from ..eval.soup import load_proj_flat
    flat, bc = load_proj_flat(out_dir)
    if sorted(flat.keys()) != sorted(tr.keys()):
        raise RuntimeError(f"ckpt roundtrip key mismatch: {sorted(set(flat) ^ set(tr))}")
    for k in tr:
        if not np.array_equal(np.asarray(flat[k]), np.asarray(tr[k])):
            raise RuntimeError(f"ckpt roundtrip value mismatch at '{k}'")
    return bc


# ----------------------------------------------------------------------------------------
# The training loop (GPU; --run via scripts/train/_run_sft_goog.sbatch).
# ----------------------------------------------------------------------------------------
def train(args):
    from ..data import checkpoint_utils as ck
    from ..critic import discriminator as D
    import lob.inference_no_errcorr as inf

    p0 = jax.process_index() == 0

    def log(msg):
        if p0:
            print(msg, flush=True)

    n_dev = jax.device_count()
    sharded = shard_decision(args.shard, n_dev) == "shard_map"
    mesh = DU.make_pop_mesh() if sharded else None
    if sharded:
        assert args.batch % n_dev == 0, f"--batch {args.batch} must divide device_count={n_dev}"
    Ms = args.batch // (n_dev if sharded else 1)
    assert Ms % args.chunk == 0, f"--chunk {args.chunk} must divide the per-shard batch {Ms}"
    assert args.n_val % args.batch == 0, f"--n_val {args.n_val} must be a --batch multiple"

    # --- anchor restore (partial_restore inference path) ---
    loaded = ck.load_pretrained_generator(args.ckpt_dir, args.ckpt_step, build_loaders=False)
    margs = loaded["args"]
    assert not bool(getattr(margs, "batchnorm", False)), \
        "SFT trainer assumes batchnorm=False (true for the s28730 anchor)"
    assert getattr(margs, "token_mode", "26tok") != "1tok", "26tok checkpoints only"
    if int(getattr(margs, "msg_seq_len", args.msg_seq_len)) != args.msg_seq_len:
        print(f"[SFT] NOTE: --msg_seq_len {args.msg_seq_len} != pretraining msg_seq_len "
              f"{margs.msg_seq_len} (SSM handles any length; 500 is the matched default)", flush=True)
    params0 = loaded["train_state"].params
    backbone = D.make_backbone(loaded["model_cls"])   # training=False forward; p_dropout=0.0 anyway

    # --- trainable scope: EXACTLY the EGGROLL proj scope (the scientific control) ---
    hs = import_hyperscalees()
    es_map = build_es_map_proj(params0, hs, perturb_glu=bool(args.perturb_glu),
                               perturb_book_proj=bool(args.perturb_book_proj),
                               perturb_fused_encoder=bool(args.perturb_fused_encoder))
    tr = {k: jnp.asarray(v) for k, v in extract_trainable(hs, params0, es_map).items()}
    tr_anchor = {k: jnp.asarray(v) for k, v in tr.items()}
    n_mm = len(tr)
    mm_params = int(sum(v.size for v in tr.values()))
    if args.expect_leaves and n_mm != args.expect_leaves:
        raise RuntimeError(f"trainable scope has {n_mm} MM_PARAM kernels, expected "
                           f"{args.expect_leaves} (the EGGROLL production scope) — toggle mismatch?")

    def merge_fn(t):
        return merge_trainable(hs, params0, es_map, t)

    L = args.msg_seq_len * MSG_LEN
    log(f"[SFT] scope: {n_mm} MM_PARAM kernels, {mm_params/1e6:.1f}M trainable / 78M total | "
        f"lr={args.lr:.1e} solver={args.solver} clip={args.grad_clip} kl_coef={args.kl_coef} "
        f"kl_anneal={args.kl_anneal} | batch={args.batch} ({Ms}/shard, chunk={args.chunk}) "
        f"L={L} tokens/window | n_steps={args.n_steps} ckpt_every={args.ckpt_every} "
        f"max_hours={args.max_hours} | devices={n_dev} shard={'shard_map' if sharded else 'vmap'}")

    # --- dataset: same lob loader family as pretraining/EGGROLL; 500-msg windows (n_eval=0) ---
    ds = inf.get_dataset(args.data_dir, args.msg_seq_len, 0, test_split=0.0)
    n_win = len(ds)
    assert n_win > args.n_val + args.batch, f"only {n_win} windows in {args.data_dir}"
    # Val split keyed by a SEED-INDEPENDENT rng: the same held-out windows across sft_s0/s1/s2.
    perm = np.random.default_rng(12345).permutation(n_win)
    val_idx = np.sort(perm[:args.n_val])
    train_idx = np.sort(perm[args.n_val:])
    log(f"[SFT] corpus: {n_win} windows x {args.msg_seq_len} msgs -> train {len(train_idx)} / "
        f"val {len(val_idx)} (val split seed-independent)")

    tick = float(CFG.rollout.tick_size)

    def fetch(idx):
        """Window ids -> (x_m [B,L] int32, book [B,n_msg+1,503] f32, y [B,L] int32) host numpy."""
        out = ds[[int(i) for i in idx]]
        m_seq = np.stack([np.asarray(a) for a in out[0]])                 # [B, L+1] (START-shifted)
        b_pv = np.stack([np.asarray(a) for a in out[2]])                  # [B, n_msg+1, k]
        assert m_seq.shape[1] == L + 1, f"window token length {m_seq.shape[1]} != L+1={L + 1}"
        b_seq = np.asarray(inf.transform_L2_state_batch(jnp.asarray(b_pv), 500, tick))
        x, y = split_xy(m_seq)
        return np.asarray(x), b_seq.astype(np.float32), np.asarray(y)

    val_batches = [fetch(val_idx[i:i + args.batch]) for i in range(0, len(val_idx), args.batch)]

    grad_fn = make_sft_grad(backbone, params0, merge_fn, with_kl=args.kl_coef > 0,
                            shard=args.shard, chunk=args.chunk)
    eval_fn = make_sft_eval(backbone, params0, merge_fn, with_kl=args.kl_coef > 0,
                            shard=args.shard, chunk=args.chunk)
    tx = make_solver(args.solver, args.lr, args.grad_clip)
    opt_state = tx.init(tr)

    def _meta(step):
        return dict(scope="proj-sft", solver=args.solver, lr=args.lr, kl_coef=args.kl_coef,
                    kl_anneal=args.kl_anneal, seed=args.seed, batch=args.batch,
                    msg_seq_len=args.msg_seq_len, n_mm=n_mm, mm_params=mm_params,
                    grad_clip=args.grad_clip, sigma=0.0, fitness_control="sft",
                    tokens_seen=int(step) * args.batch * L,
                    perturb_glu=int(args.perturb_glu),
                    perturb_book_proj=int(args.perturb_book_proj),
                    perturb_fused_encoder=int(args.perturb_fused_encoder))

    history = []
    step0 = 0
    if args.resume_dir and os.path.exists(os.path.join(args.resume_dir, "latest_checkpoint.json")):
        tr, opt_state, bc = load_sft_ckpt(args.resume_dir, tr, opt_state)
        step0 = int(bc["step"])
        history = bc.get("history", [])
        log(f"[SFT] resumed from {args.resume_dir} @ step {step0}")

    def _val():
        ces, kls = [], []
        for (vx, vb, vy) in val_batches:
            c, k = eval_fn(DU.shard_pop(vx, mesh), DU.shard_pop(vb, mesh),
                           DU.shard_pop(vy, mesh), tr)
            ces.append(np.asarray(DU.gather_host(c, mesh)))
            kls.append(np.asarray(DU.gather_host(k, mesh)))
        return float(np.mean(np.concatenate(ces))), float(np.mean(np.concatenate(kls)))

    val_ce0, _ = _val()
    log(f"[SFT] step {step0:5d}  ANCHOR val_ce {val_ce0:.4f} (nats/token, {len(val_idx)} held-out "
        f"windows)")

    t0 = time.time()
    verified = False
    diverged = False
    last_step = step0
    for step in range(step0 + 1, args.n_steps + 1):
        # deterministic per-(seed, step) draw — resume-safe, seed-symmetric with the ES arm.
        idx = np.random.default_rng([args.seed, step]).choice(train_idx, size=args.batch,
                                                              replace=False)
        t_d = time.time()
        x_m, book, y = fetch(idx)
        t_f = time.time()
        lam = kl_lambda(step, args.n_steps, args.kl_coef, anneal=args.kl_anneal)
        loss_s, grad_tr, ces, kls = grad_fn(DU.shard_pop(x_m, mesh), DU.shard_pop(book, mesh),
                                            DU.shard_pop(y, mesh), tr, jnp.float32(lam))
        grads = jax.tree_util.tree_map(lambda g: jnp.asarray(np.asarray(DU.gather_host(g, mesh)))
                                       / args.batch, grad_tr)
        updates, opt_state = tx.update(grads, opt_state, tr)
        tr = optax.apply_updates(tr, updates)
        ce_m = float(np.mean(np.asarray(DU.gather_host(ces, mesh))))
        kl_m = float(np.mean(np.asarray(DU.gather_host(kls, mesh))))
        loss = float(DU.gather_host(loss_s, mesh)) / args.batch
        last_step = step
        t_e = time.time()

        if step % args.log_every == 0 or step == 1:
            log(f"[SFT] step {step:5d}  loss {loss:.4f}  ce {ce_m:.4f}  kl {kl_m:.5f}(λ{lam:.3f})"
                f"  | {t_e - t_f:.1f}s step +{t_f - t_d:.1f}s data | "
                f"tokens {step * args.batch * L / 1e6:.1f}M")
        rec = dict(step=step, loss=loss, ce=ce_m, kl=kl_m, lam=lam,
                   step_s=round(t_e - t_f, 2), data_s=round(t_f - t_d, 2))
        if step % args.eval_every == 0 or step == 1 or step == args.n_steps:
            val_ce, val_kl = _val()
            dk = max(float(jnp.max(jnp.abs(tr[k] - tr_anchor[k]))) for k in tr)
            finite = bool(np.isfinite(loss)) and all(bool(jnp.isfinite(v).all()) for v in tr.values())
            rec.update(val_ce=val_ce, val_kl=val_kl, dkernel=dk, finite=finite)
            log(f"[SFT] step {step:5d}  val_ce {val_ce:.4f} (anchor {val_ce0:.4f})  "
                f"val_kl {val_kl:.5f}  ||Δ|| {dk:.2e}  finite={finite}")
            if not finite:
                log("[SFT] *** NON-FINITE — divergence guard tripped (lower lr) ***")
                diverged = True
        history.append(rec)
        if step % args.ckpt_every == 0 or step == args.n_steps or diverged:
            if p0:
                save_sft_ckpt(args.out_dir, step, tr, opt_state, history, _meta(step))
                sdir = os.path.join(args.out_dir, f"step{step:04d}")
                save_sft_ckpt(sdir, step, tr, opt_state, [], _meta(step))
                log(f"[SFT] checkpoint @ step {step} -> {args.out_dir} (+ {os.path.basename(sdir)})")
                if not verified:
                    _verify_ckpt_roundtrip(args.out_dir, tr)
                    log("[SFT] ckpt payload round-trips BIT-exactly through eval.soup.load_proj_flat")
                    verified = True
        if diverged:
            break
        if (time.time() - t0) / 3600.0 > args.max_hours:
            log(f"[SFT] wall budget --max_hours {args.max_hours} reached @ step {step} — stopping")
            break

    if p0:
        save_sft_ckpt(args.out_dir, last_step, tr, opt_state, history, _meta(last_step))
    log(f"[SFT] {'DIVERGED' if diverged else 'done'} @ step {last_step} — "
        f"{last_step * args.batch * L / 1e6:.1f}M supervised tokens, "
        f"{(time.time() - t0) / 3600.0:.2f}h -> {args.out_dir} "
        f"(selection: held-out LOB-Bench over the step ckpts, same protocol as the ES arm)")
    return 1 if diverged else 0


# ----------------------------------------------------------------------------------------
# CPU-safe glue checks (login node; tiny fake backbone, no model, no data).
# ----------------------------------------------------------------------------------------
def cpu_checks(seed=0):
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
              flush=True)
        if not cond:
            fails.append(name)

    print("\n[SFT] CPU checks — alignment / grad / scope / ckpt round-trip", flush=True)
    k = jax.random.PRNGKey(seed)
    n_msg, dm, dbook, M = 2, 5, 7, 4
    L = n_msg * MSG_LEN

    # (S1) pretraining alignment: inference_mask window -> no_mask (x, y) shift; book token-rate.
    START = 3
    toks = np.arange(1000, 1000 + L, dtype=np.int64) % VOCAB
    m_seq = np.concatenate([[START], toks])[None]                       # [1, L+1] (inference_mask)
    x, y = split_xy(m_seq)
    ref_x = np.concatenate([[START], toks[:-1]])                        # no_mask: [START; seq[:-1]]
    bk = np.arange((n_msg + 1) * dbook, dtype=np.float32).reshape(1, n_msg + 1, dbook)
    xb = book_to_token_rate(bk)
    align_ok = (bool(np.array_equal(np.asarray(x)[0], ref_x))
                and bool(np.array_equal(np.asarray(y)[0], toks))
                and xb.shape == (1, L, dbook)
                and bool(np.array_equal(np.asarray(xb)[0, 0], bk[0, 0]))       # msg 0 <- state 0
                and bool(np.array_equal(np.asarray(xb)[0, MSG_LEN], bk[0, 1]))  # msg 1 <- state 1
                and bool(np.array_equal(np.asarray(xb)[0, -1], bk[0, n_msg - 1])))  # last state used
    chk("(S1) no_mask x/y shift + causal token-rate book", align_ok)

    # Fake MERGEABLE backbone: features = H0(x_m, x_b) @ proj_kernel, so the
    # trainable kernel is exercised end-to-end through merge_fn without any model build.
    th0 = jnp.eye(dm) * 0.9 + 0.01
    Wh = jax.random.normal(jax.random.fold_in(k, 1), (dm, VOCAB)) * 0.1
    bh = jnp.zeros((VOCAB,))
    bb = {"proj": {"kernel": th0}, "decoder": {"kernel": Wh, "bias": bh}}

    def merge_fn(t):
        return {"proj": {"kernel": t["proj/kernel"]}, "decoder": bb["decoder"]}

    class _FakeBB:
        def apply(self, variables, xm, xb, mt, bt, *a, method=None):
            th = variables["params"]["proj"]["kernel"]
            base = (jnp.arange(xm.shape[0], dtype=jnp.float32)[:, None] / xm.shape[0]) \
                * jnp.ones((1, dm))
            H0 = base + 0.01 * xm[:, None].astype(jnp.float32) * jnp.ones((1, dm)) \
                + 0.005 * jnp.mean(xb) * jnp.ones((xm.shape[0], dm))
            return H0 @ th

    x_m = jax.random.randint(jax.random.fold_in(k, 2), (M, L), 0, VOCAB)
    book = jax.random.normal(jax.random.fold_in(k, 3), (M, n_msg + 1, dbook))
    yb = jax.random.randint(jax.random.fold_in(k, 4), (M, L), 0, VOCAB)
    tr0 = {"proj/kernel": th0}

    # (S2) KL(θ==anchor)==0; CE>0 finite; chunked == unchunked.
    outs = {}
    for c in (1, M):
        fn = make_sft_grad(_FakeBB(), bb, merge_fn, with_kl=True, shard="off", chunk=c, remat=False)
        outs[c] = fn(x_m, book, yb, tr0, jnp.float32(0.1))
    dmax = max(float(jnp.max(jnp.abs(a - b))) for a, b in
               zip(jax.tree_util.tree_leaves(outs[1]), jax.tree_util.tree_leaves(outs[M])))
    loss0, g0, ce0, kl0 = outs[M]
    chk("(S2) KL(anchor)=0; CE finite>0; chunked==unchunked",
        float(jnp.max(jnp.abs(kl0))) < 1e-6 and bool(jnp.isfinite(ce0).all())
        and float(jnp.min(ce0)) > 0.0 and dmax < 1e-5, f"max|chunk d|={dmax:.2e}")

    # (S3) d loss / d proj-kernel == finite differences (lam=0 isolates the CE term).
    fn0 = make_sft_grad(_FakeBB(), bb, merge_fn, with_kl=True, shard="off", chunk=M, remat=False)

    def loss_of(th):
        return float(fn0(x_m, book, yb, {"proj/kernel": th}, jnp.float32(0.0))[0])
    _, g, _, _ = fn0(x_m, book, yb, tr0, jnp.float32(0.0))
    i, j, eps = 1, 2, 3e-3
    fd = (loss_of(th0.at[i, j].add(eps)) - loss_of(th0.at[i, j].add(-eps))) / (2 * eps)
    ad = float(g["proj/kernel"][i, j])
    chk("(S3) d loss / d kernel matches finite differences",
        abs(ad - fd) < 2e-2 * max(abs(fd), 1e-2), f"ad={ad:.5f} fd={fd:.5f}")

    # (S4) eval_fn forward parity with grad_fn's aux; a perturbed kernel gives KL>0.
    ev = make_sft_eval(_FakeBB(), bb, merge_fn, with_kl=True, shard="off", chunk=1)
    ce_e, kl_e = ev(x_m, book, yb, tr0)
    th1 = th0 + 0.05 * jax.random.normal(jax.random.fold_in(k, 5), (dm, dm))
    _, kl_p = ev(x_m, book, yb, {"proj/kernel": th1})
    chk("(S4) eval==grad forward; perturbed kernel -> KL>0",
        float(jnp.max(jnp.abs(ce_e - ce0))) < 1e-5 and float(jnp.max(jnp.abs(kl_e - kl0))) < 1e-6
        and float(jnp.mean(kl_p)) > 0.0, f"mean_kl_pert={float(jnp.mean(kl_p)):.4f}")

    # (S5) an optimizer step moves the kernel and stays finite; grad tree structure == tr.
    tx = make_solver("adamw", 1e-3, 1.0)
    st = tx.init(tr0)
    up, st = tx.update(jax.tree_util.tree_map(lambda x: x / M, g), st, tr0)
    tr1 = optax.apply_updates(tr0, up)
    chk("(S5) adamw step moves kernel & finite; grad keys == tr keys",
        sorted(g.keys()) == sorted(tr0.keys())
        and float(jnp.max(jnp.abs(tr1["proj/kernel"] - th0))) > 0.0
        and bool(jnp.isfinite(tr1["proj/kernel"]).all()))

    # (S6) THE FORMAT CONTRACT: save -> eval.soup.load_proj_flat round-trip is bit-exact, the
    # breadcrumb carries the keys test_eval reads, and resume restores tr + opt state bit-exactly.
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"sft_ckpt_test_{os.getpid()}")
    tr_multi = {"message_encoder/layers_0/seq/in_proj/kernel":
                    jax.random.normal(jax.random.fold_in(k, 6), (4, 10)),
                "fused_s5/layers_0/seq/out_proj/kernel":
                    jax.random.normal(jax.random.fold_in(k, 7), (6, 4))}
    st_m = tx.init(tr_multi)
    meta = dict(scope="proj-sft", solver="adamw", lr=1e-4, kl_coef=0.1, sigma=0.0,
                fitness_control="sft", tokens_seen=123, seed=0)
    save_sft_ckpt(out, 7, tr_multi, st_m, history=[{"step": 7, "ce": 1.0}], meta=meta)
    bc = _verify_ckpt_roundtrip(out, tr_multi)          # raises on any bit mismatch
    tr_r, st_r, bc_r = load_sft_ckpt(out, tr_multi, st_m)
    rt_ok = (bc["step"] == 7 and bc.get("generator_proj") == "s5_generator_proj.msgpack"
             and all(kk in bc for kk in ("sigma", "lr", "fitness_control"))     # test_eval row meta
             and bc_r["tokens_seen"] == 123
             and all(bool(jnp.array_equal(tr_r[kk], tr_multi[kk])) for kk in tr_multi)
             and all(bool(jnp.array_equal(a, b)) for a, b in
                     zip(jax.tree_util.tree_leaves(st_r), jax.tree_util.tree_leaves(st_m))))
    chk("(S6) proj-ckpt round-trip via eval.soup.load_proj_flat + resume (tr/opt bit-exact)", rt_ok,
        f"step={bc['step']} keys={len(tr_multi)}")
    import shutil
    shutil.rmtree(out, ignore_errors=True)

    # (S7) KL λ schedule: const holds; cosine anneals to ~0; kl_coef=0 -> 0.
    chk("(S7) kl_lambda schedule",
        kl_lambda(5, 10, 0.1) == 0.1 and kl_lambda(10, 10, 0.1, anneal="cosine") < 1e-6
        and kl_lambda(3, 10, 0.0) == 0.0)

    print("\n[SFT] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


def main():
    ap = argparse.ArgumentParser(
        description="SFT comparison arm: supervised fine-tune of the EGGROLL proj scope + checks")
    ap.add_argument("--run", action="store_true", help="run the trainer (GH200); else CPU checks only")
    ap.add_argument("--data_dir", default=None, help="node-local GOOG day-filtered farm dir")
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "sft_out"))
    ap.add_argument("--resume_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--msg_seq_len", type=int, default=500,
                    help="messages per training window (500 = the pretraining msg_seq_len)")
    ap.add_argument("--batch", type=int, default=16, help="global windows/step (divides device count)")
    ap.add_argument("--chunk", type=int, default=1,
                    help="windows per value_and_grad chunk (per shard); memory knob")
    ap.add_argument("--n_steps", type=int, default=6000,
                    help="fills the 3.4h wall budget at the measured ~1.5-3 s/step (smoke "
                         "5477400); --max_hours governs when throughput is slower")
    ap.add_argument("--max_hours", type=float, default=3.4,
                    help="wall budget INSIDE the loop (primary budget-match criterion; "
                         "1 node x 4 GPU x ~4h wall = the 16 GPU-h EGGROLL seed budget)")
    ap.add_argument("--lr", type=float, default=1e-4, help="small SFT lr (adamw, wd=0)")
    ap.add_argument("--solver", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--kl_coef", type=float, default=0.1,
                    help="KL(π_θ||π_anchor) trust region (EGGROLL production λ=0.1; 0 disables)")
    ap.add_argument("--kl_anneal", choices=["const", "cosine"], default="const")
    ap.add_argument("--n_val", type=int, default=64,
                    help="held-out val windows (monitoring only; --batch multiple)")
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=300,
                    help="proj-ckpt cadence (latest + step-keyed stepNNNN/ dirs; 20 ckpts at the "
                         "6000-step default, ~294 MB full-kernel payload each)")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--perturb_glu", type=int, default=1)
    ap.add_argument("--perturb_book_proj", type=int, default=0)
    ap.add_argument("--perturb_fused_encoder", type=int, default=1)
    ap.add_argument("--expect_leaves", type=int, default=31,
                    help="hard gate on the trainable-kernel count (the EGGROLL scope match; 0=off)")
    ap.add_argument("--shard", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()

    fails = cpu_checks(seed=args.seed)
    if fails:
        print(f"[SFT] CPU checks FAILED: {fails}")
        sys.exit(1)
    if args.run:
        if not args.data_dir:
            print("[SFT] --run requires --data_dir (node-local GOOG dir)")
            sys.exit(2)
        sys.exit(train(args))
    print("\n[SFT] (skipped GPU run — pass --run on the GH200; CPU glue checks PASSED)")


if __name__ == "__main__":
    main()
