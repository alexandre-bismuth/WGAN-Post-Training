"""Held-out generator cross-entropy across ES post-training step checkpoints.

WHAT THIS MEASURES. The training histories log `cross_entropy` = the CRITIC's real-vs-fake
separability CE (discriminator.critic_cross_entropy), NOT the generator's likelihood. This
script produces the missing curve: token-level next-token CE (nats/token) of the GENERATOR on
held-out real data, evaluated at the anchor (ES step 0) and at every persisted step checkpoint
(step0005..step0050) of each training lane — the likelihood axis of the CE-vs-critic-score
training-dynamics figure.

PROTOCOL. Teacher-forced CE over `--n_windows` fixed 500-message windows (13,000 tokens each,
all 26 token positions, pretraining alignment: x = [START; tok..], y = tok..) drawn from the
staged eval days with a seed-independent rng(12345) permutation — the SFT trainer's val-split
convention, so every checkpoint of every lane is scored on the IDENTICAL window set. The
forward pass reuses `train_sft_lora.make_sft_eval` (frozen anchor decoder head, fp32
log_softmax) and checkpoints merge through `test_eval.apply_flat_subtree` — the same merge
path whose no-op gate (G1) is bit-exact against the anchor. KL(π_θ || π_anchor) on the same
positions is recorded as a free second diagnostic.

Per-window CE values are persisted so the plot layer can bootstrap CIs over windows and
average across lanes/seeds without re-running the GPU pass.

CLUSTER: GPU compute — launch via scripts/eval/_run_gen_ce_curve.sbatch (squashfs day
staging, breadcrumb-only checkpoint discovery, MAMBA3_EPS_COMPAT=0 for the s28730 anchors).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..config import DEFAULT as CFG

MSG_LEN = CFG.model.msg_len            # 26


def parse_lanes(spec):
    """"s0=/path/lane_root,s1=..." -> [(label, lane_root)] preserving order."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        lbl, _, root = part.partition("=")
        if not root:
            raise ValueError(f"lane spec '{part}' is not label=dir")
        out.append((lbl, root))
    return out


def run(args):
    import jax
    from ..data import checkpoint_utils as ck
    from ..critic import discriminator as D
    from ..training.train_sft_lora import make_sft_eval, split_xy
    from .test_eval import load_proj_payload, apply_flat_subtree, _tree_get
    import lob.inference_no_errcorr as inf

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    lanes = parse_lanes(args.lanes)
    assert lanes and steps, "need --lanes and --steps"
    assert args.n_windows % args.chunk == 0, "--chunk must divide --n_windows"

    loaded = ck.load_pretrained_generator(args.ckpt_dir, args.ckpt_step, build_loaders=False)
    params0 = loaded["train_state"].params
    backbone = D.make_backbone(loaded["model_cls"])

    # Fixed held-out window set: seed-independent permutation (the SFT val-split convention).
    ds = inf.get_dataset(args.data_dir, args.msg_seq_len, 0, test_split=0.0)
    n_win = len(ds)
    assert n_win >= args.n_windows, f"only {n_win} windows in {args.data_dir}"
    idx = np.sort(np.random.default_rng(12345).permutation(n_win)[:args.n_windows])
    L = args.msg_seq_len * MSG_LEN
    tick = float(CFG.rollout.tick_size)

    out = ds[[int(i) for i in idx]]
    m_seq = np.stack([np.asarray(a) for a in out[0]])                  # [B, L+1] START-shifted
    b_pv = np.stack([np.asarray(a) for a in out[2]])                   # [B, n_msg+1, k]
    assert m_seq.shape[1] == L + 1, f"window token length {m_seq.shape[1]} != L+1={L + 1}"
    import jax.numpy as jnp
    b_seq = np.asarray(inf.transform_L2_state_batch(jnp.asarray(b_pv), 500, tick)).astype(np.float32)
    x_m, y = split_xy(m_seq)
    x_m, y = np.asarray(x_m), np.asarray(y)
    print(f"[gen_ce] {args.n_windows} windows x {args.msg_seq_len} msgs "
          f"({args.n_windows * L / 1e6:.1f}M scored tokens/model) from {args.data_dir}", flush=True)

    # merge_fn: overwrite the 31 projection kernels from a flat payload — identical tree
    # surgery for every row, so the jitted eval compiles once and is reused for all rows.
    def merge_fn(t):
        return apply_flat_subtree(params0, t)

    eval_fn = make_sft_eval(backbone, params0, merge_fn, with_kl=True,
                            shard="off", chunk=args.chunk)

    # Row list: anchor (step 0, its own kernels through the same merge path) + lane ckpts.
    # Payload keys come from the first available checkpoint so the anchor row uses the
    # exact leaf set the lanes evolve.
    first_dir = os.path.join(lanes[0][1], f"step{steps[0]:04d}")
    first_flat, _ = load_proj_payload(first_dir)
    payload_keys = sorted(first_flat.keys())
    rows = [("anchor", 0, {k: _tree_get(params0, k) for k in payload_keys})]
    for lbl, root in lanes:
        for st in steps:
            d = os.path.join(root, f"step{st:04d}")
            if not os.path.isfile(os.path.join(d, "latest_checkpoint.json")):
                print(f"[gen_ce] WARN no breadcrumb at {d} — row skipped", flush=True)
                continue
            flat, bc = load_proj_payload(d)
            assert sorted(flat.keys()) == payload_keys, f"{d}: payload keys differ"
            rows.append((f"{lbl}_st{st:04d}", st, flat, lbl))
    print(f"[gen_ce] {len(rows)} rows (anchor + {len(rows) - 1} step ckpts)", flush=True)

    results = {}
    for row in rows:
        name, st, flat = row[0], row[1], row[2]
        tr = {k: jnp.asarray(v) for k, v in flat.items()}
        ces, kls = eval_fn(x_m, b_seq, y, tr)
        ces, kls = np.asarray(ces, np.float64), np.asarray(kls, np.float64)
        rec = dict(step=int(st), lane=(row[3] if len(row) > 3 else None),
                   ce_mean=float(ces.mean()), ce_std=float(ces.std(ddof=1)),
                   kl_mean=float(kls.mean()),
                   ce_per_window=[float(v) for v in ces])
        results[name] = rec
        print(f"[gen_ce] {name}: ce {rec['ce_mean']:.5f} +- {rec['ce_std']:.4f} "
              f"kl {rec['kl_mean']:.5f}", flush=True)

    payload = dict(stage="gen_ce_curve", ckpt_dir=args.ckpt_dir, ckpt_step=args.ckpt_step,
                   data_dir=args.data_dir, n_windows=args.n_windows,
                   msg_seq_len=args.msg_seq_len, window_rng="default_rng(12345) permutation",
                   steps=steps, lanes=dict(lanes), idx=[int(i) for i in idx],
                   rows=results)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    os.chmod(args.out, 0o664)
    print(f"[gen_ce] done -> {args.out}", flush=True)
    return 0


def cpu_checks():
    """Login-safe glue checks (no jax, no data): lane parsing + window-draw determinism."""
    ok = True
    lanes = parse_lanes("s0=/a/b,s1=/c/d")
    ok &= lanes == [("s0", "/a/b"), ("s1", "/c/d")]
    try:
        parse_lanes("nodir"); ok = False
    except ValueError:
        pass
    a = np.sort(np.random.default_rng(12345).permutation(1000)[:64])
    b = np.sort(np.random.default_rng(12345).permutation(1000)[:64])
    ok &= bool(np.array_equal(a, b))
    print(f"[gen_ce] CPU checks {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description="held-out generator CE across ES step checkpoints")
    ap.add_argument("--run", action="store_true", help="run the GPU eval; else CPU checks only")
    ap.add_argument("--data_dir", default=None, help="node-local staged eval-day data dir")
    ap.add_argument("--ckpt_dir", default=CFG.paths.ckpt_dir)
    ap.add_argument("--ckpt_step", type=int, default=CFG.paths.ckpt_step)
    ap.add_argument("--lanes", default=None, metavar="LBL=LANE_ROOT,...",
                    help="lane roots holding stepNNNN/ proj checkpoints")
    ap.add_argument("--steps", default="5,10,15,20,25,30,35,40,45,50")
    ap.add_argument("--n_windows", type=int, default=512)
    ap.add_argument("--msg_seq_len", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not cpu_checks():
        raise SystemExit(1)
    if args.run:
        assert args.data_dir, "--run requires --data_dir"
        raise SystemExit(run(args))
    print("[gen_ce] (skipped GPU eval — pass --run on the GH200; CPU checks PASSED)")


if __name__ == "__main__":
    main()
