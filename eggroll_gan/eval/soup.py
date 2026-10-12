"""Model-soup utility for EGGROLL-proj checkpoints (uniform weight averaging across seeds).

A proj-scope checkpoint stores the FULL evolved projection kernels — the MM_PARAM leaves of the
optimiser mean (see train_eggroll_gan_s5.save_ckpt / es_generator.extract_trainable), NOT LoRA
factors. So a uniform model soup is simply the element-wise mean of the per-seed kernel dicts:

    W_soup = mean_i(W_evolved_i) = W_frozen + mean_i(dW_i)

Identical keys/shapes across seeds (same anchor, es_map, toggles) make this exact and trivial. The
soup is written as a normal proj checkpoint directory (breadcrumb + msgpack) so it is consumed
unmodified by `test_eval.py --eggroll soup=<dir>`. Averaging the seeds collapses the between-seed
variance that dominates the per-seed result; a soup typically matches or beats the mean member.

CPU-only, login-node safe (pure numpy + msgpack, no JAX compile).

    python -m eggroll_gan.eval.soup --out_dir runs/<job>/soup \\
        --inputs runs/<job>/m_s0/best runs/<job>/m_s1/best runs/<job>/m_s2/best
    python -m eggroll_gan.eval.soup --selftest
"""
import argparse
import json
import os
import sys

import numpy as np
from flax import serialization


def load_proj_flat(ckpt_dir):
    """Breadcrumb-only load of a proj checkpoint (no ls — Lustre rule). -> (flat dict, breadcrumb)."""
    with open(os.path.join(ckpt_dir, "latest_checkpoint.json")) as f:
        bc = json.load(f)
    gen_file = bc.get("generator_proj")
    if not gen_file:
        raise ValueError(f"{ckpt_dir} is not a proj-scope checkpoint (no generator_proj in breadcrumb)")
    with open(os.path.join(ckpt_dir, gen_file), "rb") as f:
        flat = serialization.msgpack_restore(f.read())
    return flat, bc


def average_flats(flats, weights=None):
    """Element-wise (optionally weighted) mean of N flat {key: array} dicts. Keys and per-key shapes
    must match across all members (else a loud error — a structural mismatch means a wrong/foreign
    checkpoint). The accumulation is float64; the result is cast back to the members' dtype."""
    if not flats:
        raise ValueError("no checkpoints to soup")
    keys = list(flats[0].keys())
    for i, fl in enumerate(flats[1:], 1):
        if list(fl.keys()) != keys:
            raise ValueError(f"member {i} key mismatch: {sorted(set(fl) ^ set(keys))}")
        for k in keys:
            if tuple(np.shape(fl[k])) != tuple(np.shape(flats[0][k])):
                raise ValueError(f"member {i} shape mismatch at '{k}': "
                                 f"{np.shape(fl[k])} vs {np.shape(flats[0][k])}")
    n = len(flats)
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, float) / float(np.sum(weights))
    out = {}
    for k in keys:
        acc = sum(w[i] * np.asarray(flats[i][k], np.float64) for i in range(n))
        out[k] = np.asarray(acc, dtype=np.asarray(flats[0][k]).dtype)
    return out


def write_soup(out_dir, flat, sources, meta):
    """Write the soup as a proj checkpoint dir (msgpack + atomic breadcrumb). chmod g+rw (shared FS)."""
    os.makedirs(out_dir, exist_ok=True)
    gen_file = "s5_generator_proj.msgpack"
    gen_path = os.path.join(out_dir, gen_file)
    with open(gen_path, "wb") as f:
        f.write(serialization.to_bytes(flat))
    bc = dict(stage="S5_soup", generator_proj=gen_file,
              n_members=len(sources), soup_sources=list(sources), **meta)   # meta carries step/sigma/lr/solver
    tmp = os.path.join(out_dir, "latest_checkpoint.json.tmp")
    with open(tmp, "w") as f:
        json.dump(bc, f, indent=2)
    bc_path = os.path.join(out_dir, "latest_checkpoint.json")
    os.replace(tmp, bc_path)
    for p in (gen_path, bc_path):
        try:
            os.chmod(p, 0o664)
        except OSError:
            pass
    return bc


def build_soup(inputs, out_dir, weights=None):
    """Load each member checkpoint, average the kernels, write the soup checkpoint. -> breadcrumb."""
    flats, bcs = [], []
    for d in inputs:
        fl, bc = load_proj_flat(d)
        flats.append(fl)
        bcs.append(bc)
    avg = average_flats(flats, weights)
    b0 = bcs[0]
    meta = {"sigma": b0.get("sigma"), "lr": b0.get("lr"), "solver": b0.get("solver"),
            "step": b0.get("step"), "member_steps": [b.get("step") for b in bcs]}
    bc = write_soup(out_dir, avg, list(inputs), meta)
    print(f"[soup] {len(inputs)} members -> {out_dir}  "
          f"({len(avg)} kernels, solver={meta['solver']}, member_steps={meta['member_steps']})", flush=True)
    return bc


def _selftest():
    import tempfile
    rng = np.random.RandomState(0)
    shapes = {"message_encoder/layers_0/seq/in_proj/kernel": (4, 10),
              "fused_s5/layers_0/seq/out_proj/kernel": (6, 4)}
    tmp = tempfile.mkdtemp()
    flats, dirs = [], []
    for i in range(3):
        fl = {k: rng.randn(*s).astype(np.float32) for k, s in shapes.items()}
        flats.append(fl)
        d = os.path.join(tmp, f"seed{i}")
        os.makedirs(d)
        with open(os.path.join(d, "s5_generator_proj.msgpack"), "wb") as f:
            f.write(serialization.to_bytes(fl))
        with open(os.path.join(d, "latest_checkpoint.json"), "w") as f:
            json.dump({"generator_proj": "s5_generator_proj.msgpack", "step": 30,
                       "sigma": 0.003, "lr": 0.01, "solver": "muon"}, f)
        dirs.append(d)
    out = os.path.join(tmp, "soup")
    build_soup(dirs, out)
    soup_fl, soup_bc = load_proj_flat(out)

    ok = True
    for k in shapes:
        expect = np.mean([np.asarray(fl[k], np.float64) for fl in flats], axis=0)
        if not np.allclose(np.asarray(soup_fl[k], np.float64), expect, atol=1e-5):
            print(f"  [FAIL] soup[{k}] != element-wise mean")
            ok = False
    if not (soup_bc.get("n_members") == 3 and soup_bc.get("solver") == "muon"):
        print("  [FAIL] soup breadcrumb metadata wrong")
        ok = False
    try:                                                # shape mismatch must fail loudly
        bad = dict(flats[0]); bad[next(iter(shapes))] = np.zeros((4, 11), np.float32)
        average_flats([flats[0], bad])
        print("  [FAIL] shape mismatch not caught")
        ok = False
    except ValueError:
        pass
    print("[soup selftest]", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="uniform model soup of EGGROLL-proj checkpoints")
    ap.add_argument("--inputs", nargs="+", help="member checkpoint dirs (each with latest_checkpoint.json)")
    ap.add_argument("--out_dir", help="output soup checkpoint dir")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="optional per-member weights")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.inputs or not args.out_dir:
        ap.error("--inputs and --out_dir are required (or use --selftest)")
    build_soup(args.inputs, args.out_dir, args.weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
