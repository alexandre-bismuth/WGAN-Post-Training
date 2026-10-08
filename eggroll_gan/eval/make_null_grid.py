"""Selection-matched null: N random rank-r LoRA-structured perturbations of the anchor.

The selection/zero-sum control for the paper protocol (tab:sealed Control row): training
is replaced by drawing N random low-rank perturbations of the frozen anchor, matched to
the deployed picks' parameter-norm profile, and passed through the IDENTICAL selection +
sealed-test pipeline. N matches the per-seed selection-grid size, so the null's argmin
runs over the same multiplicity as each treated seed's selection.

Each member m (numpy RandomState(seed + m)) draws, for every kernel key of the picks'
payload (shape (a, b)): D = A @ B.T with A ~ N(0,1)^(a,r), B ~ N(0,1)^(b,r) — the same
rank-r structure EGGROLL perturbs with. The draw is scaled to the picks' delta norms:
  per_key (default): D_k *= mean_i ||pick_i_k - anchor_k|| / ||D_k||   (norm AND profile)
  global:            D   *= mean_i ||pick_i - anchor||_2  / ||D||_2    (total norm only)
and written as anchor + D — a normal proj checkpoint dir (soup.write_soup format), one
member per stepNNNN/ subdir, so the chain SEL wrapper consumes the grid unmodified via
    GRIDS="null=<out_dir>" STEP_GRID="0001 0002 ... 000N" \
        bash scripts/submit_chain.sh --eval-only --sel-only --submit

KL-to-anchor is verified SEPARATELY (GPU): run eggroll_gan.eval.gen_ce_curve over this
grid and over the picks; KL scales ~quadratically in the perturbation scale, so if the
null's mean KL is off, rescale via --extra_scale and regenerate (one iteration suffices).

Anchor kernel values come from the FULL anchor checkpoint (--anchor_full --anchor_step,
needs the JAX training env; CPU devices are fine) or from a proj checkpoint whose kernels
ARE anchor values (--anchor_proj, e.g. a step-0 best/ dir). Generation itself is
CPU-only.

    python -m eggroll_gan.eval.make_null_grid \
        --picks <pick_dir> [<pick_dir> ...] \
        --anchor_full checkpoints/mamba3_78M_googonly_s28730 --anchor_step 28730 \
        --out_dir <CKPT_ROOT>/null_grid_<tag> --n 8 --rank 4 --seed 0
    python -m eggroll_gan.eval.make_null_grid --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

from .soup import load_proj_flat, write_soup


def _gnorm(flat):
    return float(np.sqrt(sum(float(np.sum(np.asarray(v, np.float64) ** 2))
                             for v in flat.values())))


def anchor_flat_from_full(ckpt_dir, ckpt_step, keys):
    """Extract the anchor's values for `keys` ('a/b/c' paths) from the full checkpoint."""
    from ..data import checkpoint_utils as ck
    loaded = ck.load_pretrained_generator(ckpt_dir, ckpt_step, build_loaders=False)
    params = loaded["train_state"].params
    out = {}
    for k in keys:
        node = params
        for part in k.split("/"):
            node = node[part]
        out[k] = np.asarray(node)
    return out


def build_null_grid(pick_dirs, anchor_flat, out_dir, *, n_members=8, rank=4, seed=0,
                    norm_match="per_key", extra_scale=1.0):
    picks, srcs = [], []
    for d in pick_dirs:
        fl, _ = load_proj_flat(d)
        picks.append(fl)
        srcs.append(d)
    keys = list(picks[0].keys())
    for i, fl in enumerate(picks[1:], 1):
        assert list(fl.keys()) == keys, f"pick {i} key mismatch vs pick 0"
    for k in keys:
        assert tuple(np.shape(anchor_flat[k])) == tuple(np.shape(picks[0][k])), \
            f"anchor/pick shape mismatch at '{k}'"
        assert np.asarray(anchor_flat[k]).ndim == 2, f"'{k}' is not a 2-D kernel"

    deltas = [{k: np.asarray(p[k], np.float64) - np.asarray(anchor_flat[k], np.float64)
               for k in keys} for p in picks]
    tgt_per_key = {k: float(np.mean([np.linalg.norm(d[k]) for d in deltas])) for k in keys}
    tgt_global = float(np.mean([_gnorm(d) for d in deltas]))
    print(f"[null] {len(picks)} picks, {len(keys)} kernels; target ||delta||: "
          f"global {tgt_global:.4e}, per-key mean {np.mean(list(tgt_per_key.values())):.4e}",
          flush=True)

    manifest = {"n_members": n_members, "rank": rank, "seed": seed,
                "norm_match": norm_match, "extra_scale": extra_scale,
                "pick_sources": srcs, "target_norm_global": tgt_global,
                "members": {}}
    for m in range(n_members):
        rng = np.random.RandomState(seed + m)
        D = {}
        for k in keys:
            a, b = np.shape(anchor_flat[k])
            D[k] = rng.randn(a, rank) @ rng.randn(b, rank).T
        if norm_match == "per_key":
            for k in keys:
                D[k] *= tgt_per_key[k] / max(np.linalg.norm(D[k]), 1e-30)
        else:
            s = tgt_global / max(_gnorm(D), 1e-30)
            for k in keys:
                D[k] *= s
        if extra_scale != 1.0:
            for k in keys:
                D[k] *= extra_scale
        flat = {k: np.asarray(np.asarray(anchor_flat[k], np.float64) + D[k],
                              dtype=np.asarray(picks[0][k]).dtype) for k in keys}
        step_dir = os.path.join(out_dir, f"step{m + 1:04d}")
        write_soup(step_dir, flat, srcs,
                   {"step": m + 1, "null_member": m, "null_seed": seed + m,
                    "rank": rank, "norm_match": norm_match, "extra_scale": extra_scale})
        try:
            os.chmod(step_dir, 0o775)
        except OSError:
            pass
        manifest["members"][f"step{m + 1:04d}"] = {
            "norm_global": _gnorm(D),
            "norm_rel_to_target": _gnorm(D) / tgt_global}
        print(f"[null] member {m + 1}/{n_members} -> {step_dir}  "
              f"||D||={_gnorm(D):.4e} ({_gnorm(D) / tgt_global:.3f}x target)", flush=True)

    mp = os.path.join(out_dir, "null_grid_manifest.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=1)
    for p in (mp, out_dir):
        try:
            os.chmod(p, 0o664 if p == mp else 0o775)
        except OSError:
            pass
    return manifest


def _selftest():
    import tempfile
    from flax import serialization
    rng = np.random.RandomState(7)
    shapes = {"enc/layers_0/seq/in_proj/kernel": (12, 24),
              "fused/layers_1/seq/out_proj/kernel": (24, 12)}
    anchor = {k: rng.randn(*s).astype(np.float32) for k, s in shapes.items()}
    tmp = tempfile.mkdtemp()
    pick_dirs = []
    for i in range(3):
        fl = {k: (np.asarray(v, np.float64)
                  + 1e-3 * rng.randn(*np.shape(v))).astype(np.float32)
              for k, v in anchor.items()}
        d = os.path.join(tmp, f"pick{i}")
        os.makedirs(d)
        with open(os.path.join(d, "s5_generator_proj.msgpack"), "wb") as f:
            f.write(serialization.to_bytes(fl))
        with open(os.path.join(d, "latest_checkpoint.json"), "w") as f:
            json.dump({"generator_proj": "s5_generator_proj.msgpack", "step": 20}, f)
        pick_dirs.append(d)

    out = os.path.join(tmp, "null_grid")
    man = build_null_grid(pick_dirs, anchor, out, n_members=4, rank=4, seed=0)
    ok = True
    picks = [load_proj_flat(d)[0] for d in pick_dirs]
    tgt = {k: np.mean([np.linalg.norm(np.asarray(p[k], np.float64)
                                      - np.asarray(anchor[k], np.float64))
                       for p in picks]) for k in shapes}
    m1 = load_proj_flat(os.path.join(out, "step0001"))[0]
    m2 = load_proj_flat(os.path.join(out, "step0002"))[0]
    for k, s in shapes.items():
        d1 = np.asarray(m1[k], np.float64) - np.asarray(anchor[k], np.float64)
        d2 = np.asarray(m2[k], np.float64) - np.asarray(anchor[k], np.float64)
        if not np.isclose(np.linalg.norm(d1), tgt[k], rtol=1e-3):
            print(f"  [FAIL] member 1 '{k}' norm {np.linalg.norm(d1):.3e} != target {tgt[k]:.3e}")
            ok = False
        if np.allclose(d1, d2):
            print(f"  [FAIL] members 1 and 2 identical at '{k}' (seed not varied)")
            ok = False
        # rank check up to float32 quantization noise (the write casts to the picks' dtype)
        if np.linalg.matrix_rank(d1, tol=1e-3 * np.linalg.norm(d1, 2)) > 4:
            print(f"  [FAIL] member 1 '{k}' rank > 4")
            ok = False
    if len(man["members"]) != 4:
        print("  [FAIL] manifest member count")
        ok = False
    print("[null selftest]", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--picks", nargs="+", help="deployed pick proj-ckpt dirs (norm targets)")
    ap.add_argument("--anchor_full", help="full anchor checkpoint dir (JAX env; CPU ok)")
    ap.add_argument("--anchor_step", type=int, help="anchor checkpoint step (with --anchor_full)")
    ap.add_argument("--anchor_proj", help="proj ckpt whose kernels ARE anchor values (alt source)")
    ap.add_argument("--out_dir", help="output grid dir (stepNNNN/ members)")
    ap.add_argument("--n", type=int, default=8, help="members (= selection-grid size)")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm_match", choices=["per_key", "global"], default="per_key")
    ap.add_argument("--extra_scale", type=float, default=1.0,
                    help="uniform rescale after norm matching (KL-match iteration knob)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.picks or not args.out_dir:
        ap.error("--picks and --out_dir are required (or use --selftest)")
    keys = list(load_proj_flat(args.picks[0])[0].keys())
    if args.anchor_proj:
        anchor, _ = load_proj_flat(args.anchor_proj)
    elif args.anchor_full and args.anchor_step is not None:
        anchor = anchor_flat_from_full(args.anchor_full, args.anchor_step, keys)
    else:
        ap.error("need --anchor_proj OR --anchor_full + --anchor_step")
    build_null_grid(args.picks, anchor, args.out_dir, n_members=args.n, rank=args.rank,
                    seed=args.seed, norm_match=args.norm_match,
                    extra_scale=args.extra_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
