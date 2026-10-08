"""Muon solver gate — prove `--solver muon` applies a genuine Newton-Schulz update to every
trainable projection kernel and NEVER silently falls back to AdamW.

Why this exists: `optax.contrib.muon` internally PARTITIONS params, routing 2D matrices to the
Newton-Schulz branch and everything else to an AdamW fallback (optax/contrib/_muon.py). Nested
inside our `optax.masked(solver, mask=MM_PARAM)` wrapper, a mis-classification would silently send
the kernels through AdamW — a failure that looks like training but is the wrong optimiser.

The robust, conditioning-independent signature of Newton-Schulz is ALIGNMENT WITH THE EXACT SVD
POLAR FACTOR: for a gradient G = U S V^T, NS(G) ~= U V^T, so the muon *update* (a descent step) is
~ -U V^T and |cos(update, U V^T)| -> 1. AdamW's first step is ~ -sign(G), which aligns with the
polar factor only weakly. (Condition number is NOT a usable signature: AdamW's sign-update is
itself well-conditioned, and 5-step NS cannot fully flatten an extreme spectrum.)

  T1  bare optimiser (from the production registry) ~= polar factor; AdamW does not.
  T2  real path make_proj_noiser(solver=muon) -> do_updates: every MM kernel moves, EXCLUDED leaves
      bit-frozen, and muon deltas DIFFER from AdamW (anti-fallback: the wiring really uses muon).
  T3  per kernel, muon's delta aligns with polar(ES-grad) MORE than AdamW's — for EVERY kernel, so
      none silently fell back. The raw ES gradient is recovered with an SGD pass (delta = -lr*grad).

CPU-only, login-node safe. Run:  python -m eggroll_gan.tests.muon_gate
"""
import os
import sys
import io
import contextlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=1"

import numpy as np
import jax
import jax.numpy as jnp

from ..es.es_plumbing import import_hyperscalees, build_es_map_proj
from ..es.es_generator import make_proj_noiser, population_iterinfo
from ..es import fitness as F
from ..training.train_eggroll_gan import _SOLVERS, _solver_kwargs

_FAILED = []


def chk(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        _FAILED.append(name)


def _polar(G):
    """Exact orthogonal polar factor U V^T (the target Newton-Schulz approximates)."""
    U, _, Vt = np.linalg.svd(np.asarray(G, np.float64), full_matrices=False)
    return U @ Vt


def _abscos(A, B):
    a = np.asarray(A, np.float64).ravel()
    b = np.asarray(B, np.float64).ravel()
    return abs(float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)))


def _apply_once(tx, grad):
    p = {"k": jnp.zeros_like(grad)}
    st = tx.init(p)
    u, _ = tx.update({"k": grad}, st, p)
    return u["k"]


def t1_bare_optimizer_is_newton_schulz():
    """The muon optimiser from the production registry produces ~ the polar factor; AdamW does not."""
    # Well-conditioned skewed matrix (cond ~5) so 5-step NS converges to the polar factor.
    rng = np.random.RandomState(1)
    m = n = 16
    U, _ = np.linalg.qr(rng.randn(m, m))
    V, _ = np.linalg.qr(rng.randn(n, n))
    s = np.linspace(1.0, 0.2, m)
    g = jnp.asarray((U * s) @ V.T, jnp.float32)
    P = _polar(g)
    muon = _SOLVERS["muon"](0.1, **_solver_kwargs("muon"))
    adamw = _SOLVERS["adamw"](0.1, **_solver_kwargs("adamw"))
    a_muon = _abscos(_apply_once(muon, g), P)
    a_adamw = _abscos(_apply_once(adamw, g), P)
    print(f"    |cos(update, polar)|:  muon={a_muon:.3f}  adamw={a_adamw:.3f}")
    chk("(T1) muon update == Newton-Schulz polar factor (AdamW does not)",
        a_muon > 0.9 and a_muon > a_adamw + 0.2,
        f"muon={a_muon:.3f} adamw={a_adamw:.3f}")


def _mock_tree(k):
    def rnd(i, *s):
        return jax.random.normal(jax.random.fold_in(k, 1000 + i), s)
    return {"message_encoder": {"encoder": {"embedding": rnd(0, 7, 4)},
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


def _one_update(hs, mock, es_map, solver_name, G=6):
    """ONE EggRoll.do_updates step through the REAL noiser path. The ES gradient is
    solver-independent (same keys/fitness/iterinfo), so cross-solver delta differences are purely
    the optimiser transform. lr=1.0 so the SGD delta equals the raw ES gradient (negated)."""
    with contextlib.redirect_stdout(io.StringIO()):   # silence a stray vendored debug print
        fnp, npar, esk = make_proj_noiser(hs, mock, es_map, sigma=0.02, lr=1.0, rank=2,
                                          solver=_SOLVERS[solver_name],
                                          solver_kwargs=_solver_kwargs(solver_name), seed=7)
        it = population_iterinfo(G, 0)
        fit = F.rank_sigma_bar(jax.random.normal(jax.random.PRNGKey(123), (3, G)))
        _, cur = hs.EggRoll.do_updates(fnp, npar, mock, esk, fit, it, es_map)
    return cur


def t2_t3_production_path():
    hs = import_hyperscalees()
    k = jax.random.PRNGKey(0)
    mock = _mock_tree(k)
    es_map = build_es_map_proj(mock, hs)        # defaults: glu ON, book proj OFF, fused enc ON
    cur_m = _one_update(hs, mock, es_map, "muon")
    cur_a = _one_update(hs, mock, es_map, "adamw")
    cur_s = _one_update(hs, mock, es_map, "sgd")   # SGD delta = -lr*grad  ->  raw ES gradient

    flat0 = jax.tree_util.tree_flatten_with_path(mock)[0]
    lm = jax.tree_util.tree_leaves(cur_m)
    la = jax.tree_util.tree_leaves(cur_a)
    ls = jax.tree_util.tree_leaves(cur_s)
    maps = jax.tree_util.tree_leaves(es_map)

    mm_seen = 0
    frozen_ok = all_finite = differ_ok = True
    per_kernel_ns_ok = True
    a_muon_list, a_adamw_list = [], []
    for (path, p0), xm, xa, xs, m in zip(flat0, lm, la, ls, maps):
        all_finite = all_finite and bool(jnp.isfinite(xm).all())
        if int(m) == int(hs.MM_PARAM):
            mm_seen += 1
            d_m = np.asarray(xm - p0)
            d_a = np.asarray(xa - p0)
            grad = np.asarray(p0 - xs)            # = +lr*grad direction (lr=1)
            if float(np.max(np.abs(d_m))) <= 0.0:
                differ_ok = False                 # muon must move every kernel
            if float(np.max(np.abs(d_m - d_a))) <= 1e-8:
                differ_ok = False                 # muon != adamw  (no silent fallback)
            P = _polar(grad)
            am, aa = _abscos(d_m, P), _abscos(d_a, P)
            a_muon_list.append(am)
            a_adamw_list.append(aa)
            if not (am > aa):                     # THIS kernel got NS (closer to polar than adamw)
                per_kernel_ns_ok = False
                print(f"      kernel {'/'.join(str(p.key) for p in path)}: muon_align={am:.3f} "
                      f"<= adamw_align={aa:.3f}  (looks like AdamW!)")
        else:
            frozen_ok = frozen_ok and bool(jnp.array_equal(xm, p0))

    chk("(T2) production path: 6 MM kernels move, EXCLUDED bit-frozen, finite, muon != adamw",
        mm_seen == 6 and frozen_ok and all_finite and differ_ok,
        f"mm={mm_seen} frozen={frozen_ok} finite={all_finite} differ={differ_ok}")
    mean_m = float(np.mean(a_muon_list)) if a_muon_list else float("nan")
    mean_a = float(np.mean(a_adamw_list)) if a_adamw_list else float("nan")
    print(f"    per-kernel |cos(delta, polar(grad))| mean:  muon={mean_m:.3f}  adamw={mean_a:.3f}")
    chk("(T3) EVERY kernel got Newton-Schulz, not AdamW (per-kernel polar alignment)",
        per_kernel_ns_ok and mean_m > mean_a and mean_m > 0.6,
        f"muon={mean_m:.3f} adamw={mean_a:.3f} per_kernel_ok={per_kernel_ns_ok}")


def main():
    print("[muon_gate] CPU checks: Newton-Schulz is genuinely active (no silent AdamW fallback)")
    t1_bare_optimizer_is_newton_schulz()
    t2_t3_production_path()
    if _FAILED:
        print(f"[muon_gate] FAILED: {_FAILED}")
        return 1
    print("[muon_gate] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
