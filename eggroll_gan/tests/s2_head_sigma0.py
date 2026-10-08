"""S2, part 2 — decoder-head EGGROLL equivalence + plumbing checks (the S2 gate:
"perturb ONLY the decoder head and verify the head-perturbed forward == stock Flax forward at sigma=0").

The generator's head is `self.decoder = nn.Dense(d_output)` (lob/lob_seq_model.py): kernel (d_model, vocab),
bias (vocab,), computing `x @ kernel + bias`. We wrap it with EggRoll (`do_Tmm` for the matmul,
`get_noisy_standard` for the bias) and assert:

  (A) iterinfo=None  -> EGGROLL head == stock Flax `nn.Dense` apply, BIT-EXACT.
  (B) sigma=0 (full population) -> every member == stock, BIT-EXACT (perturbation scales with sigma).
  (C) sigma>0 -> every member differs from stock (perturbation is real); antithetic pairs (2k, 2k+1)
                 are EXACT mirrors about stock; and the antithetic-balanced population mean == stock.
  (D) es_map / freeze: on a mock *full* generator pytree, only decoder/{kernel,bias} are classified
                 trainable (MM_PARAM/PARAM); a `do_updates` step leaves every EXCLUDED leaf bit-exact
                 frozen and changes the decoder leaves. (De-risks the S3 in_axes=0 decoder-head lift.)

(A)/(B) are the required equivalence; they hold for ANY kernel/bias, so the default run uses a
freshly-initialised `nn.Dense` of the real (1024 -> 2112) shape and is fully CPU- / login-node-safe (a tiny
init + matmul; no checkpoint, no model build, no gymnax). With `--use_checkpoint` (COMPUTE -> GH200 node only)
the SAME assertions are re-run against the REAL trained decoder leaves loaded from the frozen checkpoint, so
the equivalence is verified on the actual weights S3 will perturb.

Run (login/CPU):  JAX_PLATFORMS=cpu PYTHONPATH=<exp_root> python -u -m eggroll_gan.s2_head_sigma0
Run (GH200 node): PYTHONPATH=<exp_root>:<mamba_root> python -u -m eggroll_gan.s2_head_sigma0 --use_checkpoint
"""
from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from ..config import DEFAULT as CFG
from ..es.es_plumbing import (import_hyperscalees, build_es_map, build_es_tree_key,
                          decoder_apply_stock, decoder_apply_eggroll)

_FAILS = []


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"   [{status}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        _FAILS.append(name)


def _max_abs(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def _eggroll_head(hs, fnp, npar, kernel, bias, keys, iterinfo, x):
    return decoder_apply_eggroll(hs, fnp, npar, kernel, bias, keys["kernel"], keys["bias"], iterinfo, x)


def equivalence_suite(hs, kernel, bias, x, stock, *, lr=0.01, rank=None, seed=0, n_pop=8, label="random-init"):
    """Run equivalence/plumbing checks (A)-(C) for one (kernel, bias) pair. `stock` is the reference
    output (a real `nn.Dense.apply` bound to these leaves).

    NB: run EAGER (no jax.jit). Two reasons: (1) bit-exactness — a jitted matmul reduces in a different
    order than the eager `nn.Dense`, so jit-vs-eager would show ~1e-6 float noise that is NOT a
    perturbation leak; eager-vs-eager isolates the (identically zero) sigma=0 perturbation term.
    (2) on the login node each big-matmul jit compile leaks LLVM threads and trips the 1900 ulimit.
    The jitted/vmapped path is what S3 runs on the GPU; validate that on the GH200 node, not here."""
    rank = CFG.eggroll.rank if rank is None else rank
    print(f"\n[S2.2] equivalence suite on decoder head ({label}): "
          f"kernel{tuple(kernel.shape)} bias{tuple(bias.shape)} x{tuple(x.shape)} rank={rank} (eager)", flush=True)

    leaves = {"kernel": kernel, "bias": bias}
    keys = build_es_tree_key(leaves, jax.random.key(seed + 100), hs)

    def members_at(sigma):
        """Stack of `n_pop` per-member outputs at a given sigma (thread_id = 0..n_pop-1)."""
        fnp, npar = hs.EggRoll.init_noiser(leaves, sigma, lr, solver=optax.sgd, rank=rank)
        ms = [_eggroll_head(hs, fnp, npar, kernel, bias, keys,
                            (jnp.int32(0), jnp.int32(t)), x) for t in range(n_pop)]
        return jnp.stack(ms)                                 # (n_pop, *stock.shape)

    # (A) iterinfo=None -> bit-exact stock (eager base path == nn.Dense).
    fnp0, np0 = hs.EggRoll.init_noiser(leaves, 0.0, lr, solver=optax.sgd, rank=rank)
    y_none = _eggroll_head(hs, fnp0, np0, kernel, bias, keys, None, x)
    d_none = _max_abs(y_none, stock)
    _check(f"(A) iterinfo=None == stock [{label}]", d_none == 0.0, f"max|diff|={d_none:.3e}")

    # (B) sigma=0, full population -> every member bit-exact stock (perturbation term is identically 0).
    members0 = members_at(0.0)
    d_pop0 = _max_abs(members0, stock[None])
    _check(f"(B) sigma=0 population == stock [{label}]", d_pop0 == 0.0, f"max|diff|={d_pop0:.3e}")

    # (C) sigma>0: real, mirrored, zero-mean perturbation. sigma=0.1 makes "nonzero" unambiguous
    #     (a plumbing magnitude, NOT the training sigma).
    members1 = members_at(0.1)
    per_member = jnp.max(jnp.abs(members1 - stock[None]), axis=tuple(range(1, members1.ndim)))  # (n_pop,)
    _check(f"(C1) sigma>0 every member differs from stock [{label}]",
           float(jnp.min(per_member)) > 1e-5, f"min max|member-stock|={float(jnp.min(per_member)):.3e}")
    # antithetic: (member[2k]-stock) == -(member[2k+1]-stock)
    dpos = members1[0::2] - stock[None]
    dneg = members1[1::2] - stock[None]
    d_anti = _max_abs(dpos, -dneg)
    _check(f"(C2) antithetic pairs are exact mirrors [{label}]", d_anti < 1e-3, f"max|sum|={d_anti:.3e}")
    # balanced population mean == stock
    d_mean = _max_abs(jnp.mean(members1, axis=0), stock)
    _check(f"(C3) antithetic population mean == stock [{label}]", d_mean < 1e-3, f"max|mean-stock|={d_mean:.3e}")


def es_map_freeze_check(hs, seed=0, lr=0.05):
    """(D) On a mock full generator pytree, confirm es_map marks only decoder/{kernel,bias} trainable and
    a do_updates step freezes every EXCLUDED leaf bit-exact while moving the decoder leaves."""
    print("\n[S2.2] es_map / freeze check on a mock generator pytree", flush=True)
    k = jax.random.key(seed)
    mock = {
        "message_encoder": {"l0": {"kernel": jax.random.normal(jax.random.fold_in(k, 1), (8, 8))}},
        "book_encoder": {"proj": {"kernel": jax.random.normal(jax.random.fold_in(k, 2), (8, 8))}},
        "fused_s5": {"b0": {"A": jax.random.normal(jax.random.fold_in(k, 3), (8,))}},
        "decoder": {"kernel": jax.random.normal(jax.random.fold_in(k, 4), (8, 12)),
                    "bias": jax.random.normal(jax.random.fold_in(k, 5), (12,))},
    }
    es_map = build_es_map(mock, hs)
    _check("(D1) decoder/kernel -> MM_PARAM", int(es_map["decoder"]["kernel"]) == hs.MM_PARAM)
    _check("(D2) decoder/bias -> PARAM", int(es_map["decoder"]["bias"]) == hs.PARAM)
    _check("(D3) backbone leaves -> EXCLUDED",
           int(es_map["message_encoder"]["l0"]["kernel"]) == hs.EXCLUDED
           and int(es_map["book_encoder"]["proj"]["kernel"]) == hs.EXCLUDED
           and int(es_map["fused_s5"]["b0"]["A"]) == hs.EXCLUDED)

    # One update step with SGD (no weight decay) so EXCLUDED leaves get an EXACT zero update.
    # (In the real S3 integration the frozen backbone is kept out of the noiser params entirely;
    #  EXCLUDED here is defense-in-depth and must be a true no-op under a plain solver.)
    N = 8
    es_tree_key = build_es_tree_key(mock, jax.random.key(seed + 7), hs)
    fnp, npar = hs.EggRoll.init_noiser(mock, CFG.eggroll.sigma, lr, solver=optax.sgd, rank=CFG.eggroll.rank)
    fitnesses = (jnp.arange(N, dtype=jnp.float32) - (N - 1) / 2.0)          # distinct -> nonzero grad
    iterinfos = (jnp.zeros(N, dtype=jnp.int32), jnp.arange(N, dtype=jnp.int32))
    _, new = hs.EggRoll.do_updates(fnp, npar, mock, es_tree_key, fitnesses, iterinfos, es_map)

    frozen_ok = all(_max_abs(new[g][s][p], mock[g][s][p]) == 0.0
                    for g, s, p in (("message_encoder", "l0", "kernel"),
                                    ("book_encoder", "proj", "kernel"),
                                    ("fused_s5", "b0", "A")))
    _check("(D4) EXCLUDED leaves bit-exact frozen after do_updates", frozen_ok)
    dk = _max_abs(new["decoder"]["kernel"], mock["decoder"]["kernel"])
    db = _max_abs(new["decoder"]["bias"], mock["decoder"]["bias"])
    _check("(D5) decoder/kernel updated", dk > 0.0, f"max|delta|={dk:.3e}")
    _check("(D6) decoder/bias updated", db > 0.0, f"max|delta|={db:.3e}")


def _random_head(d_model, vocab, seed=0, batch=16):
    """A freshly-initialised real flax `nn.Dense(vocab)` (== self.decoder) + an input batch.
    Returns (kernel, bias, x, stock) where stock = the real `nn.Dense.apply`."""
    dense = nn.Dense(vocab)                       # matches `self.decoder = nn.Dense(self.d_output)`
    key = jax.random.key(seed)
    x = jax.random.normal(jax.random.fold_in(key, 1), (batch, d_model), dtype=jnp.float32)
    variables = dense.init(jax.random.fold_in(key, 2), x)
    kernel = variables["params"]["kernel"]
    bias = variables["params"]["bias"]
    stock = dense.apply(variables, x)             # ground-truth Flax forward
    # Sanity: our `decoder_apply_stock` must equal Flax exactly (it claims to BE nn.Dense's math).
    assert _max_abs(decoder_apply_stock(kernel, bias, x), stock) == 0.0, \
        "decoder_apply_stock != nn.Dense.apply (matmul convention mismatch)"
    return kernel, bias, x, stock


def _checkpoint_head(batch=16):
    """Load the REAL trained decoder leaves from the frozen checkpoint (COMPUTE: GH200 node only)."""
    from ..data import checkpoint_utils as ck
    loaded = ck.load_pretrained_generator(build_loaders=False)
    params = loaded["params"]
    kernel = params["decoder"]["kernel"]
    bias = params["decoder"]["bias"]
    d_model, vocab = int(kernel.shape[0]), int(kernel.shape[1])
    x = jax.random.normal(jax.random.key(123), (batch, d_model), dtype=kernel.dtype)
    dense = nn.Dense(vocab)
    stock = dense.apply({"params": {"kernel": kernel, "bias": bias}}, x)
    print(f"[S2.2] loaded REAL decoder head from checkpoint: kernel{tuple(kernel.shape)} "
          f"bias{tuple(bias.shape)} dtype={kernel.dtype}", flush=True)
    return kernel, bias, x, stock


def main():
    ap = argparse.ArgumentParser(description="S2.2: decoder-head EGGROLL sigma=0 equivalence + plumbing")
    ap.add_argument("--use_checkpoint", action="store_true",
                    help="ALSO verify against the REAL trained decoder leaves (COMPUTE: GH200 node only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d_model = CFG.model.d_model      # 1024
    vocab = CFG.model.vocab_size     # 2112
    print(f"[S2.2] decoder head dims: d_model={d_model} vocab={vocab} "
          f"(training sigma={CFG.eggroll.sigma:.0e}, rank={CFG.eggroll.rank})", flush=True)
    hs = import_hyperscalees()

    # Default: real-shaped random nn.Dense (CPU/login-safe).
    kernel, bias, x, stock = _random_head(d_model, vocab, seed=args.seed)
    equivalence_suite(hs, kernel, bias, x, stock, seed=args.seed, label="random-init")

    # es_map / freeze plumbing (de-risks S3).
    es_map_freeze_check(hs, seed=args.seed)

    # Optional: the same equivalence on the REAL trained weights (GH200 node only).
    if args.use_checkpoint:
        ck_kernel, ck_bias, ck_x, ck_stock = _checkpoint_head()
        equivalence_suite(hs, ck_kernel, ck_bias, ck_x, ck_stock, seed=args.seed, label="real-checkpoint")

    print("\n" + "=" * 60)
    if _FAILS:
        print(f"[S2.2] FAILED checks: {_FAILS}")
        sys.exit(1)
    print("[S2.2] ALL CHECKS PASSED — decoder-head EGGROLL forward == stock Flax forward at sigma=0,")
    print("       perturbation is real/mirrored/zero-mean, and es_map freezes the backbone.")


if __name__ == "__main__":
    main()
