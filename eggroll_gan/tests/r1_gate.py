"""CPU gate for the critic R1 gradient penalty (`make_d_step(..., r1_gamma>0)`).

The backbone-hack diagnostic showed the critic reward is anti-aligned with realism (the generator
hacks a fragile critic). R1 — `γ·½·E_real||∇_h D(h)||²` on the critic's CONTINUOUS feature input —
is the standard WGAN stabiliser that smooths D so the reward can't be climbed in sharp non-realistic
directions. This gate verifies, model-free and login-safe (tiny critic, pure CPU):

  1. `r1_gamma=0.0` is deterministic and finite (bit-identical-path: the R1 branch is skipped).
  2. `r1_gamma>0` produces a DIFFERENT, finite update (the R1 branch is actually exercised).
  3. R1 SHRINKS the critic's input-gradient norm `E_real||∇_h D(h)||²` after training — i.e. it does
     what it is designed to do, on top of spectral norm.

Run: `python -m eggroll_gan.tests.r1_gate`  (exits non-zero on any failure).
"""
import sys

import jax
import jax.numpy as jnp

from eggroll_gan.config import DEFAULT as CFG
from eggroll_gan.training.train_eggroll_gan import make_critic, make_d_step


def _maxdiff(t1, t2):
    return max(float(jnp.max(jnp.abs(a - b)))
               for a, b in zip(jax.tree_util.tree_leaves(t1), jax.tree_util.tree_leaves(t2)))


def _finite(tree):
    return all(bool(jnp.isfinite(v).all()) for v in jax.tree_util.tree_leaves(tree))


def selftest(seed: int = 0) -> bool:
    d, B = 8, 64
    head, p0, sn0, tx, os0 = make_critic(CFG, d, seed=seed)
    k = jax.random.PRNGKey(1)
    real = jax.random.normal(k, (B, d))
    fake = jax.random.normal(jax.random.fold_in(k, 9), (B, d)) + 0.5

    def gradnorm(p, sn):
        f = lambda x: jnp.sum(head.apply({"params": p, **sn}, x, train=False))
        g = jax.grad(f)(real)
        return float(jnp.mean(jnp.sum(g * g, axis=-1)))

    fails = []

    # (1) gamma=0 deterministic + finite
    ds0 = make_d_step(head, tx, r1_gamma=0.0)
    pa, _, _, la, _, _ = ds0(p0, sn0, os0, real, fake)
    pb, _, _, lb, _, _ = ds0(p0, sn0, os0, real, fake)
    if not bool(jnp.allclose(jnp.asarray(la), jnp.asarray(lb))):
        fails.append("gamma=0 not deterministic")
    if not _finite(pa):
        fails.append("gamma=0 params not finite")

    # (2) gamma>0 exercises the R1 branch (different, finite update)
    pc, _, _, _, _, _ = make_d_step(head, tx, r1_gamma=5.0)(p0, sn0, os0, real, fake)
    if not (_maxdiff(pa, pc) > 1e-7):
        fails.append("gamma>0 update identical to gamma=0 (R1 branch not active)")
    if not _finite(pc):
        fails.append("gamma>0 params not finite")

    # (3) R1 shrinks the critic input-gradient norm after training
    def train(gamma, n=80):
        h, p, sn, t, o = make_critic(CFG, d, seed=seed)
        ds = make_d_step(h, t, r1_gamma=gamma)
        for _ in range(n):
            p, sn, o, _, _, _ = ds(p, sn, o, real, fake)
        return p, sn

    pn, snn = train(0.0)
    pr, snr = train(5.0)
    gn, gr = gradnorm(pn, snn), gradnorm(pr, snr)
    print(f"  input-gradnorm ||dD/dh||^2: gamma=0 -> {gn:.4f} | gamma=5 -> {gr:.4f}  ({gr/gn:.2f}x)")
    if not (gr < gn):
        fails.append(f"R1 did not shrink input-gradient norm (gamma0={gn:.4f}, gamma5={gr:.4f})")

    for f in fails:
        print(f"  [FAIL] {f}")
    print("R1 gate: ALL PASS" if not fails else f"R1 gate: {len(fails)} FAILED")
    return not fails


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
