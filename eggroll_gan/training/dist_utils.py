"""S5 — multi-process (jax.distributed) array plumbing: global-array construction + host gather.

WHY THIS MODULE EXISTS. On the 4-node path the S5 trainer initialises `jax.distributed` and builds ONE
global ("pop",) mesh spanning every process's devices. Two multi-process facts then bite:

  (a) a `jit(shard_map(...))` over a GLOBAL mesh must be fed GLOBAL `jax.Array`s whose sharding matches
      the in_specs — passing plain host-local numpy/jnp arrays (which are single-process-addressable)
      raises. So every per-rollout input must be promoted to a global array BEFORE the sharded call.
  (b) eager ops / `float(...)` on a global array whose shards live on OTHER processes fail with
      "non-addressable" — so per-rollout OUTPUTS must be explicitly gathered back to fully-replicated
      (an SPMD all-gather) before any host-side use (fitness ranking, logging, checkpoint maths).

The trainer's saving grace: every process builds IDENTICAL full host arrays (same seed, same data), so
global-array construction never needs cross-process data movement — each process just SLICES its own
full host value per local shard (`jax.make_array_from_callback(shape, sharding, lambda idx: x[idx])`).

All helpers are EXACT no-ops in single-process mode (and for mesh=None), so the validated 1-GPU/CPU
paths (S1–S5a single-host) are untouched: same objects in, same objects out. `force=True` exists only
so the global-array machinery is exercisable in single-process tests (the self-test below).

Everything is pytree-generic via `jax.tree_util.tree_map`. CPU-safe; self-test runs on a login node.
"""
from __future__ import annotations

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def is_multihost():
    """True iff jax.distributed is running with more than one process."""
    return jax.process_count() > 1


def make_pop_mesh():
    """The global 1-D ("pop",) mesh over ALL devices (all processes' devices in multi-process mode)."""
    return jax.make_mesh((jax.device_count(),), ("pop",))


def _check_divisible(n_rows, n_dev, shape):
    """Leading-axis divisibility gate for P('pop') sharding — factored out so it is unit-testable."""
    assert n_rows % n_dev == 0, (
        f"shard_pop: leading axis {n_rows} of leaf shape {tuple(shape)} is not divisible by the "
        f"{n_dev} devices on the 'pop' mesh — pad/trim the population (G*Q) to a device multiple")


def _to_global(x, mesh, spec):
    """Full-host-value -> global jax.Array with NamedSharding(mesh, spec).

    Uses `jax.make_array_from_callback(x.shape, sharding, lambda idx: x[idx])`: each process slices its
    OWN copy of the full value per local shard. Works for sharded (P('pop'): leading axis) and
    replicated (P()) specs; REQUIRES every process to hold the SAME full value of x (true in the
    trainer: identical seeds on every process)."""
    x = np.asarray(x)
    sharding = NamedSharding(mesh, spec)
    return jax.make_array_from_callback(x.shape, sharding, lambda idx: x[idx])


def shard_pop(tree, mesh, *, force=False):
    """Pytree -> leading-axis-sharded global arrays (P('pop') on axis 0, replicated elsewhere).

    No-op (returns `tree` unchanged) when mesh is None or the mesh has a SINGLE device, unless
    force=True. Gating on DEVICE count (not process count) is what makes single-process multi-GPU
    SPMD work: a `jit(shard_map)` over a >1-device mesh needs global sharded inputs even when there
    is only one process (process_count()==1). Axis-0 length of every leaf must divide the device count."""
    if mesh is None or (not force and mesh.devices.size <= 1):
        return tree
    n_dev = mesh.devices.size

    def one(x):
        x = np.asarray(x)
        _check_divisible(x.shape[0], n_dev, x.shape)
        return _to_global(x, mesh, P("pop"))

    return jax.tree_util.tree_map(one, tree)


def replicate_tree(tree, mesh, *, force=False):
    """Pytree -> fully-replicated global arrays (P()). Same no-op/force rules as `shard_pop`:
    no-op only for a None or single-device mesh; replicates for any >1-device mesh (incl.
    single-process multi-GPU), so shard_map's P() inputs are proper global arrays."""
    if mesh is None or (not force and mesh.devices.size <= 1):
        return tree
    return jax.tree_util.tree_map(lambda x: _to_global(x, mesh, P()), tree)


def _gather_leaf(x, mesh):
    """One leaf of the multi-process gather: jitted identity with replicated out_shardings (an SPMD
    all-gather), then device_get — after replication every process's local devices hold the full value,
    so device_get is legal. Plain numpy / python scalars pass straight through."""
    if not isinstance(x, jax.Array):
        return x
    rep = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh, P()))(x)
    return jax.device_get(rep)


def gather_host(tree, mesh):
    """Bring (possibly multi-host-sharded) per-rollout outputs back to host numpy.

    Single-process / mesh None: plain `jax.device_get(tree)` (the existing 1-GPU/CPU behaviour).
    Multi-process: per-leaf `_gather_leaf` (replicate-then-fetch), so host-side `float(...)`,
    ranking and logging never touch a non-addressable shard."""
    if mesh is None or not is_multihost():
        return jax.device_get(tree)
    return jax.tree_util.tree_map(lambda x: _gather_leaf(x, mesh), tree)


# ----------------------------------------------------------------------------------------
# Login-node self-test (CPU, single process): no-op fast paths + the forced global-array path.
# ----------------------------------------------------------------------------------------
def _cpu_self_test(seed=0):
    fails = []

    def chk(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        if not cond:
            fails.append(name)

    print("[dist_utils] CPU self-test — single-process no-ops + forced global-array machinery", flush=True)

    # (D1) single process; the ("pop",) mesh builds and covers all (here: 1) devices.
    mesh = make_pop_mesh()
    n_dev = mesh.devices.size
    chk("(D1) is_multihost()==False, 1-device ('pop',) mesh",
        (not is_multihost()) and mesh.axis_names == ("pop",) and n_dev == jax.device_count(),
        f"process_count={jax.process_count()} axis_names={mesh.axis_names} n_dev={n_dev}")

    tree = {"a": np.arange(12, dtype=np.float32).reshape(4, 3),
            "b": np.arange(4, dtype=np.int32)}

    # (D2) no-op fast paths: mesh=None -> literally the same objects; 1-device mesh (not forced) -> equal.
    sp_none, rt_none = shard_pop(tree, None), replicate_tree(tree, None)
    chk("(D2a) mesh=None -> same objects",
        sp_none is tree and rt_none is tree and sp_none["a"] is tree["a"])
    sp_1d, rt_1d = shard_pop(tree, mesh), replicate_tree(tree, mesh)
    gh_none, gh_1d = gather_host(tree, None), gather_host(tree, mesh)
    eq = lambda t: all(np.array_equal(t[k], tree[k]) for k in tree)
    chk("(D2b) 1-device mesh (not forced) + gather_host -> values equal inputs",
        all(eq(t) for t in (sp_1d, rt_1d, gh_none, gh_1d)))

    # (D3) forced global path: jax Arrays with NamedSharding partitioning axis 0 on 'pop'; values equal.
    g = shard_pop(tree, mesh, force=True)
    ok_shard = all(isinstance(g[k], jax.Array) and isinstance(g[k].sharding, NamedSharding)
                   and g[k].sharding.spec[0] == "pop" for k in g)
    chk("(D3) shard_pop(force=True) -> NamedSharding P('pop') on axis 0, values equal",
        ok_shard and eq({k: np.asarray(v) for k, v in g.items()}),
        f"a.spec={g['a'].sharding.spec} b.spec={g['b'].sharding.spec}")

    # (D4) replicate_tree(force=True) -> fully-replicated sharding; values equal.
    r = replicate_tree(tree, mesh, force=True)
    ok_rep = all(isinstance(r[k], jax.Array) and r[k].sharding.is_fully_replicated for k in r)
    chk("(D4) replicate_tree(force=True) -> fully replicated, values equal",
        ok_rep and eq({k: np.asarray(v) for k, v in r.items()}),
        f"a.spec={r['a'].sharding.spec}")

    # (D5) round-trip: gather_host(shard_pop(force=True)) -> host numpy equal to the originals.
    back = gather_host(g, mesh)
    chk("(D5) gather_host(shard_pop(force=True)) -> numpy equal originals",
        all(isinstance(back[k], np.ndarray) for k in back) and eq(back))
    # (D5b) the multi-process gather leaf (replicate-then-fetch) also works on one device, and
    #       passes plain numpy / python scalars through untouched.
    leaf = _gather_leaf(g["a"], mesh)
    chk("(D5b) _gather_leaf: jit-identity all-gather + numpy/scalar pass-through",
        isinstance(leaf, np.ndarray) and np.array_equal(leaf, tree["a"])
        and _gather_leaf(tree["b"], mesh) is tree["b"] and _gather_leaf(3.5, mesh) == 3.5)

    # (D6) the divisibility gate fires: 3 rows over 2 devices must raise AssertionError.
    try:
        _check_divisible(3, 2, (3, 2))
        raised = False
    except AssertionError:
        raised = True
    chk("(D6) _check_divisible raises on (3 rows, 2 devices)", raised)

    print("\n[dist_utils] " + ("ALL CPU CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _cpu_self_test() else 0)
