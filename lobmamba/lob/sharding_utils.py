"""
Sharding utilities for LOBS5 training.
Provides helpers for creating mesh and shardings for jax.jit + shardings migration.

This module replaces the implicit parallelism of jax.pmap with explicit
mesh-based sharding, following the MaxText approach.

Extracted from ssm_stable branch — only data-parallel functions (no FSDP).
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from typing import Any, Tuple, Optional


# Global mesh storage
_GLOBAL_MESH = None


def create_simple_mesh(num_devices: int, hierarchical: bool = False,
                       tp_size: int = 1) -> Mesh:
    """
    Create a data-parallel (or TP+DP) mesh.

    tp_size=1 (default): pure data parallelism.
      hierarchical=False: 1D mesh ('data',)
      hierarchical=True:  2D mesh ('nodes', 'gpus') for hierarchical AllReduce

    tp_size>1: tensor parallelism + data parallelism.
      tp_size can be intra-node (=gpus_per_node, e.g. 4) or cross-node
      (multiple of gpus_per_node, e.g. 8 = 2 nodes). Cross-node TP uses
      Slingshot (~25 GB/s) for AllReduce instead of NVLink (~478 GB/s).

      Multi-node with tp_size <= gpus_per_node:
        2D mesh ('dp', 'tp') where dp = total_gpus / tp_size.
      Multi-node with tp_size > gpus_per_node (cross-node TP):
        2D mesh ('dp', 'tp') where tp spans multiple nodes.
      Multi-node with tp_size == total_gpus:
        1D mesh ('tp',) — pure TP, no DP.
      Single-node:
        1D mesh ('tp',) for pure tensor parallelism.
    """
    from jax.experimental import mesh_utils
    import numpy as np

    if jax.process_count() > 1:
        devices = jax.devices()
        num_nodes = jax.process_count()
        gpus_per_node = len(jax.local_devices())
        total_gpus = len(devices)
        print(f"[Sharding] Multi-node mode: Process {jax.process_index()}/{num_nodes}")
        print(f"[Sharding] Global mesh with {total_gpus} devices across {num_nodes} processes")

        if tp_size > 1:
            assert total_gpus % tp_size == 0, \
                f"total_gpus={total_gpus} must be divisible by tp_size={tp_size}"
            dp_size = total_gpus // tp_size

            if dp_size == 1:
                # Pure TP across all GPUs, no DP
                mesh = Mesh(np.array(devices).reshape(tp_size), axis_names=('tp',))
                print(f"[Sharding] 1D cross-node TP mesh: {tp_size} tp")
            else:
                # TP+DP: dp_size groups of tp_size GPUs each
                mesh = Mesh(np.array(devices).reshape(dp_size, tp_size),
                            axis_names=('dp', 'tp'))
                if tp_size > gpus_per_node:
                    print(f"[Sharding] 2D cross-node TP+DP mesh: "
                          f"({dp_size} dp, {tp_size} tp spanning "
                          f"{tp_size // gpus_per_node} nodes/group)")
                else:
                    print(f"[Sharding] 2D TP+DP mesh: ({dp_size} dp, {tp_size} tp)")
            return mesh

        if hierarchical:
            devices_2d = np.array(devices).reshape(num_nodes, gpus_per_node)
            mesh = Mesh(devices_2d, axis_names=('nodes', 'gpus'))
            print(f"[Sharding] 2D hierarchical mesh: ({num_nodes} nodes, {gpus_per_node} gpus)")
            return mesh
    else:
        local_devs = jax.local_devices()
        devices = local_devs[:num_devices]
        print(f"[Sharding] Single-node mode: Using {len(devices)} local devices")

        if tp_size > 1:
            assert len(devices) == tp_size, \
                f"tp_size={tp_size} must equal num_devices={len(devices)}"
            mesh = Mesh(np.array(devices).reshape(tp_size), axis_names=('tp',))
            print(f"[Sharding] 1D TP mesh with {tp_size} devices")
            return mesh

    devices_array = np.array(devices).reshape(-1)
    mesh = Mesh(devices_array, axis_names=('data',))
    print(f"[Sharding] Created 1D mesh with {len(devices)} devices along 'data' axis")
    return mesh


def initialize_mesh(num_devices: int, hierarchical: bool = False,
                     tp_size: int = 1) -> Mesh:
    """Initialize the global mesh. Call once at training start."""
    global _GLOBAL_MESH
    _GLOBAL_MESH = create_simple_mesh(num_devices, hierarchical=hierarchical,
                                       tp_size=tp_size)
    return _GLOBAL_MESH


def get_global_mesh() -> Mesh:
    """Get the global mesh."""
    global _GLOBAL_MESH
    if _GLOBAL_MESH is None:
        raise RuntimeError("Mesh not initialized. Call initialize_mesh() first.")
    return _GLOBAL_MESH


def _get_batch_axis(mesh: Mesh):
    """Return the batch PartitionSpec axis based on mesh dimensionality.

    1D mesh ('data',):        returns 'data'
    1D mesh ('tp',):          returns None (pure TP, batch replicated)
    2D mesh ('nodes', 'gpus'): returns ('nodes', 'gpus') for pure DP
    2D mesh ('dp', 'tp'):     returns 'dp' (DP + TP)
    2D mesh ('nodes', 'tp'):  returns 'nodes' (legacy intra-node TP)
    """
    if 'tp' in mesh.axis_names:
        if 'dp' in mesh.axis_names:
            return 'dp'
        if 'nodes' in mesh.axis_names:
            return 'nodes'
        return None  # single-node TP: batch replicated
    if len(mesh.axis_names) == 2 and 'nodes' in mesh.axis_names:
        return ('nodes', 'gpus')
    return 'data'


def create_data_sharding(mesh: Mesh) -> NamedSharding:
    """
    Create sharding for data: batch dimension sharded along all mesh axes.
    1D: P('data', None), 2D: P(('nodes','gpus'), None)
    """
    batch_axis = _get_batch_axis(mesh)
    return NamedSharding(mesh, P(batch_axis, None))


def create_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Create fully replicated sharding (for model parameters)."""
    return NamedSharding(mesh, P(None))


def tree_replicate_to_devices(pytree: Any, sharding: NamedSharding) -> Any:
    """Replicate a pytree to all devices. Replacement for jax_utils.replicate()."""
    return jax.device_put(pytree, sharding)


def tree_unreplicate(pytree: Any) -> Any:
    """
    Extract a single copy from a replicated pytree.
    Replacement for jax_utils.unreplicate().
    For jit+shardings, replicated arrays are already normal arrays — no-op.
    """
    return pytree


def create_state_shardings(state: Any, mesh: Mesh) -> Any:
    """
    Create shardings for train state. All parameters replicated.
    Scalars (rank 0) use P(), arrays use P(None).
    """
    def get_sharding_for_leaf(leaf):
        if isinstance(leaf, jax.Array):
            if leaf.ndim == 0:
                return NamedSharding(mesh, P())
            else:
                return NamedSharding(mesh, P(None))
        else:
            return NamedSharding(mesh, P())

    return jax.tree_util.tree_map(get_sharding_for_leaf, state)


def get_data_shardings_for_batch(
        mesh: Mesh,
        has_book_data: bool = True,
    ) -> Tuple[Any, Any, Any]:
    """
    Create shardings for batch data components.
    Returns (inputs_sharding, labels_sharding, integration_times_sharding).
    Auto-detects 1D vs 2D mesh for correct PartitionSpecs.
    """
    batch_axis = _get_batch_axis(mesh)
    data_sharding_2d = create_data_sharding(mesh)
    data_sharding_1d = NamedSharding(mesh, P(batch_axis))

    if has_book_data:
        inputs_sharding = (data_sharding_2d, data_sharding_2d)
    else:
        inputs_sharding = (data_sharding_2d,)

    labels_sharding = data_sharding_1d

    if has_book_data:
        integration_times_sharding = (data_sharding_2d, data_sharding_2d)
    else:
        integration_times_sharding = (data_sharding_2d,)

    return inputs_sharding, labels_sharding, integration_times_sharding
