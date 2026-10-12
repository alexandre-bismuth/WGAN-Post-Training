"""K3 v3 State Scan with N-split: Python wrapper.

Two-kernel pipeline:
  1. siso_state_scan: outputs (N_SPLITS, B, L, H, P) partial results
  2. reduce_y_off:    sums N_SPLITS partials -> (B, L, H, P) final

N-split=2 enables 2 CTAs/SM on GH200 (SMEM ~100 KB each) and doubles
the grid size (24 CTAs for H=12 B=1 instead of 12).
"""
import os
import sys
import jax
import jax.numpy as jnp

_CS = 64
_N = 128  # d_state (exp_R1_Mamba3 baseline)
_P = 64
_N_SPLITS = 8  # must match SS_N_SPLITS in siso_state_scan.cu (N_LOCAL=16 at N=128, hits WMMA_K minimum)

_registered = False


def _ensure_registered():
    global _registered
    if _registered:
        return
    build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "m3_kernels", "build")
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import state_scan_ffi
    for name, capsule in state_scan_ffi.registrations().items():
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA")
    _registered = True


def _state_scan_cuda_fwd_only(K_scaled, V, Q_rot, ADT):
    """Raw forward call to the CUDA FFI kernel — no gradient defined."""
    _ensure_registered()

    B, L, H, N_dim = K_scaled.shape
    P_dim = V.shape[-1]
    assert N_dim == _N, f"d_state must be {_N}, got {N_dim}"
    assert P_dim == _P, f"headdim must be {_P}, got {P_dim}"
    assert L % _CS == 0, f"L must be divisible by chunk_size {_CS}, got {L}"

    # Step 1: Compute N-split partials
    partial_type = jax.ShapeDtypeStruct((_N_SPLITS, B, L, H, P_dim), jnp.bfloat16)
    Y_partial = jax.ffi.ffi_call(
        "siso_state_scan",
        partial_type,
        vmap_method='sequential',
    )(K_scaled, V, Q_rot, ADT)

    # Step 2: Reduce across N-splits
    out_type = jax.ShapeDtypeStruct((B, L, H, P_dim), jnp.bfloat16)
    Y_off = jax.ffi.ffi_call(
        "reduce_y_off",
        out_type,
        vmap_method='sequential',
    )(Y_partial)

    return Y_off


def _state_scan_jax_reference(K_scaled, V, Q_rot, ADT):
    """Pure JAX reference for the same compute (Phases 4+5+6).

    Used as the backward-pass fallback for state_scan_cuda since the CUDA
    kernel is forward-only. Inputs/outputs match the CUDA kernel exactly:
      K_scaled, V, Q_rot: (B, L, H, *) bf16
      ADT:                (B, H, L) f32
      Y_off:              (B, L, H, P) bf16

    Implementation mirrors mamba3_jax.py phases 4+5+6 but with batch dim and
    differentiable through standard jax.grad.
    """
    B, L, H, N_dim = K_scaled.shape
    P_dim = V.shape[-1]
    nc = L // _CS

    # Reshape to chunks (cast to f32 for stable accumulation)
    K_c = K_scaled.reshape(B, nc, _CS, H, N_dim).astype(jnp.float32)
    V_c = V.reshape(B, nc, _CS, H, P_dim).astype(jnp.float32)
    Q_c = Q_rot.reshape(B, nc, _CS, H, N_dim).astype(jnp.float32)
    ADT_c = ADT.reshape(B, H, nc, _CS)

    # Phase 4: per-chunk state accumulation
    A_cumsum = jnp.cumsum(ADT_c, axis=-1)
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
    states = jnp.einsum('bcshp,bhcs,bcshn->bchpn', V_c, decay_states, K_c)

    # Phase 5: cross-chunk sequential scan (segsum-based)
    initial = jnp.zeros((B, 1, H, P_dim, N_dim), dtype=jnp.float32)
    states_ext = jnp.concatenate([initial, states], axis=1)
    A_end = A_cumsum[..., -1]
    A_end_pad = jnp.pad(A_end, ((0, 0), (0, 0), (1, 0)))

    def _segsum_1d(x):
        T = x.shape[-1]
        xc = jnp.cumsum(x, axis=-1)
        x_segsum = xc[..., :, None] - xc[..., None, :]
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        return jnp.where(mask, x_segsum, -jnp.inf)

    decay_chunk = jnp.exp(jnp.vectorize(_segsum_1d, signature='(n)->(n,n)')(A_end_pad))
    new_states = jnp.einsum('bhzc,bchpn->bzhpn', decay_chunk, states_ext)
    states_in = new_states[:, :-1]

    # Phase 6: state -> output
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum('bclhn,bchpn,bhcl->bclhp', Q_c, states_in, state_decay_out)

    return Y_off.reshape(B, L, H, P_dim).astype(jnp.bfloat16)


@jax.custom_vjp
def state_scan_cuda(K_scaled, V, Q_rot, ADT):
    """Forward-CUDA + backward-JAX wrapper for SSD phases 4+5+6.

    Forward path uses the fused CUDA kernel for speed; backward path falls
    back to the JAX reference (autodiff'd by jax.grad). Net effect: forward
    speedup, backward unchanged. Use this in training where gradients are
    needed.

    Inputs/outputs match the FFI-only path:
      K_scaled, V, Q_rot: (B, L, H, *) bf16
      ADT:                (B, H, L) f32
      Y_off:              (B, L, H, P) bf16
    """
    return _state_scan_cuda_fwd_only(K_scaled, V, Q_rot, ADT)


def _state_scan_cuda_fwd(K_scaled, V, Q_rot, ADT):
    """Custom-VJP forward: CUDA forward, save inputs as residuals."""
    Y = _state_scan_cuda_fwd_only(K_scaled, V, Q_rot, ADT)
    residuals = (K_scaled, V, Q_rot, ADT)
    return Y, residuals


def _state_scan_cuda_bwd(residuals, dY):
    """Custom-VJP backward: AD through the JAX reference."""
    K_scaled, V, Q_rot, ADT = residuals

    def _ref(K_, V_, Q_, A_):
        return _state_scan_jax_reference(K_, V_, Q_, A_)

    _, vjp_fn = jax.vjp(_ref, K_scaled, V, Q_rot, ADT)
    return vjp_fn(dY)


state_scan_cuda.defvjp(_state_scan_cuda_fwd, _state_scan_cuda_bwd)
