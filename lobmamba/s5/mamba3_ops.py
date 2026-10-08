"""JAX triton_call wrappers + custom_vjp for Mamba-3 SISO Triton kernels.

Architecture:
    @jax.custom_vjp
      mamba3_siso_triton(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z)
        fwd: angle_dt_fwd_triton + siso_fwd_triton  -> (Out, residuals)
        bwd: 5 backward kernels + angle_dt_bwd       -> grads tuple

The forward path is fully implemented. Backward kernels are stubbed with
NotImplementedError markers -- fill them in iteratively once the forward
path is verified end-to-end on device.

NOTE: This file targets the non-varlen, non-initial-state case for the
first bring-up.  Varlen / state-passing paths can be added later by
extending the triton_call invocations with the appropriate grid and
constexpr changes.
"""
from __future__ import annotations

import sys
import os
import math
from typing import Optional, Tuple

# jax_triton_kda bridge lives in the K4 experiment
_JT_PKG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "exp_K4_KDA", "jax_triton_kda_pkg",
)
if _JT_PKG not in sys.path:
    sys.path.insert(0, _JT_PKG)

import jax
import jax.numpy as jnp
import numpy as np
import jax_triton_kda as jt

# Import Triton kernel functions (the @triton.jit decorated functions)
from .mamba3_kernels.siso_fwd import mamba3_siso_fwd_kernel
from .mamba3_kernels.siso_bwd import (
    mamba3_siso_bwd_kernel_dzdo,
    mamba3_siso_bwd_kernel_dqkv,
    mamba3_siso_bwd_kernel_rotary_bias_angles,
    mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states,
    mamba3_siso_bwd_kernel_dk_state_post,
)
from .mamba3_kernels.angle_dt import angle_dt_fwd_kernel, angle_dt_bwd_kernel


# ============================================================================
# Constants
# ============================================================================
_CHUNK_SIZE = 64  # Hard-coded to avoid JIT tracing issues with constexpr params

# ============================================================================
# Helpers
# ============================================================================

def _strides(shape):
    """Compute C-contiguous strides *in elements* for a given shape tuple.

    JAX arrays are always C-contiguous, so for shape (d0, d1, d2, d3):
        stride(0) = d1 * d2 * d3
        stride(1) = d2 * d3
        stride(2) = d3
        stride(3) = 1
    """
    ndim = len(shape)
    strides = []
    for i in range(ndim):
        strides.append(int(np.prod(shape[i + 1:])) if i < ndim - 1 else 1)
    return strides


def _next_power_of_2(x):
    """Round up to the next power of 2 (returns x if already a power of 2)."""
    return 1 << (int(x) - 1).bit_length() if x > 0 else 1


# ============================================================================
# angle_dt forward wrapper
# ============================================================================

def angle_dt_fwd_triton(
    angles: jax.Array,
    dt: jax.Array,
    chunk_size: int = 64,
) -> jax.Array:
    """Wrapper for the angle_dt forward Triton kernel.

    Computes cumulative sum of tanh(angles) * pi * dt, mod 2*pi.

    Args:
        angles: (batch, seqlen, nheads, dim)
        dt:     (batch, nheads, seqlen)
        chunk_size: chunk size for chunked processing

    Returns:
        out: (batch, seqlen, nheads, dim) -- cumulative angle output
    """
    batch, seqlen, nheads, dim = angles.shape
    BLOCK_D = _next_power_of_2(dim)

    # Dummy tensors for optional input pointers (INIT_STATE, CU_SEQLENS)
    dummy = jnp.zeros((1,), dtype=angles.dtype)

    # Reordered kernel signature:
    #   [inputs] [all strides] [dims] [outputs from out_shape] [constexpr]
    # Strides include output strides (computed from output shape, passed as scalars).
    out_s = _strides((batch, seqlen, nheads, dim))
    angle_s = _strides(angles.shape)
    dt_s = _strides(dt.shape)

    stride_args = (
        # OUT strides (output, but passed as scalar)
        *out_s,
        # OUTPUT_STATE strides (dummy)
        0, 0, 0,
        # ANGLE strides
        *angle_s,
        # DT strides
        *dt_s,
        # INIT_STATE strides (dummy)
        0, 0, 0,
        # CU_SEQLENS stride
        0,
        # Dimensions
        seqlen, dim,
    )

    result = jt.triton_call(
        # Input arrays only: ANGLE, DT, INIT_STATE, CU_SEQLENS
        angles, dt, dummy, dummy,
        # Scalar strides + dims
        *stride_args,
        kernel=angle_dt_fwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((batch, seqlen, nheads, dim), angles.dtype),  # OUT
            jax.ShapeDtypeStruct((1,), angles.dtype),                          # OUTPUT_STATE (dummy)
        ],
        grid=(nheads, batch),
        name="angle_dt_fwd",
        CHUNK_SIZE=_CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        HAS_INIT_STATE=False,
        RETURN_OUTPUT_STATE=False,
        IS_VARLEN=False,
    )
    return result[0]  # OUT


# ============================================================================
# siso_fwd forward wrapper
# ============================================================================

def siso_fwd_triton(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    ADT: jax.Array,
    DT: jax.Array,
    Trap: jax.Array,
    Q_bias: jax.Array,
    K_bias: jax.Array,
    Angles_Cumsum: jax.Array,
    D: Optional[jax.Array],
    Z: Optional[jax.Array],
    chunk_size: int = 64,
) -> Tuple[jax.Array, dict]:
    """Wrapper for mamba3_siso_fwd_kernel.

    This is the non-varlen, no-initial-state, training path (store_states=True,
    return_final_states=False).

    Args:
        Q:               (batch, seqlen, nheads_qk, headdim_qk)
        K:               (batch, seqlen, nheads_qk, headdim_qk)
        V:               (batch, seqlen, nheads, headdim_v)
        ADT:             (batch, nheads, seqlen)
        DT:              (batch, nheads, seqlen)
        Trap:            (batch, nheads, seqlen)
        Q_bias:          (nheads, headdim_qk)
        K_bias:          (nheads, headdim_qk)
        Angles_Cumsum:   (batch, seqlen, nheads, headdim_angles)
        D:               (nheads,) or None
        Z:               (batch, seqlen, nheads, headdim_v) or None
        chunk_size:      chunk size

    Returns:
        (Out, saved) where saved is a dict of intermediates for backward.
    """
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles_Cumsum.shape[-1]
    nchunks = (seqlen + _CHUNK_SIZE - 1) // _CHUNK_SIZE

    HEADDIM_QK = _next_power_of_2(headdim_qk)
    HEADDIM_V = _next_power_of_2(headdim_v)

    has_D = D is not None
    has_Z = Z is not None

    # Dummy arrays for optional input tensors
    dummy_1d = jnp.zeros((1,), dtype=Q.dtype)
    dummy_D = D if has_D else jnp.zeros((1,), dtype=jnp.float32)
    dummy_Z = Z if has_Z else jnp.zeros((1,), dtype=Q.dtype)

    # ── Compute strides for all tensors (input + output) ──
    # Input strides
    q_s = _strides(Q.shape)
    k_s = _strides(K.shape)
    v_s = _strides(V.shape)
    adt_s = _strides(ADT.shape)
    dt_s = _strides(DT.shape)
    trap_s = _strides(Trap.shape)
    qb_s = _strides(Q_bias.shape)
    kb_s = _strides(K_bias.shape)
    ang_s = _strides(Angles_Cumsum.shape)
    d_s = _strides(dummy_D.shape) if has_D else [0]
    z_s = _strides(dummy_Z.shape) if has_Z else [0, 0, 0, 0]

    # Output shapes + strides (computed from shapes, passed as scalars)
    out_shape_4d = (batch, seqlen, nheads, headdim_v)
    out_s = _strides(out_shape_4d)
    outv_s = _strides(out_shape_4d)
    ssm_states_shape = (batch, nheads, headdim_v, nchunks * headdim_qk)
    ssm_s = _strides(ssm_states_shape)
    da_cs_shape = (batch, nheads, seqlen)
    da_cs_s = _strides(da_cs_shape)
    da_cs_sum_shape = (batch, nheads, nchunks)
    da_cs_sum_s = _strides(da_cs_sum_shape)
    q_store_shape = (batch, seqlen, nheads, headdim_qk)
    q_store_s = _strides(q_store_shape)
    k_store_shape = (batch, seqlen, nheads, headdim_qk)
    k_store_s = _strides(k_store_shape)
    qk_store_shape = (batch, nheads, seqlen)
    qk_store_s = _strides(qk_store_shape)
    scale_store_shape = (batch, nheads, seqlen)
    scale_store_s = _strides(scale_store_shape)
    gamma_store_shape = (batch, nheads, seqlen)
    gamma_store_s = _strides(gamma_store_shape)

    # Reordered kernel signature:
    #   [15 inputs] [45 input strides] [43 output strides] [5 dims]
    #   [12 outputs from out_shape] [9 constexpr]

    stride_scalars = (
        # Input strides
        *q_s, *k_s, *v_s,
        *adt_s, *dt_s, *trap_s,
        *qb_s, *kb_s,
        *ang_s,
        d_s[0] if has_D else 0,
        *(z_s if has_Z else [0, 0, 0, 0]),
        0, 0, 0, 0,                       # stride_init_ssm_state (dummy)
        0, 0, 0,                           # stride_init_k_state (dummy)
        0, 0, 0,                           # stride_init_v_state (dummy)
        0,                                 # stride_cu_seqlen (dummy)
        # Output strides
        *out_s, *outv_s,
        *ssm_s, *da_cs_s, *da_cs_sum_s,
        *q_store_s, *k_store_s,
        *qk_store_s, *scale_store_s, *gamma_store_s,
        0, 0, 0, 0,                        # stride_final_ssm_state (dummy)
        0, 0, 0, 0,                        # stride_final_k_state (dummy)
        # Dimensions
        seqlen, nheads_qk, headdim_qk, headdim_v, headdim_angles,
    )

    out_shapes = [
        jax.ShapeDtypeStruct(out_shape_4d, V.dtype),              # Out
        jax.ShapeDtypeStruct(out_shape_4d, V.dtype),              # Out_v
        jax.ShapeDtypeStruct(ssm_states_shape, jnp.bfloat16),    # SSM_States
        jax.ShapeDtypeStruct(da_cs_shape, jnp.float32),          # DA_CS
        jax.ShapeDtypeStruct(da_cs_sum_shape, jnp.float32),      # DA_CS_SUM
        jax.ShapeDtypeStruct(q_store_shape, Q.dtype),             # Q_store
        jax.ShapeDtypeStruct(k_store_shape, K.dtype),             # K_store
        jax.ShapeDtypeStruct(qk_store_shape, jnp.float32),       # QK_store
        jax.ShapeDtypeStruct(scale_store_shape, jnp.float32),    # Scale_store
        jax.ShapeDtypeStruct(gamma_store_shape, jnp.float32),    # Gamma_store
        jax.ShapeDtypeStruct((1,), jnp.float32),                 # Final_SSM (dummy)
        jax.ShapeDtypeStruct((1,), jnp.float32),                 # Final_K (dummy)
    ]

    results = jt.triton_call(
        # Input arrays only (15): no output buffers in *args
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum,
        dummy_D, dummy_Z,
        dummy_1d, dummy_1d, dummy_1d, dummy_1d,  # Init_SSM, Init_K, Init_V, Cu_Seqlens
        # Scalar strides + dims
        *stride_scalars,
        kernel=mamba3_siso_fwd_kernel,
        out_shape=out_shapes,
        grid=(nheads, batch),
        zeroed_outputs=(2, 4),  # SSM_States (idx 2), DA_CS_SUM (idx 4) need zero-init
        name="mamba3_siso_fwd",
        CHUNK_SIZE=_CHUNK_SIZE,
        HEADDIM_QK=HEADDIM_QK,
        HEADDIM_V=HEADDIM_V,
        STORE_SSM_STATES_ADT_OUTV=True,
        HAS_INITIAL_STATES=False,
        RETURN_FINAL_STATES=False,
        HAS_D=has_D,
        HAS_Z=has_Z,
        IS_VARLEN=False,
    )

    Out, Out_v, SSM_States, DA_CS, DA_CS_SUM = results[0], results[1], results[2], results[3], results[4]
    Q_rot, K_scaled, QK_dot, Scale, Gamma = results[5], results[6], results[7], results[8], results[9]

    saved = {
        "Q": Q,
        "K": K,
        "V": V,
        "ADT": ADT,
        "DT": DT,
        "Trap": Trap,
        "Q_bias": Q_bias,
        "K_bias": K_bias,
        "Angles_Cumsum": Angles_Cumsum,
        "D": D,
        "Z": Z,
        "Out": Out,
        "Out_v": Out_v,
        "SSM_States": SSM_States,
        "DA_CS": DA_CS,
        "DA_CS_SUM": DA_CS_SUM,
        "Q_rot": Q_rot,
        "K_scaled": K_scaled,
        "QK_dot": QK_dot,
        "Scale": Scale,
        "Gamma": Gamma,
        "chunk_size": chunk_size,
        "has_D": has_D,
        "has_Z": has_Z,
    }

    return Out, saved


# ============================================================================
# Backward kernel wrappers
# ============================================================================

def compute_dzdo_triton(
    dOut: jax.Array,
    Z: jax.Array,
    Out_v: jax.Array,
    chunk_size: int = 64,
) -> Tuple[jax.Array, jax.Array]:
    """Backward kernel for Z-gating: computes dZ and scales dO.

    Reordered kernel signature (K4-KDA pattern):
        [inputs] DO, Z, O,
        [strides+dims] <input strides(12)>, <output strides(8)>, seqlen, headdim_v,
        [outputs from out_shape] Dz, DO_scaled,
        [constexpr] CHUNK_SIZE, HEADDIM_V
    """
    batch, seqlen, nheads, headdim_v = dOut.shape
    HEADDIM_V = _next_power_of_2(headdim_v)

    do_s = _strides(dOut.shape)       # (batch, seqlen, nheads, headdim_v)
    z_s = _strides(Z.shape)
    o_s = _strides(Out_v.shape)
    dz_shape = (batch, seqlen, nheads, headdim_v)
    dz_s = _strides(dz_shape)
    do_scaled_shape = (batch, seqlen, nheads, headdim_v)
    do_scaled_s = _strides(do_scaled_shape)

    nchunks = (seqlen + _CHUNK_SIZE - 1) // _CHUNK_SIZE

    stride_scalars = (
        *do_s, *z_s, *o_s,           # input strides (12)
        *dz_s, *do_scaled_s,          # output strides (8)
        seqlen, headdim_v,            # dims (2)
    )

    results = jt.triton_call(
        # Input arrays only (3)
        dOut, Z, Out_v,
        # Scalar strides + dims
        *stride_scalars,
        kernel=mamba3_siso_bwd_kernel_dzdo,
        out_shape=[
            jax.ShapeDtypeStruct(dz_shape, dOut.dtype),         # Dz
            jax.ShapeDtypeStruct(do_scaled_shape, dOut.dtype),  # DO_scaled
        ],
        grid=(nchunks, nheads, batch),
        name="mamba3_siso_bwd_dzdo",
        CHUNK_SIZE=_CHUNK_SIZE,
        HEADDIM_V=HEADDIM_V,
    )
    return results[0], results[1]  # dZ, dO_scaled


def compute_dqkv_triton(
    q_rot: jax.Array,
    k_scaled: jax.Array,
    v: jax.Array,
    da_cs: jax.Array,
    da_cs_sum: jax.Array,
    qk_dot: jax.Array,
    ssm_states: jax.Array,
    do: jax.Array,
    D: Optional[jax.Array],
    chunk_size: int = 64,
) -> Tuple[jax.Array, ...]:
    """Backward: dQ_mid, dK_mid, dV, dADT, dQK_dot, dD.

    Reordered kernel signature (K4-KDA pattern):
        [inputs] Q, K, V, DA_CS, DA_CS_SUM, QK_Dot, D, SSM_States, dO, d_OSSM_State, Cu_Seqlens,
        [strides+dims] <input strides>, <output strides>, <dims>,
        [outputs from out_shape] dQ, dK, dV, dADT, dQK_Dot, dD, d_ISSM_State,
        [constexpr]
    """
    batch, seqlen, nheads_qk, headdim_qk = q_rot.shape
    _, _, nheads, headdim_v = v.shape
    nchunks = (seqlen + _CHUNK_SIZE - 1) // _CHUNK_SIZE

    HEADDIM_QK = _next_power_of_2(headdim_qk)
    HEADDIM_V = _next_power_of_2(headdim_v)
    has_D = D is not None

    dummy_1d = jnp.zeros((1,), dtype=q_rot.dtype)
    dummy_D = D if has_D else jnp.zeros((1,), dtype=jnp.float32)

    # Compute strides for inputs
    q_s = _strides(q_rot.shape)       # (batch, seqlen, nheads_qk, headdim_qk)
    k_s = _strides(k_scaled.shape)
    v_s = _strides(v.shape)           # (batch, seqlen, nheads, headdim_v)
    da_cs_s = _strides(da_cs.shape)   # (batch, nheads, seqlen)
    da_cs_sum_s = _strides(da_cs_sum.shape)  # (batch, nheads, nchunks)
    qk_dot_s = _strides(qk_dot.shape)       # (batch, nheads, seqlen)
    d_s = _strides(dummy_D.shape) if has_D else [0]
    ssm_s = _strides(ssm_states.shape)      # (batch, nheads, headdim_v, nchunks*headdim_qk)
    do_s = _strides(do.shape)               # (batch, seqlen, nheads, headdim_v)

    # Output shapes
    dq_shape = (batch, seqlen, nheads, headdim_qk)
    dk_shape = (batch, seqlen, nheads, headdim_qk)
    dv_shape = (batch, seqlen, nheads, headdim_v)
    dadt_shape = (batch, nheads, seqlen)
    dqk_shape = (batch, nheads, seqlen)
    # dD: (batch, nheads) -- kernel writes per (batch, head) then we sum over batch
    dd_shape = (batch, nheads)
    d_issm_shape = (1,)  # dummy, not used (no initial states)

    dq_s = _strides(dq_shape)
    dk_s = _strides(dk_shape)
    dv_s = _strides(dv_shape)
    dadt_s = _strides(dadt_shape)
    dqk_s = _strides(dqk_shape)
    dd_s = _strides(dd_shape) if has_D else [0, 0]

    stride_scalars = (
        # Input strides
        *q_s, *k_s, *v_s,
        *da_cs_s, *da_cs_sum_s, *qk_dot_s,
        d_s[0] if has_D else 0,
        *ssm_s, *do_s,
        0, 0, 0, 0,     # d_OSSM_State strides (dummy)
        0,               # Cu_Seqlens stride (dummy)
        # Output strides
        *dq_s, *dk_s, *dv_s,
        *dadt_s, *dqk_s, *dd_s,
        0, 0, 0, 0,     # d_ISSM_State strides (dummy)
        # Dimensions
        seqlen, nheads_qk, headdim_qk, headdim_v,
    )

    out_shapes = [
        jax.ShapeDtypeStruct(dq_shape, q_rot.dtype),           # dQ
        jax.ShapeDtypeStruct(dk_shape, k_scaled.dtype),        # dK
        jax.ShapeDtypeStruct(dv_shape, v.dtype),               # dV
        jax.ShapeDtypeStruct(dadt_shape, jnp.float32),         # dADT
        jax.ShapeDtypeStruct(dqk_shape, jnp.float32),         # dQK_Dot
        jax.ShapeDtypeStruct(dd_shape if has_D else (1,), jnp.float32),  # dD
        jax.ShapeDtypeStruct((1,), jnp.float32),               # d_ISSM_State (dummy)
    ]

    results = jt.triton_call(
        # Input arrays only (11)
        q_rot, k_scaled, v, da_cs, da_cs_sum, qk_dot, dummy_D,
        ssm_states, do, dummy_1d, dummy_1d,
        # Scalar strides + dims
        *stride_scalars,
        kernel=mamba3_siso_bwd_kernel_dqkv,
        out_shape=out_shapes,
        grid=(nheads, batch),
        name="mamba3_siso_bwd_dqkv",
        CHUNK_SIZE=_CHUNK_SIZE,
        HEADDIM_QK=HEADDIM_QK,
        HEADDIM_V=HEADDIM_V,
        RECOMPUTE_MASK=False,
        HAS_D_OSSM_STATE=False,
        RETURN_D_ISSM_STATE=False,
        IS_VARLEN=False,
    )

    dQ_mid = results[0]
    dK_mid = results[1]
    dV = results[2]
    dADT = results[3]
    dQK_dot = results[4]
    # dD: (batch, nheads) -> sum over batch to get (nheads,)
    dD = jnp.sum(results[5], axis=0) if has_D else None

    return dQ_mid, dK_mid, dV, dADT, dQK_dot, dD


def compute_dqktheta_triton(
    q: jax.Array,
    k: jax.Array,
    scale: jax.Array,
    gamma: jax.Array,
    q_bias: jax.Array,
    k_bias: jax.Array,
    angles_cumsum: jax.Array,
    dq_in: jax.Array,
    dk_in: jax.Array,
    dqk: jax.Array,
    chunk_size: int = 64,
) -> Tuple[jax.Array, ...]:
    """Backward: dQ, dK, dQ_bias, dK_bias, dAngles_Cumsum, dScale, dGamma.

    Two kernels:
      1) mamba3_siso_bwd_kernel_rotary_bias_angles -- main rotary+bias grads
      2) mamba3_siso_bwd_kernel_dk_state_post -- NOT needed in non-initial-state mode
         (d_ok_state is None when HAS_INITIAL_STATES=False)
    """
    batch, seqlen, nheads_qk, headdim_qk = q.shape
    nheads = scale.shape[1]
    headdim_angles = angles_cumsum.shape[-1]
    nchunks = (seqlen + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    GQA_RATIO = nheads // nheads_qk

    HEADDIM_QK = _next_power_of_2(headdim_qk)
    BLOCK_HEADDIM_QK = min(HEADDIM_QK, 64)

    # Input strides
    q_s = _strides(q.shape)                 # (batch, seqlen, nheads_qk, headdim_qk)
    k_s = _strides(k.shape)
    scale_s = _strides(scale.shape)          # (batch, nheads, seqlen)
    gamma_s = _strides(gamma.shape)
    qb_s = _strides(q_bias.shape)            # (nheads, headdim_qk)
    kb_s = _strides(k_bias.shape)
    ang_s = _strides(angles_cumsum.shape)    # (batch, seqlen, nheads, headdim_angles)
    dq_in_s = _strides(dq_in.shape)          # (batch, seqlen, nheads, headdim_qk)
    dk_in_s = _strides(dk_in.shape)
    dqk_s = _strides(dqk.shape)              # (batch, nheads, seqlen)

    # Output shapes
    dq_shape = (batch, seqlen, nheads_qk, headdim_qk)
    dk_shape = (batch, seqlen, nheads_qk, headdim_qk)
    dangles_shape = (batch, seqlen, nheads, headdim_angles)
    n_qk_chunks = HEADDIM_QK // BLOCK_HEADDIM_QK
    dscale_shape = (batch, nheads, n_qk_chunks, seqlen)
    dgamma_shape = (batch, nheads, n_qk_chunks, seqlen)
    dq_bias_partial_shape = (batch, nchunks, nheads, headdim_qk)
    dk_bias_partial_shape = (batch, nchunks, nheads, headdim_qk)

    dq_s = _strides(dq_shape)
    dk_s_out = _strides(dk_shape)
    dangles_s = _strides(dangles_shape)
    dscale_s = _strides(dscale_shape)
    dgamma_s = _strides(dgamma_shape)
    dq_bias_p_s = _strides(dq_bias_partial_shape)
    dk_bias_p_s = _strides(dk_bias_partial_shape)

    stride_scalars = (
        # Input strides
        *q_s, *k_s, *scale_s, *gamma_s,
        *qb_s, *kb_s, *ang_s,
        *dq_in_s, *dk_in_s, *dqk_s,
        # Output strides
        *dq_s, *dk_s_out, *dangles_s,
        *dscale_s, *dgamma_s,
        *dq_bias_p_s, *dk_bias_p_s,
        # Dimensions
        seqlen, nheads_qk, nheads, headdim_qk, headdim_angles,
    )

    out_shapes = [
        jax.ShapeDtypeStruct(dq_shape, dq_in.dtype),
        jax.ShapeDtypeStruct(dk_shape, dk_in.dtype),
        jax.ShapeDtypeStruct(dangles_shape, angles_cumsum.dtype),
        jax.ShapeDtypeStruct(dscale_shape, scale.dtype),
        jax.ShapeDtypeStruct(dgamma_shape, gamma.dtype),
        jax.ShapeDtypeStruct(dq_bias_partial_shape, jnp.float32),
        jax.ShapeDtypeStruct(dk_bias_partial_shape, jnp.float32),
    ]

    results = jt.triton_call(
        # Input arrays only (10)
        q, k, scale, gamma, q_bias, k_bias, angles_cumsum, dq_in, dk_in, dqk,
        # Scalar strides + dims
        *stride_scalars,
        kernel=mamba3_siso_bwd_kernel_rotary_bias_angles,
        out_shape=out_shapes,
        grid=(nchunks, batch),
        name="mamba3_siso_bwd_rotary_bias",
        CHUNK_SIZE=_CHUNK_SIZE,
        HEADDIM_QK=HEADDIM_QK,
        BLOCK_HEADDIM_QK=BLOCK_HEADDIM_QK,
        GQA_RATIO=GQA_RATIO,
    )

    dQ = results[0]
    dK = results[1]
    dAngles = results[2]
    # dScale: (batch, nheads, n_qk_chunks, seqlen) -> sum over n_qk_chunks dim
    dScale = jnp.sum(results[3], axis=2)
    dGamma = jnp.sum(results[4], axis=2)
    # dQ_bias: (batch, nchunks, nheads, headdim_qk) -> sum over (batch, nchunks)
    dQ_bias = jnp.sum(results[5], axis=(0, 1))
    dK_bias = jnp.sum(results[6], axis=(0, 1))

    return dQ, dK, dQ_bias, dK_bias, dAngles, dScale, dGamma


def compute_ddt_dtrap_triton(
    dscale: jax.Array,
    dgamma: jax.Array,
    dt: jax.Array,
    trap: jax.Array,
    headdim_qk: int,
    headdim_v: int,
) -> Tuple[jax.Array, jax.Array]:
    """Backward: dDT, dTrap from dScale/dGamma.

    Non-varlen, non-initial-state path: only computes Part 1 (dDT, dTrap).

    headdim_qk / headdim_v come from the actual Q.shape[-1] / V.shape[-1] at
    the call site. They feed the kernel's runtime scalar dims and the
    HEADDIM_QK / HEADDIM_V constexpr block-size masks; with the prior
    hardcoded (64, 128) defaults, larger real headdims silently truncated
    memory reads, corrupting the bwd kernel's loads.

    Reordered kernel signature (K4-KDA pattern):
        [inputs] dScale, dGamma, DT, Trap, d_ISSM_State, Input_K, Input_V, Cu_Seqlens,
        [strides+dims] <input strides>, <output strides>, <dims>,
        [outputs from out_shape] dDT, dTrap, dInput_SSM, dInput_K, dInput_V,
        [constexpr]
    """
    batch, nheads, seqlen = dscale.shape

    HEADDIM_V = _next_power_of_2(headdim_v)
    HEADDIM_QK = _next_power_of_2(headdim_qk)

    dummy_1d = jnp.zeros((1,), dtype=dscale.dtype)

    # Input strides
    dscale_s = _strides(dscale.shape)   # (batch, nheads, seqlen)
    dgamma_s = _strides(dgamma.shape)
    dt_s = _strides(dt.shape)
    trap_s = _strides(trap.shape)

    # Output shapes + strides
    ddt_shape = (batch, nheads, seqlen)
    dtrap_shape = (batch, nheads, seqlen)
    ddt_s = _strides(ddt_shape)
    dtrap_s = _strides(dtrap_shape)

    stride_scalars = (
        # Input strides
        *dscale_s, *dgamma_s, *dt_s, *trap_s,
        0, 0, 0, 0,     # d_ISSM_State strides (dummy)
        0, 0, 0,         # Input_K_State strides (dummy)
        0, 0, 0,         # Input_V_State strides (dummy)
        0,               # Cu_Seqlens stride (dummy)
        # Output strides
        *ddt_s, *dtrap_s,
        0, 0, 0, 0,     # dInput_SSM_State strides (dummy)
        0, 0, 0,         # dInput_K_State strides (dummy)
        0, 0, 0,         # dInput_V_State strides (dummy)
        # Dimensions
        seqlen, headdim_v, headdim_qk,
    )

    out_shapes = [
        jax.ShapeDtypeStruct(ddt_shape, jnp.float32),
        jax.ShapeDtypeStruct(dtrap_shape, jnp.float32),
        jax.ShapeDtypeStruct((1,), jnp.float32),  # dInput_SSM (dummy)
        jax.ShapeDtypeStruct((1,), jnp.float32),  # dInput_K (dummy)
        jax.ShapeDtypeStruct((1,), jnp.float32),  # dInput_V (dummy)
    ]

    results = jt.triton_call(
        # Input arrays only (8)
        dscale, dgamma, dt, trap, dummy_1d, dummy_1d, dummy_1d, dummy_1d,
        # Scalar strides + dims
        *stride_scalars,
        kernel=mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states,
        out_shape=out_shapes,
        grid=(nheads, batch),
        name="mamba3_siso_bwd_ddt_dtrap",
        CHUNK_SIZE=_CHUNK_SIZE,  # Was missing — needed by kernel constexpr
        HEADDIM_V=HEADDIM_V,
        HEADDIM_QK=HEADDIM_QK,
        HAS_INPUT_STATE=False,
        IS_VARLEN=False,
    )

    return results[0], results[1]  # dDT, dTrap


def angle_dt_bwd_triton(
    grad_out: jax.Array,
    angles: jax.Array,
    dt: jax.Array,
    chunk_size: int = 64,
) -> Tuple[jax.Array, jax.Array]:
    """Backward for angle_dt cumsum.

    Reordered kernel signature (K4-KDA pattern):
        [inputs] GRAD_OUT, GRAD_OUTPUT_STATE, ANGLE, DT, CU_SEQLENS,
        [strides+dims] <all strides>, <dims>,
        [outputs from out_shape] GRAD_ANGLE, GRAD_DT, GRAD_INIT_STATE,
        [constexpr]
    """
    batch, seqlen, nheads, dim = angles.shape
    BLOCK_D = _next_power_of_2(dim)

    dummy = jnp.zeros((1,), dtype=angles.dtype)

    # Output shapes + strides
    grad_angle_shape = (batch, seqlen, nheads, dim)
    grad_dt_shape = (batch, nheads, seqlen)

    ga_s = _strides(grad_angle_shape)      # output strides (passed as scalars)
    gdt_s = _strides(grad_dt_shape)
    go_s = _strides(grad_out.shape)        # input strides
    ang_s = _strides(angles.shape)
    dt_s = _strides(dt.shape)

    # Reordered kernel signature:
    #   [inputs] [all strides] [dims] [outputs from out_shape] [constexpr]
    stride_scalars = (
        # GRAD_ANGLE strides (output, as scalar)
        *ga_s,
        # GRAD_DT strides (output, as scalar)
        *gdt_s,
        # GRAD_INIT_STATE strides (dummy)
        0, 0, 0,
        # GRAD_OUT strides
        *go_s,
        # GRAD_OUTPUT_STATE strides (dummy)
        0, 0, 0,
        # ANGLE strides
        *ang_s,
        # DT strides
        *dt_s,
        # CU_SEQLENS stride (dummy)
        0,
        # Dimensions
        seqlen, dim,
    )

    results = jt.triton_call(
        # Input arrays only (5): GRAD_OUT, GRAD_OUTPUT_STATE, ANGLE, DT, CU_SEQLENS
        grad_out, dummy, angles, dt, dummy,
        # Scalar strides + dims
        *stride_scalars,
        kernel=angle_dt_bwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(grad_angle_shape, angles.dtype),   # GRAD_ANGLE
            jax.ShapeDtypeStruct(grad_dt_shape, dt.dtype),          # GRAD_DT
            jax.ShapeDtypeStruct((1,), jnp.float32),                # GRAD_INIT_STATE (dummy)
        ],
        grid=(nheads, batch),
        name="angle_dt_bwd",
        CHUNK_SIZE=_CHUNK_SIZE,
        BLOCK_D=BLOCK_D,
        HAS_INIT_STATE=False,
        HAS_GRAD_OUTPUT_STATE=False,
        IS_VARLEN=False,
    )

    return results[0], results[1]  # grad_angle, grad_dt


# ============================================================================
# Combined forward / backward
# ============================================================================

def mamba3_siso_fwd(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    ADT: jax.Array,
    DT: jax.Array,
    Trap: jax.Array,
    Q_bias: jax.Array,
    K_bias: jax.Array,
    Angles: jax.Array,
    D: Optional[jax.Array],
    Z: Optional[jax.Array],
    chunk_size: int = 64,
) -> Tuple[jax.Array, dict]:
    """Full forward: angle_dt_fwd + siso_fwd.

    Returns (output, residuals_for_bwd).
    """
    # Step 1: Compute cumulative angles
    Angles_Cumsum = angle_dt_fwd_triton(Angles, DT, chunk_size=chunk_size)

    # Step 2: Run the main SISO forward kernel
    Out, saved = siso_fwd_triton(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum,
        D, Z, chunk_size=chunk_size,
    )

    # Add Angles (pre-cumsum) to saved for backward
    saved["Angles"] = Angles

    return Out, saved


def mamba3_siso_bwd(
    saved: dict,
    grad_output: jax.Array,
    chunk_size: int = 64,
) -> Tuple[jax.Array, ...]:
    """Full backward: 5 bwd kernels + angle_dt_bwd.

    Follows the same 5-step structure as _Mamba3Function.backward() in
    siso_combined.py:
        1. compute_dzdo (if Z gating)
        2. compute_dqkv
        3. compute_dqktheta
        4. compute_ddt_dtrap_dinput_states
        5. angle_dt_bwd

    Returns gradient tuple matching forward arg order:
        (dQ, dK, dV, dADT, dDT, dTrap, dQ_bias, dK_bias, dAngles, dD, dZ)
    """
    Q = saved["Q"]
    K = saved["K"]
    V = saved["V"]
    ADT = saved["ADT"]
    DT = saved["DT"]
    Trap = saved["Trap"]
    Q_bias = saved["Q_bias"]
    K_bias = saved["K_bias"]
    Angles = saved["Angles"]
    Angles_Cumsum = saved["Angles_Cumsum"]
    D = saved["D"]
    Z = saved["Z"]
    Out = saved["Out"]
    Out_v = saved["Out_v"]
    SSM_States = saved["SSM_States"]
    DA_CS = saved["DA_CS"]
    DA_CS_SUM = saved["DA_CS_SUM"]
    Q_rot = saved["Q_rot"]
    K_scaled = saved["K_scaled"]
    QK_dot = saved["QK_dot"]
    Scale = saved["Scale"]
    Gamma = saved["Gamma"]
    cs = saved["chunk_size"]
    has_D = saved["has_D"]
    has_Z = saved["has_Z"]

    # Step 1: Z-gating gradient
    if has_Z:
        dZ, grad_out_scaled = compute_dzdo_triton(grad_output, Z, Out_v, chunk_size=cs)
    else:
        dZ = None
        grad_out_scaled = grad_output

    # Step 2: Main gradients (dQ_mid, dK_mid, dV, dADT, dQK_dot, dD)
    dQ_mid, dK_mid, dV, dADT, dQK_dot, dD = compute_dqkv_triton(
        Q_rot, K_scaled, V, DA_CS, DA_CS_SUM, QK_dot, SSM_States,
        grad_out_scaled, D, chunk_size=cs,
    )

    # Step 3: Rotary + bias gradients
    dQ, dK, dQ_bias, dK_bias, dAngles_Cumsum, dScale, dGamma = compute_dqktheta_triton(
        Q, K, Scale, Gamma, Q_bias, K_bias, Angles_Cumsum,
        dQ_mid, dK_mid, dQK_dot, chunk_size=cs,
    )

    # Step 4: dDT, dTrap (pass real headdims so kernel masks/loads match
    # the actual Q/V shapes — was hardcoded to (128, 64) and silently
    # truncated for larger headdim_qk)
    headdim_qk = Q.shape[-1]
    headdim_v = V.shape[-1]
    dDT, dTrap = compute_ddt_dtrap_triton(
        dScale, dGamma, DT, Trap,
        headdim_qk=headdim_qk, headdim_v=headdim_v,
    )

    # Step 5: angle_dt backward
    dAngles, dDT_angle = angle_dt_bwd_triton(
        dAngles_Cumsum, Angles, DT, chunk_size=cs,
    )

    # Accumulate DT gradients from angle_dt backward
    dDT = dDT + dDT_angle

    return (dQ, dK, dV, dADT, dDT, dTrap, dQ_bias, dK_bias, dAngles, dD, dZ)


# ============================================================================
# custom_vjp public API
# ============================================================================

@jax.custom_vjp
def _mamba3_impl(
    Q: jax.Array,
    K: jax.Array,
    V: jax.Array,
    ADT: jax.Array,
    DT: jax.Array,
    Trap: jax.Array,
    Q_bias: jax.Array,
    K_bias: jax.Array,
    Angles: jax.Array,
    D: Optional[jax.Array] = None,
    Z: Optional[jax.Array] = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Mamba-3 SISO attention via Triton kernels.

    Drop-in replacement for the PyTorch mamba3_siso_combined().
    Supports JAX autodiff via custom_vjp.

    Args:
        Q:       (batch, seqlen, nheads_qk, headdim_qk)
        K:       (batch, seqlen, nheads_qk, headdim_qk)
        V:       (batch, seqlen, nheads, headdim_v)
        ADT:     (batch, nheads, seqlen)  -- A * dt decay
        DT:      (batch, nheads, seqlen)  -- dt time delta
        Trap:    (batch, nheads, seqlen)  -- trapezoidal factor
        Q_bias:  (nheads, headdim_qk)
        K_bias:  (nheads, headdim_qk)
        Angles:  (batch, seqlen, nheads, headdim_angles)  -- raw angle rates
        D:       (nheads,) or None  -- skip connection
        Z:       (batch, seqlen, nheads, headdim_v) or None  -- gating

    Returns:
        Out:     (batch, seqlen, nheads, headdim_v)
    """
    # Direct path (no vmap): inputs are unbatched (L, nh, dim). Add batch=1 for Triton.
    Q, K, V = Q[None], K[None], V[None]
    ADT, DT, Trap = ADT[None], DT[None], Trap[None]
    Angles = Angles[None]
    if Z is not None and Z.ndim == 3:
        Z = Z[None]
    out, _ = mamba3_siso_fwd(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, chunk_size)
    return out[0]  # remove batch dim


def _mamba3_impl_fwd(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, chunk_size=64):
    """Forward rule for custom_vjp: returns (primal_out, residuals)."""
    # Direct path: add batch=1
    Q, K, V = Q[None], K[None], V[None]
    ADT, DT, Trap = ADT[None], DT[None], Trap[None]
    Angles = Angles[None]
    if Z is not None and Z.ndim == 3:
        Z = Z[None]
    out, saved = mamba3_siso_fwd(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, chunk_size)
    # Pack saved dict as a tuple for JAX pytree compatibility.
    # custom_vjp residuals must be a valid JAX pytree (arrays + static).
    # We separate arrays from non-array metadata.
    residuals = (
        saved["Q"],
        saved["K"],
        saved["V"],
        saved["ADT"],
        saved["DT"],
        saved["Trap"],
        saved["Q_bias"],
        saved["K_bias"],
        saved["Angles"],
        saved["Angles_Cumsum"],
        saved["D"] if saved["D"] is not None else jnp.zeros((1,), dtype=jnp.float32),
        saved["Z"] if saved["Z"] is not None else jnp.zeros((1,), dtype=jnp.float32),
        saved["Out"],
        saved["Out_v"],
        saved["SSM_States"],
        saved["DA_CS"],
        saved["DA_CS_SUM"],
        saved["Q_rot"],
        saved["K_scaled"],
        saved["QK_dot"],
        saved["Scale"],
        saved["Gamma"],
    )
    return out[0], residuals  # squeeze batch dim for direct path


def _mamba3_impl_bwd(residuals, grad_output):
    """Backward rule for custom_vjp (direct/unbatched path)."""
    (
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
        D_or_dummy, Z_or_dummy,
        Out, Out_v, SSM_States, DA_CS, DA_CS_SUM,
        Q_rot, K_scaled, QK_dot, Scale, Gamma,
    ) = residuals
    # Residuals have batch=1 dim from fwd. grad_output is unbatched — add batch=1.
    grad_output = grad_output[None]

    # Detect whether D/Z were actually present from shape heuristics.
    # D is (nheads,) when present, (1,) when dummy.
    nheads = V.shape[2]
    has_D = D_or_dummy.shape == (nheads,)
    has_Z = Z_or_dummy.ndim == 4 and Z_or_dummy.shape[2] == nheads

    saved = {
        "Q": Q, "K": K, "V": V, "ADT": ADT, "DT": DT, "Trap": Trap,
        "Q_bias": Q_bias, "K_bias": K_bias,
        "Angles": Angles, "Angles_Cumsum": Angles_Cumsum,
        "D": D_or_dummy if has_D else None,
        "Z": Z_or_dummy if has_Z else None,
        "Out": Out, "Out_v": Out_v,
        "SSM_States": SSM_States, "DA_CS": DA_CS, "DA_CS_SUM": DA_CS_SUM,
        "Q_rot": Q_rot, "K_scaled": K_scaled, "QK_dot": QK_dot,
        "Scale": Scale, "Gamma": Gamma,
        "chunk_size": 64,  # default; cannot pass int through residuals
        "has_D": has_D, "has_Z": has_Z,
    }

    (dQ, dK, dV, dADT, dDT, dTrap, dQ_bias, dK_bias, dAngles, dD, dZ) = \
        mamba3_siso_bwd(saved, grad_output)

    # Gradient for D: None -> zeros
    if dD is None:
        dD = jnp.zeros_like(D_or_dummy)
    # Gradient for Z: None -> zeros
    if dZ is None:
        dZ = jnp.zeros_like(Z_or_dummy)

    # Squeeze batch=1 from gradients (direct path outputs are unbatched)
    dQ, dK, dV = dQ[0], dK[0], dV[0]
    dADT, dDT, dTrap = dADT[0], dDT[0], dTrap[0]
    dAngles = dAngles[0]
    if dZ is not None and dZ.ndim == 4:
        dZ = dZ[0]

    return (dQ, dK, dV, dADT, dDT, dTrap, dQ_bias, dK_bias, dAngles, dD, dZ, None)


_mamba3_impl.defvjp(_mamba3_impl_fwd, _mamba3_impl_bwd)


# ── Batched path (for jax.vmap) ──────────────────────────────────────────────
# Same logic as _mamba3_impl but receives tensors with leading batch dim from vmap.
# Triton kernels handle batch natively (batch is in the grid), so fwd/bwd are identical.

@jax.custom_vjp
def _mamba3_impl_batched(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z):
    out, _ = mamba3_siso_fwd(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, 64)
    return out

def _mamba3_impl_batched_fwd(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z):
    out, saved = mamba3_siso_fwd(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, 64)
    # Pack residuals — same logic as _mamba3_impl_fwd
    residuals = (
        saved["Q"], saved["K"], saved["V"],
        saved["ADT"], saved["DT"], saved["Trap"],
        saved["Q_bias"], saved["K_bias"],
        saved["Angles"], saved["Angles_Cumsum"],
        saved["D"] if saved["D"] is not None else jnp.zeros((1,), dtype=jnp.float32),
        saved["Z"] if saved["Z"] is not None else jnp.zeros((1,), dtype=jnp.float32),
        saved["Out"], saved["Out_v"], saved["SSM_States"],
        saved["DA_CS"], saved["DA_CS_SUM"],
        saved["Q_rot"], saved["K_scaled"], saved["QK_dot"],
        saved["Scale"], saved["Gamma"],
    )
    return out, residuals

def _mamba3_impl_batched_bwd(residuals, grad_output):
    # Unlike _mamba3_impl_bwd, grad_output already has batch dim — don't add [None].
    (Q, K, V, ADT, DT, Trap, Q_bias, K_bias,
     Angles, Angles_Cumsum, D_or_dummy, Z_or_dummy,
     Out, Out_v, SSM_States, DA_CS, DA_CS_SUM,
     Q_rot, K_scaled, QK_dot, Scale, Gamma,
    ) = residuals

    nheads = V.shape[2]
    has_D = D_or_dummy.shape == (nheads,)
    has_Z = Z_or_dummy.ndim == 4 and Z_or_dummy.shape[2] == nheads

    saved = {
        "Q": Q, "K": K, "V": V, "ADT": ADT, "DT": DT, "Trap": Trap,
        "Q_bias": Q_bias, "K_bias": K_bias,
        "Angles": Angles, "Angles_Cumsum": Angles_Cumsum,
        "D": D_or_dummy if has_D else None,
        "Z": Z_or_dummy if has_Z else None,
        "Out": Out, "Out_v": Out_v,
        "SSM_States": SSM_States, "DA_CS": DA_CS, "DA_CS_SUM": DA_CS_SUM,
        "Q_rot": Q_rot, "K_scaled": K_scaled, "QK_dot": QK_dot,
        "Scale": Scale, "Gamma": Gamma,
        "chunk_size": 64, "has_D": has_D, "has_Z": has_Z,
    }

    (dQ, dK, dV, dADT, dDT, dTrap, dQ_bias, dK_bias, dAngles, dD, dZ) = \
        mamba3_siso_bwd(saved, grad_output)

    if dD is None:
        dD = jnp.zeros_like(D_or_dummy)
    if dZ is None:
        dZ = jnp.zeros_like(Z_or_dummy)

    # No squeeze — batched path returns batched gradients
    return (dQ, dK, dV, dADT, dDT, dTrap, dQ_bias, dK_bias, dAngles, dD, dZ)

_mamba3_impl_batched.defvjp(_mamba3_impl_batched_fwd, _mamba3_impl_batched_bwd)


# ── Public API with custom_vmap ──────────────────────────────────────────────
from jax.custom_batching import custom_vmap

@custom_vmap
def mamba3_siso_triton(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, chunk_size=64):
    """Mamba-3 SISO via Triton kernels. Supports vmap + autodiff."""
    return _mamba3_impl(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, chunk_size)

@mamba3_siso_triton.def_vmap
def _mamba3_vmap_rule(axis_size, in_batched,
                      Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, chunk_size):
    # Under nn.vmap, batched args already have the leading B dim.
    # Triton kernels handle B>1 natively (batch is a grid dimension).
    o = _mamba3_impl_batched(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z)
    return o, True
