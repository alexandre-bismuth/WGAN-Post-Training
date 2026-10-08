"""Mamba3 SISO chunked algorithm — pure JAX reference implementation.

Implements the Mamba-3 Structured State Space Duality (SSD) forward pass
using JAX einsums, letting XLA handle kernel fusion. This is the SISO
(single-input single-output) variant without MIMO projections.

Use cases:
  1. Fallback if Triton kernels have issues on GH200 / ARM
  2. Speed comparison baseline (XLA fusion vs hand-written Triton)
  3. Reference for correctness testing against the Triton kernel

Algorithm overview (Mamba-3 vs Mamba-2):
  - Trapezoidal discretization: scale = gamma + shifted_gamma, creating a
    3-term recurrence (current K*V via gamma, previous K*V via shifted_gamma)
  - RoPE on B,C (called K,Q in attention notation): cumulative angles
    computed as cumsum(tanh(angles) * pi * dt) mod 2pi
  - BCNorm: RMSNorm on B,C before bias addition (handled by caller)
  - QK-dot diagonal: sum(Q * K, dim=-1) * gamma, added (replaces attention diagonal)
  - SiLU gating with z: out = out * silu(z)
  - D feedthrough: out += D * V

Notation mapping (SSM <-> Attention):
  Q = C (state-to-output)    K = B (input-to-state)    V = x (input)
  ADT = A * dt (log-decay)   DT = dt (time delta)

Interface: no batch dim — LOBS5 uses vmap for batching.

References:
  - Mamba-3 paper: https://arxiv.org/abs/2502.09992
  - Mamba-3 Triton kernel: mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py
  - Mamba-3 reference test: tests/ops/triton/test_mamba3_siso.py
  - Mamba-2 JAX port: experiments/exp_K3_Mamba2/s5/mamba2.py
"""
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def segsum(x):
    """Stable segment sum for 1-semiseparable causal mask.

    Computes cumulative sums along the last axis, producing a lower-triangular
    matrix of pairwise differences. Used to build the causal decay mask for SSD.

    This is the "stable" variant from the Mamba-2 paper (Listing 1), which
    avoids catastrophic cancellation by zeroing above-diagonal entries before
    the cumsum rather than after.

    Args:
        x: (..., T)
    Returns:
        (..., T, T) — lower-triangular cumulative sums, -inf above diagonal.
    """
    T = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def apply_rope(x, cos_angles, sin_angles):
    """Apply rotary position embeddings to x.

    Splits x into pairs (x0, x1) along the last dim, then applies:
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

    If cos/sin have fewer elements than x's pair count (partial RoPE),
    the remaining dimensions pass through unchanged — cos is padded with
    1.0 and sin with 0.0.

    Args:
        x: (..., d) — input tensor, d must be even.
        cos_angles: (..., d_rope) — cosine of cumulative angles.
        sin_angles: (..., d_rope) — sine of cumulative angles.
    Returns:
        (..., d) — rotated tensor, same shape as x.
    """
    # Split into even/odd pairs: (..., d) -> (..., d//2, 2) -> two (..., d//2)
    d = x.shape[-1]
    d_half = d // 2
    x_pairs = x.reshape(*x.shape[:-1], d_half, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]

    # Pad cos/sin if partial RoPE (d_rope < d_half)
    d_rope = cos_angles.shape[-1]
    if d_rope < d_half:
        pad_width = [(0, 0)] * (cos_angles.ndim - 1) + [(0, d_half - d_rope)]
        cos_angles = jnp.pad(cos_angles, pad_width, constant_values=1.0)
        sin_angles = jnp.pad(sin_angles, pad_width, constant_values=0.0)

    out0 = x0 * cos_angles - x1 * sin_angles
    out1 = x0 * sin_angles + x1 * cos_angles
    # Interleave back: (..., d_half, 2) -> (..., d)
    out = jnp.stack([out0, out1], axis=-1).reshape(*x.shape[:-1], d)
    return out


def angle_dt_cumsum(angles, dt):
    """Compute cumulative angle-dt product for RoPE.

    angles_cumsum[t] = sum_{i=0}^{t} tanh(angles[i]) * pi * dt[i]

    Then reduced mod 2*pi for numerical stability.

    Args:
        angles: (L, nheads, num_rope_angles) — raw angle rates from in_proj.
        dt: (nheads, L) — time deltas (note: transposed layout from Triton).
    Returns:
        (L, nheads, num_rope_angles) — cumulative angles mod 2*pi.
    """
    PI = jnp.float32(jnp.pi)
    TWO_PI = 2.0 * PI
    # angles_scaled[l, h, r] = tanh(angles[l,h,r]) * pi * dt[h, l]
    angles_scaled = jnp.tanh(angles) * PI * dt.T[..., None]  # (L, nheads, num_rope_angles)
    angles_cumsum = jnp.cumsum(angles_scaled, axis=0)  # cumsum over L
    # Mod 2*pi for stability
    angles_cumsum = angles_cumsum - TWO_PI * jnp.floor(angles_cumsum / TWO_PI)
    return angles_cumsum


# ---------------------------------------------------------------------------
# Main chunked forward pass
# ---------------------------------------------------------------------------

def _phases_4_5_6_jax(Q_c, K_c, V_c, ADT_c, A_cumsum):
    """JAX reference for SSD phases 4+5+6.

    Inputs (chunked): produced by phases 1-3 of mamba3_ssd_chunked_jax.
      Q_c:     (nc, CS, nheads, d_state)  rotated query (= C after RoPE)
      K_c:     (nc, CS, nheads, d_state)  scaled key   (= B after RoPE * trap_scale)
      V_c:     (nc, CS, nheads, headdim)  value (= x)
      ADT_c:   (nheads, nc, CS)           log-decay per chunk
      A_cumsum:(nheads, nc, CS)           intra-chunk cumulative log-decay

    Returns:
      Y_off:   (nc, CS, nheads, headdim)  cross-chunk contribution

    This is the swappable seam — replace with a CUDA FFI implementation that
    has the same input/output signature and produces numerically equivalent
    results within bf16 tolerance (rel_err ~ 4e-3).
    """
    nheads = Q_c.shape[2]
    headdim = V_c.shape[-1]
    d_state = Q_c.shape[-1]

    # Phase 4: per-chunk state accumulation (V * decay_to_chunk_end) @ K
    decay_states = jnp.exp(A_cumsum[:, :, -1:] - A_cumsum)  # (nheads, nc, CS)
    states = jnp.einsum('cshp,hcs,cshn->chpn', V_c, decay_states, K_c)
    # states: (nc, nheads, headdim, d_state)

    # Phase 5: cross-chunk sequential scan (state propagation)
    initial_state = jnp.zeros((1, nheads, headdim, d_state), dtype=V_c.dtype)
    states = jnp.concatenate([initial_state, states], axis=0)  # (nc+1, ...)
    A_end = A_cumsum[:, :, -1]  # (nheads, nc)
    A_end_padded = jnp.pad(A_end, ((0, 0), (1, 0)))  # (nheads, nc+1)
    decay_chunk = jnp.exp(segsum(A_end_padded))  # (nheads, nc+1, nc+1)
    new_states = jnp.einsum('hzc,chpn->zhpn', decay_chunk, states)
    states = new_states[:-1]  # state ENTERING each chunk

    # Phase 6: state -> output (cross-chunk contribution)
    state_decay_out = jnp.exp(A_cumsum)  # (nheads, nc, CS)
    Y_off = jnp.einsum('clhn,chpn,hcl->clhp', Q_c, states, state_decay_out)
    return Y_off


def mamba3_ssd_chunked_jax(
    Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z,
    chunk_size=64,
    phases_4_5_6_fn=_phases_4_5_6_jax,
):
    """Pure JAX Mamba3 SISO chunked forward pass.

    Implements the 4-stage SSD algorithm from Mamba-2, extended with Mamba-3's
    trapezoidal discretization, RoPE on Q/K, QK-dot skip, and SiLU gating.

    The computation proceeds as:
      1. Preprocessing: cumulative RoPE angles, bias addition, QK-dot skip,
         rotary embedding, trapezoidal scale/gamma.
      2. Intra-chunk: quadratic attention form within each chunk.
      3. Cross-chunk: sequential state propagation across chunks.
      4. Output: combine intra + cross-chunk, apply skip, gating, feedthrough.

    Args:
        Q: (L, nheads, d_state)     — query (= C after BCNorm, before bias/RoPE)
        K: (L, nheads, d_state)     — key   (= B after BCNorm, before bias/RoPE)
        V: (L, nheads, headdim)     — value (= x, the input)
        ADT: (nheads, L)            — log-decay A*dt (negative values)
        DT: (nheads, L)             — time deltas dt (positive)
        Trap: (nheads, L)           — trapezoidal mixing (pre-sigmoid logits)
        angles: (L, nheads, num_rope_angles) — rotation angle rates
        Q_bias: (nheads, d_state)   — query bias (= C_bias)
        K_bias: (nheads, d_state)   — key bias   (= B_bias)
        D: (nheads,)                — feedthrough weight
        Z: (L, nheads, headdim)     — gating tensor
        chunk_size: int

    Returns:
        y: (L, nheads, headdim)
    """
    L_orig = Q.shape[0]
    nheads = Q.shape[1]
    d_state = Q.shape[2]
    headdim = V.shape[2]
    CS = chunk_size

    # ===================================================================
    # Phase 1: Preprocessing — RoPE, biases, trapezoidal scale, QK-dot
    # ===================================================================

    # 1a. Cumulative angles for RoPE
    angles_cumsum = angle_dt_cumsum(angles, DT)   # (L, nheads, num_rope_angles)
    cos_angles = jnp.cos(angles_cumsum)
    sin_angles = jnp.sin(angles_cumsum)

    # 1b. Apply sigmoid to Trap (raw logits from in_proj)
    Trap_sig = jax.nn.sigmoid(Trap)  # (nheads, L)

    # 1c. Compute trapezoidal scale and gamma
    #   gamma[h, t]         = dt[h, t] * trap[h, t]
    #   shifted_gamma[h, t] = dt[h, t+1] * (1 - trap[h, t+1])
    #   scale[h, t]         = gamma[h, t] + shifted_gamma[h, t]
    gamma = DT * Trap_sig                           # (nheads, L)
    DT_shifted = jnp.pad(DT[:, 1:], ((0, 0), (0, 1)))       # shift left, pad 0
    Trap_shifted = jnp.pad(Trap_sig[:, 1:], ((0, 0), (0, 1)))
    shifted_gamma = DT_shifted * (1.0 - Trap_shifted)        # (nheads, L)
    scale = gamma + shifted_gamma                             # (nheads, L)

    # 1d. Add biases to Q, K
    Q_biased = Q + Q_bias[None, :, :]  # (L, nheads, d_state)
    K_biased = K + K_bias[None, :, :]  # (L, nheads, d_state)

    # 1e. QK-dot skip connection: sum(Q * K, dim=-1) * gamma
    #   The Triton kernel uses strictly lower-triangular attention (excluding
    #   diagonal) and then adds QK_dot*gamma on the diagonal. This avoids
    #   non-causal numerical leakage from the diagonal of QK^T.
    QK_dot = jnp.sum(Q_biased * K_biased, axis=-1)  # (L, nheads)
    QK_dot = QK_dot * gamma.T                        # (L, nheads)  -- gamma is (nheads, L)

    # 1f. Apply RoPE to Q and K (after bias, before scaling)
    Q_rot = apply_rope(Q_biased, cos_angles, sin_angles)  # (L, nheads, d_state)
    K_rot = apply_rope(K_biased, cos_angles, sin_angles)  # (L, nheads, d_state)

    # 1g. Scale K by trapezoidal scale
    K_scaled = K_rot * scale.T[..., None]  # (L, nheads, d_state), scale.T is (L, nheads)

    # ===================================================================
    # Phase 2: Pad and reshape to chunks
    # ===================================================================

    pad_len = (CS - L_orig % CS) % CS
    def pad_L(x, ndim):
        """Pad along the first (L) axis."""
        if pad_len == 0:
            return x
        pads = [(0, pad_len)] + [(0, 0)] * (ndim - 1)
        return jnp.pad(x, pads)

    Q_rot = pad_L(Q_rot, 3)
    K_scaled = pad_L(K_scaled, 3)
    V_p = pad_L(V, 3)
    Z_p = pad_L(Z, 3)
    QK_dot_p = pad_L(QK_dot, 2)

    # ADT needs padding on the L axis (axis=1, shape is (nheads, L))
    if pad_len > 0:
        ADT_p = jnp.pad(ADT, ((0, 0), (0, pad_len)))
    else:
        ADT_p = ADT

    L_padded = L_orig + pad_len
    nc = L_padded // CS

    # Reshape to (nc, CS, ...)
    Q_c = Q_rot.reshape(nc, CS, nheads, d_state)       # (nc, CS, nheads, d_state)
    K_c = K_scaled.reshape(nc, CS, nheads, d_state)     # (nc, CS, nheads, d_state)
    V_c = V_p.reshape(nc, CS, nheads, headdim)          # (nc, CS, nheads, headdim)
    Z_c = Z_p.reshape(nc, CS, nheads, headdim)          # (nc, CS, nheads, headdim)
    QK_dot_c = QK_dot_p.reshape(nc, CS, nheads)         # (nc, CS, nheads)
    ADT_c = ADT_p.reshape(nheads, nc, CS)               # (nheads, nc, CS)

    # ===================================================================
    # Phase 3: Intra-chunk — quadratic attention form
    # ===================================================================

    # Cumulative decay within each chunk
    A_cumsum = jnp.cumsum(ADT_c, axis=-1)  # (nheads, nc, CS)

    # Causal decay mask: exp(segsum(ADT_c))
    # segsum produces (nheads, nc, CS, CS) lower-triangular decay weights
    L_mat = jnp.exp(segsum(ADT_c))  # (nheads, nc, CS, CS)

    # Strictly lower-triangular attention: QK^T * L_mat
    # Y_diag[c, l, h, p] = sum_s sum_n Q[c,l,h,n] * K[c,s,h,n] * L[h,c,l,s] * V[c,s,h,p]
    # But we must use STRICTLY lower-triangular (l > s) to avoid
    # non-causal leakage on the diagonal. The diagonal contribution
    # comes from the QK-dot skip connection instead.
    strict_mask = jnp.tril(jnp.ones((CS, CS), dtype=bool), k=-1)  # (CS, CS)
    L_mat_strict = jnp.where(strict_mask[None, None, :, :], L_mat, 0.0)  # (nheads, nc, CS, CS)

    Y_diag = jnp.einsum('clhn,cshn,hcls,cshp->clhp', Q_c, K_c, L_mat_strict, V_c)

    # ===================================================================
    # Phases 4+5+6: per-chunk state accumulation, cross-chunk scan, state-to-output
    # Delegated to a swappable callable so that a CUDA FFI implementation can be
    # substituted without touching phases 1-3 or phase 7.
    # ===================================================================
    Y_off = phases_4_5_6_fn(Q_c, K_c, V_c, ADT_c, A_cumsum)

    # ===================================================================
    # Phase 7: Combine and apply skip, gating, feedthrough
    # ===================================================================

    # Combine intra-chunk + cross-chunk
    Y = Y_diag + Y_off  # (nc, CS, nheads, headdim)

    # D feedthrough + QK-dot diagonal contribution (ADDED, matching Triton kernel)
    # The Triton kernel computes: acc_o += (D + qk_dot) * V
    # where qk_dot = sum(Q*K, dim=-1) * gamma (the diagonal term)
    Y = Y + (D[None, None, :, None] + QK_dot_c[..., None]) * V_c

    # SiLU gating: out = out * silu(z) = out * z * sigmoid(z)
    Y = Y * jax.nn.silu(Z_c)

    # Reshape back to (L_padded, nheads, headdim) and trim padding
    Y = Y.reshape(L_padded, nheads, headdim)
    return Y[:L_orig]
