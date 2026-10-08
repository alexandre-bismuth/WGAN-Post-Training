"""Gated Delta Networks (GDN) / Kimi Delta Attention (KDA) SSM layer.

Drop-in replacement for S5SSM with the same interface:
  __call__(input_sequence: (L, H)) -> (L, H)
  __call_rnn__(hidden, input_sequence, resets) -> hidden, (L, H)

References:
  - GDN: Yang et al., "Gated Delta Networks", ICLR 2025
  - KDA: Kimi Team, "Kimi-VL", 2025 (per-key-dim alpha variant)
  - FLA: Triton-based Flash Linear Attention (naive chunkwise algorithm)
"""
import functools
from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

import os


# ---------------------------------------------------------------------------
# Causal depthwise Conv1d (safe path — no reliance on nn.Conv padding mode)
# ---------------------------------------------------------------------------
class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise 1D convolution: (L, C) -> (L, C) with left-padding."""
    channels: int
    kernel_size: int = 4

    def setup(self):
        k = self.kernel_size
        C = self.channels
        self.kernel = self.param('kernel', lecun_normal(), (k, C))
        self.bias = self.param('bias', nn.initializers.zeros, (C,))

    def _conv(self, x_with_context):
        """Shared depthwise conv logic. x_with_context: (k-1+L, C) → (L, C)."""
        kernel_reshaped = self.kernel[:, None, :]  # (k, 1, C)
        windows = jax.lax.conv_general_dilated(
            x_with_context[None, :, :],
            kernel_reshaped,
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NTC', 'TIO', 'NTC'),
            feature_group_count=self.channels,
        )
        return windows[0] + self.bias

    def __call__(self, x):
        # x: (L, C) — training path: zero-pad left
        x_padded = jnp.pad(x, ((self.kernel_size - 1, 0), (0, 0)))
        return self._conv(x_padded)

    def step(self, buffer, x):
        """Stateful forward for RNN inference: use buffer instead of zero-padding.

        Args:
            buffer: (k-1, C) — previous k-1 inputs
            x: (L, C) — current input (typically L=1)
        Returns:
            new_buffer: (k-1, C), y: (L, C)
        """
        x_cat = jnp.concatenate([buffer, x], axis=0)  # (k-1+L, C)
        y = self._conv(x_cat)
        new_buffer = x_cat[-(self.kernel_size - 1):]
        return new_buffer, y


# ---------------------------------------------------------------------------
# GDNSSM Module
# ---------------------------------------------------------------------------
class GDNSSM(nn.Module):
    """Gated Delta Network SSM layer.

    Same interface as S5SSM:
      __call__(input_sequence) -> output_sequence      (L, H) -> (L, H)
      __call_rnn__(hidden, input_sequence, resets) -> (hidden, output_sequence)

    Args:
        H:             feature dim (d_model). Overridable via partial(ssm, H=d_book).
        num_heads:     number of attention heads.
        head_dim:      key/query dimension per head.
        expand_v:      value expansion factor (head_v_dim = head_dim * expand_v).
        chunk_size:    chunkwise parallel chunk size.
        use_conv:      whether to apply causal depthwise Conv1d(k=4) on q,k,v.
        use_kda:       per-key-dim alpha (KDA) vs per-head scalar alpha (GDN).
        step_rescale:  ACCEPTED but IGNORED — SequenceLayer passes this to all SSMs.
    """
    H: int
    num_heads: int
    head_dim: int = 128
    expand_v: int = 2
    chunk_size: int = 64
    use_conv: bool = True
    use_kda: bool = False
    step_rescale: float = 1.0  # ignored, compatibility with SequenceLayer

    def setup(self):
        # Auto-adjust for small H (book pre-layers where H=d_book)
        self.eff_heads = min(self.num_heads, max(1, self.H // self.head_dim))
        self.eff_head_dim = min(self.head_dim, self.H)
        self.head_v_dim = self.eff_head_dim * self.expand_v

        nh = self.eff_heads
        hd = self.eff_head_dim
        hvd = self.head_v_dim

        # Projections
        self.q_proj = nn.Dense(nh * hd, use_bias=False)
        self.k_proj = nn.Dense(nh * hd, use_bias=False)
        self.v_proj = nn.Dense(nh * hvd, use_bias=False)

        # Beta gate (write strength): per-head scalar
        self.b_proj = nn.Dense(nh, use_bias=True)

        # Alpha gate (decay/erase): per-key-dim (KDA) or per-head (GDN)
        alpha_dim = nh * hd if self.use_kda else nh
        self.gk_proj = nn.Dense(alpha_dim, use_bias=True)

        # Output gate
        self.g_proj = nn.Dense(nh * hvd, use_bias=False)

        # Output projection: merge heads back to H
        self.o_proj = nn.Dense(self.H, use_bias=False)

        # Optional causal Conv1d(k=4) on q, k, v
        if self.use_conv:
            self.q_conv = CausalDepthwiseConv1d(channels=nh * hd, kernel_size=4)
            self.k_conv = CausalDepthwiseConv1d(channels=nh * hd, kernel_size=4)
            self.v_conv = CausalDepthwiseConv1d(channels=nh * hvd, kernel_size=4)

        # RMSNorm per head (applied to output before gating)
        self.out_norm = nn.RMSNorm(hvd)

        # Feedthrough parameter (matches S5 convention)
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

    def __call__(self, input_sequence):
        """Chunkwise parallel forward (training mode).

        Args:
            input_sequence: (L, H)
        Returns:
            output: (L, H)
        """
        L_orig = input_sequence.shape[0]
        x = input_sequence
        nh = self.eff_heads
        hd = self.eff_head_dim
        hvd = self.head_v_dim
        C = self.chunk_size

        # --- Projections ---
        q = self.q_proj(x)  # (L, nh*hd)
        k = self.k_proj(x)  # (L, nh*hd)
        v = self.v_proj(x)  # (L, nh*hvd)

        # Optional Conv1d + SiLU
        if self.use_conv:
            q = nn.silu(self.q_conv(q))
            k = nn.silu(self.k_conv(k))
            v = nn.silu(self.v_conv(v))

        # L2 normalize q, k; scale q by 1/sqrt(head_dim) (per FLA/Qwen3 reference)
        q = q.reshape(L_orig, nh, hd)
        k = k.reshape(L_orig, nh, hd)
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        q = q * (hd ** -0.5)

        v = v.reshape(L_orig, nh, hvd)

        # Beta (write gate): sigmoid, per-head
        beta = jax.nn.sigmoid(self.b_proj(x))  # (L, nh)

        # Alpha (decay gate): log-sigmoid
        gk_raw = self.gk_proj(x)  # (L, nh*hd) or (L, nh)
        alpha_log = jax.nn.log_sigmoid(gk_raw)  # negative values (decay)

        if self.use_kda:
            alpha_log = alpha_log.reshape(L_orig, nh, hd)  # per-key-dim
        else:
            alpha_log = alpha_log.reshape(L_orig, nh, 1)   # per-head, broadcast over hd

        # Output gate
        g = nn.silu(self.g_proj(x)).reshape(L_orig, nh, hvd)  # (L, nh, hvd)

        # --- Pad to multiple of chunk_size ---
        pad_len = (C - L_orig % C) % C
        L_padded = L_orig + pad_len
        num_chunks = L_padded // C

        if pad_len > 0:
            q = jnp.pad(q, ((0, pad_len), (0, 0), (0, 0)))
            k = jnp.pad(k, ((0, pad_len), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, pad_len), (0, 0), (0, 0)))
            beta = jnp.pad(beta, ((0, pad_len), (0, 0)))
            alpha_log = jnp.pad(alpha_log, ((0, pad_len), (0, 0), (0, 0)))
            g = jnp.pad(g, ((0, pad_len), (0, 0), (0, 0)))

        # Reshape to chunks: (num_chunks, C, nh, dim)
        q = q.reshape(num_chunks, C, nh, hd)
        k = k.reshape(num_chunks, C, nh, hd)
        v = v.reshape(num_chunks, C, nh, hvd)
        beta = beta.reshape(num_chunks, C, nh)
        alpha_log = alpha_log.reshape(num_chunks, C, nh, -1)  # (nc, C, nh, hd or 1)
        g = g.reshape(num_chunks, C, nh, hvd)

        # --- Chunkwise parallel computation ---
        o = _chunkwise_gdn(q, k, v, beta, alpha_log, nh, hd, hvd, C, num_chunks)

        # --- Post-processing ---
        # o: (num_chunks, C, nh, hvd)
        o = o.reshape(L_padded, nh, hvd)
        o = o[:L_orig]  # unpad
        g = g.reshape(L_padded, nh, hvd)[:L_orig]

        # RMSNorm per head, then multiply by output gate
        o = self.out_norm(o.reshape(-1, hvd)).reshape(L_orig, nh, hvd)
        o = o * g

        # Merge heads and project
        o = o.reshape(L_orig, nh * hvd)
        o = self.o_proj(o)  # (L, H)

        # Feedthrough
        Du = input_sequence * self.D[None, :]
        return o + Du

    def __call_rnn__(self, hidden, input_sequence, resets):
        """Fused recurrent forward (inference mode).

        Args:
            hidden: (S_carry, (q_buf, k_buf, v_buf)) if use_conv else S_carry
                    S_carry: (1, nh, hvd, hd) float32
                    q_buf/k_buf: (k-1, nh*hd), v_buf: (k-1, nh*hvd)
            input_sequence: (L, H)
            resets: (L,) or None — reset signals (unused for now, kept for interface)
        Returns:
            new_hidden: same structure as hidden
            output: (L, H)
        """
        L = input_sequence.shape[0]
        x = input_sequence
        nh = self.eff_heads
        hd = self.eff_head_dim
        hvd = self.head_v_dim

        # --- Unpack hidden state ---
        if self.use_conv:
            S_carry, (q_buf, k_buf, v_buf) = hidden
        else:
            S_carry = hidden

        # --- Projections ---
        q = self.q_proj(x)  # (L, nh*hd)
        k = self.k_proj(x)
        v = self.v_proj(x)  # (L, nh*hvd)

        # --- Conv1d with stateful buffers (key fix for per-token inference) ---
        if self.use_conv:
            q_buf, q = self.q_conv.step(q_buf, q)
            k_buf, k = self.k_conv.step(k_buf, k)
            v_buf, v = self.v_conv.step(v_buf, v)
            q = nn.silu(q)
            k = nn.silu(k)
            v = nn.silu(v)

        # L2 normalize; scale q by 1/sqrt(head_dim) (per FLA/Qwen3 reference)
        q = q.reshape(L, nh, hd)
        k = k.reshape(L, nh, hd)
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        q = q * (hd ** -0.5)

        v = v.reshape(L, nh, hvd)

        beta = jax.nn.sigmoid(self.b_proj(x))  # (L, nh)

        gk_raw = self.gk_proj(x)
        alpha_log = jax.nn.log_sigmoid(gk_raw)
        if self.use_kda:
            alpha_log = alpha_log.reshape(L, nh, hd)
        else:
            alpha_log = alpha_log.reshape(L, nh, 1)

        g = nn.silu(self.g_proj(x)).reshape(L, nh, hvd)

        # --- Sequential scan ---
        S_init = S_carry[0]  # (nh, hvd, hd)

        def rnn_step(S, inp):
            q_t, k_t, v_t, beta_t, alpha_log_t, g_t = inp
            # S: (nh, hvd, hd)
            # alpha_log_t: (nh, hd) or (nh, 1)
            alpha = jnp.exp(alpha_log_t)  # (nh, hd) or (nh, 1)
            # Decay state: broadcast alpha over hvd dim
            S = S * alpha[:, None, :]  # (nh, hvd, hd) * (nh, 1, hd_or_1)

            # Delta rule: v - S @ k
            Sk = jnp.einsum('nvk,nk->nv', S, k_t)  # (nh, hvd)
            delta = v_t - Sk  # (nh, hvd)

            # Update: S += beta * outer(delta, k)
            S = S + jnp.einsum('nv,nk->nvk', beta_t[:, None] * delta, k_t)

            # Output: o = S @ q
            o_t = jnp.einsum('nvk,nk->nv', S, q_t)  # (nh, hvd)
            return S, o_t

        # Pack inputs for scan
        scan_inputs = (
            q,                 # (L, nh, hd)
            k,                 # (L, nh, hd)
            v,                 # (L, nh, hvd)
            beta,              # (L, nh)
            alpha_log,         # (L, nh, hd or 1)
            g,                 # (L, nh, hvd) — not used in step, but need for output
        )

        S_final, o_seq = jax.lax.scan(rnn_step, S_init, scan_inputs)
        # o_seq: (L, nh, hvd)

        # RMSNorm per head, gating
        o_seq = self.out_norm(o_seq.reshape(-1, hvd)).reshape(L, nh, hvd)
        o_seq = o_seq * g  # (L, nh, hvd)

        # Merge heads
        o_seq = o_seq.reshape(L, nh * hvd)
        o_seq = self.o_proj(o_seq)  # (L, H)

        # Feedthrough
        Du = input_sequence * self.D[None, :]
        output = o_seq + Du

        # --- Pack hidden state ---
        if self.use_conv:
            return (S_final[None], (q_buf, k_buf, v_buf)), output
        else:
            return S_final[None], output


# ---------------------------------------------------------------------------
# Triton-accelerated solve_tril via FLA's hierarchical block inverse
# ---------------------------------------------------------------------------
# Triton solve_tril disabled: eval_step JIT compilation deadlocks with
# custom_vmap + Triton autotuning on multi-node shard_map. Training works
# but eval hangs indefinitely (tested 60min watchdog, jobs 3364535/3364634).
# Using JAX solve_triangular fallback until Triton-eval compat is resolved.
_HAS_TRITON_SOLVE = False


@jax.custom_batching.custom_vmap
def _solve_tril_64(A):
    """Compute (I + A)^{-1} via Triton. Supports vmap (e.g. model.init)."""
    return _solve_tril_64_impl(A)


@_solve_tril_64.def_vmap
def _solve_tril_64_vmap(axis_size, in_batched, A):
    """Handle vmap by folding extra batch dim into B."""
    (batched,) = in_batched
    if batched:
        orig_shape = A.shape
        A = A.reshape(-1, *A.shape[2:])  # merge vmap dim into B
        result = _solve_tril_64_impl(A)
        result = result.reshape(orig_shape)
        return result, True
    return _solve_tril_64_impl(A), False


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _solve_tril_and_apply(L, nc, nh, C, rhs_v, rhs_k):
    """Compute (I - L)^{-1} @ rhs via Triton hierarchical block inverse.

    Forward: Triton kernel (~15x faster than JAX solve_triangular).
    Backward: JAX matmul (backward is not performance-critical).
    """
    # Convert to FLA layout [B, T, H, BT] where B=1, T=nc*C
    A_fla = (-L).transpose(0, 1, 3, 2).reshape(1, nc * C, nh, C)
    Ai_fla = _solve_tril_64(A_fla)
    Ai_2d = Ai_fla.reshape(nc, C, nh, C).transpose(0, 2, 1, 3).reshape(nc * nh, C, C)
    rv = rhs_v.transpose(0, 2, 1, 3).reshape(nc * nh, C, -1)
    rk = rhs_k.transpose(0, 2, 1, 3).reshape(nc * nh, C, -1)
    v_corr = jnp.matmul(Ai_2d, rv).reshape(nc, nh, C, -1).transpose(0, 2, 1, 3)
    k_cumd = jnp.matmul(Ai_2d, rk).reshape(nc, nh, C, -1).transpose(0, 2, 1, 3)
    return v_corr, k_cumd


def _solve_tril_and_apply_fwd(L, nc, nh, C, rhs_v, rhs_k):
    """Forward pass: Triton solve + matmul. Save Ai for backward."""
    A_fla = (-L).transpose(0, 1, 3, 2).reshape(1, nc * C, nh, C)
    Ai_fla = _solve_tril_64(A_fla)
    Ai_2d = Ai_fla.reshape(nc, C, nh, C).transpose(0, 2, 1, 3).reshape(nc * nh, C, C)
    rv = rhs_v.transpose(0, 2, 1, 3).reshape(nc * nh, C, -1)
    rk = rhs_k.transpose(0, 2, 1, 3).reshape(nc * nh, C, -1)
    v_corr = jnp.matmul(Ai_2d, rv).reshape(nc, nh, C, -1).transpose(0, 2, 1, 3)
    k_cumd = jnp.matmul(Ai_2d, rk).reshape(nc, nh, C, -1).transpose(0, 2, 1, 3)
    return (v_corr, k_cumd), (Ai_2d, rv, rk)


def _solve_tril_and_apply_bwd(nc, nh, C, res, g):
    """Backward: pure JAX matmul. Not performance-critical."""
    (Ai_2d, rv, rk) = res
    (dv_corr, dk_cumd) = g

    dv_2d = dv_corr.transpose(0, 2, 1, 3).reshape(nc * nh, C, -1)
    dk_2d = dk_cumd.transpose(0, 2, 1, 3).reshape(nc * nh, C, -1)
    AiT = Ai_2d.transpose(0, 2, 1)

    # d/d(rhs) = Ai^T @ d_output
    d_rhs_v = jnp.matmul(AiT, dv_2d).reshape(nc, nh, C, -1).transpose(0, 2, 1, 3)
    d_rhs_k = jnp.matmul(AiT, dk_2d).reshape(nc, nh, C, -1).transpose(0, 2, 1, 3)

    # d/d(Ai) = d_output @ rhs^T
    d_Ai = jnp.matmul(dv_2d, rv.transpose(0, 2, 1)) + \
           jnp.matmul(dk_2d, rk.transpose(0, 2, 1))

    # d((I+A)^{-1})/d(A) = -Ai^T @ d_Ai @ Ai^T
    d_A = -jnp.matmul(AiT, jnp.matmul(d_Ai, AiT))

    # A_fla = (-L).transpose(0,1,3,2)... -> d_L = -d_A reshaped
    d_L = -d_A.reshape(nc, nh, C, C).transpose(0, 2, 3, 1)

    return (d_L, d_rhs_v, d_rhs_k)


_solve_tril_and_apply.defvjp(_solve_tril_and_apply_fwd, _solve_tril_and_apply_bwd)


# ---------------------------------------------------------------------------
# Chunkwise GDN computation with WY correction (true delta rule)
# ---------------------------------------------------------------------------
def _chunkwise_gdn(q, k, v, beta, alpha_log, nh, hd, hvd, C, num_chunks):
    """Chunkwise gated delta rule with WY correction.

    Implements the algorithm from Yang et al. "Gated Delta Networks" (ICLR 2025),
    matching torch_chunk_gated_delta_rule from FLA/Qwen3-next reference.

    The WY correction ensures __call__ (chunkwise) == __call_rnn__ (recurrent)
    by accounting for within-chunk sequential dependencies in the delta rule.

    Args:
        q:         (num_chunks, C, nh, hd)
        k:         (num_chunks, C, nh, hd)
        v:         (num_chunks, C, nh, hvd)
        beta:      (num_chunks, C, nh)
        alpha_log: (num_chunks, C, nh, hd_or_1)

    Returns:
        o: (num_chunks, C, nh, hvd)
    """
    nc = num_chunks

    # Cumulative decay within each chunk
    decay_cum = jnp.cumsum(alpha_log, axis=1)  # (nc, C, nh, hd_or_1)
    alpha_dim = alpha_log.shape[-1]

    # Masks
    causal_mask = jnp.tril(jnp.ones((C, C)))        # (C, C) incl diagonal
    strict_lower = jnp.tril(jnp.ones((C, C)), k=-1)  # (C, C) excl diagonal

    # Beta-scaled keys and values
    k_beta = k * beta[:, :, :, None]   # (nc, C, nh, hd)
    v_beta = v * beta[:, :, :, None]   # (nc, C, nh, hvd)

    # =====================================================================
    # Step 1: Compute causal decay mask + intra-chunk attention + WY L matrix
    # =====================================================================
    if alpha_dim == 1:
        # GDN: scalar decay per head
        decay_cum_s = decay_cum[:, :, :, 0]  # (nc, C, nh)
        decay_diff = decay_cum_s[:, :, None, :] - decay_cum_s[:, None, :, :]
        causal_4d = causal_mask[None, :, :, None]
        decay_diff_safe = jnp.where(causal_4d, decay_diff, 0.0)
        decay_mask_4d = jnp.exp(decay_diff_safe) * causal_4d  # (nc, C, C, nh)

        # q@k^T via batched matmul (for intra-chunk output attention)
        q_mat = q.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)
        k_mat = k.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)
        qk = jnp.matmul(q_mat, k_mat.swapaxes(-1, -2))
        qk = qk.reshape(nc, nh, C, C).transpose(0, 2, 3, 1)
        intra_attn = qk * decay_mask_4d  # (nc, C, C, nh) — NO beta

        # WY L matrix: -(k_beta @ k^T) * decay_mask, strictly lower triangular
        kb_mat = k_beta.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)
        kk = jnp.matmul(kb_mat, k_mat.swapaxes(-1, -2))
        kk = kk.reshape(nc, nh, C, C).transpose(0, 2, 3, 1)
        L = -kk * decay_mask_4d * strict_lower[None, :, :, None]
    else:
        # KDA: per-dim decay
        decay_cum_i = decay_cum[:, :, None, :, :]  # (nc, C, 1, nh, hd)
        decay_cum_j = decay_cum[:, None, :, :, :]  # (nc, 1, C, nh, hd)
        decay_diff_kda = decay_cum_i - decay_cum_j
        causal_5d = causal_mask[None, :, :, None, None]
        decay_diff_safe = jnp.where(causal_5d, decay_diff_kda, 0.0)
        decay_weights = jnp.exp(decay_diff_safe)  # (nc, C, C, nh, hd)

        # Intra-chunk attention (q@k with per-dim decay, NO beta)
        intra_attn = jnp.einsum('bihd,bjhd,bijhd->bijh',
                                q, k, decay_weights)
        intra_attn = intra_attn * causal_mask[None, :, :, None]

        # WY L matrix: -(k_beta . k * per-dim decay), strictly lower tri
        L = -jnp.einsum('bihd,bjhd,bijhd->bijh',
                         k_beta, k, decay_weights)
        L = L * strict_lower[None, :, :, None]

    # =====================================================================
    # Step 2: WY correction — solve (I - L) x = b
    # =====================================================================
    k_with_decay = k_beta * jnp.exp(decay_cum)  # (nc, C, nh, hd)

    if _HAS_TRITON_SOLVE:
        # Triton hierarchical block inverse (FLA kernel) with custom_vjp
        v_corrected, k_cumdecay = _solve_tril_and_apply(
            L, nc, nh, C, v_beta, k_with_decay)
    else:
        # Fallback: JAX solve_triangular (no Triton available)
        L_mat = L.transpose(0, 3, 1, 2).reshape(nc * nh, C, C)
        I_minus_L = jnp.eye(C)[None, :, :] - L_mat

        vb_mat = v_beta.transpose(0, 2, 1, 3).reshape(nc * nh, C, hvd)
        v_corrected = jax.scipy.linalg.solve_triangular(
            I_minus_L, vb_mat, lower=True)
        v_corrected = v_corrected.reshape(nc, nh, C, hvd).transpose(0, 2, 1, 3)

        kwd_mat = k_with_decay.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)
        k_cumdecay = jax.scipy.linalg.solve_triangular(
            I_minus_L, kwd_mat, lower=True)
        k_cumdecay = k_cumdecay.reshape(nc, nh, C, hd).transpose(0, 2, 1, 3)

    # =====================================================================
    # Step 3: Precompute per-chunk quantities for cross-chunk scan
    # =====================================================================
    chunk_total_decay = jnp.exp(decay_cum[:, -1, :, :])  # (nc, nh, hd_or_1)

    # Raw keys decayed to end of chunk (NOT k_beta — matches recurrent path)
    decay_to_end = jnp.exp(decay_cum[:, -1:, :, :] - decay_cum)  # (nc, C, nh, hd_or_1)
    k_decayed = k * decay_to_end  # (nc, C, nh, hd)

    # --- Precompute state-independent parts (batched across all chunks) ---
    # Split: o_intra = intra_attn @ v_new = intra_attn @ v_corrected - (intra_attn @ k_cumdecay) @ S
    # Split: delta_S = k_decayed^T @ v_new = k_decayed^T @ v_corrected - (k_decayed^T @ k_cumdecay) @ S
    # This moves the large matmuls out of the sequential scan.

    # o_intra_base[c] = intra_attn[c] @ v_corrected[c]  (nc, C, nh, hvd)
    ia_mat = intra_attn.transpose(0, 3, 1, 2).reshape(nc * nh, C, C)       # (nc*nh, C, C)
    vc_mat = v_corrected.transpose(0, 2, 1, 3).reshape(nc * nh, C, hvd)    # (nc*nh, C, hvd)
    o_intra_base = jnp.matmul(ia_mat, vc_mat)                              # (nc*nh, C, hvd)
    o_intra_base = o_intra_base.reshape(nc, nh, C, hvd).transpose(0, 2, 1, 3)  # (nc, C, nh, hvd)

    # A_k[c] = intra_attn[c] @ k_cumdecay[c]  (nc, C, nh, hd)
    kc_mat = k_cumdecay.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)      # (nc*nh, C, hd)
    A_k = jnp.matmul(ia_mat, kc_mat)                                       # (nc*nh, C, hd)
    A_k = A_k.reshape(nc, nh, C, hd).transpose(0, 2, 1, 3)                 # (nc, C, nh, hd)

    # delta_S_base[c] = k_decayed[c]^T @ v_corrected[c]  (nc, nh, hd, hvd)
    kd_mat = k_decayed.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)       # (nc*nh, C, hd)
    delta_S_base = jnp.matmul(kd_mat.swapaxes(-1, -2), vc_mat)             # (nc*nh, hd, hvd)
    delta_S_base = delta_S_base.reshape(nc, nh, hd, hvd).transpose(0, 1, 3, 2)  # (nc, nh, hvd, hd)

    # B_k[c] = k_decayed[c]^T @ k_cumdecay[c]  (nc, nh, hd, hd)
    B_k = jnp.matmul(kd_mat.swapaxes(-1, -2), kc_mat)                      # (nc*nh, hd, hd)
    B_k = B_k.reshape(nc, nh, hd, hd)                                       # (nc, nh, hd, hd)

    # =====================================================================
    # Step 4: Cross-chunk scan with delta correction (precomputed split)
    # =====================================================================
    def scan_fn(S, chunk_data):
        (o_intra_base_c, A_k_c, delta_S_base_c, B_k_c,
         q_c, decay_cum_c, chunk_decay_c) = chunk_data

        # Inter-chunk output: (q * exp(decay_cum)) @ S
        decay_exp = jnp.exp(decay_cum_c)  # (C, nh, hd_or_1)
        dq = decay_exp * q_c  # (C, nh, hd)
        o_inter = jnp.einsum('nvk,cnk->cnv', S, dq)  # (C, nh, hvd)

        # Intra-chunk: precomputed base - correction from state
        # o_intra = intra_attn @ v_corrected - (intra_attn @ k_cumdecay) @ S
        intra_corr = jnp.einsum('cnk,nvk->cnv', A_k_c, S)  # (C, nh, hvd)
        o_intra = o_intra_base_c - intra_corr

        o_c = o_inter + o_intra

        # State update: precomputed base - correction from state
        # delta_S = k_decayed^T @ v_corrected - (k_decayed^T @ k_cumdecay) @ S
        state_corr = jnp.einsum('nkm,nvm->nvk', B_k_c, S)  # (nh, hvd, hd)
        delta_S = delta_S_base_c - state_corr
        S_new = S * chunk_decay_c[:, None, :] + delta_S

        return S_new, o_c

    S_init = jnp.zeros((nh, hvd, hd), dtype=jnp.float32)
    _, o_chunks = jax.lax.scan(
        scan_fn, S_init,
        (o_intra_base, A_k, delta_S_base, B_k,
         q, decay_cum, chunk_total_decay))

    return o_chunks


# ---------------------------------------------------------------------------
# Factory function (matches init_S5SSM interface)
# ---------------------------------------------------------------------------
def init_GDN_SSM(H, num_heads, head_dim=128, expand_v=2, chunk_size=64,
                 use_conv=True, use_kda=False):
    """Create a GDNSSM partial — same pattern as init_S5SSM.

    Returns:
        functools.partial[GDNSSM] with all config bound except step_rescale.
    """
    return partial(GDNSSM, H=H, num_heads=num_heads, head_dim=head_dim,
                   expand_v=expand_v, chunk_size=chunk_size,
                   use_conv=use_conv, use_kda=use_kda)
