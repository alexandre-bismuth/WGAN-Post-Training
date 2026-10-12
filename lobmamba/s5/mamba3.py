"""Mamba3 SISO SSM module for LOBS5.

Implements the Mamba-3 (Dao & Gu, ICLR 2026) SISO variant as a Flax nn.Module
with the same interface as GDNSSM:
  __call__(input_sequence)              -> output_sequence     (L, H) -> (L, H)
  __call_rnn__(hidden, input_seq, rst)  -> (hidden, output)

Two forward paths:
  use_triton=False -> pure JAX einsums via XLA (default / stable path)
  use_triton=True  -> fused Triton kernels via jax_triton_kda (experimental)

Factory: init_Mamba3SSM(H, ...) -> functools.partial[Mamba3SSM]
"""

import math
import os
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import normal, zeros, ones, lecun_normal

from s5 import es_fold


class _FullScaleParam(nn.Module):
    """Holds RMSNorm 'scale' param at FULL d_inner (TP-agnostic shape).

    Param path under parent: <parent>/<name>/scale  (matches nn.RMSNorm).
    """
    full_dim: int

    @nn.compact
    def __call__(self):
        return self.param('scale', ones, (self.full_dim,))


class _FullKernelParam(nn.Module):
    """Holds out_proj 'kernel' param at FULL (d_inner, H) (TP-agnostic shape).

    Param path under parent: <parent>/<name>/kernel  (matches nn.Dense).
    """
    in_full: int
    out_dim: int

    @nn.compact
    def __call__(self):
        return self.param('kernel', lecun_normal(), (self.in_full, self.out_dim))


class Mamba3SSM(nn.Module):
    """Mamba-3 SISO (Single-Input Single-Output) SSM layer.

    Args:
        H:              feature dim (d_model). Overridable via partial(ssm, H=d_book).
        d_state:        SSM state dimension N.
        expand:         expansion factor (d_inner = expand * H).
        headdim:        per-head dimension P.
        chunk_size:     chunkwise parallel chunk size.
        rope_fraction:  fraction of d_state used for RoPE (0.5 or 1.0).
        A_floor:        minimum magnitude for decay (clamp).
        dt_min/dt_max:  range for dt_bias initialization.
        use_triton:     if True, use fused Triton kernels; if False, pure JAX.
        step_rescale:   ACCEPTED but IGNORED — SequenceLayer passes this to all SSMs.
    """
    H: int
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 64
    rope_fraction: float = 0.5
    A_floor: float = 1e-4
    dt_min: float = 0.001
    dt_max: float = 0.1
    use_triton: bool = False
    use_cuda: bool = False  # if True, route SSD phases 4+5+6 through CUDA FFI kernel
    tp_size: int = 1  # tensor parallelism: split heads across this many GPUs
    step_rescale: float = 1.0  # ignored

    def setup(self):
        # Auto-adjust for small H (book pre-layers)
        d_inner_target = self.expand * self.H
        self.eff_headdim = min(self.headdim, d_inner_target)
        self.n_heads = max(1, d_inner_target // self.eff_headdim)
        self.d_inner = self.n_heads * self.eff_headdim

        # TP: number of heads processed by each GPU.
        # Book pre-layers have small H (d_book) giving n_heads not divisible by tp_size.
        # Fall back to tp_size=1 for those layers (they're cheap, TP not needed).
        if self.tp_size > 1 and self.n_heads % self.tp_size != 0:
            self._effective_tp = 1
        else:
            self._effective_tp = self.tp_size
        self.nh_local = self.n_heads // self._effective_tp

        nh = self.n_heads
        hd = self.eff_headdim
        N = self.d_state

        # RoPE config
        self.split_tensor_size = int(N * self.rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2

        # Single fused input projection (full size, each GPU computes all then slices):
        # [z, x, B, C, dd_dt, dd_A, trap, angles]
        d_in_proj = (2 * self.d_inner +       # z, x
                     2 * N +                   # B, C (ngroups=1, mimo_rank=1)
                     3 * nh +                  # dd_dt, dd_A, trap
                     self.num_rope_angles)     # angles
        self.in_proj = nn.Dense(d_in_proj, use_bias=False)

        # dt_bias: initialized from uniform in [dt_min, dt_max] (full nh)
        def dt_bias_init(key, shape):
            u = jax.random.uniform(key, shape)
            dt = jnp.exp(u * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min))
            dt = jnp.clip(dt, a_min=1e-4)
            return dt + jnp.log(-jnp.expm1(-dt))
        self.dt_bias = self.param("dt_bias", dt_bias_init, (nh,))

        # B and C biases (full nh)
        self.B_bias = self.param("B_bias", ones, (nh, N))
        self.C_bias = self.param("C_bias", ones, (nh, N))

        # BCNorm (RMSNorm on B and C, shared across heads)
        # MAMBA3_EPS_COMPAT=1: reproduce the pre-2026-04-22 "epsilon trap"
        # (upstream ffe45d66) — ckpts trained before that fix (e.g. the 78M
        # s46050 anchor, trained 2026-03-28) used nn.RMSNorm(N) which sets
        # epsilon=N (d_state), making BCNorm a near-constant scaling. Those
        # weights require the SAME epsilon at inference; the upstream "fix"
        # silently changes the forward pass under them.
        _eps_compat = os.environ.get("MAMBA3_EPS_COMPAT", "0") == "1"
        if _eps_compat:
            self.B_norm = nn.RMSNorm(epsilon=float(N))
            self.C_norm = nn.RMSNorm(epsilon=float(N))
        else:
            self.B_norm = nn.RMSNorm()
            self.C_norm = nn.RMSNorm()

        # D feedthrough (full nh)
        self.D = self.param("D", ones, (nh,))

        # Output norm + projection: params held at FULL d_inner regardless of TP.
        # Under TP, scale and kernel are runtime-sliced per device, and the
        # RMSNorm sum-of-squares is AllReduced across the tp axis so the
        # normalization matches single-device math exactly. This makes ckpts
        # interchangeable across tp_size values.
        self.out_norm = _FullScaleParam(self.d_inner)
        self.out_proj = _FullKernelParam(self.d_inner, self.H)
        # Pre-2026-04-22 ckpts: out_norm was nn.RMSNorm(self.d_inner), i.e.
        # epsilon=d_inner (=1024 for the 78M anchor). See BCNorm note above.
        self._out_norm_eps = float(self.d_inner) if _eps_compat else 1e-6

    def _tp_head_start(self):
        """Return starting head index for this TP shard (0 when tp_size=1)."""
        if self._effective_tp <= 1:
            return 0
        if self.is_initializing():
            return 0  # at init time, no shard_map context
        try:
            return jax.lax.axis_index('tp') * self.nh_local
        except NameError:
            # Outside shard_map (e.g., eval step with jax.jit).
            # Default to shard 0. Eval produces partial output (1/tp of heads).
            # TODO: make eval use shard_map for correct TP eval.
            return 0

    def _preprocess(self, input_sequence, es=None):
        """Split input projection and compute SSM inputs.

        With tp_size>1, slices head-dependent outputs to local heads only.
        Returns: (z, x, B, C, ADT, DT, trap, angles) all as (L, ...) tensors.
        es: optional EGGROLL LoRA factors for this SSM (es_fold; None = stock path).
        """
        L, _ = input_sequence.shape
        nh = self.n_heads
        nhl = self.nh_local
        hd = self.eff_headdim
        N = self.d_state

        # Project (full d_in_proj on every GPU, then slice)
        proj = self.in_proj(input_sequence)  # (L, d_in_proj)
        proj = es_fold.fold_kernel(es, "in_proj", input_sequence, proj)

        # Split: [z, x, B, C, dd_dt, dd_A, trap, angles]
        s = [self.d_inner, self.d_inner, N, N, nh, nh, nh, self.num_rope_angles]
        cum = []
        c = 0
        for v in s[:-1]:
            c += v
            cum.append(c)
        z, x, B, C, dd_dt, dd_A, trap, angles = jnp.split(proj, cum, axis=-1)

        # Reshape x, z to (L, nh, hd), then slice to local heads
        x = x.reshape(-1, nh, hd)
        z = z.reshape(-1, nh, hd)
        if self._effective_tp > 1:
            hs = self._tp_head_start()
            x = jax.lax.dynamic_slice_in_dim(x, hs, nhl, axis=1)
            z = jax.lax.dynamic_slice_in_dim(z, hs, nhl, axis=1)
            dd_dt = dd_dt.reshape(-1, nh)
            dd_dt = jax.lax.dynamic_slice_in_dim(dd_dt, hs, nhl, axis=1)
            dd_A = dd_A.reshape(-1, nh)
            dd_A = jax.lax.dynamic_slice_in_dim(dd_A, hs, nhl, axis=1)
            trap = trap.reshape(-1, nh)
            trap = jax.lax.dynamic_slice_in_dim(trap, hs, nhl, axis=1)

        # B, C: (L, N) -> (L, nhl, N) via broadcast (ngroups=1, only local heads)
        B = jnp.repeat(B[:, None, :], nhl, axis=1)
        C = jnp.repeat(C[:, None, :], nhl, axis=1)

        # BCNorm
        B = self.B_norm(B)
        C = self.C_norm(C)

        # Slice per-head params to local heads
        if self._effective_tp > 1:
            hs = self._tp_head_start()
            dt_bias_l = jax.lax.dynamic_slice(self.dt_bias, (hs,), (nhl,))
        else:
            dt_bias_l = self.dt_bias

        # Compute A, DT, ADT
        A = jnp.clip(-jax.nn.softplus(dd_A.astype(jnp.float32)), a_max=-self.A_floor)
        DT = jax.nn.softplus(dd_dt.astype(jnp.float32) + dt_bias_l)
        ADT = A * DT

        # Transpose for kernel convention: (L, nhl) -> (nhl, L)
        ADT_t = ADT.T
        DT_t = DT.T
        trap_t = trap.T

        # angles: (L, num_rope_angles) -> (L, nhl, num_rope_angles) via broadcast
        angles = jnp.repeat(angles[:, None, :], nhl, axis=1)

        return z, x, B, C, ADT_t, DT_t, trap_t, angles

    def _tp_local_biases(self):
        """Return per-head biases sliced to local heads for TP."""
        if self._effective_tp <= 1:
            return self.C_bias, self.B_bias, self.D
        hs = self._tp_head_start()
        nhl = self.nh_local
        N = self.d_state
        C_bias_l = jax.lax.dynamic_slice(self.C_bias, (hs, 0), (nhl, N))
        B_bias_l = jax.lax.dynamic_slice(self.B_bias, (hs, 0), (nhl, N))
        D_l = jax.lax.dynamic_slice(self.D, (hs,), (nhl,))
        return C_bias_l, B_bias_l, D_l

    def _post_norm_and_project(self, y, es=None):
        """RMSNorm with global mean-of-squares + row-parallel out_proj.

        y: (L, nh_local*hd) per device. Under TP, AllReduces sum-of-squares
        across the tp axis so the RMS denominator matches single-device math,
        slices scale and kernel rows to the local head group, and AllReduces
        partial out_proj outputs across tp. Mathematically equivalent to
        TP=1 forward when all devices hold identical params.
        es: optional EGGROLL LoRA factors (es_fold). ES folding is only supported
        at tp=1 (the ES path never runs under TP; A rows are not TP-sliced).
        """
        if es is not None and es_fold.subtree(es, "out_proj") is not None:
            assert self._effective_tp == 1, "EGGROLL out_proj fold requires tp_size==1"
        hd = self.eff_headdim

        # Global RMS via local sum-of-squares + cross-tp psum
        scale_full = self.out_norm()  # (d_inner,)
        ss_local = jnp.sum(jax.lax.square(y.astype(jnp.float32)), axis=-1,
                            keepdims=True)
        if self._effective_tp > 1 and not self.is_initializing():
            try:
                ss_global = jax.lax.psum(ss_local, axis_name='tp')
            except NameError:
                ss_global = ss_local  # outside shard_map (eval), local only
        else:
            ss_global = ss_local
        rms_inv = jax.lax.rsqrt(ss_global / self.d_inner + self._out_norm_eps)

        if self._effective_tp > 1:
            hs = self._tp_head_start()
            scale_local = jax.lax.dynamic_slice(
                scale_full, (hs * hd,), (self.nh_local * hd,))
        else:
            scale_local = scale_full
        y = (y * rms_inv.astype(y.dtype)) * scale_local

        # Row-parallel out_proj: slice kernel rows to local heads, matmul, psum
        kernel_full = self.out_proj()  # (d_inner, H)
        if self._effective_tp > 1:
            hs = self._tp_head_start()
            kernel_local = jax.lax.dynamic_slice(
                kernel_full, (hs * hd, 0), (self.nh_local * hd, self.H))
        else:
            kernel_local = kernel_full
        y = es_fold.fold_kernel(es, "out_proj", y, y @ kernel_local)

        if self._effective_tp > 1 and not self.is_initializing():
            try:
                y = jax.lax.psum(y, axis_name='tp')
            except NameError:
                pass  # outside shard_map (eval), skip AllReduce
        return y

    def __call__(self, input_sequence, es=None):
        """Chunkwise parallel forward (training mode).

        Args:
            input_sequence: (L, H)
            es: optional EGGROLL LoRA factors for this SSM (None = stock path)
        Returns:
            output: (L, H)
        """
        z, x, B, C, ADT, DT, trap, angles = self._preprocess(input_sequence, es=es)
        C_bias_l, B_bias_l, D_l = self._tp_local_biases()

        if self.use_cuda:
            try:
                y = self._forward_cuda(C, B, x, ADT, DT, trap, angles, z,
                                       C_bias_l, B_bias_l, D_l)
            except (ImportError, FileNotFoundError) as e:
                y = self._forward_jax(C, B, x, ADT, DT, trap, angles, z,
                                      C_bias_l, B_bias_l, D_l)
        elif self.use_triton and self.rope_fraction > 0:
            try:
                from s5.mamba3_ops import mamba3_siso_triton
                y = mamba3_siso_triton(
                    C, B, x, ADT, DT, trap,
                    C_bias_l, B_bias_l, angles,
                    D_l, z, self.chunk_size)
            except ImportError:
                y = self._forward_jax(C, B, x, ADT, DT, trap, angles, z,
                                      C_bias_l, B_bias_l, D_l)
        else:
            y = self._forward_jax(C, B, x, ADT, DT, trap, angles, z,
                                  C_bias_l, B_bias_l, D_l)

        # Reshape (L, nhl, hd) -> (L, nhl*hd) and apply post-SSM norm+project.
        y = y.reshape(-1, self.nh_local * self.eff_headdim)
        return self._post_norm_and_project(y, es=es)

    def _forward_jax(self, C, B, x, ADT, DT, trap, angles, z,
                     Q_bias, K_bias, D):
        """Pure JAX forward path."""
        from s5.mamba3_jax import mamba3_ssd_chunked_jax
        return mamba3_ssd_chunked_jax(
            Q=C, K=B, V=x,
            ADT=ADT, DT=DT, Trap=trap,
            angles=angles,
            Q_bias=Q_bias, K_bias=K_bias,
            D=D, Z=z,
            chunk_size=self.chunk_size,
        )

    def _forward_cuda(self, C, B, x, ADT, DT, trap, angles, z,
                      Q_bias, K_bias, D):
        """CUDA FFI forward path: phases 1-3 + 7 in JAX, phases 4-5-6 in CUDA.

        The CUDA kernel (state_scan_cuda) replaces the JAX einsum block that
        does per-chunk state accumulation, cross-chunk sequential scan, and
        state-to-output projection. All other phases stay in JAX.
        """
        from s5.mamba3_jax import mamba3_ssd_chunked_jax
        from s5.state_scan_ops import state_scan_cuda

        def _phases_4_5_6_cuda(Q_c, K_c, V_c, ADT_c, A_cumsum):
            # Q_c, K_c: (nc, CS, H, N); V_c: (nc, CS, H, P); ADT_c: (H, nc, CS)
            # CUDA kernel takes flat (B=1, L, H, *) inputs.
            nc, CS, H, N = Q_c.shape
            P = V_c.shape[-1]
            L = nc * CS
            # Cast bf16 (kernel asserts bf16 inputs) and add batch dim.
            Q_b = Q_c.reshape(1, L, H, N).astype(jnp.bfloat16)
            K_b = K_c.reshape(1, L, H, N).astype(jnp.bfloat16)
            V_b = V_c.reshape(1, L, H, P).astype(jnp.bfloat16)
            # ADT_c is (H, nc, CS); kernel wants (B, H, L) f32.
            ADT_b = ADT_c.reshape(1, H, L).astype(jnp.float32)
            Y_off_b = state_scan_cuda(K_b, V_b, Q_b, ADT_b)  # (1, L, H, P) bf16
            # Reshape back to chunks and cast to V's dtype to match JAX path.
            return Y_off_b[0].reshape(nc, CS, H, P).astype(V_c.dtype)

        return mamba3_ssd_chunked_jax(
            Q=C, K=B, V=x,
            ADT=ADT, DT=DT, Trap=trap,
            angles=angles,
            Q_bias=Q_bias, K_bias=K_bias,
            D=D, Z=z,
            chunk_size=self.chunk_size,
            phases_4_5_6_fn=_phases_4_5_6_cuda,
        )

    def __call_rnn__(self, hidden, input_sequence, resets, es=None):
        """Sequential scan forward (inference/generation mode).

        Args:
            hidden: (angle_state, ssm_state, k_state, v_state) — 4-tuple
            input_sequence: (L, H)
            resets: (L,) or None — unused
            es: optional EGGROLL LoRA factors for this SSM (None = stock path)
        Returns:
            (new_hidden, output): same hidden structure, output (L, H)
        """
        from s5.mamba3_jax import apply_rope

        angle_state, ssm_state, k_state, v_state = hidden
        # Remove leading singleton dims: (1, ...) -> (...)
        angle_state = angle_state[0]  # (nh, num_rope_angles)
        ssm_state = ssm_state[0]      # (nh, hd, N)
        k_state = k_state[0, 0]       # (nh, N)
        v_state = v_state[0]          # (nh, hd)

        z, x, B, C, ADT, DT, trap_logits, angles = self._preprocess(input_sequence, es=es)
        # Transpose back: (nh, L) -> (L, nh)
        ADT_seq = ADT.T   # (L, nh)
        DT_seq = DT.T     # (L, nh)
        trap_seq = jax.nn.sigmoid(trap_logits.T)  # (L, nh) — apply sigmoid here for RNN

        C_bias_l, B_bias_l, D_l = self._tp_local_biases()
        nh = self.n_heads
        hd = self.eff_headdim
        N = self.d_state
        PI = jnp.float32(math.pi)

        def rnn_step(carry, t_inputs):
            ang_s, ssm_s, k_prev, v_prev = carry
            x_t, z_t, B_t, C_t, adt_t, dt_t, trap_t, angle_t = t_inputs

            # Update angle state
            angle_delta = jnp.tanh(angle_t) * PI * dt_t[:, None]  # (nh, num_rope_angles)
            ang_s = ang_s + angle_delta

            # Apply RoPE to B, C with biases
            cos_a = jnp.cos(ang_s)
            sin_a = jnp.sin(ang_s)
            B_biased = B_t + B_bias_l  # (nhl, N)
            C_biased = C_t + C_bias_l
            B_rot = apply_rope(B_biased, cos_a, sin_a)
            C_rot = apply_rope(C_biased, cos_a, sin_a)

            # Trapezoidal 3-term recurrence:
            # h = exp(A*dt)*h + (1-trap)*dt*exp(A*dt)*(k_prev outer v_prev) + trap*dt*(B_rot outer x_t)
            decay = jnp.exp(adt_t)[:, None, None]  # (nh, 1, 1)
            ssm_s = decay * ssm_s  # decay old state

            # Trapezoidal terms
            trap_t_expanded = trap_t[:, None, None]  # (nh, 1, 1)
            dt_expanded = dt_t[:, None, None]

            # Term from previous step (shifted)
            prev_outer = v_prev[:, :, None] * k_prev[:, None, :]  # (nh, hd, N)
            ssm_s = ssm_s + (1 - trap_t_expanded) * dt_expanded * decay * prev_outer

            # Term from current step
            curr_outer = x_t[:, :, None] * B_rot[:, None, :]  # (nh, hd, N)
            ssm_s = ssm_s + trap_t_expanded * dt_expanded * curr_outer

            # Output: y = C @ h + D * x
            y_t = jnp.einsum('hn,hpn->hp', C_rot, ssm_s)  # (nh, hd)
            y_t = y_t + D_l[:, None] * x_t

            # QK-dot skip (gamma * sum(Q*K) * V)
            gamma = dt_t * trap_t  # (nh,)
            qk_dot = jnp.sum(C_rot * B_rot, axis=-1)  # (nh,)
            y_t = y_t + (qk_dot * gamma)[:, None] * x_t

            # SiLU gating
            y_t = y_t * (z_t * jax.nn.sigmoid(z_t))

            new_carry = (ang_s, ssm_s, B_rot, x_t)
            return new_carry, y_t

        # Stack inputs along L dimension for scan
        scan_inputs = (x, z, B, C, ADT_seq, DT_seq, trap_seq, angles)

        init_carry = (angle_state, ssm_state, k_state, v_state)
        final_carry, y_seq = jax.lax.scan(rnn_step, init_carry, scan_inputs)

        # Re-add leading singleton dims for hidden state
        ang_f, ssm_f, k_f, v_f = final_carry
        new_hidden = (ang_f[None], ssm_f[None], k_f[None, None], v_f[None])

        # Reshape output: (L, nhl, hd) -> (L, nhl*hd), then norm + project
        y_seq = y_seq.reshape(-1, self.nh_local * self.eff_headdim)
        output = self._post_norm_and_project(y_seq, es=es)

        return new_hidden, output


def init_Mamba3SSM(H, d_state=128, expand=2, headdim=64, chunk_size=64,
                   rope_fraction=0.5, A_floor=1e-4, dt_min=0.001, dt_max=0.1,
                   use_triton=False, use_cuda=False, tp_size=1):
    """Create a Mamba3SSM partial — same pattern as init_GDN_SSM.

    Returns functools.partial[Mamba3SSM] with all config bound except step_rescale.
    """
    return partial(
        Mamba3SSM,
        H=H,
        d_state=d_state,
        expand=expand,
        headdim=headdim,
        chunk_size=chunk_size,
        rope_fraction=rope_fraction,
        A_floor=A_floor,
        dt_min=dt_min,
        dt_max=dt_max,
        use_triton=use_triton,
        use_cuda=use_cuda,
        tp_size=tp_size,
    )
