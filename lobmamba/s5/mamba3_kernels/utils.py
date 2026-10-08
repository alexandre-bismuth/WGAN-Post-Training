"""
Mamba-3 Util Functions.

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

import triton
import triton.language as tl

# We use PTX approximations instead of triton built-in functions
# to trade off a bit of accuracy for much faster speed.

@triton.jit
def cos_approx(x):
    """
    (Fast) Cosine approximation using PTX inline assembly.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate cosine values in float32
    """
    return tl.inline_asm_elementwise(
        "cos.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sin_approx(x):
    """
    (Fast) Sine approximation using PTX inline assembly.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate sine values in float32
    """
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def tanh_approx(x):
    """
    (Fast) hyperbolic tangent approximation using PTX inline assembly.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate tanh values in float32
    """
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def sech2_approx(x):
    """
    (Fast) square of the hyperbolic secant approximation using PTX inline assembly.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate sech^2 values in float32
    """
    tanh_x = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return 1.0 - tanh_x * tanh_x

@triton.jit
def sigmoid_approx(x):
    """
    (Fast) Sigmoid approximation using PTX inline assembly.

    Formula: sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))
    Leverages fast tanh approximation for speed.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate sigmoid values in float32
    """
    # tanh_half_x = tl.inline_asm_elementwise(
    #     "tanh.approx.f32 $0, $1;",
    #     constraints="=f,f",
    #     args=[0.5 * x],
    #     dtype=tl.float32,
    #     is_pure=True,
    #     pack=1,
    # )
    # return 0.5 * (1.0 + tanh_half_x)
    # NOTE: We ended up using the built-in sigmoid for better performance, as the PTX approximation was not faster in this case.
    return tl.sigmoid(x)

@triton.jit
def silu(x):
    """
    SiLU (Swish) activation function: x * sigmoid(x).

    Formula: silu(x) = 0.5*x * (1 + tanh(0.5*x)) + 0.5*x.
    Leverages fast tanh_approx for speed.

    Args:
        x: Input triton tensor (any shape) in float32

    Returns:
        SiLU activation output in float32
    """
    # x_half = 0.5 * x
    # return x_half * tanh_approx(x_half) + x_half
    # NOTE: We ended up using the built-in sigmoid for better performance, as the PTX approximation was not faster in this case.
    return x*tl.sigmoid(x)


# =============================================================================
# Diagnostic-only autotune logger (for debugging Bug #7 / kernel OOM)
# =============================================================================
#
# Used as `prune_configs_by={'early_config_prune': log_kernel_autotune}` on
# @triton.autotune decorators. Does NOT prune anything; just logs every
# kernel's constexprs (HEADDIM_QK, HEADDIM_V, etc.) and the candidate config
# list to stderr. Each unique kernel signature is logged ONCE.
#
# Requires the local jax_triton_kda kwargs-forwarding patch
# (~/local_packages/jax_triton_kda_pkg/jax_triton_kda/triton_lib.py:564-579)
# without which `kwargs` is empty and the constexprs are invisible.

# Counter for invocation order (so we can correlate prints with crash position).
_AUTOTUNE_CALL_COUNT = [0]


def log_kernel_autotune(configs, named_args=None, **kwargs):
    """
    Diagnostic: log kernel fingerprint and configs, return configs unchanged.

    Used as `prune_configs_by={'early_config_prune': log_kernel_autotune}`.
    Required for debugging Bug #7 (gpuLaunchKernel OOM at small mesh).

    Args:
        configs: list of triton.Config objects from the autotune decorator
        named_args: positional-arg dict from Triton autotuner (`self.nargs`).
                    Maps kernel arg names to values, INCLUDING constexpr
                    args that the kernel function declares positionally.
                    This is where HEADDIM_QK, HEADDIM_V etc. live — NOT in kwargs.
        **kwargs: explicit metaparams passed to the autotuner call (e.g.,
                  CHUNK_SIZE=64 from jax_triton_kda's metaparam dict).

    Returns:
        configs unchanged (this is a logger, not a pruner)
    """
    import sys

    _AUTOTUNE_CALL_COUNT[0] += 1
    call_idx = _AUTOTUNE_CALL_COUNT[0]

    # Constexpr keys we care about for fingerprinting Mamba3 SSD kernels.
    # These come from named_args (the autotuner's nargs dict), NOT from kwargs.
    sig_keys = (
        "HEADDIM_QK", "HEADDIM_V", "BLOCK_D", "BLOCK_HEADDIM_QK",
        "STORE_SSM_STATES_ADT_OUTV", "HAS_INPUT_STATE", "HAS_INIT_STATE",
        "RETURN_OUTPUT_STATE", "RETURN_FINAL_STATES", "GQA_RATIO",
        "IS_VARLEN", "HAS_D", "HAS_Z", "HAS_INITIAL_STATES",
        "HAS_GRAD_OUTPUT_STATE", "seqlen", "headdim_v",
    )
    nargs_dict = named_args or {}
    sig_from_nargs = {k: nargs_dict[k] for k in sig_keys if k in nargs_dict}
    sig_from_kwargs = {k: kwargs[k] for k in sig_keys if k in kwargs}

    cs_set = sorted({c.kwargs.get("CHUNK_SIZE", "fixed") for c in configs})
    config_list = [
        f"({c.num_stages}s,{c.num_warps}w"
        + (f",cs={c.kwargs['CHUNK_SIZE']}" if "CHUNK_SIZE" in c.kwargs else "")
        + ")"
        for c in configs
    ]

    # Also dump the FULL set of available keys so we can see what's there
    # in case our sig_keys list misses something.
    all_nargs_keys = sorted(nargs_dict.keys()) if nargs_dict else []
    all_kwargs_keys = sorted(kwargs.keys())

    print(
        f"[mamba3_kernels.log_kernel_autotune] call={call_idx} "
        f"nargs_sig={sig_from_nargs} kwargs_sig={sig_from_kwargs} "
        f"cs={cs_set} ncfg={len(configs)} "
        f"all_nargs_keys={all_nargs_keys} all_kwargs_keys={all_kwargs_keys}",
        file=sys.stderr,
        flush=True,
    )

    # Return configs unchanged — pure diagnostic, no pruning
    return list(configs)