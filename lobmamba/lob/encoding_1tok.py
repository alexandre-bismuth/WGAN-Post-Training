"""Field-level vocab specs for 1-token-per-message encoding.

Reuses the existing 24-token encoding (encode_msg_24) but maps global
token IDs (0-2111) to per-field local indices for independent embedding
and classification heads.

Global token ID layout (from Vocab.__init__ order):
    0-3:      special tokens (MASK, HIDDEN, NA, START)
    4-1003:   time       (1000 values)
    1004-1007: event_type (4 values)
    1008-1107: size_digit (100 values)
    1108-2107: price      (1000 values)
    2108-2109: sign       (2 values)
    2110-2111: direction  (2 values)

Per-field local index layout:
    0: MASK, 1: HIDDEN, 2: NA, 3: START, 4..(V_i+3): real values
"""
import numpy as np
import jax.numpy as jnp


# Per-position field type (24 positions in a message)
FIELD_TYPES = (
    'event_type',   # 0
    'direction',    # 1
    'sign',         # 2  (price_sign)
    'price',        # 3  (price_mag)
    'size_digit',   # 4  (size_high)
    'size_digit',   # 5  (size_low)
    'time',         # 6  (delta_t_s)
    'time',         # 7  (delta_t_ns[0])
    'time',         # 8  (delta_t_ns[1])
    'time',         # 9  (delta_t_ns[2])
    'time',         # 10 (time_s[0])
    'time',         # 11 (time_s[1])
    'time',         # 12 (time_ns[0])
    'time',         # 13 (time_ns[1])
    'time',         # 14 (time_ns[2])
    'sign',         # 15 (price_ref_sign)
    'price',        # 16 (price_ref_mag)
    'size_digit',   # 17 (size_ref_high)
    'size_digit',   # 18 (size_ref_low)
    'time',         # 19 (time_s_ref[0])
    'time',         # 20 (time_s_ref[1])
    'time',         # 21 (time_ns_ref[0])
    'time',         # 22 (time_ns_ref[1])
    'time',         # 23 (time_ns_ref[2])
)

FIELD_NAMES = (
    'event_type', 'direction', 'price_sign', 'price_mag',
    'size_high', 'size_low',
    'delta_t_s', 'delta_t_ns_0', 'delta_t_ns_1', 'delta_t_ns_2',
    'time_s_0', 'time_s_1', 'time_ns_0', 'time_ns_1', 'time_ns_2',
    'price_ref_sign', 'price_ref_mag',
    'size_ref_high', 'size_ref_low',
    'time_s_ref_0', 'time_s_ref_1',
    'time_ns_ref_0', 'time_ns_ref_1', 'time_ns_ref_2',
)

N_FIELDS = 24
N_SPECIAL_TOKENS = 4  # MASK=0, HIDDEN=1, NA=2, START=3

# Number of real values per field type (excluding 4 special tokens)
_FIELD_REAL_VOCAB = {
    'event_type': 4,
    'direction': 2,
    'sign': 2,
    'price': 1000,
    'size_digit': 100,
    'time': 1000,
}

# Per-position real vocab size (excluding specials)
FIELD_VOCAB_SIZES = tuple(_FIELD_REAL_VOCAB[ft] for ft in FIELD_TYPES)

# Per-position total vocab size (including 4 special tokens)
FIELD_VOCAB_SIZES_WITH_SPECIAL = tuple(v + N_SPECIAL_TOKENS for v in FIELD_VOCAB_SIZES)

# Global token ID offset for each field type (from Vocab.__init__ order)
_FIELD_GLOBAL_OFFSETS = {
    'time': 4,
    'event_type': 1004,
    'size_digit': 1008,
    'price': 1108,
    'sign': 2108,
    'direction': 2110,
}

# Per-position global offset (for vectorized mapping)
FIELD_OFFSETS = np.array(
    [_FIELD_GLOBAL_OFFSETS[ft] for ft in FIELD_TYPES], dtype=np.int32
)


def global_to_local(tokens):
    """Convert global token IDs (0-2111) to per-field local indices.

    Args:
        tokens: np.array of shape (..., 24) with global token IDs
    Returns:
        np.array of same shape with per-field local indices
    """
    is_special = tokens < N_SPECIAL_TOKENS
    local = np.where(is_special, tokens, tokens - FIELD_OFFSETS + N_SPECIAL_TOKENS)
    return local.astype(np.int32)


def local_to_global(tokens):
    """Convert per-field local indices back to global token IDs.

    Args:
        tokens: np.array of shape (..., 24) with local per-field indices
    Returns:
        np.array of same shape with global token IDs
    """
    is_special = tokens < N_SPECIAL_TOKENS
    global_ids = np.where(is_special, tokens, tokens + FIELD_OFFSETS - N_SPECIAL_TOKENS)
    return global_ids.astype(np.int32)


# --- JAX-compatible versions for use inside jax.lax.scan / jit ---
FIELD_OFFSETS_JAX = jnp.array(FIELD_OFFSETS)


def local_to_global_jax(tokens):
    """JAX version of local_to_global for use inside jit/scan."""
    is_special = tokens < N_SPECIAL_TOKENS
    return jnp.where(is_special, tokens, tokens + FIELD_OFFSETS_JAX - N_SPECIAL_TOKENS).astype(jnp.int32)


def global_to_local_jax(tokens):
    """JAX version of global_to_local for use inside jit/scan."""
    is_special = tokens < N_SPECIAL_TOKENS
    return jnp.where(is_special, tokens, tokens - FIELD_OFFSETS_JAX + N_SPECIAL_TOKENS).astype(jnp.int32)
