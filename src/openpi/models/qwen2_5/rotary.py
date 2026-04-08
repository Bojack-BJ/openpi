import einops
import jax.numpy as jnp

import openpi.shared.array_typing as at


QWEN2_ROPE_THETA = 1_000_000.0


def repeat_kv(hidden_states: at.Array, n_rep: int) -> at.Array:
    if n_rep == 1:
        return hidden_states
    return jnp.repeat(hidden_states, n_rep, axis=2)


def apply_rotary_embedding(
    query: at.Array,
    key: at.Array,
    *,
    positions: at.Array,
    theta: float = QWEN2_ROPE_THETA,
) -> tuple[at.Array, at.Array]:
    """Applies 1D rotary embeddings to query/key tensors.

    Args:
        query: [B, T, N, H]
        key: [B, T, K, H]
        positions: [B, T]
    """
    return _apply_rope(query, positions=positions, theta=theta), _apply_rope(key, positions=positions, theta=theta)


def _apply_rope(x: at.Array, *, positions: at.Array, theta: float) -> at.Array:
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")

    freq_exponents = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    inv_freq = 1.0 / (theta**freq_exponents)
    radians = positions[..., None] * inv_freq[None, None, :]
    sin = jnp.sin(radians)[..., None, :]
    cos = jnp.cos(radians)[..., None, :]

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = jnp.stack([x_even * cos - x_odd * sin, x_odd * cos + x_even * sin], axis=-1)
    rotated = einops.rearrange(rotated, "... d two -> ... (d two)")
    return rotated.astype(x.dtype)
