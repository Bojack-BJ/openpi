import einops
import jax.numpy as jnp

import openpi.shared.array_typing as at


QWEN2_ROPE_THETA = 1_000_000.0
QWEN2_5_DEFAULT_MROPE_SECTION = (16, 24, 24)


def repeat_kv(hidden_states: at.Array, n_rep: int) -> at.Array:
    if n_rep == 1:
        return hidden_states
    return jnp.repeat(hidden_states, n_rep, axis=2)


def repeat_text_positions(positions: at.Array) -> at.Array:
    if positions.ndim == 3:
        return positions
    if positions.ndim != 2:
        raise ValueError(f"Qwen positions must have rank 2 or 3, got shape {positions.shape}")
    return jnp.broadcast_to(positions[None, :, :], (3, *positions.shape))


def default_mrope_section(head_dim: int) -> tuple[int, int, int]:
    if head_dim == 128:
        return QWEN2_5_DEFAULT_MROPE_SECTION

    half_dim = head_dim // 2
    base = half_dim // 3
    remainder = half_dim % 3
    return tuple(base + (1 if i < remainder else 0) for i in range(3))


def apply_rotary_embedding(
    query: at.Array,
    key: at.Array,
    *,
    positions: at.Array,
    theta: float = QWEN2_ROPE_THETA,
    mrope_section: tuple[int, int, int] | None = None,
) -> tuple[at.Array, at.Array]:
    positions = repeat_text_positions(positions)
    if positions.shape[0] != 3:
        raise ValueError(f"Qwen multimodal positions must have 3 streams, got shape {positions.shape}")

    head_dim = query.shape[-1]
    sections = mrope_section or default_mrope_section(head_dim)
    if sum(sections) != head_dim // 2:
        raise ValueError(
            "Qwen mRoPE sections must sum to half the head dim: "
            f"{sections} vs head_dim={head_dim}"
        )

    cos, sin = _compute_multimodal_cos_sin(positions, head_dim=head_dim, theta=theta, sections=sections)
    return _apply_rope(query, cos=cos, sin=sin), _apply_rope(key, cos=cos, sin=sin)


def _compute_multimodal_cos_sin(positions: at.Array, *, head_dim: int, theta: float, sections):
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")

    freq_exponents = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    inv_freq = 1.0 / (theta**freq_exponents)
    radians = positions[..., None] * inv_freq[None, None, None, :]
    cos_axes = jnp.cos(radians)
    sin_axes = jnp.sin(radians)

    start = 0
    selected_cos = []
    selected_sin = []
    for axis, size in enumerate(sections):
        end = start + size
        selected_cos.append(cos_axes[axis, ..., start:end])
        selected_sin.append(sin_axes[axis, ..., start:end])
        start = end
    return jnp.concatenate(selected_cos, axis=-1), jnp.concatenate(selected_sin, axis=-1)


def _apply_rope(x: at.Array, *, cos: at.Array, sin: at.Array) -> at.Array:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    rotated = jnp.stack([x_even * cos - x_odd * sin, x_odd * cos + x_even * sin], axis=-1)
    rotated = einops.rearrange(rotated, "... d two -> ... (d two)")
    return rotated.astype(x.dtype)
