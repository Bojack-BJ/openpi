import einops
import jax.numpy as jnp

import openpi.shared.array_typing as at


QWEN3_5_ROPE_THETA = 10_000_000.0
QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR = 0.25
QWEN3_5_DEFAULT_MROPE_SECTION = (11, 11, 10)


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


def rotary_dim(head_dim: int, partial_rotary_factor: float = QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR) -> int:
    dim = int(head_dim * partial_rotary_factor)
    return dim - (dim % 2)


def default_mrope_section(rotary_head_dim: int) -> tuple[int, int, int]:
    if rotary_head_dim == 64:
        return QWEN3_5_DEFAULT_MROPE_SECTION

    half_dim = rotary_head_dim // 2
    base = half_dim // 3
    remainder = half_dim % 3
    return tuple(base + (1 if i < remainder else 0) for i in range(3))


def resolve_mrope_section(
    rotary_head_dim: int, mrope_section: tuple[int, int, int] | None = None
) -> tuple[int, int, int]:
    default_sections = default_mrope_section(rotary_head_dim)
    if mrope_section is None:
        return default_sections

    # Qwen3.5 mixes full-attention and linear-attention blocks with different effective
    # rotary widths. Keep an explicit config when it matches the active layer width, but
    # fall back to a width-specific split when the configured tuple belongs to a different
    # rotary geometry.
    if sum(mrope_section) == rotary_head_dim // 2:
        return mrope_section
    return default_sections


def apply_rotary_embedding(
    query: at.Array,
    key: at.Array,
    *,
    positions: at.Array,
    theta: float = QWEN3_5_ROPE_THETA,
    mrope_section: tuple[int, int, int] | None = None,
    partial_rotary_factor: float = QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR,
) -> tuple[at.Array, at.Array]:
    positions = repeat_text_positions(positions)
    if positions.shape[0] != 3:
        raise ValueError(f"Qwen multimodal positions must have 3 streams, got shape {positions.shape}")

    head_dim = query.shape[-1]
    rot_dim = rotary_dim(head_dim, partial_rotary_factor)
    if rot_dim == 0:
        return query, key

    sections = resolve_mrope_section(rot_dim, mrope_section)

    cos, sin = _compute_multimodal_cos_sin(positions, rotary_head_dim=rot_dim, theta=theta, sections=sections)
    return _apply_rope_partial(query, rot_dim=rot_dim, cos=cos, sin=sin), _apply_rope_partial(
        key, rot_dim=rot_dim, cos=cos, sin=sin
    )


def _compute_multimodal_cos_sin(positions: at.Array, *, rotary_head_dim: int, theta: float, sections):
    freq_exponents = jnp.arange(0, rotary_head_dim, 2, dtype=jnp.float32) / rotary_head_dim
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


def _apply_rope_partial(x: at.Array, *, rot_dim: int, cos: at.Array, sin: at.Array) -> at.Array:
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    x_rot_even = x_rot[..., ::2]
    x_rot_odd = x_rot[..., 1::2]
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    rotated = jnp.stack([x_rot_even * cos - x_rot_odd * sin, x_rot_odd * cos + x_rot_even * sin], axis=-1)
    rotated = einops.rearrange(rotated, "... d two -> ... (d two)")
    out = jnp.concatenate([rotated, x_pass], axis=-1)
    return out.astype(x.dtype)


__all__ = [
    "QWEN3_5_DEFAULT_MROPE_SECTION",
    "QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR",
    "QWEN3_5_ROPE_THETA",
    "apply_rotary_embedding",
    "default_mrope_section",
    "resolve_mrope_section",
    "repeat_kv",
    "repeat_text_positions",
    "rotary_dim",
]
