import math

import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


QWEN3_5_VISION_PATCH_SIZE = 16
QWEN3_5_TEMPORAL_PATCH_SIZE = 2
QWEN3_5_SPATIAL_MERGE_SIZE = 2
QWEN3_5_VISION_HIDDEN_SIZE = 1024
QWEN3_5_VISION_DEPTH = 24
QWEN3_5_VISION_NUM_HEADS = 16
QWEN3_5_VISION_MLP_DIM = 4096
QWEN3_5_VISION_NUM_POSITIONS = 2304
QWEN3_5_VISION_MERGER_DIM = 4096


def _as_dtype(dtype_like: str | jnp.dtype) -> jnp.dtype:
    return jnp.dtype(dtype_like)


def _resize_positional_embedding(positional_embedding: jax.Array, *, grid_h: int, grid_w: int) -> jax.Array:
    source_positions, hidden_size = positional_embedding.shape
    source_side = int(math.isqrt(source_positions))
    if source_side * source_side != source_positions:
        raise ValueError(f"Qwen3.5 vision pos embedding expects a square grid, got {source_positions} positions.")

    resized = jax.image.resize(
        positional_embedding.reshape(source_side, source_side, hidden_size),
        (grid_h, grid_w, hidden_size),
        method="bicubic",
    )
    return resized.reshape(grid_h * grid_w, hidden_size)


def _repeat_image_temporally(image: jax.Array, temporal_patch_size: int) -> jax.Array:
    return jnp.repeat(image[:, None, :, :, :], temporal_patch_size, axis=1)


def _vision_rotary_cache(
    grid_h: int, grid_w: int, *, head_dim: int, theta: float = 10_000.0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if head_dim % 4 != 0:
        raise ValueError(f"Qwen3.5 vision rotary expects head_dim divisible by 4, got {head_dim}")

    half = head_dim // 2
    h_pairs = half // 2
    w_pairs = (head_dim - half) // 2

    y, x = jnp.mgrid[:grid_h, :grid_w]
    y = y.reshape(-1)
    x = x.reshape(-1)

    h_inv_freq = 1.0 / (theta ** (jnp.arange(h_pairs, dtype=jnp.float32) / max(h_pairs, 1)))
    w_inv_freq = 1.0 / (theta ** (jnp.arange(w_pairs, dtype=jnp.float32) / max(w_pairs, 1)))
    h_angles = y[:, None] * h_inv_freq[None, :]
    w_angles = x[:, None] * w_inv_freq[None, :]

    return (
        jnp.cos(h_angles),
        jnp.sin(h_angles),
        jnp.cos(w_angles),
        jnp.sin(w_angles),
    )


def _apply_axis_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([x_even * cos - x_odd * sin, x_odd * cos + x_even * sin], axis=-1)
    return einops.rearrange(rotated, "b t n d two -> b t n (d two)")


def _apply_vision_rotary(query: jax.Array, key: jax.Array, *, grid_h: int, grid_w: int) -> tuple[jax.Array, jax.Array]:
    head_dim = query.shape[-1]
    half = head_dim // 2
    q_h, q_w = query[..., :half], query[..., half:]
    k_h, k_w = key[..., :half], key[..., half:]
    cos_h, sin_h, cos_w, sin_w = _vision_rotary_cache(grid_h, grid_w, head_dim=head_dim)
    q = jnp.concatenate([_apply_axis_rope(q_h, cos_h, sin_h), _apply_axis_rope(q_w, cos_w, sin_w)], axis=-1)
    k = jnp.concatenate([_apply_axis_rope(k_h, cos_h, sin_h), _apply_axis_rope(k_w, cos_w, sin_w)], axis=-1)
    return q.astype(query.dtype), k.astype(key.dtype)


@at.typecheck
class Qwen3_5VisionPatchEmbed(nn.Module):
    hidden_size: int
    patch_size: int
    temporal_patch_size: int
    dtype_mm: jnp.dtype

    @nn.compact
    def __call__(self, image: at.Float[at.Array, "b h w c"]) -> at.Float[at.Array, "b gh gw d"]:
        image = _repeat_image_temporally(image.astype(jnp.float32), self.temporal_patch_size)
        patches = nn.Conv(
            self.hidden_size,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            strides=(self.temporal_patch_size, self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
            name="proj",
            dtype=self.dtype_mm,
        )(image)
        return patches[:, 0].astype(self.dtype_mm)


@at.typecheck
class Qwen3_5VisionAttention(nn.Module):
    hidden_size: int
    num_heads: int
    dtype_mm: jnp.dtype

    @nn.compact
    def __call__(self, x: at.Float[at.Array, "b s d"], *, grid_h: int, grid_w: int) -> at.Float[at.Array, "b s d"]:
        head_dim = self.hidden_size // self.num_heads
        qkv = nn.Dense(
            self.hidden_size * 3,
            use_bias=True,
            name="qkv",
            dtype=self.dtype_mm,
        )(x)
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.num_heads, head_dim)
        query = qkv[:, :, 0]
        key = qkv[:, :, 1]
        value = qkv[:, :, 2]
        query, key = _apply_vision_rotary(query, key, grid_h=grid_h, grid_w=grid_w)
        query = query * (head_dim**-0.5)
        attn = jnp.einsum("btnd,bsnd->bnts", query, key, preferred_element_type=jnp.float32)
        attn = jax.nn.softmax(attn, axis=-1).astype(x.dtype)
        encoded = jnp.einsum("bnts,bsnd->btnd", attn, value).reshape(x.shape[0], x.shape[1], self.hidden_size)
        return nn.Dense(
            self.hidden_size,
            use_bias=True,
            name="proj",
            dtype=self.dtype_mm,
        )(encoded)


@at.typecheck
class Qwen3_5VisionMLP(nn.Module):
    hidden_size: int
    mlp_dim: int
    dtype_mm: jnp.dtype

    @nn.compact
    def __call__(self, x: at.Float[at.Array, "b s d"]) -> at.Float[at.Array, "b s d"]:
        x = nn.Dense(self.mlp_dim, use_bias=True, name="linear_fc1", dtype=self.dtype_mm)(x)
        x = nn.gelu(x, approximate=True)
        return nn.Dense(self.hidden_size, use_bias=True, name="linear_fc2", dtype=self.dtype_mm)(x)


@at.typecheck
class Qwen3_5VisionBlock(nn.Module):
    hidden_size: int
    mlp_dim: int
    num_heads: int
    dtype_mm: jnp.dtype

    @nn.compact
    def __call__(self, x: at.Float[at.Array, "b s d"], *, grid_h: int, grid_w: int) -> at.Float[at.Array, "b s d"]:
        x = x + Qwen3_5VisionAttention(
            self.hidden_size,
            self.num_heads,
            self.dtype_mm,
            name="attn",
        )(nn.LayerNorm(name="norm1", dtype=self.dtype_mm)(x), grid_h=grid_h, grid_w=grid_w)
        x = x + Qwen3_5VisionMLP(
            self.hidden_size,
            self.mlp_dim,
            self.dtype_mm,
            name="mlp",
        )(nn.LayerNorm(name="norm2", dtype=self.dtype_mm)(x))
        return x


@at.typecheck
class Qwen3_5VisionPatchMerger(nn.Module):
    input_width: int
    hidden_width: int
    output_width: int
    merge_size: int
    dtype_mm: jnp.dtype

    @nn.compact
    def __call__(self, image_tokens: at.Float[at.Array, "b s d"], *, grid_h: int, grid_w: int):
        if grid_h % self.merge_size != 0 or grid_w % self.merge_size != 0:
            raise ValueError(
                "Qwen3.5 vision merger requires the spatial grid to be divisible by the merge size: "
                f"{grid_h}x{grid_w} vs merge={self.merge_size}"
            )

        merged_h = grid_h // self.merge_size
        merged_w = grid_w // self.merge_size
        batch_size = image_tokens.shape[0]
        image_tokens = image_tokens.reshape(batch_size, grid_h, grid_w, self.input_width)
        image_tokens = image_tokens.reshape(
            batch_size,
            merged_h,
            self.merge_size,
            merged_w,
            self.merge_size,
            self.input_width,
        )
        image_tokens = jnp.transpose(image_tokens, (0, 1, 3, 2, 4, 5))
        image_tokens = image_tokens.reshape(
            batch_size,
            merged_h * merged_w,
            self.input_width * self.merge_size * self.merge_size,
        )
        image_tokens = nn.LayerNorm(name="norm", dtype=self.dtype_mm)(image_tokens)
        image_tokens = nn.Dense(
            self.hidden_width,
            use_bias=True,
            name="linear_fc1",
            dtype=self.dtype_mm,
        )(image_tokens)
        image_tokens = nn.gelu(image_tokens, approximate=True)
        image_tokens = nn.Dense(
            self.output_width,
            use_bias=True,
            name="linear_fc2",
            dtype=self.dtype_mm,
        )(image_tokens)
        return image_tokens, (1, merged_h, merged_w)


class _Qwen3_5VisionModule(nn.Module):
    output_width: int
    hidden_size: int
    depth: int
    mlp_dim: int
    num_heads: int
    patch_size: int
    temporal_patch_size: int
    num_positions: int
    spatial_merge_size: int
    merger_dim: int
    dtype_mm: jnp.dtype

    @nn.compact
    def __call__(self, image: at.Float[at.Array, "b h w c"], *, train: bool = False):
        del train
        patches = Qwen3_5VisionPatchEmbed(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            dtype_mm=self.dtype_mm,
            name="patch_embed",
        )(image)
        batch_size, grid_h, grid_w, hidden_size = patches.shape
        tokens = patches.reshape(batch_size, grid_h * grid_w, hidden_size)

        pos_embed = nn.Embed(
            num_embeddings=self.num_positions,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="pos_embed",
        )(jnp.arange(self.num_positions, dtype=jnp.int32))
        tokens = tokens + _resize_positional_embedding(pos_embed.astype(tokens.dtype), grid_h=grid_h, grid_w=grid_w)[None, :, :]

        for i in range(self.depth):
            tokens = Qwen3_5VisionBlock(
                hidden_size=self.hidden_size,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dtype_mm=self.dtype_mm,
                name=f"blocks_{i}",
            )(tokens, grid_h=grid_h, grid_w=grid_w)

        merged_tokens, merged_grid_thw = Qwen3_5VisionPatchMerger(
            input_width=self.hidden_size,
            hidden_width=self.merger_dim,
            output_width=self.output_width,
            merge_size=self.spatial_merge_size,
            dtype_mm=self.dtype_mm,
            name="merger",
        )(tokens, grid_h=grid_h, grid_w=grid_w)
        grid_thw = (merged_grid_thw,) * batch_size
        return merged_tokens, grid_thw


class Qwen3_5VisionTower(nnx.Module):
    """Official-style JAX Qwen3.5 vision tower.

    This now uses a dedicated Conv3D patch embed + learned vision embeddings + ViT block stack
    + official-style spatial patch merger, rather than the older SigLIP fallback.
    """

    def __init__(
        self,
        *,
        output_width: int,
        precision: str,
        image_example,
        rngs: nnx.Rngs,
        hidden_size: int = QWEN3_5_VISION_HIDDEN_SIZE,
        depth: int = QWEN3_5_VISION_DEPTH,
        mlp_dim: int = QWEN3_5_VISION_MLP_DIM,
        num_heads: int = QWEN3_5_VISION_NUM_HEADS,
        patch_size: int = QWEN3_5_VISION_PATCH_SIZE,
        temporal_patch_size: int = QWEN3_5_TEMPORAL_PATCH_SIZE,
        num_positions: int = QWEN3_5_VISION_NUM_POSITIONS,
        spatial_merge_size: int = QWEN3_5_SPATIAL_MERGE_SIZE,
        merger_dim: int = QWEN3_5_VISION_MERGER_DIM,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.encoder = nnx_bridge.ToNNX(
            _Qwen3_5VisionModule(
                output_width=output_width,
                hidden_size=hidden_size,
                depth=depth,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                num_positions=num_positions,
                spatial_merge_size=spatial_merge_size,
                merger_dim=merger_dim,
                dtype_mm=_as_dtype(precision),
            )
        )
        self.encoder.lazy_init(image_example, train=False, rngs=rngs)

    def embed_image(self, image):
        return self.encoder(image, train=False)


__all__ = [
    "QWEN3_5_SPATIAL_MERGE_SIZE",
    "QWEN3_5_TEMPORAL_PATCH_SIZE",
    "QWEN3_5_VISION_DEPTH",
    "QWEN3_5_VISION_HIDDEN_SIZE",
    "QWEN3_5_VISION_MERGER_DIM",
    "QWEN3_5_VISION_MLP_DIM",
    "QWEN3_5_VISION_NUM_HEADS",
    "QWEN3_5_VISION_NUM_POSITIONS",
    "QWEN3_5_VISION_PATCH_SIZE",
    "Qwen3_5VisionTower",
]
