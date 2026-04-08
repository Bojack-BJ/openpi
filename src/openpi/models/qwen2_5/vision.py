import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax.numpy as jnp

import openpi.models.siglip as _siglip


QWEN2_5_VISION_VARIANT = "So400m/14"
QWEN2_5_VISION_PATCH_SIZE = 14
QWEN2_5_SPATIAL_MERGE_SIZE = 2


class SpatialPatchMerger(nnx.Module):
    def __init__(self, input_width: int, output_width: int, *, merge_size: int, rngs: nnx.Rngs):
        self.merge_size = merge_size
        self.proj = nnx.Linear(input_width * merge_size * merge_size, output_width, rngs=rngs)

    def __call__(self, image_tokens, *, grid_h: int, grid_w: int):
        if grid_h % self.merge_size != 0 or grid_w % self.merge_size != 0:
            raise ValueError(
                "Qwen JAX vision tower requires the spatial grid to be divisible by the merge size: "
                f"{grid_h}x{grid_w} vs merge={self.merge_size}"
            )

        batch_size, _, width = image_tokens.shape
        merged_h = grid_h // self.merge_size
        merged_w = grid_w // self.merge_size
        image_tokens = image_tokens.reshape(batch_size, grid_h, grid_w, width)
        image_tokens = image_tokens.reshape(
            batch_size,
            merged_h,
            self.merge_size,
            merged_w,
            self.merge_size,
            width,
        )
        image_tokens = jnp.transpose(image_tokens, (0, 1, 3, 2, 4, 5))
        image_tokens = image_tokens.reshape(
            batch_size,
            merged_h * merged_w,
            width * self.merge_size * self.merge_size,
        )
        return self.proj(image_tokens), (1, merged_h, merged_w)


class Qwen2_5VisionTower(nnx.Module):
    """Pragmatic JAX vision path for Qwen2.5-VL.

    This is not a weight-compatible port of the HF Qwen vision tower. It reuses the
    repository's existing SigLIP encoder, then performs a Qwen-style spatial merge and
    projection into the text hidden width so the JAX Pi0 path can run end-to-end.
    """

    def __init__(
        self,
        *,
        output_width: int,
        precision: str,
        image_example,
        rngs: nnx.Rngs,
        variant: str = QWEN2_5_VISION_VARIANT,
        patch_size: int = QWEN2_5_VISION_PATCH_SIZE,
        spatial_merge_size: int = QWEN2_5_SPATIAL_MERGE_SIZE,
    ):
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        encoder_width = _siglip.decode_variant(variant)["width"]

        encoder = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=None,
                variant=variant,
                pool_type="none",
                scan=True,
                dtype_mm=precision,
            )
        )
        encoder.lazy_init(image_example, train=False, rngs=rngs)
        self.encoder = encoder
        self.patch_merger = SpatialPatchMerger(
            encoder_width,
            output_width,
            merge_size=spatial_merge_size,
            rngs=rngs,
        )

    def embed_image(self, image):
        batch_size, image_h, image_w, _ = image.shape
        grid_h = image_h // self.patch_size
        grid_w = image_w // self.patch_size
        encoded, _ = self.encoder(image, train=False)
        projected, merged_grid_thw = self.patch_merger(encoded, grid_h=grid_h, grid_w=grid_w)
        grid_thw = (merged_grid_thw,) * batch_size
        return projected, grid_thw
