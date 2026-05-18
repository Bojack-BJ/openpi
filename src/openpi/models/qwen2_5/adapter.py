import jax.numpy as jnp

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge

import openpi.models.qwen2_5.text as _qwen_text
import openpi.models.qwen2_5.vision as _qwen_vision


class Qwen2_5_VLWithExpertModel(nnx.Module):
    """JAX Qwen2.5-VL adapter with native vision/projector and backend-owned positions."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        *,
        use_adarms,
        precision: str,
        image_example,
        rngs: nnx.Rngs,
        hf_model_id: str | None = None,
    ):
        self.hf_model_id = hf_model_id
        if vlm_config.depth != action_expert_config.depth:
            raise ValueError(
                "JAX Qwen adapter currently requires matching prefix/expert depth: "
                f"{vlm_config.depth} != {action_expert_config.depth}"
            )
        if (
            vlm_config.num_heads != action_expert_config.num_heads
            or vlm_config.num_kv_heads != action_expert_config.num_kv_heads
            or vlm_config.head_dim != action_expert_config.head_dim
        ):
            raise ValueError(
                "JAX Qwen adapter currently requires matching attention geometry for prefix and expert."
            )

        llm = nnx_bridge.ToNNX(
            _qwen_text.Module(
                configs=[vlm_config, action_expert_config],
                embed_dtype=precision,
                vocab_size=vlm_config.vocab_size or _qwen_text.QWEN2_5_VL_VOCAB_SIZE,
                rope_theta=vlm_config.rope_theta or _qwen_text.qwen_rotary.QWEN2_ROPE_THETA,
                mrope_section=vlm_config.mrope_section,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
        self.llm = llm
        self.vision = _qwen_vision.Qwen2_5VisionTower(
            output_width=vlm_config.width,
            precision=precision,
            image_example=image_example,
            rngs=rngs,
        )

    def embed_image(self, image):
        image_tokens, _ = self.vision.embed_image(image)
        return image_tokens

    def embed_image_with_metadata(self, image):
        image_tokens, grid_thw = self.vision.embed_image(image)
        grid_metadata = grid_thw[0] if grid_thw else None
        return image_tokens, {"grid_thw": grid_metadata}

    def embed_language_tokens(self, tokens):
        return self.llm(tokens, method="embed")

    def forward(self, inputs_embeds, mask, positions, kv_cache=None, adarms_cond=None):
        if adarms_cond is None:
            adarms_cond = [None, None]
        return self.llm(inputs_embeds, mask=mask, positions=positions, kv_cache=kv_cache, adarms_cond=adarms_cond)

    @staticmethod
    def _build_text_position_chunk(length: int, *, start_positions):
        return start_positions[:, None] + jnp.arange(length, dtype=jnp.int32)[None, :]

    @staticmethod
    def _build_vision_position_chunk(
        grid_thw: tuple[int, int, int] | None,
        *,
        start_positions,
        token_length: int,
    ) -> tuple[jnp.ndarray, int]:
        if grid_thw is None:
            text_positions = Qwen2_5_VLWithExpertModel._build_text_position_chunk(
                token_length,
                start_positions=start_positions,
            )
            return (
                jnp.broadcast_to(text_positions[None, :, :], (3, text_positions.shape[0], token_length)),
                token_length,
            )

        grid_t, grid_h, grid_w = grid_thw
        expected_count = grid_t * grid_h * grid_w
        if expected_count != token_length:
            raise ValueError(
                "Qwen JAX vision position chunk expected token length to match grid_thw: "
                f"{expected_count} != {token_length}"
            )

        temporal = jnp.repeat(jnp.arange(grid_t, dtype=jnp.int32), grid_h * grid_w)
        height = jnp.tile(jnp.repeat(jnp.arange(grid_h, dtype=jnp.int32), grid_w), grid_t)
        width = jnp.tile(jnp.arange(grid_w, dtype=jnp.int32), grid_t * grid_h)
        chunk = jnp.stack([temporal, height, width], axis=0)
        return chunk[:, None, :] + start_positions[None, :, None], max(grid_t, grid_h, grid_w)

    def _build_prefix_position_ids(self, prefix_pad_mask, *, prefix_layout):
        batch_size, prefix_len = prefix_pad_mask.shape
        position_ids = jnp.zeros((3, batch_size, prefix_len), dtype=jnp.int32)
        next_positions = jnp.zeros((batch_size,), dtype=jnp.int32)

        cursor = 0
        for image_idx, image_token_length in enumerate(prefix_layout.image_token_lengths):
            if image_token_length == 0:
                continue
            valid = prefix_pad_mask[:, cursor]
            grid_thw = None
            if image_idx < len(prefix_layout.image_grid_thw):
                grid_thw = prefix_layout.image_grid_thw[image_idx]
            chunk, chunk_increment = self._build_vision_position_chunk(
                grid_thw,
                start_positions=next_positions,
                token_length=image_token_length,
            )
            position_ids = position_ids.at[:, :, cursor : cursor + image_token_length].set(chunk)
            next_positions = jnp.where(valid, next_positions + chunk_increment, next_positions)
            cursor += image_token_length

        if prefix_layout.language_token_length > 0:
            text_positions = self._build_text_position_chunk(
                prefix_layout.language_token_length,
                start_positions=next_positions,
            )
            text_position_ids = jnp.broadcast_to(
                text_positions[None, :, :],
                (3, batch_size, prefix_layout.language_token_length),
            )
            position_ids = position_ids.at[:, :, cursor : cursor + prefix_layout.language_token_length].set(
                text_position_ids
            )
            next_positions = next_positions + prefix_pad_mask[:, cursor : cursor + prefix_layout.language_token_length].sum(
                axis=1, dtype=jnp.int32
            )

        return position_ids, next_positions

    def _build_suffix_position_ids(self, suffix_pad_mask, *, start_positions):
        suffix_len = suffix_pad_mask.shape[1]
        text_positions = self._build_text_position_chunk(suffix_len, start_positions=start_positions)
        return jnp.broadcast_to(text_positions[None, :, :], (3, suffix_pad_mask.shape[0], suffix_len))

    def build_prefix_positions(self, prefix_pad_mask, *, prefix_layout):
        position_ids, _ = self._build_prefix_position_ids(prefix_pad_mask, prefix_layout=prefix_layout)
        return position_ids

    def build_joint_positions(self, prefix_pad_mask, suffix_pad_mask, *, prefix_layout):
        prefix_position_ids, next_positions = self._build_prefix_position_ids(prefix_pad_mask, prefix_layout=prefix_layout)
        suffix_position_ids = self._build_suffix_position_ids(suffix_pad_mask, start_positions=next_positions)
        return jnp.concatenate([prefix_position_ids, suffix_position_ids], axis=2)

    def build_decode_positions(self, prefix_pad_mask, suffix_pad_mask, *, prefix_layout):
        _, next_positions = self._build_prefix_position_ids(prefix_pad_mask, prefix_layout=prefix_layout)
        return self._build_suffix_position_ids(suffix_pad_mask, start_positions=next_positions)
