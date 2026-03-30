import math
from typing import Literal

import pytest
import torch
from torch import nn
from transformers import AutoProcessor
from transformers import Qwen2ForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen2 import modeling_qwen2

from openpi.models_pytorch.vlm_backbone_base import AttentionContext
from openpi.models_pytorch.vlm_backbone_base import PrefixBatch
from openpi.models_pytorch.vlm_backbone_base import VLMWithExpertModel


QWEN2_5_VL_VOCAB_SIZE = 152_064
QWEN_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
QWEN_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)


def _get_layer_past_key_value(past_key_values, layer_idx: int):
    if past_key_values is None:
        return None, None

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]

    if hasattr(past_key_values, "__getitem__"):
        layer_past = past_key_values[layer_idx]
        if isinstance(layer_past, (tuple, list)) and len(layer_past) >= 2:
            return layer_past[0], layer_past[1]

    raise TypeError(f"Unsupported Qwen past_key_values type: {type(past_key_values)!r}")


class Qwen2_5_VLWithExpertModel(VLMWithExpertModel):
    """Qwen2.5-VL adapter that preserves OpenPI's existing prefix/suffix contract.

    Images and prompt tokens are embedded separately, then mixed through the same shared
    prefix/suffix self-attention pattern used by the PaliGemma path. Image preprocessing
    can already route through the official Qwen processor, but the joint-attention path
    still uses OpenPI's 1D prefix/suffix positions rather than Qwen's full multimodal
    position semantics.
    """

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        hf_model_id: str | None = None,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        # TODO: support pi05 by adding a Qwen-compatible conditional norm path for the suffix expert.
        if any(use_adarms):
            raise NotImplementedError("Qwen2.5-VL backbone does not support pi05/AdaRMS expert conditioning yet.")
        super().__init__()
        self.hf_model_id = hf_model_id
        self.qwen_processor = (
            AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True) if hf_model_id is not None else None
        )

        # The current adapter reuses native Qwen blocks and directly copies the text-tower weights
        # into the action expert, so it currently requires the full prefix/expert text geometry to
        # match. A future shared-attention bridge could relax the hidden-size constraint.
        if vlm_config.width != action_expert_config.width:
            raise ValueError(
                f"Qwen2.5-VL requires matching hidden sizes for prefix and expert: "
                f"{vlm_config.width} != {action_expert_config.width}"
            )
        if vlm_config.depth != action_expert_config.depth:
            raise ValueError(
                f"Qwen2.5-VL requires matching layer counts for prefix and expert: "
                f"{vlm_config.depth} != {action_expert_config.depth}"
            )
        if (
            vlm_config.num_heads != action_expert_config.num_heads
            or vlm_config.num_kv_heads != action_expert_config.num_kv_heads
            or vlm_config.head_dim != action_expert_config.head_dim
        ):
            raise ValueError("Qwen2.5-VL requires matching attention geometry for prefix and expert.")

        if hf_model_id is not None:
            dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
            self.qwen_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                hf_model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            loaded_text_config = self.qwen_vl.config.text_config
            if (
                loaded_text_config.hidden_size != action_expert_config.width
                or loaded_text_config.num_hidden_layers != action_expert_config.depth
                or loaded_text_config.num_attention_heads != action_expert_config.num_heads
                or loaded_text_config.num_key_value_heads != action_expert_config.num_kv_heads
            ):
                raise ValueError(
                    "The loaded Qwen2.5-VL checkpoint does not match the configured expert geometry. "
                    "Use `vlm_backbone_variant=\"qwen2_5_7b\"` and `action_expert_variant=\"qwen2_5_7b\"` "
                    "with `Qwen/Qwen2.5-VL-7B-Instruct`."
                )
            action_expert_config_hf = CONFIG_MAPPING["qwen2"](
                hidden_size=action_expert_config.width,
                intermediate_size=action_expert_config.mlp_dim,
                num_attention_heads=action_expert_config.num_heads,
                num_hidden_layers=action_expert_config.depth,
                num_key_value_heads=action_expert_config.num_kv_heads,
                vocab_size=QWEN2_5_VL_VOCAB_SIZE,
                hidden_act="silu",
                torch_dtype="float32",
            )
            self.qwen_expert = Qwen2ForCausalLM(config=action_expert_config_hf)
            self.qwen_expert.model.load_state_dict(self.qwen_vl.model.state_dict(), strict=False)
            self.qwen_expert.model.embed_tokens = None
        else:
            vlm_config_hf = CONFIG_MAPPING["qwen2_5_vl"]()
            vlm_config_hf.vocab_size = QWEN2_5_VL_VOCAB_SIZE
            vlm_config_hf.image_token_id = 151655
            vlm_config_hf.vision_start_token_id = 151652
            vlm_config_hf.vision_end_token_id = 151653
            vlm_config_hf.text_config.hidden_size = vlm_config.width
            vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
            vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
            vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
            vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
            vlm_config_hf.text_config.vocab_size = QWEN2_5_VL_VOCAB_SIZE
            vlm_config_hf.text_config.hidden_act = "silu"
            vlm_config_hf.text_config.torch_dtype = "float32"

            action_expert_config_hf = CONFIG_MAPPING["qwen2"](
                hidden_size=action_expert_config.width,
                intermediate_size=action_expert_config.mlp_dim,
                num_attention_heads=action_expert_config.num_heads,
                num_hidden_layers=action_expert_config.depth,
                num_key_value_heads=action_expert_config.num_kv_heads,
                vocab_size=QWEN2_5_VL_VOCAB_SIZE,
                hidden_act="silu",
                torch_dtype="float32",
            )

            self.qwen_vl = Qwen2_5_VLForConditionalGeneration(config=vlm_config_hf)
            self.qwen_expert = Qwen2ForCausalLM(config=action_expert_config_hf)
            self.qwen_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def set_gradient_checkpointing_enabled(self, enabled: bool):
        self.qwen_vl.model.gradient_checkpointing = enabled
        if hasattr(self.qwen_vl, "visual") and hasattr(self.qwen_vl.visual, "gradient_checkpointing"):
            self.qwen_vl.visual.gradient_checkpointing = enabled
        self.qwen_expert.model.gradient_checkpointing = enabled

    def prefix_q_proj_dtype(self):
        return self.qwen_vl.model.layers[0].self_attn.q_proj.weight.dtype

    def set_prefix_attention_implementation(self, implementation: str):
        self.qwen_vl.model.config._attn_implementation = implementation  # noqa: SLF001

    def set_suffix_attention_implementation(self, implementation: str):
        self.qwen_expert.model.config._attn_implementation = implementation  # noqa: SLF001

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "visual.patch_embed",
            "input_layernorm",
            "post_attention_layernorm",
            ".norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _normalize_images_for_qwen(self, image: torch.Tensor) -> torch.Tensor:
        # OpenPI images arrive in the same normalized range used by the PaliGemma path. Convert them
        # back to [0, 1] and re-normalize with Qwen's image statistics before calling the visual tower.
        image = (image + 1.0) / 2.0
        image = torch.clamp(image, 0.0, 1.0)
        mean = torch.tensor(QWEN_IMAGE_MEAN, dtype=image.dtype, device=image.device)[None, :, None, None]
        std = torch.tensor(QWEN_IMAGE_STD, dtype=image.dtype, device=image.device)[None, :, None, None]
        return (image - mean) / std

    def _image_grid_thw(self, image: torch.Tensor) -> torch.Tensor:
        patch_size = getattr(self.qwen_vl.config.vision_config, "patch_size", 14)
        height = image.shape[-2] // patch_size
        width = image.shape[-1] // patch_size
        grid = torch.tensor([1, height, width], dtype=torch.long, device=image.device)
        return grid[None, :].expand(image.shape[0], -1)

    @staticmethod
    def _extract_processor_value(batch_feature, key: str):
        if batch_feature is None:
            return None
        if isinstance(batch_feature, dict):
            return batch_feature.get(key)
        if hasattr(batch_feature, key):
            return getattr(batch_feature, key)
        try:
            return batch_feature[key]
        except (KeyError, TypeError):
            return None

    def _preprocess_image_with_processor(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        if self.qwen_processor is None or not hasattr(self.qwen_processor, "image_processor"):
            normalized = self._normalize_images_for_qwen(image)
            return normalized, self._image_grid_thw(normalized), "manual"

        # The processor expects raw image values rather than the OpenPI [-1, 1] normalization.
        processor_images = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)
        image_inputs = self.qwen_processor.image_processor(
            images=[img.detach().cpu() for img in processor_images],
            return_tensors="pt",
        )

        pixel_values = self._extract_processor_value(image_inputs, "pixel_values")
        if pixel_values is None:
            raise ValueError("Qwen image processor did not return `pixel_values`.")
        pixel_values = pixel_values.to(device=image.device, dtype=torch.float32)

        image_grid_thw = self._extract_processor_value(image_inputs, "image_grid_thw")
        if image_grid_thw is None:
            image_grid_thw = self._image_grid_thw(pixel_values)
        else:
            image_grid_thw = image_grid_thw.to(device=image.device, dtype=torch.long)

        return pixel_values, image_grid_thw, "official_image_processor"

    def _embed_processed_image(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        image_features = self.qwen_vl.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        if isinstance(image_features, torch.Tensor):
            return image_features
        if hasattr(image_features, "last_hidden_state"):
            return image_features.last_hidden_state
        if isinstance(image_features, tuple):
            return image_features[0]
        raise TypeError(f"Unexpected image feature output type: {type(image_features)!r}")

    def build_prefix_batch(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        raw_prompts=None,
        *,
        checkpoint_fn=None,
    ) -> PrefixBatch:
        if checkpoint_fn is None:
            checkpoint_fn = lambda func, *args, **kwargs: func(*args, **kwargs)

        embs = []
        pad_masks = []
        att_masks = []
        image_grid_thws = []
        image_processor_modes = []

        for img, img_mask in zip(images, img_masks, strict=True):
            pixel_values, image_grid_thw, processor_mode = self._preprocess_image_with_processor(img)
            img_emb = checkpoint_fn(self._embed_processed_image, pixel_values, image_grid_thw)

            batch_size, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs))
            att_masks += [0] * num_img_embs
            image_grid_thws.append(image_grid_thw)
            image_processor_modes.append(processor_mode)

        def embed_language(lang_tokens):
            lang_emb = self.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * (lang_emb_dim**0.5)

        lang_emb = checkpoint_fn(embed_language, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))

        metadata = {
            "image_grid_thw": torch.stack(image_grid_thws, dim=1) if image_grid_thws else None,
            "image_token_counts": tuple(img_emb.shape[1] for img_emb in embs[:-1]),
            "language_token_length": lang_emb.shape[1],
            "image_processor_modes": tuple(image_processor_modes),
            # Preserve original prompt text so the backend can later switch from text-only token ids
            # to Qwen's official multimodal chat-template construction without changing PI0Pytorch again.
            "raw_prompts": raw_prompts,
            # TODO: populate multimodal token types and RoPE deltas once the joint-attention path
            # consumes official Qwen multimodal position metadata instead of only 1D position_ids.
            "mm_token_type_ids": None,
            "rope_deltas": None,
        }

        return PrefixBatch(
            embeds=embs,
            pad_masks=pad_masks,
            att_masks=att_masks,
            metadata=metadata,
        )

    def _get_image_token_counts(self, prefix_batch: PrefixBatch) -> tuple[int, ...]:
        counts = prefix_batch.metadata.get("image_token_counts")
        if counts is None:
            return ()
        return tuple(int(count) for count in counts)

    def _get_language_token_length(self, prefix_batch: PrefixBatch) -> int:
        return int(prefix_batch.metadata.get("language_token_length", 0))

    def _compute_llm_grid(self, grid_thw: torch.Tensor, token_count: int) -> tuple[int, int, int]:
        grid_t = int(grid_thw[0].item())
        grid_h = int(grid_thw[1].item())
        grid_w = int(grid_thw[2].item())
        merge_size = int(getattr(self.qwen_vl.config.vision_config, "spatial_merge_size", 1))

        llm_t = max(grid_t, 1)
        llm_h = max(grid_h // merge_size, 1)
        llm_w = max(grid_w // merge_size, 1)
        if llm_t * llm_h * llm_w == token_count:
            return llm_t, llm_h, llm_w

        if grid_t * grid_h * grid_w == token_count:
            return max(grid_t, 1), max(grid_h, 1), max(grid_w, 1)

        return 1, 1, token_count

    def _make_image_position_ids(
        self,
        grid_thw: torch.Tensor,
        token_count: int,
        offset: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        llm_t, llm_h, llm_w = self._compute_llm_grid(grid_thw, token_count)
        if llm_t * llm_h * llm_w != token_count:
            flat_pos = torch.arange(token_count, device=device, dtype=torch.long) + offset
            return flat_pos[None, :].expand(3, -1)

        time_ids = torch.arange(llm_t, device=device, dtype=torch.long)[:, None, None].expand(llm_t, llm_h, llm_w)
        height_ids = torch.arange(llm_h, device=device, dtype=torch.long)[None, :, None].expand(llm_t, llm_h, llm_w)
        width_ids = torch.arange(llm_w, device=device, dtype=torch.long)[None, None, :].expand(llm_t, llm_h, llm_w)
        return torch.stack(
            [
                time_ids.reshape(-1) + offset,
                height_ids.reshape(-1) + offset,
                width_ids.reshape(-1) + offset,
            ],
            dim=0,
        )

    def _make_text_position_ids(
        self,
        valid_token_count: int,
        offset: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        flat_pos = torch.arange(valid_token_count, device=device, dtype=torch.long) + offset
        return flat_pos[None, :].expand(3, -1)

    def _build_prefix_position_metadata(self, prefix_batch: PrefixBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # This mirrors Qwen's multimodal idea (image tokens get grid-aware 3-axis positions and
        # text/suffix tokens continue from the resulting offset), but it is still an OpenPI-side
        # construction because prefix tokens are not yet serialized through Qwen's official
        # chat-template/image-placeholder path.
        image_grid_thw = prefix_batch.metadata.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError("Qwen prefix metadata is missing `image_grid_thw`.")

        image_token_counts = self._get_image_token_counts(prefix_batch)
        language_token_length = self._get_language_token_length(prefix_batch)
        batch_size, total_prefix_len = prefix_batch.pad_masks.shape
        device = prefix_batch.pad_masks.device

        position_ids = torch.ones(3, batch_size, total_prefix_len, dtype=torch.long, device=device)
        continuation_offsets = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        rope_deltas = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            cursor = 0
            next_offset = 0

            for image_idx, token_count in enumerate(image_token_counts):
                token_slice = slice(cursor, cursor + token_count)
                image_pad_mask = prefix_batch.pad_masks[batch_idx, token_slice]
                if torch.any(image_pad_mask):
                    image_positions = self._make_image_position_ids(
                        image_grid_thw[batch_idx, image_idx],
                        token_count,
                        next_offset,
                        device=device,
                    )
                    position_ids[:, batch_idx, token_slice] = image_positions
                    next_offset = int(image_positions.max().item()) + 1
                cursor += token_count

            text_slice = slice(cursor, cursor + language_token_length)
            text_pad_mask = prefix_batch.pad_masks[batch_idx, text_slice]
            valid_text_tokens = int(text_pad_mask.sum().item())
            if valid_text_tokens > 0:
                text_positions = self._make_text_position_ids(valid_text_tokens, next_offset, device=device)
                position_ids[:, batch_idx, cursor : cursor + valid_text_tokens] = text_positions
                next_offset += valid_text_tokens

            continuation_offsets[batch_idx, 0] = next_offset
            rope_deltas[batch_idx, 0] = next_offset - int(prefix_batch.pad_masks[batch_idx].sum().item())

        return position_ids, continuation_offsets, rope_deltas

    def _build_continuation_position_ids(
        self,
        continuation_offsets: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, suffix_len = suffix_pad_masks.shape
        device = suffix_pad_masks.device
        position_ids = torch.ones(3, batch_size, suffix_len, dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            valid_suffix_tokens = int(suffix_pad_masks[batch_idx].sum().item())
            if valid_suffix_tokens == 0:
                continue
            suffix_positions = self._make_text_position_ids(
                valid_suffix_tokens,
                int(continuation_offsets[batch_idx, 0].item()),
                device=device,
            )
            position_ids[:, batch_idx, :valid_suffix_tokens] = suffix_positions

        return position_ids

    def build_prefix_prefill_context(self, prefix_batch: PrefixBatch) -> AttentionContext:
        prefill_context = super().build_prefix_prefill_context(prefix_batch)
        position_ids, continuation_offsets, rope_deltas = self._build_prefix_position_metadata(prefix_batch)
        prefill_context.forward_kwargs["position_ids"] = position_ids
        prefill_context.metadata["prefix_position_ids"] = position_ids
        prefill_context.metadata["continuation_offsets"] = continuation_offsets
        prefill_context.metadata["rope_deltas"] = rope_deltas
        return prefill_context

    def build_joint_attention_context(
        self,
        prefix_batch: PrefixBatch,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> AttentionContext:
        joint_context = super().build_joint_attention_context(prefix_batch, suffix_pad_masks, suffix_att_masks)
        prefix_position_ids, continuation_offsets, rope_deltas = self._build_prefix_position_metadata(prefix_batch)
        suffix_position_ids = self._build_continuation_position_ids(continuation_offsets, suffix_pad_masks)
        joint_context.forward_kwargs["position_ids"] = torch.cat([prefix_position_ids, suffix_position_ids], dim=2)
        joint_context.metadata["continuation_offsets"] = continuation_offsets
        joint_context.metadata["rope_deltas"] = rope_deltas
        return joint_context

    def build_suffix_decode_context(
        self,
        prefix_cache,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> AttentionContext:
        decode_context = super().build_suffix_decode_context(prefix_cache, suffix_pad_masks, suffix_att_masks)
        continuation_offsets = prefix_cache.metadata.get("continuation_offsets")
        if continuation_offsets is None:
            raise ValueError("Qwen prefix cache is missing `continuation_offsets`.")
        rope_deltas = prefix_cache.metadata.get("rope_deltas")
        suffix_position_ids = self._build_continuation_position_ids(continuation_offsets, suffix_pad_masks)
        decode_context.forward_kwargs["position_ids"] = suffix_position_ids
        decode_context.metadata["rope_deltas"] = rope_deltas
        return decode_context

    def build_prefix_cache_metadata(
        self,
        prefix_batch: PrefixBatch,
        *,
        prefill_context: AttentionContext,
    ) -> dict[str, object]:
        metadata = dict(prefix_batch.metadata)
        metadata.update(prefill_context.metadata)
        # TODO: replace these OpenPI-style prefix positions with Qwen continuation positions once
        # suffix denoising consumes backend-specific cache/position metadata end-to-end.
        return metadata

    def embed_image(self, image: torch.Tensor):
        pixel_values, image_grid_thw, _ = self._preprocess_image_with_processor(image)
        return self._embed_processed_image(pixel_values, image_grid_thw)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwen_vl.model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        del adarms_cond
        if inputs_embeds[1] is None:
            prefix_output = self.qwen_vl.model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            hidden_states = inputs_embeds[1]
            num_layers = len(self.qwen_expert.model.layers)

            for layer_idx in range(num_layers):
                layer = self.qwen_expert.model.layers[layer_idx]
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

                batch_size, seq_len, _ = hidden_states.shape
                head_dim = layer.self_attn.head_dim
                num_heads = layer.self_attn.q_proj.out_features // head_dim
                num_kv_heads = layer.self_attn.k_proj.out_features // head_dim

                query_states = layer.self_attn.q_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim)
                key_states = layer.self_attn.k_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim)
                value_states = layer.self_attn.v_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim)

                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                dummy_hidden_states = torch.zeros(
                    batch_size,
                    seq_len,
                    query_states.shape[-1] * query_states.shape[1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.qwen_vl.model.rotary_emb(dummy_hidden_states, position_ids)
                query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

                past_key_states, past_value_states = _get_layer_past_key_value(past_key_values, layer_idx)
                if past_key_states is not None and past_value_states is not None:
                    key_states = torch.cat([past_key_states, key_states], dim=2)
                    value_states = torch.cat([past_value_states, value_states], dim=2)

                key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
                value_states = _repeat_kv(value_states, num_heads // num_kv_heads)

                attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(
                    query_states.shape[-1]
                )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)

                out_emb = layer.self_attn.o_proj(attn_output)
                hidden_states = residual + out_emb
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            suffix_output = self.qwen_expert.model.norm(hidden_states)
            prefix_output = None
            prefix_past_key_values = None
        else:
            # Joint mode mirrors the existing OpenPI path: each branch computes its own q/k/v, the
            # sequence dimension is concatenated for one shared attention pass, then each slice runs
            # through its own residual/norm/MLP stack.
            models = [self.qwen_vl.model, self.qwen_expert.model]
            num_layers = len(self.qwen_vl.model.layers)

            use_gradient_checkpointing = (
                hasattr(self.qwen_expert.model, "gradient_checkpointing")
                and self.qwen_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids):
                models = [self.qwen_vl.model, self.qwen_expert.model]

                query_states = []
                key_states = []
                value_states = []
                residuals = []
                seq_lens = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    residuals.append(hidden_states)
                    hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901
                    batch_size, seq_len, _ = hidden_states.shape
                    seq_lens.append(seq_len)
                    head_dim = layer.self_attn.head_dim
                    num_heads = layer.self_attn.q_proj.out_features // head_dim
                    num_kv_heads = layer.self_attn.k_proj.out_features // head_dim

                    query_state = layer.self_attn.q_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim)
                    key_state = layer.self_attn.k_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim)
                    value_state = layer.self_attn.v_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim)

                    query_states.append(query_state.transpose(1, 2))
                    key_states.append(key_state.transpose(1, 2))
                    value_states.append(value_state.transpose(1, 2))

                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_hidden_states = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1] * query_states.shape[1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.qwen_vl.model.rotary_emb(dummy_hidden_states, position_ids)
                query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

                num_kv_heads = key_states.shape[1]
                num_heads = query_states.shape[1]
                key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
                value_states = _repeat_kv(value_states, num_heads // num_kv_heads)

                attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(
                    query_states.shape[-1]
                )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)

                outputs_embeds = []
                start_pos = 0
                for i, residual in enumerate(residuals):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + seq_lens[i]
                    out_emb = layer.self_attn.o_proj(attn_output[:, start_pos:end_pos])
                    hidden_states = residual + out_emb
                    residual = hidden_states
                    hidden_states = layer.post_attention_layernorm(hidden_states)
                    hidden_states = layer.mlp(hidden_states)
                    hidden_states = residual + hidden_states
                    outputs_embeds.append(hidden_states)
                    start_pos = end_pos

                return outputs_embeds

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids)

            outputs_embeds = []
            for i, hidden_states in enumerate(inputs_embeds):
                outputs_embeds.append(models[i].norm(hidden_states))

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
