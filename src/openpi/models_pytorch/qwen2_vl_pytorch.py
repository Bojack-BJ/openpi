import math
from typing import Literal

import pytest
import torch
from torch import nn
from transformers import Qwen2ForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen2 import modeling_qwen2


QWEN2_5_VL_VOCAB_SIZE = 152_064
QWEN_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
QWEN_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)


class Qwen2_5_VLWithExpertModel(nn.Module):
    """Qwen2.5-VL adapter that preserves OpenPI's existing prefix/suffix contract.

    Images and prompt tokens are embedded separately, then mixed through the same shared
    prefix/suffix self-attention pattern used by the PaliGemma path. This is intentionally
    not a drop-in reproduction of the official Qwen multimodal input stack.
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

    def embed_image(self, image: torch.Tensor):
        image = self._normalize_images_for_qwen(image)
        image_grid_thw = self._image_grid_thw(image)
        # TODO: move to the full Qwen processor/input path when we support image placeholders and
        # multimodal position ids end-to-end instead of OpenPI's separate prefix embeddings.
        image_features = self.qwen_vl.get_image_features(pixel_values=image, image_grid_thw=image_grid_thw)
        if isinstance(image_features, torch.Tensor):
            return image_features
        if hasattr(image_features, "last_hidden_state"):
            return image_features.last_hidden_state
        if isinstance(image_features, tuple):
            return image_features[0]
        raise TypeError(f"Unexpected image feature output type: {type(image_features)!r}")

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
            suffix_output = self.qwen_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            suffix_output = suffix_output.last_hidden_state
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
