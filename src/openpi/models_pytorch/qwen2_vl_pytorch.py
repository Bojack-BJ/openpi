import math
from contextlib import nullcontext
from typing import Literal

import pytest
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers import AutoProcessor
from transformers import Qwen2ForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

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
    can already route through the official Qwen processor. For VLA-focused training we keep
    the simpler OpenPI contract: concatenate image and text embeddings into the prefix and
    use ordinary 1D prefix/suffix positions. This intentionally does not reproduce Qwen's
    full multimodal placeholder/template or position semantics yet.
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
        self.precision = precision
        self.qwen_processor = (
            AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True) if hf_model_id is not None else None
        )
        self.qwen_tokenizer = (
            self.qwen_processor.tokenizer
            if self.qwen_processor is not None and hasattr(self.qwen_processor, "tokenizer")
            else self.qwen_processor
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
                    "Use a matching pair such as `qwen2_5_3b` + `Qwen/Qwen2.5-VL-3B-Instruct` "
                    "or `qwen2_5_7b` + `Qwen/Qwen2.5-VL-7B-Instruct` for both prefix and expert."
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
            self.qwen_expert.model.load_state_dict(self._get_qwen_expert_init_state_dict(), strict=False)
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

    def _get_qwen_text_model(self) -> nn.Module:
        candidates = []

        decoder_getter = getattr(self.qwen_vl, "get_decoder", None)
        if callable(decoder_getter):
            candidates.append(decoder_getter())

        candidates.extend(
            [
                getattr(self.qwen_vl, "model", None),
                getattr(getattr(self.qwen_vl, "model", None), "model", None),
                getattr(getattr(self.qwen_vl, "model", None), "language_model", None),
                getattr(self.qwen_vl, "language_model", None),
            ]
        )

        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "layers"):
                return candidate

        raise AttributeError("Could not locate the Qwen text decoder layers on the loaded VLM backbone.")

    def _get_qwen_expert_init_state_dict(self) -> dict[str, torch.Tensor]:
        state_dict = self._get_qwen_text_model().state_dict()
        # The suffix expert reuses decoder layers and final norm only. Skip the token embedding
        # table because HF VL/text wrappers can expose different vocab sizes even when the decoder
        # geometry matches.
        return {key: value for key, value in state_dict.items() if not key.startswith("embed_tokens.")}

    def _get_qwen_mrope_section(self) -> list[int]:
        rope_scaling = getattr(self.qwen_vl.config, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = getattr(getattr(self._get_qwen_text_model(), "config", None), "rope_scaling", None)
        if rope_scaling is None or "mrope_section" not in rope_scaling:
            raise ValueError("Qwen2.5-VL config did not expose `rope_scaling['mrope_section']`.")
        return rope_scaling["mrope_section"]

    def _autocast_context(self, device: torch.device):
        if device.type == "cuda" and self.precision == "bfloat16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _prepare_qwen_position_ids(position_ids: torch.Tensor | None) -> torch.Tensor | None:
        if position_ids is None:
            return None
        if position_ids.ndim == 3:
            return position_ids
        if position_ids.ndim != 2:
            raise ValueError(f"Expected Qwen position_ids to have rank 2 or 3, got {tuple(position_ids.shape)}")
        # Qwen2.5-VL rotary embeddings expect 3 position streams. For the current OpenPI-style
        # simple prefix/suffix path we intentionally reuse the same 1D positions on all three axes
        # instead of reproducing Qwen's full multimodal THW position semantics.
        return position_ids.unsqueeze(0).expand(3, -1, -1)

    def set_gradient_checkpointing_enabled(self, enabled: bool):
        self._get_qwen_text_model().gradient_checkpointing = enabled
        if hasattr(self.qwen_vl, "visual") and hasattr(self.qwen_vl.visual, "gradient_checkpointing"):
            self.qwen_vl.visual.gradient_checkpointing = enabled
        self.qwen_expert.model.gradient_checkpointing = enabled

    def prefix_q_proj_dtype(self):
        return self._get_qwen_text_model().layers[0].self_attn.q_proj.weight.dtype

    def set_prefix_attention_implementation(self, implementation: str):
        self._get_qwen_text_model().config._attn_implementation = implementation  # noqa: SLF001

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
            do_rescale=False,
            return_tensors="pt",
        )

        pixel_values = self._extract_processor_value(image_inputs, "pixel_values")
        if pixel_values is None:
            raise ValueError("Qwen image processor did not return `pixel_values`.")
        visual_dtype = next(self.qwen_vl.visual.parameters()).dtype
        pixel_values = pixel_values.to(device=image.device, dtype=visual_dtype)

        image_grid_thw = self._extract_processor_value(image_inputs, "image_grid_thw")
        if image_grid_thw is None:
            image_grid_thw = self._image_grid_thw(pixel_values)
        else:
            image_grid_thw = image_grid_thw.to(device=image.device, dtype=torch.long)

        return pixel_values, image_grid_thw, "official_image_processor"

    def _embed_processed_image(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        with self._autocast_context(pixel_values.device):
            image_features = self.qwen_vl.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        if isinstance(image_features, torch.Tensor):
            return image_features
        if hasattr(image_features, "last_hidden_state"):
            return image_features.last_hidden_state
        if isinstance(image_features, tuple):
            return image_features[0]
        raise TypeError(f"Unexpected image feature output type: {type(image_features)!r}")

    def _get_image_token_counts(self, image_grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = int(getattr(self.qwen_vl.visual, "spatial_merge_size", 1))
        return image_grid_thw.prod(dim=-1) // (merge_size**2)

    def _pack_image_features(
        self,
        image_features: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_counts = self._get_image_token_counts(image_grid_thw)
        batch_size = image_grid_thw.shape[0]

        if image_features.ndim == 3:
            if image_features.shape[0] != batch_size:
                raise ValueError(
                    f"Expected batched image features for {batch_size} samples, got {tuple(image_features.shape)}."
                )
            max_tokens = image_features.shape[1]
            token_pad_mask = torch.arange(max_tokens, device=image_features.device)[None, :] < token_counts[:, None]
            return image_features, token_pad_mask

        if image_features.ndim != 2:
            raise ValueError(f"Unexpected Qwen image feature shape: {tuple(image_features.shape)}")

        split_sizes = token_counts.tolist()
        if sum(split_sizes) != image_features.shape[0]:
            # In the current OpenPI pipeline all images are resized to a fixed resolution, so Qwen
            # image features are expected to have a uniform token count per sample. Some processor /
            # transformers combinations return flattened features whose total length does not match
            # the naive `image_grid_thw`-derived count, so fall back to an equal split across the
            # batch instead of crashing.
            if batch_size > 0 and image_features.shape[0] % batch_size == 0:
                uniform_tokens = image_features.shape[0] // batch_size
                split_sizes = [uniform_tokens] * batch_size
            else:
                raise ValueError(
                    "Qwen image features do not match `image_grid_thw`: "
                    f"sum(token_counts)={sum(split_sizes)} vs feature_rows={image_features.shape[0]}"
                )

        chunks = torch.split(image_features, split_sizes, dim=0)
        max_tokens = max(split_sizes, default=0)
        hidden_size = image_features.shape[-1]
        packed = image_features.new_zeros((batch_size, max_tokens, hidden_size))
        token_pad_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=image_features.device)

        for idx, chunk in enumerate(chunks):
            packed[idx, : chunk.shape[0]] = chunk
            token_pad_mask[idx, : chunk.shape[0]] = True

        return packed, token_pad_mask

    @staticmethod
    def _normalize_raw_prompts(raw_prompts, batch_size: int) -> list[str] | None:
        if raw_prompts is None:
            return None

        if isinstance(raw_prompts, str):
            prompts = [raw_prompts] * batch_size
        elif isinstance(raw_prompts, (list, tuple)):
            prompts = list(raw_prompts)
        elif hasattr(raw_prompts, "shape") and getattr(raw_prompts, "ndim", 0) > 0:
            prompts = list(raw_prompts.tolist())
        elif hasattr(raw_prompts, "tolist"):
            prompts = raw_prompts.tolist()
            if not isinstance(prompts, list):
                prompts = [prompts] * batch_size
        else:
            prompts = [raw_prompts] * batch_size

        normalized = []
        for prompt in prompts:
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            elif not isinstance(prompt, str) and hasattr(prompt, "item"):
                prompt = prompt.item()
            if not isinstance(prompt, str):
                prompt = str(prompt)
            normalized.append(prompt)

        if len(normalized) == 1 and batch_size > 1:
            normalized *= batch_size
        if len(normalized) != batch_size:
            raise ValueError(f"Expected {batch_size} raw prompts, got {len(normalized)}.")
        return normalized

    def _tokenize_raw_prompts(
        self,
        raw_prompts,
        *,
        batch_size: int,
        max_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if raw_prompts is None or self.qwen_tokenizer is None:
            return None

        prompts = self._normalize_raw_prompts(raw_prompts, batch_size=batch_size)
        if prompts is None:
            return None

        cleaned_prompts = [prompt.strip().replace("_", " ").replace("\n", " ") for prompt in prompts]
        rendered_prompts = []
        for prompt in cleaned_prompts:
            if hasattr(self.qwen_tokenizer, "apply_chat_template"):
                # TODO: if we later adopt backend-specific multimodal templates, build the full
                # image-aware chat content here instead of this text-only fallback.
                rendered_prompts.append(
                    self.qwen_tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                rendered_prompts.append(f"{prompt}\n")

        pad_token_id = getattr(self.qwen_tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("Qwen tokenizer must define `pad_token_id` for backend-owned prompt assembly.")

        tokenized = self.qwen_tokenizer(
            rendered_prompts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return (
            tokenized["input_ids"].to(device=device, dtype=torch.long),
            tokenized["attention_mask"].to(device=device, dtype=torch.bool),
        )

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
            img_emb, img_token_pad_mask = self._pack_image_features(img_emb, image_grid_thw)

            batch_size, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs) & img_token_pad_mask)
            att_masks += [0] * num_img_embs
            image_grid_thws.append(image_grid_thw)
            image_processor_modes.append(processor_mode)

        prompt_token_source = "tokenized_prompt"
        if raw_prompts is not None:
            backend_tokens = self._tokenize_raw_prompts(
                raw_prompts,
                batch_size=lang_tokens.shape[0],
                max_len=lang_tokens.shape[1],
                device=lang_tokens.device,
            )
            if backend_tokens is not None:
                lang_tokens, lang_masks = backend_tokens
                prompt_token_source = "backend_raw_prompt"

        def embed_language(lang_tokens):
            lang_emb = self.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * (lang_emb_dim**0.5)

        lang_emb = checkpoint_fn(embed_language, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        image_token_lengths = tuple(emb.shape[1] for emb in embs[:-1])
        language_token_length = lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))

        metadata = {
            "image_grid_thw": torch.stack(image_grid_thws, dim=1) if image_grid_thws else None,
            "image_token_lengths": image_token_lengths,
            "language_token_length": language_token_length,
            "image_processor_modes": tuple(image_processor_modes),
            "prompt_token_source": prompt_token_source,
            # Preserve original prompt text so future backend-specific multimodal templates can
            # replace this simple text-only prompt assembly without changing PI0Pytorch again.
            "raw_prompts": raw_prompts,
        }

        return PrefixBatch(
            embeds=embs,
            pad_masks=pad_masks,
            att_masks=att_masks,
            metadata=metadata,
        )

    def embed_image(self, image: torch.Tensor):
        pixel_values, image_grid_thw, _ = self._preprocess_image_with_processor(image)
        return self._embed_processed_image(pixel_values, image_grid_thw)

    def embed_language_tokens(self, tokens: torch.Tensor):
        embedding_layer = self.qwen_vl.get_input_embeddings()
        if embedding_layer is None:
            raise ValueError("Qwen2.5-VL model did not expose input embeddings.")
        return embedding_layer(tokens)

    def _build_qwen_text_position_chunk(
        self,
        valid_count: int,
        *,
        start_position: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(start_position, start_position + valid_count, device=device, dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1)

    def _build_qwen_vision_position_chunk(
        self,
        grid_thw: torch.Tensor | None,
        *,
        valid_count: int,
        start_position: int,
        device: torch.device,
    ) -> torch.Tensor:
        if grid_thw is None:
            return self._build_qwen_text_position_chunk(valid_count, start_position=start_position, device=device)

        merge_size = int(getattr(self.qwen_vl.visual, "spatial_merge_size", 1))
        grid_t, grid_h, grid_w = (int(x) for x in grid_thw.tolist())
        llm_grid_t = max(grid_t, 1)
        llm_grid_h = max(grid_h // merge_size, 1)
        llm_grid_w = max(grid_w // merge_size, 1)
        expected_count = llm_grid_t * llm_grid_h * llm_grid_w

        if expected_count != valid_count:
            # Some processor/transformers combinations expose image features whose flattened token
            # count does not line up with the naive THW-derived expectation. Keep the backend-owned
            # path robust by falling back to the simpler 1D positions for that image block.
            return self._build_qwen_text_position_chunk(valid_count, start_position=start_position, device=device)

        temporal = torch.arange(llm_grid_t, device=device, dtype=torch.long).repeat_interleave(llm_grid_h * llm_grid_w)
        height = (
            torch.arange(llm_grid_h, device=device, dtype=torch.long)
            .repeat_interleave(llm_grid_w)
            .repeat(llm_grid_t)
        )
        width = torch.arange(llm_grid_w, device=device, dtype=torch.long).repeat(llm_grid_t * llm_grid_h)
        return torch.stack([temporal, height, width], dim=0) + start_position

    def _build_qwen_prefix_position_ids(self, prefix_batch: PrefixBatch) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, prefix_len = prefix_batch.pad_masks.shape
        device = prefix_batch.pad_masks.device
        position_ids = torch.zeros((3, batch_size, prefix_len), dtype=torch.long, device=device)
        next_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

        image_grid_thw = prefix_batch.metadata.get("image_grid_thw")
        image_token_lengths = prefix_batch.metadata.get("image_token_lengths", ())
        language_token_length = int(prefix_batch.metadata.get("language_token_length", 0))

        cursor = 0
        for image_idx, image_token_length in enumerate(image_token_lengths):
            token_slice = slice(cursor, cursor + image_token_length)
            for batch_idx in range(batch_size):
                valid_count = int(prefix_batch.pad_masks[batch_idx, token_slice].sum().item())
                if valid_count == 0:
                    continue

                grid_thw = None
                if image_grid_thw is not None:
                    grid_thw = image_grid_thw[batch_idx, image_idx]

                chunk_position_ids = self._build_qwen_vision_position_chunk(
                    grid_thw,
                    valid_count=valid_count,
                    start_position=int(next_positions[batch_idx].item()),
                    device=device,
                )
                position_ids[:, batch_idx, cursor : cursor + valid_count] = chunk_position_ids
                next_positions[batch_idx] = int(chunk_position_ids.max().item()) + 1
            cursor += image_token_length

        if language_token_length > 0:
            token_slice = slice(cursor, cursor + language_token_length)
            for batch_idx in range(batch_size):
                valid_count = int(prefix_batch.pad_masks[batch_idx, token_slice].sum().item())
                if valid_count == 0:
                    continue
                chunk_position_ids = self._build_qwen_text_position_chunk(
                    valid_count,
                    start_position=int(next_positions[batch_idx].item()),
                    device=device,
                )
                position_ids[:, batch_idx, cursor : cursor + valid_count] = chunk_position_ids
                next_positions[batch_idx] = int(chunk_position_ids[0, -1].item()) + 1

        return position_ids, next_positions

    def _build_qwen_suffix_position_ids(
        self,
        suffix_pad_masks: torch.Tensor,
        start_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, suffix_len = suffix_pad_masks.shape
        device = suffix_pad_masks.device
        position_ids = torch.zeros((3, batch_size, suffix_len), dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            valid_count = int(suffix_pad_masks[batch_idx].sum().item())
            if valid_count == 0:
                continue
            chunk_position_ids = self._build_qwen_text_position_chunk(
                valid_count,
                start_position=int(start_positions[batch_idx].item()),
                device=device,
            )
            position_ids[:, batch_idx, :valid_count] = chunk_position_ids

        return position_ids

    def build_prefix_prefill_context(self, prefix_batch: PrefixBatch) -> AttentionContext:
        prefix_att_2d_masks = self.make_att_2d_masks(prefix_batch.pad_masks, prefix_batch.att_masks)
        prefix_position_ids, prefix_next_positions = self._build_qwen_prefix_position_ids(prefix_batch)
        prefix_att_2d_masks_4d = self.prepare_attention_mask_4d(prefix_att_2d_masks)
        return AttentionContext(
            forward_kwargs={
                "attention_mask": prefix_att_2d_masks_4d,
                "position_ids": prefix_position_ids,
            },
            metadata={
                "prefix_att_2d_masks": prefix_att_2d_masks,
                "prefix_att_2d_masks_4d": prefix_att_2d_masks_4d,
                "prefix_position_ids": prefix_position_ids,
                "prefix_next_positions": prefix_next_positions,
            },
        )

    def build_joint_attention_context(
        self,
        prefix_batch: PrefixBatch,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> AttentionContext:
        pad_masks = torch.cat([prefix_batch.pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_batch.att_masks, suffix_att_masks], dim=1)
        att_2d_masks = self.make_att_2d_masks(pad_masks, att_masks)
        prefix_position_ids, prefix_next_positions = self._build_qwen_prefix_position_ids(prefix_batch)
        suffix_position_ids = self._build_qwen_suffix_position_ids(suffix_pad_masks, prefix_next_positions)
        position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=2)
        att_2d_masks_4d = self.prepare_attention_mask_4d(att_2d_masks)
        return AttentionContext(
            forward_kwargs={
                "attention_mask": att_2d_masks_4d,
                "position_ids": position_ids,
            },
            metadata={
                "att_2d_masks": att_2d_masks,
                "att_masks": att_masks,
                "pad_masks": pad_masks,
                "prefix_next_positions": prefix_next_positions,
                "prefix_position_ids": prefix_position_ids,
                "suffix_position_ids": suffix_position_ids,
            },
        )

    def build_suffix_decode_context(
        self,
        prefix_cache,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> AttentionContext:
        prefix_pad_masks = prefix_cache.pad_masks
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = self.make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self.prepare_attention_mask_4d(full_att_2d_masks)

        prefix_next_positions = prefix_cache.metadata.get("prefix_next_positions")
        if prefix_next_positions is None:
            prefix_next_positions = torch.sum(prefix_pad_masks, dim=-1)
        suffix_position_ids = self._build_qwen_suffix_position_ids(suffix_pad_masks, prefix_next_positions)
        return AttentionContext(
            forward_kwargs={
                "attention_mask": full_att_2d_masks_4d,
                "position_ids": suffix_position_ids,
            },
            metadata={
                "full_att_2d_masks": full_att_2d_masks,
                "prefix_next_positions": prefix_next_positions,
                "suffix_position_ids": suffix_position_ids,
            },
        )

    def build_prefix_cache_metadata(
        self,
        prefix_batch: PrefixBatch,
        *,
        prefill_context: AttentionContext,
    ) -> dict[str, torch.Tensor]:
        metadata = dict(prefix_batch.metadata)
        metadata["prefix_next_positions"] = prefill_context.metadata["prefix_next_positions"]
        metadata["prefix_position_ids"] = prefill_context.metadata["prefix_position_ids"]
        return metadata

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
        prefix_text_model = self._get_qwen_text_model()
        qwen_position_ids = self._prepare_qwen_position_ids(position_ids)
        mrope_section = self._get_qwen_mrope_section()
        if inputs_embeds[1] is None:
            with self._autocast_context(inputs_embeds[0].device):
                prefix_output = prefix_text_model.forward(
                    inputs_embeds=inputs_embeds[0],
                    attention_mask=attention_mask,
                    position_ids=qwen_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            hidden_states = inputs_embeds[1]
            num_layers = len(self.qwen_expert.model.layers)

            with self._autocast_context(hidden_states.device):
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
                    cos, sin = prefix_text_model.rotary_emb(dummy_hidden_states, qwen_position_ids)
                    query_states, key_states = modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb(
                        query_states,
                        key_states,
                        cos,
                        sin,
                        mrope_section,
                    )

                    past_key_states, past_value_states = _get_layer_past_key_value(past_key_values, layer_idx)
                    if past_key_states is not None and past_value_states is not None:
                        key_states = torch.cat([past_key_states, key_states], dim=2)
                        value_states = torch.cat([past_value_states, value_states], dim=2)

                    key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
                    value_states = _repeat_kv(value_states, num_heads // num_kv_heads)

                    attn_output = F.scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                    )
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
            models = [prefix_text_model, self.qwen_expert.model]
            num_layers = len(prefix_text_model.layers)

            use_gradient_checkpointing = (
                hasattr(self.qwen_expert.model, "gradient_checkpointing")
                and self.qwen_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids):
                models = [prefix_text_model, self.qwen_expert.model]
                with self._autocast_context(inputs_embeds[0].device):
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
                    cos, sin = prefix_text_model.rotary_emb(dummy_hidden_states, qwen_position_ids)
                    query_states, key_states = modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb(
                        query_states,
                        key_states,
                        cos,
                        sin,
                        mrope_section,
                    )

                    num_kv_heads = key_states.shape[1]
                    num_heads = query_states.shape[1]
                    key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
                    value_states = _repeat_kv(value_states, num_heads // num_kv_heads)

                    attn_output = F.scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                    )
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

            with self._autocast_context(inputs_embeds[0].device):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    outputs_embeds.append(models[i].norm(hidden_states))

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
