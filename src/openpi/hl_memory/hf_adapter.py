from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from PIL import Image
import torch

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.schema import HLMemoryPrediction


def create_hf_adapter(config: HLMemoryConfig) -> "BaseHLVLMAdapter":
    if config.vlm_backend == "paligemma":
        return PaliGemmaHLAdapter(config)
    if config.vlm_backend == "qwen2_5_vl":
        return Qwen25HLAdapter(config)
    if config.vlm_backend == "qwen3_5_vl":
        raise NotImplementedError(
            "V1 HL memory runtime does not yet implement `qwen3_5_vl`. "
            "Use `paligemma` or `qwen2_5_vl` for now."
        )
    raise ValueError(f"Unsupported HL VLM backend: {config.vlm_backend}")


@dataclasses.dataclass(frozen=True)
class LoadedHLVLM:
    processor: Any
    model: Any


class BaseHLVLMAdapter:
    def __init__(self, config: HLMemoryConfig):
        self.config = config

    def load(self, *, model_path: str | None = None, device: str | torch.device = "cpu") -> LoadedHLVLM:
        transformers = _import_transformers()
        torch_dtype = self._resolve_torch_dtype()
        if torch.device(device).type == "cpu" and torch_dtype == torch.bfloat16:
            torch_dtype = torch.float32
        pretrained_path = model_path or self.config.resolved_model_id
        processor = transformers.AutoProcessor.from_pretrained(pretrained_path, trust_remote_code=True)
        model = self._load_model(transformers, pretrained_path, torch_dtype=torch_dtype)
        model.to(device)
        model.eval()
        return LoadedHLVLM(processor=processor, model=model)

    def prepare_training_inputs(
        self,
        loaded: LoadedHLVLM,
        sample: ExportedHLMemorySample,
        panel_image: Image.Image,
        *,
        device: str | torch.device,
    ) -> Mapping[str, torch.Tensor]:
        prompt_inputs = self._encode_prompt_only(loaded.processor, sample, panel_image)
        full_inputs = self._encode_prompt_and_target(loaded.processor, sample, panel_image)
        labels = full_inputs["input_ids"].clone()
        prompt_length = min(int(prompt_inputs["input_ids"].shape[-1]), int(labels.shape[-1]))
        labels[..., :prompt_length] = -100
        tensors = {
            key: value.to(device)
            for key, value in full_inputs.items()
            if isinstance(value, torch.Tensor)
        }
        tensors["labels"] = labels.to(device)
        return tensors

    def predict(
        self,
        loaded: LoadedHLVLM,
        sample: ExportedHLMemorySample,
        panel_image: Image.Image,
        *,
        device: str | torch.device,
    ) -> HLMemoryPrediction:
        inputs = self._encode_prompt_only(loaded.processor, sample, panel_image)
        input_ids = inputs["input_ids"].to(device)
        generation_inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        generated_ids = loaded.model.generate(
            **generation_inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        generated_suffix = generated_ids[:, input_ids.shape[-1] :]
        tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
        decoded = tokenizer.decode(generated_suffix[0], skip_special_tokens=True)
        return HLMemoryPrediction.from_json(decoded)

    def build_prompt(self, sample: ExportedHLMemorySample) -> str:
        memory_count = len(sample.memory_frame_paths)
        recent_count = len(sample.recent_frame_paths)
        previous_memory = sample.language_memory.strip() or "No progress has been recorded yet."
        return (
            "You are the high-level planning policy for a robot.\n"
            "The input image is a context panel: historical memory frames are shown first, "
            "and recent observation frames are shown last.\n"
            "Update the language memory and predict the current subtask.\n"
            "Return exactly one JSON object with keys "
            "`updated_language_memory`, `current_subtask`, `keyframe_positions`, `phase`, "
            "`target_query`, `goal_query`.\n"
            "`keyframe_positions` must be 1-indexed positions over the recent frames only.\n"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
            f"Previous language memory: {previous_memory}\n"
            f"Historical keyframe count: {memory_count}\n"
            f"Recent frame count: {recent_count}\n"
        )

    def build_target_text(self, sample: ExportedHLMemorySample) -> str:
        return sample.target_prediction().to_json()

    def _resolve_torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.precision == "bfloat16" else torch.float32

    def _load_model(self, transformers: Any, pretrained_path: str, *, torch_dtype: torch.dtype) -> Any:
        raise NotImplementedError

    def _encode_prompt_only(self, processor: Any, sample: ExportedHLMemorySample, panel_image: Image.Image) -> Mapping[str, Any]:
        raise NotImplementedError

    def _encode_prompt_and_target(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        panel_image: Image.Image,
    ) -> Mapping[str, Any]:
        raise NotImplementedError


class PaliGemmaHLAdapter(BaseHLVLMAdapter):
    def _load_model(self, transformers: Any, pretrained_path: str, *, torch_dtype: torch.dtype) -> Any:
        return transformers.PaliGemmaForConditionalGeneration.from_pretrained(
            pretrained_path,
            torch_dtype=torch_dtype,
        )

    def _encode_prompt_only(self, processor: Any, sample: ExportedHLMemorySample, panel_image: Image.Image) -> Mapping[str, Any]:
        prompt = self.build_prompt(sample)
        return processor(images=panel_image, text=prompt, return_tensors="pt")

    def _encode_prompt_and_target(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        panel_image: Image.Image,
    ) -> Mapping[str, Any]:
        prompt = self.build_prompt(sample)
        target_text = self.build_target_text(sample)
        return processor(images=panel_image, text=f"{prompt}{target_text}", return_tensors="pt")


class Qwen25HLAdapter(BaseHLVLMAdapter):
    def _load_model(self, transformers: Any, pretrained_path: str, *, torch_dtype: torch.dtype) -> Any:
        return transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    def _encode_prompt_only(self, processor: Any, sample: ExportedHLMemorySample, panel_image: Image.Image) -> Mapping[str, Any]:
        rendered = self._render_messages(processor, sample, include_target=False)
        return processor(text=[rendered], images=[panel_image], padding=True, return_tensors="pt")

    def _encode_prompt_and_target(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        panel_image: Image.Image,
    ) -> Mapping[str, Any]:
        rendered = self._render_messages(processor, sample, include_target=True)
        return processor(text=[rendered], images=[panel_image], padding=True, return_tensors="pt")

    def _render_messages(self, processor: Any, sample: ExportedHLMemorySample, *, include_target: bool) -> str:
        user_content = [
            {"type": "image"},
            {"type": "text", "text": self.build_prompt(sample)},
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
        if include_target:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.build_target_text(sample)}],
                }
            )
        tokenizer = getattr(processor, "tokenizer", processor)
        apply_chat_template = getattr(processor, "apply_chat_template", None) or getattr(tokenizer, "apply_chat_template")
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=not include_target)


def _import_transformers() -> Any:
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The HL memory adapters require `transformers` to be installed in the active environment."
        ) from exc
    return transformers
