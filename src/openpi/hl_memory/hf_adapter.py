from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import LoadedVideoClips
from openpi.hl_memory.schema import HLMemoryPrediction


def create_hf_adapter(config: HLMemoryConfig) -> "BaseHLVLMAdapter":
    if config.vlm_backend == "paligemma":
        raise NotImplementedError(
            "HL memory V1 now expects ordered video-style clips and no longer supports `paligemma`. "
            "Use `qwen2_5_vl`."
        )
    if config.vlm_backend == "qwen2_5_vl":
        return Qwen25HLAdapter(config)
    if config.vlm_backend == "qwen3_5_vl":
        raise NotImplementedError(
            "HL memory V1 does not yet implement `qwen3_5_vl`. Use `qwen2_5_vl` for now."
        )
    raise ValueError(f"Unsupported HL VLM backend: {config.vlm_backend}")


@dataclasses.dataclass(frozen=True)
class LoadedHLVLM:
    processor: Any
    model: Any


@dataclasses.dataclass(frozen=True)
class HLVLMGeneration:
    prediction: HLMemoryPrediction
    raw_output: str


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
        clips: LoadedVideoClips,
        *,
        device: str | torch.device,
    ) -> Mapping[str, torch.Tensor]:
        prompt_inputs = self._encode_prompt_only(loaded.processor, sample, clips)
        full_inputs = self._encode_prompt_and_target(loaded.processor, sample, clips)
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
        clips: LoadedVideoClips,
        *,
        device: str | torch.device,
    ) -> HLMemoryPrediction:
        return self.generate_prediction(loaded, sample, clips, device=device).prediction

    def generate_prediction(
        self,
        loaded: LoadedHLVLM,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        device: str | torch.device,
    ) -> HLVLMGeneration:
        inputs = self._encode_prompt_only(loaded.processor, sample, clips)
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
        prediction = HLMemoryPrediction.from_json(decoded).with_recent_position_limit(clips.recent_valid_length)
        return HLVLMGeneration(prediction=prediction, raw_output=decoded)

    def build_prompt(self, sample: ExportedHLMemorySample, clips: LoadedVideoClips) -> str:
        previous_memory = sample.language_memory.strip() or "No progress has been recorded yet."
        return (
            "You are the high-level planning policy for a robot.\n"
            "You receive two ordered video clips.\n"
            "The first clip is historical memory keyframes.\n"
            "The second clip is the recent observation window.\n"
            f"The historical memory clip has {clips.memory_valid_length} valid frames out of {self.config.memory_length}.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of {self.config.recent_frames_length}.\n"
            "In each clip, positions are ordered from oldest to newest.\n"
            "Update the long-term language memory and predict the current subtask.\n"
            "Return exactly one JSON object with keys "
            "`updated_language_memory`, `current_subtask`, `keyframe_candidate_positions`, `phase`, "
            "`target_query`, `goal_query`.\n"
            "`updated_language_memory` must summarize observed task progress; do not just repeat the task instruction.\n"
            "`current_subtask` must be a short executable robot subtask.\n"
            "`target_query` and `goal_query` must be short grounding phrases or noun phrases, not questions.\n"
            "`keyframe_candidate_positions` must be 1-indexed positions inside the recent observation clip only.\n"
            f"Valid keyframe candidate positions are integers from 1 to {clips.recent_valid_length} inclusive.\n"
            f"Positions {clips.recent_valid_length + 1} to {self.config.recent_frames_length} are padding/fallback frames "
            "and must never be returned.\n"
            "If no valid recent frame is a useful keyframe candidate, return an empty list for "
            "`keyframe_candidate_positions`.\n"
            "Do not include markdown, explanation text, or extra keys outside the JSON object.\n"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
            f"Previous language memory: {previous_memory}\n"
        )

    def build_target_text(self, sample: ExportedHLMemorySample) -> str:
        return sample.target_prediction().to_json()

    def _resolve_torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.precision == "bfloat16" else torch.float32

    def _load_model(self, transformers: Any, pretrained_path: str, *, torch_dtype: torch.dtype) -> Any:
        raise NotImplementedError

    def _encode_prompt_only(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def _encode_prompt_and_target(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        raise NotImplementedError


class Qwen25HLAdapter(BaseHLVLMAdapter):
    def _load_model(self, transformers: Any, pretrained_path: str, *, torch_dtype: torch.dtype) -> Any:
        return transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    def _encode_prompt_only(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        rendered = self._render_messages(processor, sample, clips, include_target=False)
        return processor(
            text=[rendered],
            videos=self._prepare_videos(clips),
            padding=True,
            return_tensors="pt",
        )

    def _encode_prompt_and_target(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        rendered = self._render_messages(processor, sample, clips, include_target=True)
        return processor(
            text=[rendered],
            videos=self._prepare_videos(clips),
            padding=True,
            return_tensors="pt",
        )

    def _render_messages(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        include_target: bool,
    ) -> str:
        user_content = [
            {"type": "text", "text": "Historical memory clip."},
            {"type": "video"},
            {"type": "text", "text": "Recent observation clip."},
            {"type": "video"},
            {"type": "text", "text": self.build_prompt(sample, clips)},
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
        apply_chat_template = getattr(processor, "apply_chat_template", None) or getattr(
            tokenizer, "apply_chat_template"
        )
        return apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_target,
            video_fps=1,
        )

    def _prepare_videos(self, clips: LoadedVideoClips) -> list[list[Any]]:
        return [
            list(clips.memory_frames),
            list(clips.recent_frames),
        ]


def _import_transformers() -> Any:
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The HL memory adapters require `transformers` to be installed in the active environment."
        ) from exc
    return transformers
