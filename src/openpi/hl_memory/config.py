from __future__ import annotations

import dataclasses
from typing import Literal


HLVLMBackend = Literal["paligemma", "qwen2_5_vl", "qwen3_5_vl"]
Precision = Literal["bfloat16", "float16", "float32"]
HLVLMParallelMode = Literal["none", "device_map", "tensor_parallel"]
HLTargetProtocol = Literal["hl_v1", "memer_objective", "subtask_keyframe"]

_DEFAULT_VARIANTS: dict[HLVLMBackend, str | None] = {
    "paligemma": None,
    "qwen2_5_vl": "qwen2_5_3b",
    "qwen3_5_vl": "qwen3_5_2b",
}

_MODEL_IDS_BY_VARIANT: dict[str, str] = {
    "paligemma": "google/paligemma2-3b-mix-224",
    "qwen2_5_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3_5_2b": "Qwen/Qwen3.5-2B",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
    "qwen3_5_27b": "Qwen/Qwen3.5-27B",
}

_VARIANT_ALIASES: dict[str, str] = {
    "2b": "qwen3_5_2b",
    "4b": "qwen3_5_4b",
    "27b": "qwen3_5_27b",
    "qwen35_2b": "qwen3_5_2b",
    "qwen35_4b": "qwen3_5_4b",
    "qwen35_27b": "qwen3_5_27b",
}

_SUPPORTED_RUNTIME_BACKENDS = {"qwen2_5_vl", "qwen3_5_vl"}


@dataclasses.dataclass(frozen=True)
class HLMemoryConfig:
    vlm_backend: HLVLMBackend = "qwen2_5_vl"
    vlm_variant: str | None = None
    vlm_hf_model_id: str | None = None
    precision: Precision = "bfloat16"
    recent_frames_length: int = 8
    training_fps: float = 20.0
    frame_subsample: int = 5
    recent_sample_hz: float = 2.0
    memory_length: int = 8
    merge_distance: int = 5
    frame_height: int = 224
    frame_width: int = 456
    allow_single_frame_fallback: bool = True
    max_new_tokens: int = 256
    enable_thinking: bool = False
    thinking_budget_tokens: int = 128
    thinking_max_new_tokens: int = 1024
    parallel_mode: HLVLMParallelMode = "none"
    device_map: str = "auto"
    tensor_parallel_plan: str = "auto"
    target_protocol: HLTargetProtocol = "hl_v1"

    def __post_init__(self) -> None:
        if self.vlm_backend not in _DEFAULT_VARIANTS:
            raise ValueError(f"Unsupported HL VLM backend: {self.vlm_backend}")
        if self.precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError("`precision` must be one of `bfloat16`, `float16`, or `float32`.")
        if self.parallel_mode not in {"none", "device_map", "tensor_parallel"}:
            raise ValueError("`parallel_mode` must be one of `none`, `device_map`, or `tensor_parallel`.")
        if self.target_protocol not in {"hl_v1", "memer_objective", "subtask_keyframe"}:
            raise ValueError("`target_protocol` must be one of `hl_v1`, `memer_objective`, or `subtask_keyframe`.")
        if not self.device_map.strip():
            raise ValueError("device_map must be non-empty.")
        if not self.tensor_parallel_plan.strip():
            raise ValueError("tensor_parallel_plan must be non-empty.")
        resolved_variant = _resolve_variant(self.vlm_backend, self.vlm_variant)
        object.__setattr__(self, "vlm_variant", resolved_variant)
        if self.vlm_hf_model_id is None:
            default_model_id = _default_model_id(self.vlm_backend, resolved_variant)
            object.__setattr__(self, "vlm_hf_model_id", default_model_id)
        if self.recent_frames_length <= 0:
            raise ValueError("recent_frames_length must be positive.")
        if self.training_fps <= 0.0:
            raise ValueError("training_fps must be positive.")
        if self.frame_subsample <= 0:
            raise ValueError("frame_subsample must be positive.")
        if self.recent_sample_hz <= 0.0:
            raise ValueError("recent_sample_hz must be positive.")
        if self.memory_length <= 0:
            raise ValueError("memory_length must be positive.")
        if self.merge_distance < 0:
            raise ValueError("merge_distance must be non-negative.")
        if self.frame_height <= 0 or self.frame_width <= 0:
            raise ValueError("frame_height and frame_width must be positive.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.thinking_budget_tokens <= 0:
            raise ValueError("thinking_budget_tokens must be positive.")
        if self.thinking_max_new_tokens <= 0:
            raise ValueError("thinking_max_new_tokens must be positive.")

    @property
    def resolved_model_id(self) -> str:
        if self.vlm_hf_model_id is None:
            raise ValueError(
                f"`vlm_backend={self.vlm_backend}` does not have a default HF model id. "
                "Set `vlm_hf_model_id` explicitly."
            )
        return self.vlm_hf_model_id

    @property
    def supports_runtime_backend(self) -> bool:
        return self.vlm_backend in _SUPPORTED_RUNTIME_BACKENDS

    @property
    def video_fps(self) -> float:
        return float(self.recent_sample_hz)

    @property
    def recent_step_sec(self) -> float:
        return 1.0 / float(self.recent_sample_hz)

    @property
    def recent_window_sec(self) -> float:
        return float(max(self.recent_frames_length - 1, 0)) / float(self.recent_sample_hz)


def _resolve_variant(vlm_backend: HLVLMBackend, variant: str | None) -> str | None:
    if variant is None:
        return _DEFAULT_VARIANTS[vlm_backend]

    normalized = _VARIANT_ALIASES.get(variant.lower(), variant.lower())
    if vlm_backend == "paligemma":
        if normalized not in {"paligemma", "none"}:
            raise ValueError(f"`vlm_backend=paligemma` does not support variant `{variant}`.")
        return None
    if vlm_backend == "qwen2_5_vl":
        if normalized != "qwen2_5_3b":
            raise ValueError("`vlm_backend=qwen2_5_vl` currently supports only `vlm_variant=qwen2_5_3b`.")
        return normalized
    if vlm_backend == "qwen3_5_vl":
        if normalized not in {"qwen3_5_2b", "qwen3_5_4b", "qwen3_5_27b"}:
            raise ValueError(
                "`vlm_backend=qwen3_5_vl` supports `vlm_variant=qwen3_5_2b`, `qwen3_5_4b`, "
                "or `qwen3_5_27b` (aliases: `2b`, `4b`, `27b`)."
            )
        return normalized
    raise ValueError(f"Unsupported HL VLM backend: {vlm_backend}")


def _default_model_id(vlm_backend: HLVLMBackend, variant: str | None) -> str | None:
    if vlm_backend == "paligemma":
        return _MODEL_IDS_BY_VARIANT["paligemma"]
    if variant is None:
        return None
    return _MODEL_IDS_BY_VARIANT[variant]
