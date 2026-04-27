from __future__ import annotations

import dataclasses
from typing import Literal


HLVLMBackend = Literal["paligemma", "qwen2_5_vl", "qwen3_5_vl"]
Precision = Literal["bfloat16", "float32"]

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
}

_VARIANT_ALIASES: dict[str, str] = {
    "2b": "qwen3_5_2b",
    "4b": "qwen3_5_4b",
    "qwen35_2b": "qwen3_5_2b",
    "qwen35_4b": "qwen3_5_4b",
}

_SUPPORTED_RUNTIME_BACKENDS = {"qwen2_5_vl", "qwen3_5_vl"}


@dataclasses.dataclass(frozen=True)
class HLMemoryConfig:
    vlm_backend: HLVLMBackend = "qwen2_5_vl"
    vlm_variant: str | None = None
    vlm_hf_model_id: str | None = None
    precision: Precision = "bfloat16"
    recent_frames_length: int = 8
    frame_subsample: int = 5
    memory_length: int = 8
    merge_distance: int = 5
    frame_height: int = 224
    frame_width: int = 224
    allow_single_frame_fallback: bool = True
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if self.vlm_backend not in _DEFAULT_VARIANTS:
            raise ValueError(f"Unsupported HL VLM backend: {self.vlm_backend}")
        resolved_variant = _resolve_variant(self.vlm_backend, self.vlm_variant)
        object.__setattr__(self, "vlm_variant", resolved_variant)
        if self.vlm_hf_model_id is None:
            default_model_id = _default_model_id(self.vlm_backend, resolved_variant)
            object.__setattr__(self, "vlm_hf_model_id", default_model_id)
        if self.recent_frames_length <= 0:
            raise ValueError("recent_frames_length must be positive.")
        if self.frame_subsample <= 0:
            raise ValueError("frame_subsample must be positive.")
        if self.memory_length <= 0:
            raise ValueError("memory_length must be positive.")
        if self.merge_distance < 0:
            raise ValueError("merge_distance must be non-negative.")
        if self.frame_height <= 0 or self.frame_width <= 0:
            raise ValueError("frame_height and frame_width must be positive.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

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
        if normalized not in {"qwen3_5_2b", "qwen3_5_4b"}:
            raise ValueError(
                "`vlm_backend=qwen3_5_vl` supports `vlm_variant=qwen3_5_2b` or `qwen3_5_4b` "
                "(aliases: `2b`, `4b`)."
            )
        return normalized
    raise ValueError(f"Unsupported HL VLM backend: {vlm_backend}")


def _default_model_id(vlm_backend: HLVLMBackend, variant: str | None) -> str | None:
    if vlm_backend == "paligemma":
        return _MODEL_IDS_BY_VARIANT["paligemma"]
    if variant is None:
        return None
    return _MODEL_IDS_BY_VARIANT[variant]
