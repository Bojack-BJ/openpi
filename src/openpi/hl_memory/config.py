from __future__ import annotations

import dataclasses
from typing import Literal


HLVLMBackend = Literal["paligemma", "qwen2_5_vl", "qwen3_5_vl"]
Precision = Literal["bfloat16", "float32"]

_DEFAULT_MODEL_IDS: dict[HLVLMBackend, str | None] = {
    "paligemma": "google/paligemma2-3b-mix-224",
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3_5_vl": None,
}

_SUPPORTED_RUNTIME_BACKENDS = {"qwen2_5_vl"}


@dataclasses.dataclass(frozen=True)
class HLMemoryConfig:
    vlm_backend: HLVLMBackend = "qwen2_5_vl"
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
        if self.vlm_hf_model_id is None:
            default_model_id = _DEFAULT_MODEL_IDS[self.vlm_backend]
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
