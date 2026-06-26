from __future__ import annotations

import dataclasses
from typing import Literal


HLVLMBackend = Literal["paligemma", "qwen2_5_vl", "qwen3_5_vl", "qwen3_vl"]
Precision = Literal["bfloat16", "float16", "float32"]
HLVLMParallelMode = Literal["none", "device_map", "tensor_parallel"]
HLTargetProtocol = Literal[
    "hl_v1",
    "memer_objective",
    "subtask_keyframe",
    "known_prior_tracker",
    "objective_memory_state",
    "objective_last_objective",
    "objective_prev_stage",
    "keyframe_gated_memory",
    "keyframe_gated_memory_typed_mask",
    "keyframe_gated_memory_two_pass",
    "memer_film_progress_two_pass",
]
HLProprioTokenMode = Literal["per_frame", "summary", "per_frame_plus_summary"]
HLKeyframeCandidateLabelMode = Literal["canonical", "event_band"]
HLProgressConditionInputMode = Literal["completed_only", "structured", "full"]
HLStateConditionMode = Literal["off", "film", "token", "both"]

_DEFAULT_VARIANTS: dict[HLVLMBackend, str | None] = {
    "paligemma": None,
    "qwen2_5_vl": "qwen2_5_3b",
    "qwen3_5_vl": "qwen3_5_2b",
    "qwen3_vl": "qwen3_vl_4b",
}

_MODEL_IDS_BY_VARIANT: dict[str, str] = {
    "paligemma": "google/paligemma2-3b-mix-224",
    "qwen2_5_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3_5_2b": "Qwen/Qwen3.5-2B",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
    "qwen3_5_27b": "Qwen/Qwen3.5-27B",
    "qwen3_vl_4b": "Qwen/Qwen3-VL-4B-Instruct",
}

_VARIANT_ALIASES: dict[str, str] = {
    "2b": "qwen3_5_2b",
    "4b": "qwen3_5_4b",
    "27b": "qwen3_5_27b",
    "qwen35_2b": "qwen3_5_2b",
    "qwen35_4b": "qwen3_5_4b",
    "qwen35_27b": "qwen3_5_27b",
    "qwen3vl_4b": "qwen3_vl_4b",
    "qwen3-vl-4b": "qwen3_vl_4b",
    "qwen3_vl_4b": "qwen3_vl_4b",
}

_SUPPORTED_RUNTIME_BACKENDS = {"qwen2_5_vl", "qwen3_5_vl", "qwen3_vl"}


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
    proprio_enabled: bool = False
    proprio_token_mode: HLProprioTokenMode = "per_frame_plus_summary"
    proprio_state_dim: int = 14
    proprio_hidden_dim: int = 512
    proprio_dropout: float = 0.0
    proprio_noise_std: float = 0.0
    keyframe_event_band_before_sec: float = 1.0
    keyframe_event_band_after_sec: float = 0.5
    keyframe_candidate_label_mode: HLKeyframeCandidateLabelMode = "event_band"
    two_pass_training_proposal_noise_probability: float = 0.25
    keyframe_aux_enabled: bool = False
    keyframe_aux_hidden_dim: int = 512
    keyframe_aux_timing_sigma_sec: float = 0.5
    progress_condition_enabled: bool = False
    progress_condition_input_mode: HLProgressConditionInputMode = "completed_only"
    progress_condition_dim: int = 128
    progress_condition_hidden_dim: int = 512
    progress_condition_dropout: float = 0.3
    progress_condition_predict_strength: float = 0.5
    progress_condition_horizon_strength: float | None = None
    progress_condition_confirm_strength: float = 1.0
    state_condition_enabled: bool = False
    state_condition_mode: HLStateConditionMode = "film"
    state_condition_dim: int = 128
    state_condition_hidden_dim: int = 512
    state_condition_dropout: float = 0.0
    typed_mask_suppress_language_memory: bool = False

    def __post_init__(self) -> None:
        if self.vlm_backend not in _DEFAULT_VARIANTS:
            raise ValueError(f"Unsupported HL VLM backend: {self.vlm_backend}")
        if self.precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError("`precision` must be one of `bfloat16`, `float16`, or `float32`.")
        if self.parallel_mode not in {"none", "device_map", "tensor_parallel"}:
            raise ValueError("`parallel_mode` must be one of `none`, `device_map`, or `tensor_parallel`.")
        if self.target_protocol not in {
            "hl_v1",
            "memer_objective",
            "subtask_keyframe",
            "known_prior_tracker",
            "objective_memory_state",
            "objective_last_objective",
            "objective_prev_stage",
            "keyframe_gated_memory",
            "keyframe_gated_memory_typed_mask",
            "keyframe_gated_memory_two_pass",
            "memer_film_progress_two_pass",
        }:
            raise ValueError(
                "`target_protocol` must be one of `hl_v1`, `memer_objective`, `subtask_keyframe`, "
                "`known_prior_tracker`, `objective_memory_state`, `objective_last_objective`, or "
                "`objective_prev_stage`, `keyframe_gated_memory`, `keyframe_gated_memory_typed_mask`, or "
                "`keyframe_gated_memory_two_pass`, or `memer_film_progress_two_pass`."
            )
        if self.target_protocol == "keyframe_gated_memory_typed_mask" and self.vlm_backend not in {
            "qwen2_5_vl",
            "qwen3_vl",
        }:
            raise ValueError(
                "`keyframe_gated_memory_typed_mask` is only supported by `qwen2_5_vl` and `qwen3_vl`; "
                "Qwen3.5-VL uses two-pass/staged decoding instead of 4D typed masks."
            )
        if self.proprio_token_mode not in {"per_frame", "summary", "per_frame_plus_summary"}:
            raise ValueError("`proprio_token_mode` must be one of `per_frame`, `summary`, or `per_frame_plus_summary`.")
        if self.keyframe_candidate_label_mode not in {"canonical", "event_band"}:
            raise ValueError("`keyframe_candidate_label_mode` must be one of `canonical` or `event_band`.")
        if self.keyframe_event_band_before_sec < 0.0:
            raise ValueError("keyframe_event_band_before_sec must be non-negative.")
        if self.keyframe_event_band_after_sec < 0.0:
            raise ValueError("keyframe_event_band_after_sec must be non-negative.")
        if not 0.0 <= self.two_pass_training_proposal_noise_probability <= 1.0:
            raise ValueError("two_pass_training_proposal_noise_probability must be in [0, 1].")
        if self.keyframe_aux_hidden_dim <= 0:
            raise ValueError("keyframe_aux_hidden_dim must be positive.")
        if self.keyframe_aux_timing_sigma_sec <= 0.0:
            raise ValueError("keyframe_aux_timing_sigma_sec must be positive.")
        if self.progress_condition_input_mode not in {"completed_only", "structured", "full"}:
            raise ValueError(
                "`progress_condition_input_mode` must be one of `completed_only`, `structured`, or `full`."
            )
        if self.progress_condition_dim <= 0:
            raise ValueError("progress_condition_dim must be positive.")
        if self.progress_condition_hidden_dim <= 0:
            raise ValueError("progress_condition_hidden_dim must be positive.")
        if not 0.0 <= self.progress_condition_dropout <= 1.0:
            raise ValueError("progress_condition_dropout must be in [0, 1].")
        if self.progress_condition_predict_strength < 0.0:
            raise ValueError("progress_condition_predict_strength must be non-negative.")
        if self.progress_condition_horizon_strength is None:
            object.__setattr__(self, "progress_condition_horizon_strength", self.progress_condition_predict_strength)
        elif self.progress_condition_horizon_strength < 0.0:
            raise ValueError("progress_condition_horizon_strength must be non-negative.")
        if self.progress_condition_confirm_strength < 0.0:
            raise ValueError("progress_condition_confirm_strength must be non-negative.")
        if self.state_condition_mode not in {"off", "film", "token", "both"}:
            raise ValueError("`state_condition_mode` must be one of `off`, `film`, `token`, or `both`.")
        if self.state_condition_dim <= 0:
            raise ValueError("state_condition_dim must be positive.")
        if self.state_condition_hidden_dim <= 0:
            raise ValueError("state_condition_hidden_dim must be positive.")
        if not 0.0 <= self.state_condition_dropout <= 1.0:
            raise ValueError("state_condition_dropout must be in [0, 1].")
        if self.proprio_state_dim <= 0:
            raise ValueError("proprio_state_dim must be positive.")
        if self.proprio_hidden_dim <= 0:
            raise ValueError("proprio_hidden_dim must be positive.")
        if not 0.0 <= self.proprio_dropout <= 1.0:
            raise ValueError("proprio_dropout must be in [0, 1].")
        if self.proprio_noise_std < 0.0:
            raise ValueError("proprio_noise_std must be non-negative.")
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

    variant_lower = variant.lower()
    normalized = _VARIANT_ALIASES.get(variant_lower, variant_lower)
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
    if vlm_backend == "qwen3_vl":
        if variant_lower == "4b":
            normalized = "qwen3_vl_4b"
        if normalized not in {"qwen3_vl_4b"}:
            raise ValueError(
                "`vlm_backend=qwen3_vl` currently supports `vlm_variant=qwen3_vl_4b` "
                "(aliases: `4b`, `qwen3vl_4b`, `qwen3-vl-4b`)."
            )
        return normalized
    raise ValueError(f"Unsupported HL VLM backend: {vlm_backend}")


def _default_model_id(vlm_backend: HLVLMBackend, variant: str | None) -> str | None:
    if vlm_backend == "paligemma":
        return _MODEL_IDS_BY_VARIANT["paligemma"]
    if variant is None:
        return None
    return _MODEL_IDS_BY_VARIANT[variant]
