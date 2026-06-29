from __future__ import annotations

import dataclasses
import difflib
import json
import logging
import os
import pathlib
import re
import sys

import numpy as np
from PIL import Image
import torch
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.config_io import resolve_cli_args_with_yaml
from openpi.hl_memory.hf_adapter import create_hf_adapter
from openpi.hl_memory.proprio import PROPRIO_CONFIG_FILENAME
from openpi.hl_memory.schema import HLMemoryPrediction
from openpi.hl_memory.schema import render_language_memory_fields
from openpi.hl_memory.zero_shot import apply_rollout_language_memory_rule
from openpi.hl_memory.zero_shot import build_rollout_end_seconds
from openpi.hl_memory.zero_shot import build_zero_shot_clips_from_video
from openpi.hl_memory.zero_shot import build_zero_shot_clips_from_video_paths
from openpi.hl_memory.zero_shot import build_zero_shot_sample
from openpi.hl_memory.zero_shot import keyframe_candidate_seconds
from openpi.hl_memory.zero_shot import parse_seconds_argument
from openpi.hl_memory.zero_shot import read_video_paths_duration_sec
from openpi.hl_memory.zero_shot import save_keyframe_candidate_frames
from openpi.hl_memory.zero_shot import save_prediction_debug_panel
from openpi.hl_memory.zero_shot import save_zero_shot_debug_frames
from openpi.hl_memory.zero_shot import update_rollout_memory_seconds
from openpi.hl_memory.zero_shot import write_debug_video


_GROUND_TRUTH_ANNOTATION_CACHE: dict[tuple[str, int, str], tuple[tuple[int, str], ...]] = {}
_OBJECT_MASK_INDEX_CACHE: dict[str, tuple[tuple[int, pathlib.Path], ...]] = {}
_KEYFRAME_GATED_PROTOCOLS = {
    "keyframe_gated_memory",
    "keyframe_gated_memory_typed_mask",
    "keyframe_gated_memory_two_pass",
    "memer_film_progress_two_pass",
}
_PROGRESS_UPDATE_PROTOCOLS = {
    "keyframe_gated_memory",
    "keyframe_gated_memory_typed_mask",
    "keyframe_gated_memory_two_pass",
    "memer_film_progress_two_pass",
}


@dataclasses.dataclass(frozen=True)
class AcceptedKeyframeEvent:
    completed_objective: str
    keyframe_second: float


@dataclasses.dataclass(frozen=True)
class KeyframeGatedRolloutState:
    accepted_events: tuple[AcceptedKeyframeEvent, ...] = ()
    task_progress: str = ""

    @property
    def completed_events(self) -> tuple[str, ...]:
        return tuple(event.completed_objective for event in self.accepted_events if event.completed_objective)

    @property
    def accepted_keyframe_seconds(self) -> tuple[float, ...]:
        return tuple(event.keyframe_second for event in self.accepted_events)

    @property
    def completed_event_log(self) -> str:
        if self.task_progress.strip():
            return self.task_progress.strip()
        return _render_completed_event_log(self.completed_events)


@dataclasses.dataclass
class ZeroShotArgs:
    """Run one-shot or interval rollout HL-memory inference on single-view or dual-view videos."""

    # Task and input videos.
    instruction: str
    video_path: pathlib.Path | None = None
    left_video_path: pathlib.Path | None = None
    right_video_path: pathlib.Path | None = None
    session_path: pathlib.Path | None = None
    task_config_path: pathlib.Path | None = None
    ground_truth_annotations_path: pathlib.Path | None = None
    ground_truth_episode_index: int | None = None
    ground_truth_field: str = "current_objective"
    ground_truth_fps: float | None = None
    object_context_enabled: bool = False
    object_name: str = ""
    object_mask_dir: pathlib.Path | None = None
    object_mask_frame_fps: float | None = None

    # Optional YAML config and model checkpoint/source.
    config_yaml: pathlib.Path | None = None
    model_path: str | None = None
    local_vlm_ckpt_path: pathlib.Path | None = None

    # Output artifacts.
    output_json: pathlib.Path | None = None
    rollout_jsonl: pathlib.Path | None = None
    rollout_pretty_json: pathlib.Path | None = None
    debug_dir: pathlib.Path | None = None
    embedding_debug_dir: pathlib.Path | None = None
    embedding_debug_max_tokens: int = 160
    debug_video_fps: float = 1.0

    # Runtime memory and clip selection.
    language_memory: str = ""
    last_objective: str = ""
    previous_stage_objective: str = ""
    memory_input_mode: str = "full"
    memory_seconds: str | None = None
    recent_seconds: str | None = None
    recent_end_sec: float | None = None
    recent_step_sec: float | None = None
    training_fps: float = 20.0
    frame_subsample: int = 5
    recent_sample_hz: float = 2.0

    # If rollout_interval_sec is set, run recurrent HL memory rollout over the whole video range.
    rollout_interval_sec: float | None = None
    rollout_start_sec: float = 0.0
    rollout_end_sec: float | None = None
    keyframe_merge_distance_sec: float = 2.0
    auto_memory: bool = True
    known_prior_mode: bool = False
    known_prior_start_index: int = 0
    known_prior_advance_threshold: float = 0.65
    known_prior_match_threshold: float = 0.62
    known_prior_max_advance_steps: int = 3
    known_prior_next_step_require_completion: bool = False
    known_prior_next_step_confirm_steps: int = 0
    known_prior_safe_skip_mode: bool = False
    known_prior_skip_match_threshold: float = 0.95
    known_prior_skip_min_progress: float = 0.8
    known_prior_skip_min_stall_steps: int = 2

    # VLM backend and generation settings.
    vlm_backend: str = "qwen2_5_vl"
    vlm_variant: str | None = None
    vlm_hf_model_id: str | None = None
    precision: str = "bfloat16"
    recent_frames_length: int = 8
    memory_length: int = 8
    frame_height: int = 224
    frame_width: int = 456
    allow_single_frame_fallback: bool = True
    max_new_tokens: int = 256
    enable_thinking: bool = False
    thinking_budget_tokens: int = 128
    thinking_max_new_tokens: int = 1024

    # Large-model loading options.
    parallel_mode: str = "none"
    device_map: str = "auto"
    tensor_parallel_plan: str = "auto"
    target_protocol: str = "hl_v1"
    typed_mask_suppress_language_memory: bool = False
    proprio_enabled: bool = False
    proprio_token_mode: str = "per_frame_plus_summary"
    proprio_state_dim: int = 14
    proprio_hidden_dim: int = 512
    proprio_dropout: float = 0.0
    proprio_noise_std: float = 0.0
    progress_condition_enabled: bool = False
    progress_condition_input_mode: str = "completed_only"
    progress_condition_dim: int = 128
    progress_condition_hidden_dim: int = 512
    progress_condition_dropout: float = 0.3
    progress_condition_predict_strength: float = 0.5
    progress_condition_horizon_strength: float | None = None
    progress_condition_confirm_strength: float = 1.0
    state_condition_enabled: bool = False
    state_condition_mode: str = "film"
    state_condition_dim: int = 128
    state_condition_hidden_dim: int = 512
    state_condition_dropout: float = 0.0
    proprio_norm_stats_path: pathlib.Path | None = None
    keyframe_event_band_before_sec: float = 1.0
    keyframe_event_band_after_sec: float = 0.5
    keyframe_candidate_label_mode: str = "canonical"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: ZeroShotArgs) -> None:
    is_primary_process = _is_primary_process()
    if not is_primary_process:
        args = dataclasses.replace(
            args,
            output_json=None,
            rollout_jsonl=None,
            rollout_pretty_json=None,
            debug_dir=None,
        )
    args = _resolve_checkpoint_proprio_args(args)
    _resolve_session_paths(args)
    _resolve_video_paths(args)
    if args.task_config_path is not None:
        _load_task_config(args.task_config_path)
    if args.ground_truth_annotations_path is not None:
        _load_ground_truth_annotations(
            args.ground_truth_annotations_path,
            episode_index=args.ground_truth_episode_index,
            field=args.ground_truth_field,
        )
    if args.ground_truth_fps is not None and args.ground_truth_fps <= 0:
        raise ValueError("`--ground-truth-fps` must be positive when set.")
    if args.object_mask_frame_fps is not None and args.object_mask_frame_fps <= 0:
        raise ValueError("`--object-mask-frame-fps` must be positive when set.")
    if args.memory_input_mode not in {"full", "completed_only", "empty"}:
        raise ValueError("`--memory-input-mode` must be one of `full`, `completed_only`, or `empty`.")
    if args.target_protocol == "memer_objective" and args.known_prior_mode:
        raise ValueError(
            "`--target-protocol memer_objective` does not support `--known-prior-mode`; "
            "use `--target-protocol known_prior_tracker` for state-machine rollout."
        )
    if args.proprio_enabled and args.session_path is None:
        raise ValueError("Zero-shot proprio input requires `--session-path` for RGB/trajectory alignment.")
    config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_variant=args.vlm_variant,
        vlm_hf_model_id=args.vlm_hf_model_id,
        precision=args.precision,
        recent_frames_length=args.recent_frames_length,
        training_fps=args.training_fps,
        frame_subsample=args.frame_subsample,
        recent_sample_hz=args.recent_sample_hz,
        memory_length=args.memory_length,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        allow_single_frame_fallback=args.allow_single_frame_fallback,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
        thinking_max_new_tokens=args.thinking_max_new_tokens,
        parallel_mode=args.parallel_mode,
        device_map=args.device_map,
        tensor_parallel_plan=args.tensor_parallel_plan,
        target_protocol=args.target_protocol,
        typed_mask_suppress_language_memory=args.typed_mask_suppress_language_memory,
        proprio_enabled=args.proprio_enabled,
        proprio_token_mode=args.proprio_token_mode,
        proprio_state_dim=args.proprio_state_dim,
        proprio_hidden_dim=args.proprio_hidden_dim,
        proprio_dropout=args.proprio_dropout,
        proprio_noise_std=args.proprio_noise_std,
        progress_condition_enabled=args.progress_condition_enabled,
        progress_condition_input_mode=args.progress_condition_input_mode,
        progress_condition_dim=args.progress_condition_dim,
        progress_condition_hidden_dim=args.progress_condition_hidden_dim,
        progress_condition_dropout=args.progress_condition_dropout,
        progress_condition_predict_strength=args.progress_condition_predict_strength,
        progress_condition_horizon_strength=args.progress_condition_horizon_strength,
        progress_condition_confirm_strength=args.progress_condition_confirm_strength,
        state_condition_enabled=args.state_condition_enabled,
        state_condition_mode=args.state_condition_mode,
        state_condition_dim=args.state_condition_dim,
        state_condition_hidden_dim=args.state_condition_hidden_dim,
        state_condition_dropout=args.state_condition_dropout,
        keyframe_event_band_before_sec=args.keyframe_event_band_before_sec,
        keyframe_event_band_after_sec=args.keyframe_event_band_after_sec,
        keyframe_candidate_label_mode=args.keyframe_candidate_label_mode,
    )
    adapter = create_hf_adapter(config)
    resolved_model_path = args.model_path
    if args.local_vlm_ckpt_path is not None:
        resolved_model_path = str(args.local_vlm_ckpt_path)
    loaded = adapter.load(model_path=resolved_model_path, device=args.device)

    if args.rollout_interval_sec is not None:
        payload = _run_rollout(
            args,
            config=config,
            adapter=adapter,
            loaded=loaded,
            resolved_model_path=resolved_model_path,
        )
    else:
        payload = _run_single_prediction(
            args,
            config=config,
            adapter=adapter,
            loaded=loaded,
            resolved_model_path=resolved_model_path,
        )

    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    if is_primary_process:
        print(rendered)
    if args.output_json is not None:
        _write_json_atomic(args.output_json, payload)


def _resolve_checkpoint_proprio_args(args: ZeroShotArgs) -> ZeroShotArgs:
    checkpoint_dir = _local_checkpoint_dir(args)
    if checkpoint_dir is None:
        return args
    config_path = checkpoint_dir / PROPRIO_CONFIG_FILENAME
    if not config_path.is_file():
        return args
    payload = json.loads(config_path.read_text())
    updates: dict[str, object] = {}
    _maybe_set_checkpoint_arg(
        args,
        updates,
        payload,
        field_name="proprio_enabled",
        cli_names=("--proprio-enabled", "--no-proprio-enabled"),
        cast=bool,
    )
    _maybe_set_checkpoint_arg(
        args,
        updates,
        payload,
        field_name="proprio_token_mode",
        cli_names=("--proprio-token-mode",),
        cast=str,
    )
    _maybe_set_checkpoint_arg(
        args,
        updates,
        payload,
        field_name="proprio_state_dim",
        cli_names=("--proprio-state-dim",),
        cast=int,
    )
    _maybe_set_checkpoint_arg(
        args,
        updates,
        payload,
        field_name="proprio_hidden_dim",
        cli_names=("--proprio-hidden-dim",),
        cast=int,
    )
    _maybe_set_checkpoint_arg(
        args,
        updates,
        payload,
        field_name="proprio_dropout",
        cli_names=("--proprio-dropout",),
        cast=float,
    )
    _maybe_set_checkpoint_arg(
        args,
        updates,
        payload,
        field_name="proprio_noise_std",
        cli_names=("--proprio-noise-std",),
        cast=float,
    )
    if updates:
        logging.info("Loaded proprio runtime config from %s: %s", config_path, updates)
        return dataclasses.replace(args, **updates)
    return args


def _local_checkpoint_dir(args: ZeroShotArgs) -> pathlib.Path | None:
    if args.local_vlm_ckpt_path is not None:
        return args.local_vlm_ckpt_path
    if args.model_path is None:
        return None
    path = pathlib.Path(args.model_path)
    return path if path.exists() else None


def _maybe_set_checkpoint_arg(
    args: ZeroShotArgs,
    updates: dict[str, object],
    payload: dict[str, object],
    *,
    field_name: str,
    cli_names: tuple[str, ...],
    cast,
) -> None:
    if field_name not in payload:
        return
    checkpoint_value = cast(payload[field_name])
    current_value = getattr(args, field_name)
    if _cli_option_was_set(*cli_names):
        if current_value != checkpoint_value:
            names = "/".join(cli_names)
            raise ValueError(
                f"{names}={current_value!r} conflicts with checkpoint {PROPRIO_CONFIG_FILENAME} "
                f"{field_name}={checkpoint_value!r}. Use the checkpoint value or a matching checkpoint."
            )
        return
    if current_value != checkpoint_value:
        updates[field_name] = checkpoint_value


def _cli_option_was_set(*option_names: str) -> bool:
    for arg in sys.argv[1:]:
        for option_name in option_names:
            if arg == option_name or arg.startswith(f"{option_name}="):
                return True
    return False


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _run_single_prediction(
    args: ZeroShotArgs,
    *,
    config: HLMemoryConfig,
    adapter,
    loaded,
    resolved_model_path: str | None,
) -> dict[str, object]:
    clips, selection = _build_clips_from_args(
        args,
        config=config,
        recent_end_sec=args.recent_end_sec,
        recent_step_sec=_resolved_recent_step_sec(args),
        recent_seconds=parse_seconds_argument(args.recent_seconds) or None,
        memory_seconds=parse_seconds_argument(args.memory_seconds) or None,
        auto_memory=args.auto_memory,
    )
    sample = build_zero_shot_sample(
        video_path=_sample_video_path(args),
        instruction=_instruction_with_task_plan(args),
        language_memory=_protocol_language_memory_input(
            config.target_protocol,
            args.language_memory,
            args.memory_input_mode,
        ),
        last_objective=args.last_objective if config.target_protocol == "objective_last_objective" else "",
        previous_stage_objective=(
            args.previous_stage_objective if config.target_protocol == "objective_prev_stage" else ""
        ),
        step_prior=_task_config_steps(args.task_config_path),
        memory_seconds=selection.memory_seconds,
        recent_seconds=selection.recent_seconds,
    )
    sample = _attach_runtime_proprio(sample, args=args, config=config, recent_seconds=selection.recent_seconds)
    sample = _attach_runtime_object_context(sample, args=args, recent_seconds=selection.recent_seconds)
    generation = adapter.generate_prediction(loaded, sample, clips, device=args.device)
    recent_end_sec = selection.recent_seconds[-1] if selection.recent_seconds else 0.0
    prediction, memory_rule_applied = _apply_protocol_runtime_prediction(
        config.target_protocol,
        generation.prediction,
        previous_memory=args.language_memory,
        recent_end_sec=recent_end_sec,
    )
    ground_truth_subtask = _ground_truth_subtask_at(args, recent_end_sec)
    candidate_seconds = keyframe_candidate_seconds(
        prediction,
        selection,
        recent_valid_length=clips.recent_valid_length,
    )

    payload = {
        "video_paths": _payload_video_paths_from_args(args),
        "session_path": None if args.session_path is None else str(args.session_path),
        "model_path": args.model_path,
        "local_vlm_ckpt_path": None if args.local_vlm_ckpt_path is None else str(args.local_vlm_ckpt_path),
        "resolved_model_id": config.resolved_model_id if resolved_model_path is None else resolved_model_path,
        "instruction": args.instruction,
        "task_config_path": None if args.task_config_path is None else str(args.task_config_path),
        "ground_truth_annotations_path": (
            None if args.ground_truth_annotations_path is None else str(args.ground_truth_annotations_path)
        ),
        "ground_truth_episode_index": args.ground_truth_episode_index,
        "ground_truth_field": args.ground_truth_field,
        "ground_truth_fps": _ground_truth_fps(args),
        "input": _protocol_input_payload(
            config.target_protocol,
            sample=sample,
            memory_seconds=selection.memory_seconds,
            recent_seconds=selection.recent_seconds,
            proprio_enabled=config.proprio_enabled,
        ),
        "duration_sec": selection.duration_sec,
        "recent_step_sec": _resolved_recent_step_sec(args),
        "recent_sample_hz": args.recent_sample_hz,
        "recent_window_sec": config.recent_window_sec,
        "training_fps": args.training_fps,
        "frame_subsample": args.frame_subsample,
        "video_fps": config.video_fps,
        "target_protocol": config.target_protocol,
        "memory_seconds": list(selection.memory_seconds),
        "recent_seconds": list(selection.recent_seconds),
        "memory_valid_length": clips.memory_valid_length,
        "recent_valid_length": clips.recent_valid_length,
        "proprio_enabled": config.proprio_enabled,
        "proprio_norm_stats_path": _payload_proprio_norm_stats_path(args),
        "object_context_enabled": args.object_context_enabled,
        "object_name": sample.object_name,
        "recent_object_center_points": [list(point) for point in sample.recent_object_center_points],
        "output": _protocol_prediction_payload(config.target_protocol, prediction),
        "ground_truth_subtask": ground_truth_subtask,
        "diagnostics": {
            "raw_model_output": generation.raw_output,
            "parse_error": generation.parse_error,
            "language_memory_rule_applied": memory_rule_applied,
            "keyframe_candidate_seconds": list(candidate_seconds),
        },
    }

    if args.debug_dir is not None:
        debug_dir = pathlib.Path(args.debug_dir)
        save_zero_shot_debug_frames(debug_dir, clips=clips, selection=selection)
        saved_keyframes = save_keyframe_candidate_frames(
            debug_dir / "keyframe_candidates",
            clips=clips,
            selection=selection,
            positions=prediction.keyframe_candidate_positions,
        )
        debug_panel_path = save_prediction_debug_panel(
            debug_dir / "debug_panel.png",
            clips=clips,
            selection=selection,
            prediction=prediction,
            recent_end_sec=recent_end_sec,
            target_protocol=config.target_protocol,
            instruction=sample.instruction,
            state_input=_protocol_state_input(config.target_protocol, sample),
            language_memory_before=sample.language_memory,
            language_memory_after=prediction.updated_language_memory,
            memory_seconds_before=selection.memory_seconds,
            memory_seconds_after=selection.memory_seconds,
            keyframe_candidate_seconds=candidate_seconds,
            ground_truth_subtask=ground_truth_subtask,
            parse_error=generation.parse_error,
        )
        payload["debug_dir"] = str(args.debug_dir)
        payload["saved_keyframe_candidate_paths"] = [str(path) for path in saved_keyframes]
        payload["debug_panel_path"] = str(debug_panel_path)
    if args.embedding_debug_dir is not None:
        embedding_debug_dir = pathlib.Path(args.embedding_debug_dir)
        embedding_payload = _save_embedding_debug(
            embedding_debug_dir,
            adapter=adapter,
            loaded=loaded,
            sample=sample,
            clips=clips,
            device=args.device,
            max_tokens=args.embedding_debug_max_tokens,
        )
        payload["embedding_debug_dir"] = str(embedding_debug_dir)
        payload["embedding_debug"] = embedding_payload

    return payload


def _run_rollout(
    args: ZeroShotArgs,
    *,
    config: HLMemoryConfig,
    adapter,
    loaded,
    resolved_model_path: str | None,
) -> dict[str, object]:
    if args.recent_seconds is not None:
        raise ValueError("`--recent-seconds` is not supported with interval rollout; use `--recent-step-sec` instead.")

    video_paths = _resolve_video_paths(args)
    duration_sec = read_video_paths_duration_sec(video_paths)
    rollout_seconds = build_rollout_end_seconds(
        duration_sec,
        interval_sec=float(args.rollout_interval_sec),
        start_sec=args.rollout_start_sec,
        end_sec=args.rollout_end_sec,
    )
    language_memory = args.language_memory
    recent_step_sec = _resolved_recent_step_sec(args)
    memory_seconds = tuple(parse_seconds_argument(args.memory_seconds))
    keyframe_vote_seconds = list(memory_seconds)
    parsed_completed_events = _parse_completed_event_log(args.language_memory)
    paired_initial_events = tuple(
        AcceptedKeyframeEvent(
            completed_objective=parsed_completed_events[index] if index < len(parsed_completed_events) else "",
            keyframe_second=second,
        )
        for index, second in enumerate(memory_seconds)
    )
    if len(parsed_completed_events) > len(memory_seconds):
        logging.warning(
            "Ignoring initial completed events without paired keyframe seconds: completed_events=%d memory_seconds=%d.",
            len(parsed_completed_events),
            len(memory_seconds),
        )
    keyframe_gated_state = KeyframeGatedRolloutState(
        accepted_events=paired_initial_events,
        task_progress=args.language_memory.strip() if parsed_completed_events else "",
    )
    known_prior_steps = _known_prior_steps(args)
    known_prior_index = _initial_known_prior_index(args, known_prior_steps)
    known_prior_stall_steps = 0
    known_prior_next_step_confirmation_index: int | None = None
    known_prior_next_step_confirmation_steps = 0
    last_objective = args.last_objective.strip()
    previous_stage_objective = args.previous_stage_objective.strip()
    current_stage_objective = last_objective
    if known_prior_steps and not language_memory.strip():
        language_memory = _known_prior_language_memory(
            known_prior_steps,
            known_prior_index,
            task_progress="No completed subtask yet.",
            notes="Known-prior rollout initialized.",
        )
    steps: list[dict[str, object]] = []
    debug_dir = pathlib.Path(args.debug_dir) if args.debug_dir is not None else None
    debug_panel_paths: list[pathlib.Path] = []

    for step_index, end_sec in enumerate(rollout_seconds):
        memory_before = memory_seconds
        known_prior_index_before = known_prior_index
        known_prior_match: dict[str, object] | None = None
        language_memory_before = language_memory
        language_memory_input = _protocol_language_memory_input(
            config.target_protocol,
            keyframe_gated_state.completed_event_log
            if config.target_protocol in _KEYFRAME_GATED_PROTOCOLS
            else language_memory_before,
            args.memory_input_mode,
        )
        clips, selection = _build_clips_from_args(
            args,
            config=config,
            recent_end_sec=end_sec,
            recent_step_sec=recent_step_sec,
            memory_seconds=memory_before,
            auto_memory=False,
        )
        sample = build_zero_shot_sample(
            video_path=_sample_video_path(args),
            instruction=_instruction_with_task_plan(
                args,
                known_prior_steps=known_prior_steps,
                known_prior_index=known_prior_index if known_prior_steps else None,
            ),
            language_memory=language_memory_input,
            last_objective=last_objective if config.target_protocol == "objective_last_objective" else "",
            previous_stage_objective=(
                previous_stage_objective if config.target_protocol == "objective_prev_stage" else ""
            ),
            step_prior=known_prior_steps or _task_config_steps(args.task_config_path),
            memory_seconds=selection.memory_seconds,
            recent_seconds=selection.recent_seconds,
        )
        sample = _attach_runtime_proprio(sample, args=args, config=config, recent_seconds=selection.recent_seconds)
        sample = _attach_runtime_object_context(sample, args=args, recent_seconds=selection.recent_seconds)
        generation = adapter.generate_prediction(loaded, sample, clips, device=args.device)
        prediction, memory_rule_applied = _apply_protocol_runtime_prediction(
            config.target_protocol,
            generation.prediction,
            previous_memory=language_memory_before,
            recent_end_sec=end_sec,
        )
        if known_prior_steps:
            prediction, known_prior_index, known_prior_match = _apply_known_prior_rollout_state(
                prediction,
                known_prior_steps=known_prior_steps,
                current_index=known_prior_index,
                advance_threshold=args.known_prior_advance_threshold,
                match_threshold=args.known_prior_match_threshold,
                max_advance_steps=args.known_prior_max_advance_steps,
                next_step_require_completion=args.known_prior_next_step_require_completion,
                next_step_confirm_steps=args.known_prior_next_step_confirm_steps,
                next_step_confirmation_index=known_prior_next_step_confirmation_index,
                next_step_confirmation_steps=known_prior_next_step_confirmation_steps,
                safe_skip_mode=args.known_prior_safe_skip_mode,
                skip_match_threshold=args.known_prior_skip_match_threshold,
                skip_min_progress=args.known_prior_skip_min_progress,
                skip_min_stall_steps=args.known_prior_skip_min_stall_steps,
                stall_steps=known_prior_stall_steps,
            )
            if known_prior_index != known_prior_index_before:
                known_prior_stall_steps = 0
                known_prior_next_step_confirmation_index = None
                known_prior_next_step_confirmation_steps = 0
            else:
                known_prior_stall_steps += 1
                known_prior_next_step_confirmation_index = known_prior_match["next_step_confirmation_index"]
                known_prior_next_step_confirmation_steps = int(known_prior_match["next_step_confirmation_steps"])
        predicted_objective = prediction.current_objective.strip()
        predicted_previous_stage_objective = prediction.previous_stage_objective.strip()
        if predicted_objective:
            if predicted_previous_stage_objective:
                previous_stage_objective = predicted_previous_stage_objective
                if _normalize_text(predicted_objective) != _normalize_text(current_stage_objective):
                    current_stage_objective = predicted_objective
            elif current_stage_objective and _normalize_text(predicted_objective) != _normalize_text(current_stage_objective):
                previous_stage_objective = current_stage_objective
                current_stage_objective = predicted_objective
            elif not current_stage_objective:
                current_stage_objective = predicted_objective
        ground_truth_subtask = _ground_truth_subtask_at(args, end_sec)
        candidate_seconds = keyframe_candidate_seconds(
            prediction,
            selection,
            recent_valid_length=clips.recent_valid_length,
        )
        keyframe_gated_update: dict[str, object] | None = None
        if config.target_protocol in _KEYFRAME_GATED_PROTOCOLS:
            keyframe_gated_state, keyframe_gated_update = _update_keyframe_gated_rollout_state(
                keyframe_gated_state,
                prediction=prediction,
                candidate_seconds=candidate_seconds,
                memory_length=config.memory_length,
                merge_distance_sec=args.keyframe_merge_distance_sec,
            )
            memory_seconds = keyframe_gated_state.accepted_keyframe_seconds
            language_memory = keyframe_gated_state.completed_event_log
        else:
            keyframe_vote_seconds.extend(candidate_seconds)
            memory_seconds = update_rollout_memory_seconds(
                keyframe_vote_seconds,
                (),
                memory_length=config.memory_length,
                merge_distance_sec=args.keyframe_merge_distance_sec,
            )
        if config.target_protocol in {"hl_v1", "known_prior_tracker", "objective_memory_state"}:
            language_memory = prediction.updated_language_memory

        saved_keyframes: list[str] = []
        debug_panel_path: pathlib.Path | None = None
        step_debug_dir: pathlib.Path | None = None
        if debug_dir is not None:
            step_debug_dir = debug_dir / f"rollout_step_{step_index:03d}"
            save_zero_shot_debug_frames(step_debug_dir, clips=clips, selection=selection)
            saved_keyframes = [
                str(path)
                for path in save_keyframe_candidate_frames(
                    step_debug_dir / "keyframe_candidates",
                    clips=clips,
                    selection=selection,
                    positions=prediction.keyframe_candidate_positions,
                )
            ]
            debug_panel_path = save_prediction_debug_panel(
                step_debug_dir / "debug_panel.png",
                clips=clips,
                selection=selection,
                prediction=prediction,
                step_index=step_index,
                recent_end_sec=end_sec,
                target_protocol=config.target_protocol,
                instruction=sample.instruction,
                state_input=_protocol_state_input(config.target_protocol, sample),
                language_memory_before=language_memory_input,
                language_memory_after=language_memory,
                memory_seconds_before=memory_before,
                memory_seconds_after=memory_seconds,
                keyframe_candidate_seconds=candidate_seconds,
                state_update=_format_keyframe_gated_state_update(keyframe_gated_update),
                ground_truth_subtask=ground_truth_subtask,
                parse_error=generation.parse_error,
            )
            debug_panel_paths.append(debug_panel_path)
        embedding_debug_payload = None
        if args.embedding_debug_dir is not None:
            step_embedding_dir = pathlib.Path(args.embedding_debug_dir) / f"rollout_step_{step_index:03d}"
            embedding_debug_payload = _save_embedding_debug(
                step_embedding_dir,
                adapter=adapter,
                loaded=loaded,
                sample=sample,
                clips=clips,
                device=args.device,
                max_tokens=args.embedding_debug_max_tokens,
            )

        steps.append(
            {
                "step_index": step_index,
                "recent_end_sec": end_sec,
                "input": _protocol_input_payload(
                    config.target_protocol,
                    sample=sample,
                    memory_seconds=memory_before,
                    recent_seconds=selection.recent_seconds,
                    proprio_enabled=config.proprio_enabled,
                ),
                "output": _protocol_prediction_payload(config.target_protocol, prediction),
                "ground_truth_subtask": ground_truth_subtask,
                "diagnostics": {
                    "raw_model_output": generation.raw_output,
                    "parse_error": generation.parse_error,
                    "keyframe_candidate_seconds": list(candidate_seconds),
                    **({"state_update": keyframe_gated_update} if keyframe_gated_update is not None else {}),
                    "debug_dir": None if step_debug_dir is None else str(step_debug_dir),
                    "debug_panel_path": None if debug_panel_path is None else str(debug_panel_path),
                    "saved_keyframe_candidate_paths": saved_keyframes,
                },
                **({"known_prior_match": known_prior_match} if known_prior_steps else {}),
                **({"embedding_debug": embedding_debug_payload} if embedding_debug_payload is not None else {}),
            }
        )
        if predicted_objective:
            last_objective = predicted_objective
        _write_live_rollout_summary(
            args.output_json,
            _build_rollout_payload(
                args,
                config=config,
                resolved_model_path=resolved_model_path,
                duration_sec=duration_sec,
                recent_step_sec=recent_step_sec,
                known_prior_steps=known_prior_steps,
                known_prior_index=known_prior_index,
                language_memory=language_memory,
                memory_seconds=memory_seconds,
                steps=steps,
                debug_dir=debug_dir,
            ),
        )

    debug_video_path: pathlib.Path | None = None
    debug_video_error: str | None = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(debug_dir / "rollout.jsonl", steps)
        _write_pretty_json(debug_dir / "rollout_pretty.json", steps)
        try:
            debug_video_path = write_debug_video(
                debug_panel_paths,
                debug_dir / "rollout_debug.mp4",
                fps=args.debug_video_fps,
            )
        except (ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
            debug_video_error = f"{type(exc).__name__}: {exc}"
    if args.rollout_jsonl is not None:
        _write_jsonl(args.rollout_jsonl, steps)
    if args.rollout_pretty_json is not None:
        _write_pretty_json(args.rollout_pretty_json, steps)

    return _build_rollout_payload(
        args,
        config=config,
        resolved_model_path=resolved_model_path,
        duration_sec=duration_sec,
        recent_step_sec=recent_step_sec,
        known_prior_steps=known_prior_steps,
        known_prior_index=known_prior_index,
        language_memory=language_memory,
        memory_seconds=memory_seconds,
        steps=steps,
        debug_dir=debug_dir,
        debug_video_path=debug_video_path,
        debug_video_error=debug_video_error,
    )


def _build_rollout_payload(
    args: ZeroShotArgs,
    *,
    config: HLMemoryConfig,
    resolved_model_path: str | None,
    duration_sec: float,
    recent_step_sec: float,
    known_prior_steps: tuple[str, ...],
    known_prior_index: int,
    language_memory: str,
    memory_seconds: tuple[float, ...],
    steps: list[dict[str, object]],
    debug_dir: pathlib.Path | None,
    debug_video_path: pathlib.Path | None = None,
    debug_video_error: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "video_paths": _payload_video_paths_from_args(args),
        "session_path": None if args.session_path is None else str(args.session_path),
        "model_path": args.model_path,
        "local_vlm_ckpt_path": None if args.local_vlm_ckpt_path is None else str(args.local_vlm_ckpt_path),
        "resolved_model_id": config.resolved_model_id if resolved_model_path is None else resolved_model_path,
        "instruction": args.instruction,
        "task_config_path": None if args.task_config_path is None else str(args.task_config_path),
        "ground_truth_annotations_path": (
            None if args.ground_truth_annotations_path is None else str(args.ground_truth_annotations_path)
        ),
        "ground_truth_episode_index": args.ground_truth_episode_index,
        "ground_truth_field": args.ground_truth_field,
        "ground_truth_fps": _ground_truth_fps(args),
        "initial_language_memory": args.language_memory,
        "duration_sec": duration_sec,
        "rollout_interval_sec": args.rollout_interval_sec,
        "rollout_start_sec": args.rollout_start_sec,
        "rollout_end_sec": args.rollout_end_sec,
        "recent_step_sec": recent_step_sec,
        "recent_sample_hz": args.recent_sample_hz,
        "recent_window_sec": config.recent_window_sec,
        "training_fps": args.training_fps,
        "frame_subsample": args.frame_subsample,
        "video_fps": config.video_fps,
        "target_protocol": config.target_protocol,
        "proprio_enabled": config.proprio_enabled,
        "keyframe_merge_distance_sec": args.keyframe_merge_distance_sec,
        "final_memory_seconds": list(memory_seconds),
        "debug_dir": None if debug_dir is None else str(debug_dir),
        "debug_video_path": None if debug_video_path is None else str(debug_video_path),
        "debug_video_error": debug_video_error,
        "steps": steps,
    }
    if config.proprio_enabled:
        payload["proprio_norm_stats_path"] = _payload_proprio_norm_stats_path(args)
    if args.object_context_enabled:
        payload["object_context"] = {
            "object_name": args.object_name,
            "object_mask_dir": None if args.object_mask_dir is None else str(args.object_mask_dir),
            "object_mask_frame_fps": _object_mask_frame_fps(args),
        }
    if known_prior_steps:
        payload["known_prior"] = {
            "steps": list(known_prior_steps),
            "final_index": known_prior_index,
            "advance_threshold": args.known_prior_advance_threshold,
            "match_threshold": args.known_prior_match_threshold,
            "max_advance_steps": args.known_prior_max_advance_steps,
        }
    final_state = _protocol_final_state(
        config.target_protocol,
        language_memory=language_memory,
        previous_stage_objective=_last_step_output_value(steps, "previous_stage_objective"),
    )
    if final_state:
        payload["final_state"] = final_state
    return payload


def _protocol_language_memory_input(protocol: str, memory: str, memory_input_mode: str) -> str:
    if protocol in _KEYFRAME_GATED_PROTOCOLS:
        return memory.strip()
    if protocol not in {
        "hl_v1",
        "known_prior_tracker",
        "subtask_keyframe",
        "objective_memory_state",
    }:
        return ""
    return _language_memory_for_model(memory, memory_input_mode)


def _apply_protocol_runtime_prediction(
    protocol: str,
    prediction,
    *,
    previous_memory: str,
    recent_end_sec: float,
):
    if protocol in {"hl_v1", "known_prior_tracker"}:
        return apply_rollout_language_memory_rule(
            prediction,
            previous_memory=previous_memory,
            recent_end_sec=recent_end_sec,
        )
    return prediction, False


def _protocol_state_input(protocol: str, sample) -> str:
    if protocol == "objective_memory_state":
        return sample.language_memory
    if protocol == "objective_prev_stage":
        return sample.previous_stage_objective
    if protocol in _KEYFRAME_GATED_PROTOCOLS:
        return sample.language_memory
    return ""


def _protocol_input_payload(
    protocol: str,
    *,
    sample,
    memory_seconds,
    recent_seconds,
    proprio_enabled: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "instruction": sample.instruction,
        "historical_keyframe_seconds": list(memory_seconds),
        "recent_frame_seconds": list(recent_seconds),
    }
    if protocol == "objective_memory_state":
        payload["completed_subtasks_memory"] = sample.language_memory
    elif protocol == "objective_prev_stage":
        payload["previous_stage_objective"] = sample.previous_stage_objective
    elif protocol == "memer_film_progress_two_pass":
        payload["progress_condition_input"] = sample.language_memory
    elif protocol in {"keyframe_gated_memory", "keyframe_gated_memory_typed_mask"}:
        payload["task_progress_input"] = sample.language_memory
    elif protocol in _KEYFRAME_GATED_PROTOCOLS:
        payload["completed_event_log"] = sample.language_memory
    if proprio_enabled:
        payload["recent_robot_states"] = [list(row) for row in sample.recent_robot_states]
        payload["recent_robot_state_masks"] = [list(row) for row in sample.recent_robot_state_masks]
    if sample.object_name or sample.recent_object_center_points:
        payload["object_context"] = {
            "object_name": sample.object_name,
            "recent_center_points": [list(point) for point in sample.recent_object_center_points],
        }
    return payload


def _protocol_prediction_payload(protocol: str, prediction) -> dict[str, object]:
    if protocol in _PROGRESS_UPDATE_PROTOCOLS:
        return prediction.to_runtime_schema_dict()

    payload: dict[str, object] = {
        "current_objective": prediction.current_objective,
        "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
    }
    if protocol in {
        "memer_objective",
        "memer_objective_grounding",
        "objective_memory_state",
        "objective_last_objective",
        "objective_prev_stage",
        "keyframe_gated_memory",
    }:
        payload["horizon_current_objective"] = prediction.horizon_current_objective
    if protocol == "memer_objective_grounding":
        payload["target_object"] = prediction.target_object
        if prediction.target_slot:
            payload["target_slot"] = prediction.target_slot
    elif protocol in {"keyframe_gated_memory_typed_mask", "keyframe_gated_memory_two_pass", "memer_film_progress_two_pass"}:
        payload["horizon_current_objective"] = prediction.horizon_current_objective
    if protocol == "objective_memory_state":
        payload["updated_language_memory"] = prediction.updated_language_memory
    elif protocol == "objective_last_objective":
        payload["last_objective"] = prediction.last_objective
    elif protocol == "objective_prev_stage":
        payload["previous_stage_objective"] = prediction.previous_stage_objective
    elif protocol in _KEYFRAME_GATED_PROTOCOLS:
        payload["new_completed_objective"] = prediction.new_completed_objective
        payload["task_progress"] = prediction.task_progress
    elif protocol == "known_prior_tracker":
        payload["subtask_progress"] = prediction.subtask_progress
        payload["should_advance_objective"] = prediction.should_advance_objective
    elif protocol == "hl_v1":
        return prediction.to_dict(include_legacy=False)
    return payload


def _protocol_final_state(
    protocol: str,
    *,
    language_memory: str,
    previous_stage_objective: str,
) -> dict[str, str]:
    if protocol == "objective_memory_state":
        return {"completed_subtasks_memory": language_memory}
    if protocol in _KEYFRAME_GATED_PROTOCOLS:
        return {"completed_event_log": language_memory}
    if protocol == "objective_prev_stage":
        return {"previous_stage_objective": previous_stage_objective}
    return {}


def _update_keyframe_gated_rollout_state(
    state: KeyframeGatedRolloutState,
    *,
    prediction: HLMemoryPrediction,
    candidate_seconds: tuple[float, ...] | list[float],
    memory_length: int,
    merge_distance_sec: float,
) -> tuple[KeyframeGatedRolloutState, dict[str, object]]:
    completed_objective = prediction.completed_objective.strip()
    task_progress = prediction.task_progress.strip()
    candidate_seconds = tuple(float(second) for second in candidate_seconds)
    if not candidate_seconds:
        return state, {
            "accepted": False,
            "reason": "no_keyframe_candidates",
            "completed_objective": completed_objective,
            "task_progress_after": state.completed_event_log,
            "candidate_seconds": [],
            "completed_events_after": list(state.completed_events),
            "accepted_keyframe_seconds_after": list(state.accepted_keyframe_seconds),
        }

    # Event-band training can produce several candidate frames for one transition.
    # Keep one compact representative so long-term memory does not grow with band width.
    representative_second = max(candidate_seconds)
    normalized_completed = _normalize_text(completed_objective)
    duplicate_seconds = [
        event.keyframe_second
        for event in state.accepted_events
        if abs(representative_second - event.keyframe_second) < merge_distance_sec
        and (
            not normalized_completed
            or _normalize_text(event.completed_objective) == normalized_completed
        )
    ]
    if duplicate_seconds:
        reason = (
            "duplicate_completed_objective_near_existing_keyframe"
            if normalized_completed
            else "duplicate_keyframe_near_existing_keyframe"
        )
        return state, {
            "accepted": False,
            "reason": reason,
            "completed_objective": completed_objective,
            "task_progress_after": state.completed_event_log,
            "candidate_seconds": list(candidate_seconds),
            "representative_keyframe_second": representative_second,
            "completed_events_after": list(state.completed_events),
            "accepted_keyframe_seconds_after": list(state.accepted_keyframe_seconds),
        }

    next_task_progress = _merge_task_progress(state.task_progress, task_progress, completed_objective)
    accepted_events = (
        *state.accepted_events,
        AcceptedKeyframeEvent(
            completed_objective=completed_objective,
            keyframe_second=representative_second,
        ),
    )
    accepted_events = accepted_events[-memory_length:] if memory_length > 0 else ()
    new_state = KeyframeGatedRolloutState(accepted_events=accepted_events, task_progress=next_task_progress)
    reason = (
        "accepted_completed_objective_with_keyframes"
        if completed_objective
        else "accepted_keyframe_candidate_without_completed_objective"
    )
    return new_state, {
        "accepted": True,
        "reason": reason,
        "completed_objective": completed_objective,
        "task_progress_after": new_state.completed_event_log,
        "candidate_seconds": list(candidate_seconds),
        "representative_keyframe_second": representative_second,
        "completed_events_after": list(new_state.completed_events),
        "accepted_keyframe_seconds_after": list(new_state.accepted_keyframe_seconds),
    }


def _parse_completed_event_log(memory: str) -> tuple[str, ...]:
    events: list[str] = []
    for raw_line in memory.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("completed events:"):
            line = line.split(":", 1)[1].strip()
            for item in line.split(";"):
                event = item.strip(" .")
                if event:
                    events.append(event)
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if ":" in line:
            _, line = line.split(":", 1)
            line = line.strip()
        if line and line.lower() not in {"none", "no accepted completed event yet"}:
            events.append(line.strip(" ."))
    return tuple(event for event in events if event)


def _merge_task_progress(previous: str, predicted: str, new_completed_objective: str = "") -> str:
    previous = previous.strip()
    predicted = predicted.strip()
    new_completed_objective = new_completed_objective.strip()
    if not previous:
        return predicted or _render_completed_event_log((new_completed_objective,) if new_completed_objective else ())
    if not predicted:
        return previous
    normalized_previous = _normalize_text(previous)
    normalized_predicted = _normalize_text(predicted)
    if normalized_previous and normalized_previous in normalized_predicted:
        return predicted
    normalized_completed = _normalize_text(new_completed_objective)
    if normalized_completed and normalized_completed in normalized_previous:
        return previous
    return f"{previous.rstrip(' .')}; {predicted.strip(' .')}."


def _render_completed_event_log(events: tuple[str, ...] | list[str], *, max_events: int = 8) -> str:
    if not events:
        return "No accepted completed event yet."
    compact_events = [str(event).strip() for event in events[-max_events:] if str(event).strip()]
    if not compact_events:
        return "No accepted completed event yet."
    prefix = "Recent completed events"
    if len(events) > len(compact_events):
        prefix += f" (last {len(compact_events)} of {len(events)})"
    return prefix + ": " + "; ".join(compact_events) + "."


def _format_keyframe_gated_state_update(update: dict[str, object] | None) -> str:
    if update is None:
        return ""
    accepted = "accepted" if update.get("accepted") else "rejected"
    reason = str(update.get("reason", "unknown"))
    completed_objective = str(update.get("completed_objective", "")).strip() or "none"
    events_after = update.get("completed_events_after", [])
    event_count = len(events_after) if isinstance(events_after, list) else 0
    return f"{accepted}; reason={reason}; completed_objective={completed_objective}; events_after={event_count}"


def _last_step_output_value(steps: list[dict[str, object]], field: str) -> str:
    if not steps:
        return ""
    output = steps[-1].get("output")
    if not isinstance(output, dict):
        return ""
    return str(output.get(field, "")).strip()


def _build_clips_from_args(
    args: ZeroShotArgs,
    *,
    config: HLMemoryConfig,
    recent_end_sec: float | None,
    recent_step_sec: float,
    recent_seconds: list[float] | None = None,
    memory_seconds: tuple[float, ...] | list[float] | None = None,
    auto_memory: bool,
):
    if args.video_path is not None:
        return build_zero_shot_clips_from_video(
            args.video_path,
            config=config,
            recent_end_sec=recent_end_sec,
            recent_step_sec=recent_step_sec,
            recent_seconds=recent_seconds,
            memory_seconds=memory_seconds,
            auto_memory=auto_memory,
        )
    return build_zero_shot_clips_from_video_paths(
        _resolve_video_paths(args),
        config=config,
        recent_end_sec=recent_end_sec,
        recent_step_sec=recent_step_sec,
        recent_seconds=recent_seconds,
        memory_seconds=memory_seconds,
        auto_memory=auto_memory,
    )


def _resolved_recent_step_sec(args: ZeroShotArgs) -> float:
    if args.recent_step_sec is not None:
        if args.recent_step_sec <= 0.0:
            raise ValueError("--recent-step-sec must be positive.")
        return float(args.recent_step_sec)
    if args.training_fps <= 0.0:
        raise ValueError("--training-fps must be positive when --recent-step-sec is omitted.")
    if args.frame_subsample <= 0:
        raise ValueError("--frame-subsample must be positive when --recent-step-sec is omitted.")
    if args.recent_sample_hz <= 0.0:
        raise ValueError("--recent-sample-hz must be positive when --recent-step-sec is omitted.")
    return 1.0 / float(args.recent_sample_hz)


def _resolve_session_paths(args: ZeroShotArgs) -> None:
    if args.session_path is None:
        return
    session_path = args.session_path
    if not session_path.is_dir():
        raise FileNotFoundError(f"Session path does not exist: {session_path}")

    if args.task_config_path is None:
        task_config_path = session_path / "subtask.json"
        if task_config_path.is_file():
            args.task_config_path = task_config_path

    if args.video_path is not None or args.left_video_path is not None or args.right_video_path is not None:
        return

    hand_dirs = _session_hand_dirs(session_path)
    left_dir = next((path for path in hand_dirs if path.name.startswith("left_hand_")), None)
    right_dir = next((path for path in hand_dirs if path.name.startswith("right_hand_")), None)
    if left_dir is not None and right_dir is not None:
        args.left_video_path = _required_session_file(left_dir / "RGB_Images" / "video.mp4")
        args.right_video_path = _required_session_file(right_dir / "RGB_Images" / "video.mp4")
        return
    if len(hand_dirs) == 1:
        args.video_path = _required_session_file(hand_dirs[0] / "RGB_Images" / "video.mp4")
        return

    root_video_path = session_path / "RGB_Images" / "video.mp4"
    if root_video_path.is_file():
        args.video_path = root_video_path
        return
    raise ValueError(
        f"Could not resolve video input from session path {session_path}. Expected left/right hand subdirs "
        "or root RGB_Images/video.mp4."
    )


def _required_session_file(path: pathlib.Path) -> pathlib.Path:
    if not path.is_file():
        raise FileNotFoundError(f"Expected session file does not exist: {path}")
    return path


def _session_hand_dirs(session_path: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in session_path.iterdir()
        if path.is_dir() and (path.name.startswith("left_hand_") or path.name.startswith("right_hand_"))
    )


def _resolve_video_paths(args: ZeroShotArgs) -> dict[str, pathlib.Path]:
    has_single_video = args.video_path is not None
    has_left_video = args.left_video_path is not None
    has_right_video = args.right_video_path is not None

    if has_single_video:
        if has_left_video or has_right_video:
            raise ValueError("Use either `--video-path` or `--left-video-path` + `--right-video-path`, not both.")
        assert args.video_path is not None
        return {"front": args.video_path}

    if has_left_video != has_right_video:
        raise ValueError("Dual-view mode requires both `--left-video-path` and `--right-video-path`.")

    if not has_left_video:
        raise ValueError("Provide either `--video-path` or both `--left-video-path` and `--right-video-path`.")

    assert args.left_video_path is not None
    assert args.right_video_path is not None
    return {
        "robot_0": args.left_video_path,
        "robot_1": args.right_video_path,
    }


def _payload_video_paths_from_args(args: ZeroShotArgs) -> dict[str, str]:
    return {view_name: str(path) for view_name, path in _resolve_video_paths(args).items()}


def _attach_runtime_object_context(
    sample,
    *,
    args: ZeroShotArgs,
    recent_seconds: tuple[float, ...] | list[float],
):
    if not args.object_context_enabled:
        return sample
    mask_dir = _resolve_object_mask_dir(args)
    mask_index = _load_object_mask_index(mask_dir)
    fps = _object_mask_frame_fps(args)
    centers = tuple(_object_center_for_second(mask_index, second=float(second), fps=fps) for second in recent_seconds)
    return dataclasses.replace(
        sample,
        object_name=args.object_name.strip() or _infer_object_name(args),
        recent_object_center_points=centers,
    )


def _resolve_object_mask_dir(args: ZeroShotArgs) -> pathlib.Path:
    if args.object_mask_dir is not None:
        if not args.object_mask_dir.is_dir():
            raise FileNotFoundError(f"Object mask dir does not exist: {args.object_mask_dir}")
        return args.object_mask_dir
    if args.session_path is None:
        raise ValueError("`--object-context-enabled` requires `--session-path` or `--object-mask-dir`.")
    candidates = [
        args.session_path / "mask_stage_results" / "single_hand" / "masks",
        args.session_path / "mask_stage_results" / "masks",
    ]
    for hand_dir in _session_hand_dirs(args.session_path):
        candidates.extend(
            [
                hand_dir / "mask_stage_results" / "single_hand" / "masks",
                hand_dir / "mask_stage_results" / "masks",
            ]
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No object mask directory found for session path: {args.session_path}")


def _load_object_mask_index(mask_dir: pathlib.Path) -> tuple[tuple[int, pathlib.Path], ...]:
    cache_key = str(mask_dir.resolve())
    cached = _OBJECT_MASK_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    entries: list[tuple[int, pathlib.Path]] = []
    for path in sorted(mask_dir.glob("*.png")):
        try:
            frame_index = int(path.stem)
        except ValueError:
            continue
        entries.append((frame_index, path))
    if not entries:
        raise FileNotFoundError(f"No numeric PNG object masks found in {mask_dir}")
    result = tuple(entries)
    _OBJECT_MASK_INDEX_CACHE[cache_key] = result
    return result


def _object_mask_frame_fps(args: ZeroShotArgs) -> float:
    if args.object_mask_frame_fps is not None:
        return float(args.object_mask_frame_fps)
    if args.task_config_path is not None:
        task_config = _load_task_config(args.task_config_path)
        video_fps = _optional_float(task_config.get("video_fps"))
        if video_fps is not None and video_fps > 0:
            return video_fps
    return float(args.training_fps)


def _object_center_for_second(
    mask_index: tuple[tuple[int, pathlib.Path], ...],
    *,
    second: float,
    fps: float,
) -> tuple[float | None, float | None]:
    target_frame = int(round(max(0.0, second) * fps))
    nearest_path = _nearest_mask_path(mask_index, target_frame)
    if nearest_path is None:
        return (None, None)
    return _mask_center_xy_norm(nearest_path)


def _nearest_mask_path(mask_index: tuple[tuple[int, pathlib.Path], ...], target_frame: int) -> pathlib.Path | None:
    if not mask_index:
        return None
    frame_indices = [frame for frame, _ in mask_index]
    insertion = int(np.searchsorted(frame_indices, int(target_frame)))
    if insertion <= 0:
        return mask_index[0][1]
    if insertion >= len(mask_index):
        return mask_index[-1][1]
    before = mask_index[insertion - 1]
    after = mask_index[insertion]
    return before[1] if abs(before[0] - target_frame) <= abs(after[0] - target_frame) else after[1]


def _mask_center_xy_norm(path: pathlib.Path) -> tuple[float | None, float | None]:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (None, None)
    height, width = mask.shape[:2]
    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    return (float(np.mean(xs) / denom_x), float(np.mean(ys) / denom_y))


def _infer_object_name(args: ZeroShotArgs) -> str:
    instruction = args.instruction.lower()
    if "stick" in instruction:
        return "stick"
    if "calculator" in instruction:
        return "calculator"
    if "bottle" in instruction:
        return "bottle"
    return "tracked object"


def _attach_runtime_proprio(
    sample,
    *,
    args: ZeroShotArgs,
    config: HLMemoryConfig,
    recent_seconds: tuple[float, ...] | list[float],
):
    if not config.proprio_enabled:
        return sample
    if args.session_path is None:
        raise ValueError("Runtime proprio requires --session-path.")
    states, masks = _build_runtime_proprio_states(args, config=config, recent_seconds=recent_seconds)
    return dataclasses.replace(
        sample,
        recent_robot_states=tuple(tuple(float(value) for value in row) for row in states),
        recent_robot_state_masks=tuple(tuple(float(value) for value in row) for row in masks),
        robot_state_dim_names=tuple(f"fastumi14drpy_{index}" for index in range(config.proprio_state_dim)),
    )


def _build_runtime_proprio_states(
    args: ZeroShotArgs,
    *,
    config: HLMemoryConfig,
    recent_seconds: tuple[float, ...] | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    if config.proprio_state_dim != 14:
        raise ValueError("Runtime trajectory proprio currently expects proprio_state_dim=14.")
    session_path = pathlib.Path(args.session_path) if args.session_path is not None else None
    if session_path is None:
        raise ValueError("Runtime proprio requires --session-path.")

    hand_sources = _resolve_session_proprio_sources(session_path)
    raw_states = np.zeros((len(recent_seconds), config.proprio_state_dim), dtype=np.float64)
    masks = np.zeros_like(raw_states)
    if set(hand_sources) >= {"left", "right"}:
        _fill_hand_runtime_states(raw_states, masks, seconds=recent_seconds, source=hand_sources["left"], offset=0)
        _fill_hand_runtime_states(raw_states, masks, seconds=recent_seconds, source=hand_sources["right"], offset=7)
    elif "single" in hand_sources:
        _fill_hand_runtime_states(raw_states, masks, seconds=recent_seconds, source=hand_sources["single"], offset=0)
    elif "left" in hand_sources:
        _fill_hand_runtime_states(raw_states, masks, seconds=recent_seconds, source=hand_sources["left"], offset=0)
    elif "right" in hand_sources:
        # Single-arm runs always occupy the first 7 fastumi dims.
        _fill_hand_runtime_states(raw_states, masks, seconds=recent_seconds, source=hand_sources["right"], offset=0)
    else:
        raise ValueError(f"No runtime proprio source found under {session_path}")

    stats = _load_runtime_proprio_norm_stats(args)
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if mean.shape[0] != config.proprio_state_dim or std.shape[0] != config.proprio_state_dim:
        raise ValueError(
            f"Proprio norm stats dim mismatch: expected {config.proprio_state_dim}, got {mean.shape[0]}/{std.shape[0]}."
        )
    normalized = ((raw_states - mean) / std) * masks
    return normalized, masks


@dataclasses.dataclass(frozen=True)
class _RuntimeProprioSource:
    timestamps_path: pathlib.Path
    trajectory_path: pathlib.Path


def _resolve_session_proprio_sources(session_path: pathlib.Path) -> dict[str, _RuntimeProprioSource]:
    sources: dict[str, _RuntimeProprioSource] = {}
    for hand_dir in _session_hand_dirs(session_path):
        if hand_dir.name.startswith("left_hand_"):
            key = "left"
        elif hand_dir.name.startswith("right_hand_"):
            key = "right"
        else:
            continue
        sources[key] = _runtime_source_from_dir(hand_dir)
    if not sources and (session_path / "RGB_Images" / "timestamps.csv").is_file():
        sources["single"] = _runtime_source_from_dir(session_path)
    return sources


def _runtime_source_from_dir(directory: pathlib.Path) -> _RuntimeProprioSource:
    return _RuntimeProprioSource(
        timestamps_path=_required_session_file(directory / "RGB_Images" / "timestamps.csv"),
        trajectory_path=_required_session_file(directory / "Merged_Trajectory" / "merged_trajectory.txt"),
    )


def _fill_hand_runtime_states(
    raw_states: np.ndarray,
    masks: np.ndarray,
    *,
    seconds: tuple[float, ...] | list[float],
    source: _RuntimeProprioSource,
    offset: int,
) -> None:
    video_start_stamp = _read_first_video_timestamp(source.timestamps_path)
    trajectory = _read_merged_trajectory(source.trajectory_path)
    for index, second in enumerate(seconds):
        pose = _nearest_trajectory_pose(trajectory, video_start_stamp + float(second))
        raw_states[index, offset : offset + 7] = pose
        masks[index, offset : offset + 7] = 1.0


def _read_first_video_timestamp(path: pathlib.Path) -> float:
    with path.open() as handle:
        header = next(handle, None)
        line = next(handle, None)
    if line is None:
        raise ValueError(f"Video timestamps CSV is empty: {path}")
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) < 3:
        raise ValueError(f"Expected frame_index,seq,header_stamp columns in {path}, got: {line!r}")
    return float(parts[2])


def _read_merged_trajectory(path: pathlib.Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = [float(value) for value in stripped.split()]
            if len(values) < 8:
                continue
            rows.append(values[:8])
    if not rows:
        raise ValueError(f"Merged trajectory is empty or malformed: {path}")
    trajectory = np.asarray(rows, dtype=np.float64)
    order = np.argsort(trajectory[:, 0])
    return trajectory[order]


def _nearest_trajectory_pose(trajectory: np.ndarray, timestamp: float) -> np.ndarray:
    stamps = trajectory[:, 0]
    index = int(np.searchsorted(stamps, timestamp))
    if index <= 0:
        nearest = 0
    elif index >= len(stamps):
        nearest = len(stamps) - 1
    else:
        previous_index = index - 1
        nearest = previous_index if abs(stamps[previous_index] - timestamp) <= abs(stamps[index] - timestamp) else index
    return trajectory[nearest, 1:8]


def _load_runtime_proprio_norm_stats(args: ZeroShotArgs) -> dict[str, object]:
    path = _resolved_proprio_norm_stats_path(args)
    if path is None:
        raise ValueError(
            "Runtime proprio requires --proprio-norm-stats-path or a session path containing a task id "
            "with /root/Users/dataset/hl_memory/subtask/<task_id>/train/proprio_norm_stats.json."
        )
    with path.open() as handle:
        return json.load(handle)


def _resolved_proprio_norm_stats_path(args: ZeroShotArgs) -> pathlib.Path | None:
    if args.proprio_norm_stats_path is not None:
        return args.proprio_norm_stats_path
    if args.session_path is None:
        return None
    task_id = _task_id_from_session_path(args.session_path)
    if task_id is None:
        return None
    path = pathlib.Path("/root/Users/dataset/hl_memory/subtask") / task_id / "train" / "proprio_norm_stats.json"
    return path if path.is_file() else None


def _payload_proprio_norm_stats_path(args: ZeroShotArgs) -> str | None:
    path = _resolved_proprio_norm_stats_path(args)
    return None if path is None else str(path)


def _task_id_from_session_path(path: pathlib.Path) -> str | None:
    for part in path.parts:
        if re.fullmatch(r"\d{8}[A-Z]\d{3}[A-Z]?", part):
            return part
    return None


def _is_primary_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _sample_video_path(args: ZeroShotArgs) -> pathlib.Path:
    if args.video_path is not None:
        return args.video_path
    if args.left_video_path is None or args.right_video_path is None:
        raise ValueError("Cannot build sample path without a valid single-view or dual-view input.")
    return pathlib.Path(f"{args.left_video_path.stem}__{args.right_video_path.stem}")


def _instruction_with_task_plan(
    args: ZeroShotArgs,
    *,
    known_prior_steps: tuple[str, ...] = (),
    known_prior_index: int | None = None,
) -> str:
    instruction = args.instruction.strip()
    if args.task_config_path is None:
        return instruction

    task_config = _load_task_config(args.task_config_path)
    description = _task_config_description(task_config)
    subtasks = list(known_prior_steps) or _task_config_subtasks(task_config, path=args.task_config_path)
    plan_lines = [
        instruction,
        "",
        (
            "A nominal manipulation step prior may be provided separately in this prompt. Use it as a segmentation "
            "prior, not as a substitute for visual evidence."
        ),
    ]
    if description:
        plan_lines.append(f"Task description: {description}")
    if known_prior_index is not None and 0 <= known_prior_index < len(subtasks):
        plan_lines.extend(
            [
                "",
                (
                    "Known-prior tracker state: the controller is currently evaluating this primitive "
                    f"step index {known_prior_index + 1}/{len(subtasks)}: {subtasks[known_prior_index]}"
                ),
                (
                    "Estimate progress and completion for this current known-prior step from the recent video. "
                    "The rollout controller will advance to the next listed step when completion is high."
                ),
                (
                    "If the recent video is already clearly at a later listed primitive step, set "
                    "current_objective/current_subtask/phase to the closest matching step text from the list. "
                    "Do not invent a step outside the list."
                ),
            ]
        )
    else:
        plan_lines.append(
            "If the recent video shows a different active hand/object/phase than this nominal plan, "
            "report the observed step."
        )
    return "\n".join(plan_lines)


def _task_config_steps(path: pathlib.Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    task_config = _load_task_config(path)
    return tuple(_task_config_subtasks(task_config, path=path))


def _load_task_config(path: pathlib.Path) -> dict[str, object]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Task config must be a JSON object: {path}")
    return data


def _known_prior_steps(args: ZeroShotArgs) -> tuple[str, ...]:
    if not args.known_prior_mode:
        return ()
    if args.task_config_path is None:
        raise ValueError("--known-prior-mode requires --task-config-path with `steps` or `segments[*].subtask`.")
    task_config = _load_task_config(args.task_config_path)
    steps = tuple(_task_config_subtasks(task_config, path=args.task_config_path))
    if not steps:
        raise ValueError(f"--known-prior-mode found no steps/subtasks in {args.task_config_path}.")
    return steps


def _initial_known_prior_index(args: ZeroShotArgs, steps: tuple[str, ...]) -> int:
    if not steps:
        return 0
    if args.known_prior_start_index < 0 or args.known_prior_start_index >= len(steps):
        raise ValueError(
            f"--known-prior-start-index must be in [0, {len(steps) - 1}], got {args.known_prior_start_index}."
        )
    return int(args.known_prior_start_index)


def _apply_known_prior_rollout_state(
    prediction: HLMemoryPrediction,
    *,
    known_prior_steps: tuple[str, ...],
    current_index: int,
    advance_threshold: float,
    match_threshold: float,
    max_advance_steps: int,
    next_step_require_completion: bool = False,
    next_step_confirm_steps: int = 0,
    next_step_confirmation_index: int | None = None,
    next_step_confirmation_steps: int = 0,
    safe_skip_mode: bool = False,
    skip_match_threshold: float = 0.95,
    skip_min_progress: float = 0.8,
    skip_min_stall_steps: int = 2,
    stall_steps: int = 0,
) -> tuple[HLMemoryPrediction, int, dict[str, object]]:
    if not known_prior_steps:
        return prediction, current_index, {}
    current_index = max(0, min(current_index, len(known_prior_steps) - 1))
    match = _match_prediction_to_prior_step(
        prediction,
        known_prior_steps=known_prior_steps,
        current_index=current_index,
        max_advance_steps=max_advance_steps,
    )
    should_advance = _known_prior_should_advance(prediction, threshold=advance_threshold)
    next_index = current_index
    advance_reason = "hold"
    matched_index = int(match["index"])
    matched_score = float(match["score"])
    next_step_confirmation_index, next_step_confirmation_steps = _update_next_step_confirmation(
        current_index=current_index,
        matched_index=matched_index,
        matched_score=matched_score,
        match_threshold=match_threshold,
        previous_index=next_step_confirmation_index,
        previous_steps=next_step_confirmation_steps,
    )
    next_step_confirmed = (
        next_step_confirm_steps > 0 and next_step_confirmation_steps >= next_step_confirm_steps
    )
    if matched_index > current_index and matched_score >= match_threshold:
        if not safe_skip_mode:
            if (
                matched_index == current_index + 1
                and next_step_require_completion
                and not should_advance
                and not next_step_confirmed
            ):
                advance_reason = "matched_next_prior_step_waiting_for_completion"
            else:
                next_index = matched_index
                advance_reason = (
                    "matched_next_prior_step_confirmed"
                    if matched_index == current_index + 1 and next_step_confirmed and not should_advance
                    else "matched_later_prior_step"
                )
        elif matched_index == current_index + 1:
            if next_step_require_completion and not should_advance and not next_step_confirmed:
                advance_reason = "matched_next_prior_step_waiting_for_completion"
            else:
                next_index = matched_index
                advance_reason = (
                    "matched_next_prior_step_confirmed"
                    if next_step_confirmed and not should_advance
                    else "matched_next_prior_step"
                )
        elif _known_prior_can_safe_skip(
            prediction,
            should_advance=should_advance,
            matched_score=matched_score,
            skip_match_threshold=skip_match_threshold,
            skip_min_progress=skip_min_progress,
            skip_min_stall_steps=skip_min_stall_steps,
            stall_steps=stall_steps,
        ):
            next_index = matched_index
            advance_reason = "safe_skip_matched_later_prior_step"
        elif next_step_require_completion and not should_advance:
            advance_reason = "safe_skip_waiting_for_completion"
        else:
            next_index = current_index + 1
            advance_reason = "safe_skip_clamped_to_next_step"
    elif should_advance and current_index + 1 < len(known_prior_steps):
        next_index = current_index + 1
        advance_reason = "progress_or_advance_flag"

    current_objective = known_prior_steps[next_index]
    progress = 0.0 if next_index != current_index else prediction.subtask_progress
    advance_flag = False if next_index != current_index else prediction.should_advance_objective
    task_progress = _known_prior_task_progress(known_prior_steps, next_index)
    relevant_objects = prediction.relevant_objects or tuple(
        item for item in (prediction.target_query, prediction.goal_query) if item and item.lower() != "none"
    )
    updated_language_memory = render_language_memory_fields(
        task_progress=task_progress,
        current_objective=current_objective,
        relevant_objects=relevant_objects,
        notes=prediction.notes or "Known-prior tracker output.",
    )
    return (
        dataclasses.replace(
            prediction,
            task_progress=task_progress,
            current_objective=current_objective,
            current_subtask=current_objective,
            phase=current_objective,
            updated_language_memory=updated_language_memory,
            subtask_progress=progress,
            should_advance_objective=advance_flag,
        ),
        next_index,
        {
            **match,
            "advance_reason": advance_reason,
            "advance_threshold": advance_threshold,
            "match_threshold": match_threshold,
            "max_advance_steps": max_advance_steps,
            "next_step_require_completion": next_step_require_completion,
            "next_step_confirm_steps": next_step_confirm_steps,
            "next_step_confirmation_index": next_step_confirmation_index,
            "next_step_confirmation_steps": next_step_confirmation_steps,
            "next_step_confirmed": next_step_confirmed,
            "safe_skip_mode": safe_skip_mode,
            "skip_match_threshold": skip_match_threshold,
            "skip_min_progress": skip_min_progress,
            "skip_min_stall_steps": skip_min_stall_steps,
            "stall_steps": stall_steps,
            "should_advance_by_progress": should_advance,
        },
    )


def _update_next_step_confirmation(
    *,
    current_index: int,
    matched_index: int,
    matched_score: float,
    match_threshold: float,
    previous_index: int | None,
    previous_steps: int,
) -> tuple[int | None, int]:
    expected_index = current_index + 1
    if matched_index != expected_index or matched_score < match_threshold:
        return None, 0
    if previous_index == matched_index:
        return matched_index, max(0, int(previous_steps)) + 1
    return matched_index, 1


def _known_prior_can_safe_skip(
    prediction: HLMemoryPrediction,
    *,
    should_advance: bool,
    matched_score: float,
    skip_match_threshold: float,
    skip_min_progress: float,
    skip_min_stall_steps: int,
    stall_steps: int,
) -> bool:
    if matched_score < skip_match_threshold:
        return False
    if should_advance:
        return True
    if prediction.subtask_progress is not None and float(prediction.subtask_progress) >= skip_min_progress:
        return True
    return stall_steps >= max(0, int(skip_min_stall_steps))


def _known_prior_should_advance(prediction: HLMemoryPrediction, *, threshold: float) -> bool:
    if prediction.should_advance_objective:
        return True
    if prediction.subtask_progress is None:
        return False
    return float(prediction.subtask_progress) >= threshold


def _match_prediction_to_prior_step(
    prediction: HLMemoryPrediction,
    *,
    known_prior_steps: tuple[str, ...],
    current_index: int,
    max_advance_steps: int,
) -> dict[str, object]:
    max_advance_steps = max(0, int(max_advance_steps))
    last_index = min(len(known_prior_steps) - 1, current_index + max_advance_steps)
    candidate_texts = [
        prediction.current_objective,
        prediction.current_subtask,
        prediction.phase,
    ]
    best: dict[str, object] = {
        "index": current_index,
        "score": 0.0,
        "source": "",
        "matched_step": known_prior_steps[current_index],
    }
    for source in candidate_texts:
        source_text = str(source or "").strip()
        if not source_text:
            continue
        for index in range(current_index, last_index + 1):
            score = _prior_step_similarity(source_text, known_prior_steps[index])
            if score > float(best["score"]):
                best = {
                    "index": index,
                    "score": score,
                    "source": source_text,
                    "matched_step": known_prior_steps[index],
                }
    return best


def _prior_step_similarity(left: str, right: str) -> float:
    left_norm = _normalize_prior_step_text(left)
    right_norm = _normalize_prior_step_text(right)
    if not left_norm or not right_norm:
        return 0.0
    sequence_score = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if not left_tokens or not right_tokens:
        return sequence_score
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(sequence_score, token_score)


def _normalize_prior_step_text(text: str) -> str:
    lowered = text.lower()
    normalized_chars = [char if char.isalnum() else " " for char in lowered]
    return " ".join("".join(normalized_chars).split())


def _known_prior_task_progress(steps: tuple[str, ...], current_index: int) -> str:
    completed = [step for step in steps[:current_index] if step.strip()]
    if not completed:
        return "No completed subtask yet."
    rendered = "; ".join(completed[-4:])
    prefix = "Completed known-prior subtasks"
    if len(completed) > 4:
        return f"{prefix}: ...; {rendered}."
    return f"{prefix}: {rendered}."


def _known_prior_language_memory(
    steps: tuple[str, ...],
    current_index: int,
    *,
    task_progress: str,
    notes: str,
) -> str:
    current_index = max(0, min(current_index, len(steps) - 1))
    return render_language_memory_fields(
        task_progress=task_progress,
        current_objective=steps[current_index],
        relevant_objects=(),
        notes=notes,
    )


def _language_memory_for_model(memory: str, mode: str) -> str:
    memory = memory.strip()
    if mode == "full":
        return memory
    if mode == "empty":
        return ""
    if mode != "completed_only":
        raise ValueError(f"Unsupported memory input mode: {mode!r}")
    progress = "No completed subtask yet."
    for line in memory.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() == "task progress" and value.strip():
            progress = value.strip()
            break
    return f"Task progress: {progress}"


def _task_config_description(task_config: dict[str, object]) -> str:
    return str(task_config.get("task_description", task_config.get("description", ""))).strip()


def _task_config_subtasks(task_config: dict[str, object], *, path: pathlib.Path) -> list[str]:
    """Returns only subtask text for model input; segment timestamps stay debug-only."""

    steps = task_config.get("steps")
    if steps is not None:
        if not isinstance(steps, list):
            raise ValueError(f"`steps` must be a list in task config: {path}")
        return _dedupe_consecutive_strings(str(step).strip() for step in steps)

    segments = task_config.get("segments")
    if segments is None:
        return []
    if not isinstance(segments, list):
        raise ValueError(f"`segments` must be a list in task config: {path}")

    subtasks: list[str] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"`segments[{index}]` must be an object in task config: {path}")
        subtasks.append(str(segment.get("subtask", "")).strip())
    return _dedupe_consecutive_strings(subtasks)


def _dedupe_consecutive_strings(values) -> list[str]:
    result: list[str] = []
    previous_normalized = ""
    for value in values:
        if not value:
            continue
        normalized = " ".join(value.lower().split())
        if normalized == previous_normalized:
            continue
        result.append(value)
        previous_normalized = normalized
    return result


def _ground_truth_subtask_at(args: ZeroShotArgs, second: float | None) -> str | None:
    if args.ground_truth_annotations_path is not None:
        return _annotation_ground_truth_at(
            args.ground_truth_annotations_path,
            second=second,
            episode_index=args.ground_truth_episode_index,
            field=args.ground_truth_field,
            fps=_ground_truth_fps(args),
        )
    return _task_config_subtask_at(args.task_config_path, second)


def _ground_truth_fps(args: ZeroShotArgs) -> float:
    return float(args.ground_truth_fps if args.ground_truth_fps is not None else args.training_fps)


def _annotation_ground_truth_at(
    path: pathlib.Path,
    *,
    second: float | None,
    episode_index: int | None,
    field: str,
    fps: float,
) -> str | None:
    if second is None:
        return None
    rows = _load_ground_truth_annotations(path, episode_index=episode_index, field=field)
    if not rows:
        return None

    target_frame = int(round(second * fps))
    best_frame, best_value = min(rows, key=lambda item: (abs(item[0] - target_frame), item[0] > target_frame))
    del best_frame
    return best_value


def _load_ground_truth_annotations(
    path: pathlib.Path,
    *,
    episode_index: int | None,
    field: str,
) -> tuple[tuple[int, str], ...]:
    if episode_index is None:
        raise ValueError("`--ground-truth-episode-index` is required with `--ground-truth-annotations-path`.")
    path = pathlib.Path(path)
    cache_key = (str(path.resolve()), int(episode_index), field)
    cached = _GROUND_TRUTH_ANNOTATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows: list[tuple[int, str]] = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row {line_number} in {path}: {exc}") from exc
            if _optional_int(payload.get("episode_index")) != episode_index:
                continue
            frame_index = _optional_int(payload.get("frame_index"))
            if frame_index is None:
                continue
            value = str(payload.get(field, "")).strip()
            if not value and field != "current_objective":
                value = str(payload.get("current_objective", "")).strip()
            if not value:
                continue
            rows.append((frame_index, value))

    rows = sorted(rows, key=lambda item: item[0])
    result = tuple(rows)
    if not result:
        raise ValueError(f"No usable GT rows found in {path} for episode_index={episode_index}, field={field!r}.")
    _GROUND_TRUTH_ANNOTATION_CACHE[cache_key] = result
    return result


def _task_config_subtask_at(path: pathlib.Path | None, second: float | None) -> str | None:
    if path is None or second is None:
        return None
    task_config = _load_task_config(path)
    segments = task_config.get("segments")
    if not isinstance(segments, list):
        return None

    fallback: str | None = None
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        subtask = str(segment.get("subtask", "")).strip()
        if not subtask:
            continue
        start_time = _optional_float(segment.get("start_time"))
        end_time = _optional_float(segment.get("end_time"))
        if start_time is None or second < start_time:
            continue
        fallback = subtask
        if end_time is None or second < end_time:
            return subtask
    return fallback


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _save_embedding_debug(
    output_dir: pathlib.Path,
    *,
    adapter,
    loaded,
    sample,
    clips,
    device: str | torch.device,
    max_tokens: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"output_dir": str(output_dir)}
    try:
        inputs = adapter._encode_prompt_only(loaded.processor, sample, clips)  # pylint: disable=protected-access
        input_device = adapter._resolve_input_device(loaded.model, device)  # pylint: disable=protected-access
        tensor_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
        input_ids = tensor_inputs["input_ids"]
        tokens = _decode_input_tokens(tokenizer, input_ids[0].detach().cpu())
        image_positions = _image_like_token_positions(tokens, input_ids[0].detach().cpu(), loaded.model)
        text_positions = [index for index in range(len(tokens)) if index not in set(image_positions)]
        _write_tokens_file(output_dir / "tokens.txt", tokens=tokens, image_positions=set(image_positions))
        with torch.no_grad():
            try:
                outputs = loaded.model(
                    **tensor_inputs,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            except Exception as exc:
                logging.warning("Attention debug forward with attentions failed; retrying hidden_states only: %s", exc)
                outputs = loaded.model(
                    **tensor_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                payload["attention_error"] = f"{type(exc).__name__}: {exc}"
        hidden_states = getattr(outputs, "hidden_states", None)
        attentions = getattr(outputs, "attentions", None)
        payload["input_shapes"] = {
            key: list(value.shape)
            for key, value in tensor_inputs.items()
            if isinstance(value, torch.Tensor)
        }
        payload["num_tokens"] = len(tokens)
        payload["num_image_like_tokens"] = len(image_positions)

        if hidden_states:
            hidden = hidden_states[-1][0].detach().float().cpu()
            torch.save(hidden, output_dir / "last_hidden_state.pt")
            if image_positions:
                _save_latent_pca(output_dir / "image_latent_pca.png", hidden[image_positions])
                payload["image_latent_pca"] = str(output_dir / "image_latent_pca.png")

        if attentions:
            attention = attentions[-1][0].detach().float().cpu().mean(dim=0)
            torch.save(attention, output_dir / "last_layer_mean_attention.pt")
            crop = attention[:max_tokens, :max_tokens]
            _save_heatmap(output_dir / "token_attention_heatmap.png", crop)
            payload["token_attention_heatmap"] = str(output_dir / "token_attention_heatmap.png")
            query_index = text_positions[-1] if text_positions else len(tokens) - 1
            _write_top_attention_json(
                output_dir / "text_attention_top.json",
                attention[query_index],
                tokens=tokens,
                candidate_positions=text_positions,
            )
            if image_positions:
                _write_top_attention_json(
                    output_dir / "image_attention_top.json",
                    attention[query_index],
                    tokens=tokens,
                    candidate_positions=image_positions,
                )
                _save_1d_heatmap(output_dir / "image_attention_1d.png", attention[query_index, image_positions])
                payload["image_attention_1d"] = str(output_dir / "image_attention_1d.png")
        else:
            (output_dir / "attention_note.txt").write_text(
                "Model did not return attentions. Some HF attention implementations require eager attention.\n"
            )
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
        return payload
    except Exception as exc:  # Debug output should not break normal inference.
        error_payload = {"output_dir": str(output_dir), "error": f"{type(exc).__name__}: {exc}"}
        (output_dir / "error.json").write_text(json.dumps(error_payload, indent=2, ensure_ascii=True) + "\n")
        logging.exception("Failed to save embedding debug artifacts to %s", output_dir)
        return error_payload


def _decode_input_tokens(tokenizer, input_ids: torch.Tensor) -> list[str]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return [str(token) for token in tokenizer.convert_ids_to_tokens(input_ids.tolist())]
    return [str(int(token_id)) for token_id in input_ids.tolist()]


def _image_like_token_positions(tokens: list[str], input_ids: torch.Tensor, model) -> list[int]:
    config = getattr(model, "config", None)
    candidate_ids = {
        int(value)
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
        for value in [getattr(config, name, None)]
        if value is not None
    }
    positions: list[int] = []
    for index, (token, token_id) in enumerate(zip(tokens, input_ids.tolist(), strict=False)):
        lowered = token.lower()
        if int(token_id) in candidate_ids or "image" in lowered or "video" in lowered or "vision" in lowered:
            positions.append(index)
    return positions


def _write_tokens_file(path: pathlib.Path, *, tokens: list[str], image_positions: set[int]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index, token in enumerate(tokens):
            tag = "image" if index in image_positions else "text"
            stream.write(f"{index}\t{tag}\t{token}\n")


def _write_top_attention_json(
    path: pathlib.Path,
    scores: torch.Tensor,
    *,
    tokens: list[str],
    candidate_positions: list[int],
    limit: int = 50,
) -> None:
    rows = [
        {"index": index, "token": tokens[index], "score": float(scores[index])}
        for index in candidate_positions
    ]
    rows.sort(key=lambda row: row["score"], reverse=True)
    path.write_text(json.dumps(rows[:limit], indent=2, ensure_ascii=True) + "\n")


def _save_heatmap(path: pathlib.Path, matrix: torch.Tensor) -> None:
    values = matrix.detach().float().cpu().numpy()
    if values.size == 0:
        return
    values = values - values.min()
    values = values / max(float(values.max()), 1e-8)
    image = Image.fromarray(np.uint8(values * 255), mode="L")
    image = image.resize((max(1, image.width * 4), max(1, image.height * 4)))
    image.save(path)


def _save_1d_heatmap(path: pathlib.Path, values: torch.Tensor) -> None:
    arr = values.detach().float().cpu().numpy()[None, :]
    _save_heatmap(path, torch.from_numpy(arr))


def _save_latent_pca(path: pathlib.Path, embeddings: torch.Tensor) -> None:
    if embeddings.numel() == 0:
        return
    x = embeddings.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        projected = torch.zeros((x.shape[0], 3), dtype=torch.float32)
    else:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        components = vh[: min(3, vh.shape[0])].T
        projected = x @ components
        if projected.shape[1] < 3:
            projected = torch.nn.functional.pad(projected, (0, 3 - projected.shape[1]))
    arr = projected.numpy()
    arr = arr - arr.min(axis=0, keepdims=True)
    arr = arr / np.maximum(arr.max(axis=0, keepdims=True), 1e-8)
    colors = np.uint8(arr[:, :3] * 255)
    side = int(np.ceil(np.sqrt(max(len(colors), 1))))
    canvas = np.zeros((side * side, 3), dtype=np.uint8)
    canvas[: len(colors)] = colors
    image = Image.fromarray(canvas.reshape(side, side, 3), mode="RGB")
    image = image.resize((side * 24, side * 24), resample=Image.Resampling.NEAREST)
    image.save(path)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def _write_pretty_json(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n")


def _write_live_rollout_summary(path: pathlib.Path | None, payload: dict[str, object]) -> None:
    if path is not None:
        _write_json_atomic(path, payload)


def _write_json_atomic(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    temporary_path.replace(path)


if __name__ == "__main__":
    main(resolve_cli_args_with_yaml(ZeroShotArgs, tyro))
