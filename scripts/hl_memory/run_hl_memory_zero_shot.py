from __future__ import annotations

import dataclasses
import difflib
import json
import logging
import os
import pathlib

import numpy as np
from PIL import Image
import torch
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.config_io import resolve_cli_args_with_yaml
from openpi.hl_memory.hf_adapter import create_hf_adapter
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


@dataclasses.dataclass
class ZeroShotArgs:
    """Run one-shot or interval rollout HL-memory inference on single-view or dual-view videos."""

    # Task and input videos.
    instruction: str
    video_path: pathlib.Path | None = None
    left_video_path: pathlib.Path | None = None
    right_video_path: pathlib.Path | None = None
    task_config_path: pathlib.Path | None = None

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
    memory_seconds: str | None = None
    recent_seconds: str | None = None
    recent_end_sec: float | None = None
    recent_step_sec: float | None = None
    training_fps: float = 20.0
    frame_subsample: int = 5

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

    # VLM backend and generation settings.
    vlm_backend: str = "qwen2_5_vl"
    vlm_variant: str | None = None
    vlm_hf_model_id: str | None = None
    precision: str = "bfloat16"
    recent_frames_length: int = 8
    memory_length: int = 8
    frame_height: int = 224
    frame_width: int = 224
    allow_single_frame_fallback: bool = True
    max_new_tokens: int = 256
    enable_thinking: bool = False
    thinking_budget_tokens: int = 128
    thinking_max_new_tokens: int = 1024

    # Large-model loading options.
    parallel_mode: str = "none"
    device_map: str = "auto"
    tensor_parallel_plan: str = "auto"
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
    _resolve_video_paths(args)
    if args.task_config_path is not None:
        _load_task_config(args.task_config_path)
    config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_variant=args.vlm_variant,
        vlm_hf_model_id=args.vlm_hf_model_id,
        precision=args.precision,
        recent_frames_length=args.recent_frames_length,
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
        args.output_json.write_text(rendered + "\n")


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
        language_memory=args.language_memory,
        memory_seconds=selection.memory_seconds,
        recent_seconds=selection.recent_seconds,
    )
    generation = adapter.generate_prediction(loaded, sample, clips, device=args.device)
    recent_end_sec = selection.recent_seconds[-1] if selection.recent_seconds else 0.0
    prediction, memory_rule_applied = apply_rollout_language_memory_rule(
        generation.prediction,
        previous_memory=args.language_memory,
        recent_end_sec=recent_end_sec,
    )
    ground_truth_subtask = _task_config_subtask_at(args.task_config_path, recent_end_sec)
    candidate_seconds = keyframe_candidate_seconds(
        prediction,
        selection,
        recent_valid_length=clips.recent_valid_length,
    )

    payload = {
        "video_paths": _payload_video_paths_from_args(args),
        "model_path": args.model_path,
        "local_vlm_ckpt_path": None if args.local_vlm_ckpt_path is None else str(args.local_vlm_ckpt_path),
        "resolved_model_id": config.resolved_model_id if resolved_model_path is None else resolved_model_path,
        "instruction": args.instruction,
        "task_config_path": None if args.task_config_path is None else str(args.task_config_path),
        "language_memory": args.language_memory,
        "duration_sec": selection.duration_sec,
        "recent_step_sec": _resolved_recent_step_sec(args),
        "training_fps": args.training_fps,
        "frame_subsample": args.frame_subsample,
        "memory_seconds": list(selection.memory_seconds),
        "recent_seconds": list(selection.recent_seconds),
        "memory_valid_length": clips.memory_valid_length,
        "recent_valid_length": clips.recent_valid_length,
        "raw_model_output": generation.raw_output,
        "parse_error": generation.parse_error,
        "model_prediction": generation.prediction.to_dict(),
        "prediction": prediction.to_dict(),
        "ground_truth_subtask": ground_truth_subtask,
        "language_memory_rule_applied": memory_rule_applied,
        "keyframe_candidate_seconds": list(candidate_seconds),
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
            language_memory_before=args.language_memory,
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
    known_prior_steps = _known_prior_steps(args)
    known_prior_index = _initial_known_prior_index(args, known_prior_steps)
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
            language_memory=language_memory_before,
            memory_seconds=selection.memory_seconds,
            recent_seconds=selection.recent_seconds,
        )
        generation = adapter.generate_prediction(loaded, sample, clips, device=args.device)
        prediction, memory_rule_applied = apply_rollout_language_memory_rule(
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
            )
        ground_truth_subtask = _task_config_subtask_at(args.task_config_path, end_sec)
        candidate_seconds = keyframe_candidate_seconds(
            prediction,
            selection,
            recent_valid_length=clips.recent_valid_length,
        )
        keyframe_vote_seconds.extend(candidate_seconds)
        memory_seconds = update_rollout_memory_seconds(
            keyframe_vote_seconds,
            (),
            memory_length=config.memory_length,
            merge_distance_sec=args.keyframe_merge_distance_sec,
        )
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
                language_memory_before=language_memory_before,
                language_memory_after=language_memory,
                memory_seconds_before=memory_before,
                memory_seconds_after=memory_seconds,
                keyframe_candidate_seconds=candidate_seconds,
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
                "language_memory_before": language_memory_before,
                "language_memory_after": language_memory,
                "known_prior_mode": bool(known_prior_steps),
                "known_prior_steps": list(known_prior_steps),
                "known_prior_index_before": known_prior_index_before if known_prior_steps else None,
                "known_prior_index_after": known_prior_index if known_prior_steps else None,
                "known_prior_current_step": known_prior_steps[known_prior_index] if known_prior_steps else None,
                "known_prior_match": known_prior_match,
                "memory_seconds_before": list(memory_before),
                "memory_seconds_after": list(memory_seconds),
                "keyframe_vote_seconds": list(keyframe_vote_seconds),
                "recent_seconds": list(selection.recent_seconds),
                "memory_valid_length": clips.memory_valid_length,
                "recent_valid_length": clips.recent_valid_length,
                "raw_model_output": generation.raw_output,
                "parse_error": generation.parse_error,
                "model_prediction": generation.prediction.to_dict(),
                "prediction": prediction.to_dict(),
                "ground_truth_subtask": ground_truth_subtask,
                "language_memory_rule_applied": memory_rule_applied,
                "keyframe_candidate_seconds": list(candidate_seconds),
                "debug_dir": None if step_debug_dir is None else str(step_debug_dir),
                "debug_panel_path": None if debug_panel_path is None else str(debug_panel_path),
                "saved_keyframe_candidate_paths": saved_keyframes,
                "embedding_debug": embedding_debug_payload,
            }
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

    return {
        "video_paths": _payload_video_paths_from_args(args),
        "model_path": args.model_path,
        "local_vlm_ckpt_path": None if args.local_vlm_ckpt_path is None else str(args.local_vlm_ckpt_path),
        "resolved_model_id": config.resolved_model_id if resolved_model_path is None else resolved_model_path,
        "instruction": args.instruction,
        "task_config_path": None if args.task_config_path is None else str(args.task_config_path),
        "initial_language_memory": args.language_memory,
        "duration_sec": duration_sec,
        "rollout_interval_sec": args.rollout_interval_sec,
        "rollout_start_sec": args.rollout_start_sec,
        "rollout_end_sec": args.rollout_end_sec,
        "recent_step_sec": recent_step_sec,
        "training_fps": args.training_fps,
        "frame_subsample": args.frame_subsample,
        "keyframe_merge_distance_sec": args.keyframe_merge_distance_sec,
        "known_prior_mode": bool(known_prior_steps),
        "known_prior_steps": list(known_prior_steps),
        "known_prior_advance_threshold": args.known_prior_advance_threshold,
        "known_prior_match_threshold": args.known_prior_match_threshold,
        "known_prior_max_advance_steps": args.known_prior_max_advance_steps,
        "final_known_prior_index": known_prior_index if known_prior_steps else None,
        "final_language_memory": language_memory,
        "final_memory_seconds": list(memory_seconds),
        "debug_dir": None if debug_dir is None else str(debug_dir),
        "debug_video_path": None if debug_video_path is None else str(debug_video_path),
        "debug_video_error": debug_video_error,
        "steps": steps,
    }


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
    return float(args.frame_subsample) / float(args.training_fps)


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
    rendered_steps = [f"{index}. {subtask}" for index, subtask in enumerate(subtasks, start=1)]
    plan_lines = [
        instruction,
        "",
        (
            "Nominal manipulation plan from task config. Use it as a segmentation prior, "
            "not as a substitute for visual evidence."
        ),
    ]
    if description:
        plan_lines.append(f"Task description: {description}")
    if rendered_steps:
        plan_lines.append("Expected primitive sequence:")
        plan_lines.extend(rendered_steps)
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
    if match["index"] > current_index and match["score"] >= match_threshold:
        next_index = int(match["index"])
        advance_reason = "matched_later_prior_step"
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
            "should_advance_by_progress": should_advance,
        },
    )


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


if __name__ == "__main__":
    main(resolve_cli_args_with_yaml(ZeroShotArgs, tyro))
