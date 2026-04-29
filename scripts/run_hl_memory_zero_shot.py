from __future__ import annotations

import dataclasses
import json
import pathlib

import torch
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.config_io import resolve_cli_args_with_yaml
from openpi.hl_memory.hf_adapter import create_hf_adapter
from openpi.hl_memory.zero_shot import apply_rollout_language_memory_rule
from openpi.hl_memory.zero_shot import build_rollout_end_seconds
from openpi.hl_memory.zero_shot import build_zero_shot_clips_from_video
from openpi.hl_memory.zero_shot import build_zero_shot_clips_from_video_paths
from openpi.hl_memory.zero_shot import build_zero_shot_sample
from openpi.hl_memory.zero_shot import keyframe_candidate_seconds
from openpi.hl_memory.zero_shot import parse_seconds_argument
from openpi.hl_memory.zero_shot import read_video_paths_duration_sec
from openpi.hl_memory.zero_shot import save_keyframe_candidate_frames
from openpi.hl_memory.zero_shot import save_zero_shot_debug_frames
from openpi.hl_memory.zero_shot import update_rollout_memory_seconds


@dataclasses.dataclass
class ZeroShotArgs:
    instruction: str
    video_path: pathlib.Path | None = None
    left_video_path: pathlib.Path | None = None
    right_video_path: pathlib.Path | None = None
    task_config_path: pathlib.Path | None = None
    config_yaml: pathlib.Path | None = None
    model_path: str | None = None
    local_vlm_ckpt_path: pathlib.Path | None = None
    output_json: pathlib.Path | None = None
    rollout_jsonl: pathlib.Path | None = None
    rollout_pretty_json: pathlib.Path | None = None
    debug_dir: pathlib.Path | None = None
    language_memory: str = ""
    memory_seconds: str | None = None
    recent_seconds: str | None = None
    recent_end_sec: float | None = None
    recent_step_sec: float = 1.0
    rollout_interval_sec: float | None = None
    rollout_start_sec: float = 0.0
    rollout_end_sec: float | None = None
    keyframe_merge_distance_sec: float = 2.0
    auto_memory: bool = True
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: ZeroShotArgs) -> None:
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
        recent_step_sec=args.recent_step_sec,
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
        "memory_seconds": list(selection.memory_seconds),
        "recent_seconds": list(selection.recent_seconds),
        "memory_valid_length": clips.memory_valid_length,
        "recent_valid_length": clips.recent_valid_length,
        "raw_model_output": generation.raw_output,
        "parse_error": generation.parse_error,
        "model_prediction": generation.prediction.to_dict(),
        "prediction": prediction.to_dict(),
        "language_memory_rule_applied": memory_rule_applied,
        "keyframe_candidate_seconds": list(candidate_seconds),
    }

    if args.debug_dir is not None:
        save_zero_shot_debug_frames(args.debug_dir, clips=clips, selection=selection)
        saved_keyframes = save_keyframe_candidate_frames(
            pathlib.Path(args.debug_dir) / "keyframe_candidates",
            clips=clips,
            selection=selection,
            positions=prediction.keyframe_candidate_positions,
        )
        payload["debug_dir"] = str(args.debug_dir)
        payload["saved_keyframe_candidate_paths"] = [str(path) for path in saved_keyframes]

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
    memory_seconds = tuple(parse_seconds_argument(args.memory_seconds))
    steps: list[dict[str, object]] = []
    debug_dir = pathlib.Path(args.debug_dir) if args.debug_dir is not None else None

    for step_index, end_sec in enumerate(rollout_seconds):
        memory_before = memory_seconds
        language_memory_before = language_memory
        clips, selection = _build_clips_from_args(
            args,
            config=config,
            recent_end_sec=end_sec,
            recent_step_sec=args.recent_step_sec,
            memory_seconds=memory_before,
            auto_memory=False,
        )
        sample = build_zero_shot_sample(
            video_path=_sample_video_path(args),
            instruction=_instruction_with_task_plan(args),
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
        candidate_seconds = keyframe_candidate_seconds(
            prediction,
            selection,
            recent_valid_length=clips.recent_valid_length,
        )
        memory_seconds = update_rollout_memory_seconds(
            memory_before,
            candidate_seconds,
            memory_length=config.memory_length,
            merge_distance_sec=args.keyframe_merge_distance_sec,
        )
        language_memory = prediction.updated_language_memory

        saved_keyframes: list[str] = []
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

        steps.append(
            {
                "step_index": step_index,
                "recent_end_sec": end_sec,
                "language_memory_before": language_memory_before,
                "language_memory_after": language_memory,
                "memory_seconds_before": list(memory_before),
                "memory_seconds_after": list(memory_seconds),
                "recent_seconds": list(selection.recent_seconds),
                "memory_valid_length": clips.memory_valid_length,
                "recent_valid_length": clips.recent_valid_length,
                "raw_model_output": generation.raw_output,
                "parse_error": generation.parse_error,
                "model_prediction": generation.prediction.to_dict(),
                "prediction": prediction.to_dict(),
                "language_memory_rule_applied": memory_rule_applied,
                "keyframe_candidate_seconds": list(candidate_seconds),
                "debug_dir": None if step_debug_dir is None else str(step_debug_dir),
                "saved_keyframe_candidate_paths": saved_keyframes,
            }
        )

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(debug_dir / "rollout.jsonl", steps)
        _write_pretty_json(debug_dir / "rollout_pretty.json", steps)
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
        "keyframe_merge_distance_sec": args.keyframe_merge_distance_sec,
        "final_language_memory": language_memory,
        "final_memory_seconds": list(memory_seconds),
        "debug_dir": None if debug_dir is None else str(debug_dir),
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


def _sample_video_path(args: ZeroShotArgs) -> pathlib.Path:
    if args.video_path is not None:
        return args.video_path
    if args.left_video_path is None or args.right_video_path is None:
        raise ValueError("Cannot build sample path without a valid single-view or dual-view input.")
    return pathlib.Path(f"{args.left_video_path.stem}__{args.right_video_path.stem}")


def _instruction_with_task_plan(args: ZeroShotArgs) -> str:
    instruction = args.instruction.strip()
    if args.task_config_path is None:
        return instruction

    task_config = _load_task_config(args.task_config_path)
    description = str(task_config.get("task_description", "")).strip()
    steps = task_config.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError(f"`steps` must be a list in task config: {args.task_config_path}")
    rendered_steps = [
        f"{index}. {str(step).strip()}"
        for index, step in enumerate(steps, start=1)
        if str(step).strip()
    ]
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
