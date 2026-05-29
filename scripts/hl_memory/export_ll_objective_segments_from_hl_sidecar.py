#!/usr/bin/env python3
"""Build LL current-objective segment sidecar from LeRobot subtasks and HL task sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUBTASK_SEGMENTS_NAME = "subtask_segments.json"
DEFAULT_HL_SIDECAR_NAME = "hl_segments_llm_sidecar.json"
DEFAULT_OUTPUT_NAME = "ll_current_objective_segments.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map original LeRobot episode/frame subtask segments to HL-normalized current_objective text. "
            "The output keeps the standard subtask_segments.json shape so LL VLA training can consume it via "
            "DataConfig.subtask_segments_path."
        )
    )
    parser.add_argument("--task-dir", type=Path, default=None, help="Task/repo directory containing the sidecars.")
    parser.add_argument("--subtask-segments-json", type=Path, default=None)
    parser.add_argument("--hl-sidecar-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subtask_segments_json = _resolve_path(args.subtask_segments_json, args.task_dir, DEFAULT_SUBTASK_SEGMENTS_NAME)
    hl_sidecar_json = _resolve_path(args.hl_sidecar_json, args.task_dir, DEFAULT_HL_SIDECAR_NAME)
    output_json = _resolve_path(args.output_json, args.task_dir, DEFAULT_OUTPUT_NAME)
    if output_json.exists() and not args.overwrite:
        raise FileExistsError(f"{output_json} already exists. Use --overwrite to replace it.")

    payload = build_ll_objective_segments(
        subtask_segments=json.loads(subtask_segments_json.read_text(encoding="utf-8")),
        hl_sidecar=json.loads(hl_sidecar_json.read_text(encoding="utf-8")),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    episode_count = len(payload.get("episodes", {}))
    segment_count = sum(len(episode.get("segments", [])) for episode in payload.get("episodes", {}).values())
    print(f"Wrote {segment_count} LL objective segments across {episode_count} episodes to {output_json}")


def build_ll_objective_segments(*, subtask_segments: dict[str, Any], hl_sidecar: dict[str, Any]) -> dict[str, Any]:
    objective_segments = _objective_segments_from_hl_sidecar(hl_sidecar)
    episodes = _episodes_from_subtask_segments(subtask_segments)
    output_episodes: dict[str, dict[str, Any]] = {}
    for episode_index, episode in episodes.items():
        original_segments = _episode_segments(episode)
        mapped_segments: list[dict[str, Any]] = []
        for segment_index, segment in enumerate(original_segments):
            objective = _resolve_objective(segment, segment_index=segment_index, objective_segments=objective_segments)
            mapped_segments.append(
                {
                    "start_frame": int(segment["start_frame"]),
                    "end_frame": int(segment["end_frame"]),
                    "subtask": objective,
                    "source_subtask": str(segment.get("subtask", "")).strip(),
                    "source_segment_index": segment_index,
                }
            )
        output_episodes[str(episode_index)] = {"segments": mapped_segments}
    return {
        "source": DEFAULT_HL_SIDECAR_NAME,
        "field_semantics": {"subtask": "current_objective"},
        "episodes": output_episodes,
    }


def _resolve_path(path: Path | None, task_dir: Path | None, default_name: str) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    if task_dir is None:
        raise ValueError(f"Provide --{default_name.replace('_', '-').replace('.json', '-json')} or --task-dir.")
    return (task_dir / default_name).expanduser().resolve()


def _objective_segments_from_hl_sidecar(payload: dict[str, Any]) -> list[dict[str, str]]:
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("HL sidecar must contain a list field `segments`.")
    objectives: list[dict[str, str]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        objective = str(segment.get("current_objective", segment.get("raw_subtask", ""))).strip()
        raw_subtask = str(segment.get("raw_subtask", "")).strip()
        if objective:
            objectives.append({"current_objective": objective, "raw_subtask": raw_subtask})
    if not objectives:
        raise ValueError("HL sidecar has no usable current_objective segments.")
    return objectives


def _episodes_from_subtask_segments(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episodes_payload = payload.get("episodes")
    if isinstance(episodes_payload, dict):
        return {int(index): dict(episode) for index, episode in episodes_payload.items()}
    if isinstance(episodes_payload, list):
        return {int(episode.get("episode_index", index)): dict(episode) for index, episode in enumerate(episodes_payload)}
    if "segments" in payload:
        return {int(payload.get("episode_index", 0)): payload}
    raise ValueError("Subtask segments payload must contain `episodes` or `segments`.")


def _episode_segments(episode: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(episode.get("segments"), list):
        return [
            {
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "subtask": str(segment.get("subtask", "")).strip(),
            }
            for segment in episode["segments"]
            if str(segment.get("subtask", "")).strip() and int(segment["start_frame"]) < int(segment["end_frame"])
        ]

    subtasks = [str(item).strip() for item in episode.get("subtask", [])]
    boundaries = [int(item) for item in episode.get("boundaries_frame_indices", [])]
    if not subtasks:
        return []
    episode_end = int(episode.get("num_frames") or episode.get("end_frame") or episode.get("episode_length"))
    starts = [0, *boundaries]
    ends = [*boundaries, episode_end]
    return [
        {"start_frame": start, "end_frame": end, "subtask": subtask}
        for start, end, subtask in zip(starts, ends, subtasks, strict=False)
        if subtask and start < end
    ]


def _resolve_objective(
    segment: dict[str, Any],
    *,
    segment_index: int,
    objective_segments: list[dict[str, str]],
) -> str:
    source_subtask = str(segment.get("subtask", "")).strip().lower()
    matches = [
        item["current_objective"]
        for item in objective_segments
        if item.get("raw_subtask", "").strip().lower() == source_subtask
    ]
    if len(matches) == 1:
        return matches[0]
    if segment_index < len(objective_segments):
        return objective_segments[segment_index]["current_objective"]
    return objective_segments[-1]["current_objective"]


if __name__ == "__main__":
    main()
