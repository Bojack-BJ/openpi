#!/usr/bin/env python3
"""Report short coarse HL annotation runs with optional raw-session matches."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any


DEFAULT_ANNOTATION_NAME = "hl_annotations_llm_normalized_coarse.jsonl"


def main() -> None:
    args = parse_args()
    result = build_report(
        annotation_root=args.annotation_root.expanduser().resolve(),
        raw_root=args.raw_root.expanduser().resolve() if args.raw_root else None,
        annotation_name=args.annotation_name,
        threshold_frames=args.threshold_frames,
        quantile=args.quantile,
        task_id_glob=args.task_id_glob,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.output_csv:
        write_csv(args.output_csv, result["short_runs"])

    print(
        json.dumps(
            {
                "num_runs": result["num_runs"],
                "threshold_frames": result["threshold_frames"],
                "num_short_runs": result["num_short_runs"],
                "span_stats": result["span_stats"],
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv) if args.output_csv else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for row in result["short_runs"][: args.max_print]:
        session = Path(row.get("session_path") or "").name
        print(
            f"{row['span_frames']:>3} {row['task_id']} ep={row['episode_index']} "
            f"{session} {row['run_start_frame']}-{row['run_end_frame']} "
            f"{row['current_objective']} | {row['coarse_action_type']} | {row['coarse_merge_reason']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None, help="Optional raw segmentation_data_dtw root.")
    parser.add_argument("--annotation-name", default=DEFAULT_ANNOTATION_NAME)
    parser.add_argument("--task-id-glob", default="*")
    parser.add_argument(
        "--threshold-frames",
        type=int,
        default=None,
        help="Report runs with span <= this value. Defaults to round(quantile).",
    )
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-print", type=int, default=20)
    return parser.parse_args()


def build_report(
    *,
    annotation_root: Path,
    raw_root: Path | None,
    annotation_name: str,
    threshold_frames: int | None,
    quantile: float,
    task_id_glob: str,
) -> dict[str, Any]:
    all_runs: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir() and path.match(task_id_glob)):
        all_runs.extend(_task_runs(task_dir, annotation_name=annotation_name))

    spans = [int(row["span_frames"]) for row in all_runs]
    threshold = threshold_frames if threshold_frames is not None else int(round(_quantile(spans, quantile) or 0))
    short_runs = [row for row in all_runs if int(row["span_frames"]) <= threshold]

    session_map_cache: dict[str, dict[int, dict[str, Any]]] = {}
    if raw_root is not None:
        for row in short_runs:
            task_id = str(row["task_id"])
            if task_id not in session_map_cache:
                session_map_cache[task_id] = _match_sessions(
                    task_id=task_id,
                    task_dir=annotation_root / task_id,
                    raw_root=raw_root,
                )
            match = session_map_cache[task_id].get(int(row["episode_index"]), {})
            row["session_path"] = match.get("session_path", "")
            row["session_match_error"] = match.get("session_match_error", "")

    short_runs.sort(key=lambda row: (int(row["span_frames"]), str(row["task_id"]), int(row["episode_index"]), int(row["run_start_frame"])))
    return {
        "annotation_root": str(annotation_root),
        "raw_root": str(raw_root) if raw_root else None,
        "annotation_name": annotation_name,
        "num_runs": len(all_runs),
        "threshold_frames": threshold,
        "quantile": quantile,
        "num_short_runs": len(short_runs),
        "span_stats": {
            "min": min(spans) if spans else None,
            "p10": _quantile(spans, 0.10),
            "median": _quantile(spans, 0.50),
            "p90": _quantile(spans, 0.90),
            "max": max(spans) if spans else None,
        },
        "short_runs": short_runs,
    }


def _task_runs(task_dir: Path, *, annotation_name: str) -> list[dict[str, Any]]:
    path = task_dir / annotation_name
    if not path.exists():
        return []
    rows = sorted(
        _iter_jsonl(path),
        key=lambda row: (
            int(row.get("episode_index", -1)),
            int(row.get("frame_index", -1)),
            str(row.get("event_type", "")),
        ),
    )
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in rows:
        key = (int(row.get("episode_index", -1)), str(row.get("current_objective") or row.get("current_subtask") or ""))
        if current is None or current["key"] != key:
            if current is not None:
                runs.append(current)
            current = {"key": key, "rows": [row]}
        else:
            current["rows"].append(row)
    if current is not None:
        runs.append(current)

    output: list[dict[str, Any]] = []
    for run in runs:
        run_rows = run["rows"]
        frames = [int(row.get("frame_index", 0)) for row in run_rows]
        first = run_rows[0]
        output.append(
            {
                "task_id": task_dir.name,
                "episode_index": int(first.get("episode_index", -1)),
                "run_start_frame": min(frames),
                "run_end_frame": max(frames),
                "span_frames": max(frames) - min(frames),
                "num_annotation_rows": len(run_rows),
                "current_objective": first.get("current_objective", ""),
                "fine_current_objective": first.get("fine_current_objective", ""),
                "coarse_action_type": first.get("coarse_action_type", ""),
                "coarse_merge_reason": first.get("coarse_merge_reason", ""),
                "coarse_source_objective": first.get("coarse_source_objective", ""),
                "event_types": ",".join(sorted({str(row.get("event_type", "")) for row in run_rows if row.get("event_type") is not None})),
                "frame_indices": ",".join(str(frame) for frame in sorted(set(frames))[:20]),
            }
        )
    return output


def _match_sessions(*, task_id: str, task_dir: Path, raw_root: Path) -> dict[int, dict[str, Any]]:
    raws = _raw_sessions(raw_root / task_id)
    episodes = _converted_episodes(task_dir)
    output: dict[int, dict[str, Any]] = {}
    for episode_index, converted in episodes.items():
        best: tuple[float, str] | None = None
        for raw in raws:
            score = _session_match_score(raw, converted)
            candidate = (score, raw["session_path"])
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            output[episode_index] = {"session_path": best[1], "session_match_error": round(best[0], 3)}
    return output


def _session_match_score(raw: dict[str, Any], converted: dict[str, Any]) -> float:
    seq_penalty = abs(len(raw["seq"]) - len(converted["seq"])) * 10_000
    count = min(len(raw["seq"]), len(converted["seq"]))
    seq_penalty += sum(100_000 for index in range(count) if raw["seq"][index] != converted["seq"][index])
    frame_penalty = 0.0
    for index in range(count):
        frame_penalty += abs(raw["starts"][index] - converted["starts"][index])
        frame_penalty += abs(raw["ends"][index] - converted["ends"][index])
    return seq_penalty + frame_penalty


def _raw_sessions(task_root: Path) -> list[dict[str, Any]]:
    if not task_root.exists():
        return []
    output: list[dict[str, Any]] = []
    for path in sorted(task_root.rglob("subtask.json")):
        if ".sam3_mask_stage_work" in path.parts:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        segments = data.get("segments") or []
        if not segments:
            continue
        output.append(
            {
                "session_path": str(path.parent),
                "seq": [_normalize_text(segment.get("subtask")) for segment in segments],
                # Raw segmentation_data_dtw annotations are 60 Hz; LeRobot subtask repos are 20 Hz.
                "starts": [float(segment.get("start_frame") or 0) / 3.0 for segment in segments],
                "ends": [float(segment.get("end_frame") or 0) / 3.0 for segment in segments],
            }
        )
    return output


def _converted_episodes(task_dir: Path) -> dict[int, dict[str, Any]]:
    path = task_dir / "subtask_segments.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    output: dict[int, dict[str, Any]] = {}
    for episode_id, episode in (data.get("episodes") or {}).items():
        segments = episode.get("segments") if isinstance(episode, dict) else episode
        if not segments:
            continue
        output[int(episode_id)] = {
            "seq": [_normalize_text(segment.get("subtask")) for segment in segments],
            "starts": [float(segment.get("start_frame") or 0) for segment in segments],
            "ends": [float(segment.get("end_frame") or 0) for segment in segments],
        }
    return output


def _normalize_text(text: Any) -> str:
    normalized = str(text or "").strip().lower().replace("_", " ")
    normalized = re.sub(r"\bthe\b", " ", normalized)
    normalized = re.sub(r"[.。]+$", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _quantile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = quantile * (len(ordered) - 1)
    low = math.floor(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "episode_index",
        "session_path",
        "session_match_error",
        "run_start_frame",
        "run_end_frame",
        "span_frames",
        "num_annotation_rows",
        "current_objective",
        "fine_current_objective",
        "coarse_action_type",
        "coarse_merge_reason",
        "coarse_source_objective",
        "event_types",
        "frame_indices",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
