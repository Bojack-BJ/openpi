from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
from typing import Any


Segment = tuple[int, int, str]


def main() -> None:
    args = _parse_args()
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_jsonl} already exists. Use --overwrite to replace it.")

    if args.raw_dir is not None:
        rows = _rows_from_raw_dirs(args)
    else:
        rows = _rows_from_sidecar(args)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} HL annotations to {args.output_jsonl}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FastUMI subtask.json / subtask_segments.json files into the annotations.jsonl "
            "format consumed by scripts/hl_memory/export_hl_memory_dataset.py."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--lerobot-dir",
        type=pathlib.Path,
        help="LeRobot dataset root containing subtask_segments.json.",
    )
    input_group.add_argument(
        "--repo-id",
        help="LeRobot repo id under HF_LEROBOT_HOME, for example fastumi/sponge_visual_guided.",
    )
    input_group.add_argument(
        "--raw-dir",
        action="append",
        type=pathlib.Path,
        help=(
            "Raw data root or session_* directory. Repeat for multiple roots. This is only a fallback because raw "
            "session ordering may differ from the final LeRobot episode ordering."
        ),
    )
    input_group.add_argument(
        "--subtask-segments-json",
        type=pathlib.Path,
        help="LeRobot dataset sidecar, usually <HF_LEROBOT_HOME>/<repo_id>/subtask_segments.json.",
    )
    parser.add_argument("--output-jsonl", type=pathlib.Path, required=True)
    parser.add_argument("--instruction", default="", help="Optional task instruction copied to every annotation.")
    parser.add_argument("--target-query", default="", help="Optional target query copied to every annotation.")
    parser.add_argument("--goal-query", default="", help="Optional goal query copied to every annotation.")
    parser.add_argument(
        "--start-episode-index",
        type=int,
        default=0,
        help="Episode index assigned to the first raw session when using --raw-dir.",
    )
    parser.add_argument(
        "--emit-success-events",
        action="store_true",
        help="Also emit a success annotation at end_frame - 1 for each segment.",
    )
    parser.add_argument(
        "--progress-sample-stride",
        type=int,
        default=0,
        help=(
            "If > 0, emit additional progress annotations every N frames inside each segment, capped by "
            "--max-progress-samples-per-segment. The segment midpoint is always included when possible."
        ),
    )
    parser.add_argument(
        "--progress-sample-fractions",
        default="",
        help=(
            "Comma-separated relative positions inside each segment, excluding endpoints, for example "
            "'0.2,0.4,0.6,0.8'. When set, this takes precedence over --progress-sample-stride."
        ),
    )
    parser.add_argument(
        "--progress-extra-fractions",
        default="",
        help=(
            "Comma-separated additional relative positions inside each segment, for example '0.85,0.9,0.95'. "
            "These are added on top of dynamic/stride/default progress sampling and still respect "
            "--progress-min-gap and --max-progress-samples-per-segment."
        ),
    )
    parser.add_argument(
        "--progress-sample-target-frames",
        type=int,
        default=0,
        help=(
            "If > 0 and --progress-sample-fractions is unset, dynamically choose the number of progress samples as "
            "segment_length / N, then place them at evenly spaced internal fractions."
        ),
    )
    parser.add_argument(
        "--progress-sample-jitter",
        type=float,
        default=0.0,
        help=(
            "Optional deterministic fraction jitter in [0, 0.5), applied to fraction-based/dynamic sampling. "
            "Example: 0.05 perturbs each internal fraction by up to +/- 5%% of segment length."
        ),
    )
    parser.add_argument(
        "--progress-sample-seed",
        type=int,
        default=0,
        help="Seed for deterministic progress sample jitter.",
    )
    parser.add_argument(
        "--min-progress-samples-per-segment",
        type=int,
        default=0,
        help="Minimum number of dynamic progress samples per segment when --progress-sample-target-frames > 0.",
    )
    parser.add_argument(
        "--max-progress-samples-per-segment",
        type=int,
        default=1,
        help="Maximum number of progress annotations per segment.",
    )
    parser.add_argument(
        "--progress-min-gap",
        type=int,
        default=0,
        help="Minimum frame distance between progress annotations inside the same segment. Use 0 to disable.",
    )
    parser.add_argument(
        "--empty-last-end-policy",
        choices=("skip", "use-start", "error"),
        default="skip",
        help="How to handle a segment whose end_frame cannot be inferred.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _rows_from_raw_dirs(args: argparse.Namespace) -> list[dict[str, object]]:
    sessions = _find_sessions(args.raw_dir)
    if not sessions:
        raise FileNotFoundError(f"No valid session_* directories found under: {args.raw_dir}")

    rows: list[dict[str, object]] = []
    for offset, session_path in enumerate(sessions):
        episode_index = args.start_episode_index + offset
        subtask_path = _find_subtask_file(session_path)
        if subtask_path is None:
            print(f"[warn] skip episode={episode_index}: no subtask.json or segments.json under {session_path}")
            continue
        segments = _segments_from_subtask_json(
            subtask_path,
            empty_last_end_policy=args.empty_last_end_policy,
        )
        if not segments:
            print(f"[warn] skip episode={episode_index}: no valid segments in {subtask_path}")
            continue
        rows.extend(_segments_to_rows(episode_index, segments, args))
    return _sort_rows(rows)


def _rows_from_sidecar(args: argparse.Namespace) -> list[dict[str, object]]:
    path = _resolve_sidecar_path(args)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    episodes = payload.get("episodes", {})
    if not isinstance(episodes, dict):
        raise ValueError(f"Expected `episodes` object in {path}")

    rows: list[dict[str, object]] = []
    for episode_key, episode_payload in sorted(episodes.items(), key=lambda item: int(item[0])):
        if not isinstance(episode_payload, dict):
            continue
        segments = _segments_from_payload(
            episode_payload,
            empty_last_end_policy=args.empty_last_end_policy,
        )
        rows.extend(_segments_to_rows(int(episode_key), segments, args))
    return _sort_rows(rows)


def _resolve_sidecar_path(args: argparse.Namespace) -> pathlib.Path:
    if args.subtask_segments_json is not None:
        return args.subtask_segments_json
    if args.lerobot_dir is not None:
        return args.lerobot_dir / "subtask_segments.json"
    if args.repo_id:
        hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
        if not hf_lerobot_home:
            raise ValueError("--repo-id requires HF_LEROBOT_HOME to be set.")
        return pathlib.Path(hf_lerobot_home) / args.repo_id / "subtask_segments.json"
    raise ValueError("Expected one LeRobot sidecar input.")


def _find_sessions(raw_dirs: list[pathlib.Path]) -> list[pathlib.Path]:
    found: dict[pathlib.Path, pathlib.Path] = {}
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            raise FileNotFoundError(raw_dir)
        if _is_valid_session(raw_dir):
            found[_safe_resolve(raw_dir)] = raw_dir
        for root, dirs, _files in os.walk(raw_dir):
            dirs.sort()
            for dirname in dirs:
                if not dirname.startswith("session"):
                    continue
                session_path = pathlib.Path(root) / dirname
                if _is_valid_session(session_path):
                    found.setdefault(_safe_resolve(session_path), session_path)
    return [found[key] for key in sorted(found)]


def _is_valid_session(path: pathlib.Path) -> bool:
    if not path.is_dir() or not path.name.startswith("session"):
        return False
    if (path / "RGB_Images").is_dir():
        return True
    subdirs = [item for item in path.iterdir() if item.is_dir()]
    has_left = any(item.name.startswith("left_hand") for item in subdirs)
    has_right = any(item.name.startswith("right_hand") for item in subdirs)
    return has_left and has_right


def _safe_resolve(path: pathlib.Path) -> pathlib.Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _find_subtask_file(session_path: pathlib.Path) -> pathlib.Path | None:
    candidates = [
        session_path / "subtask.json",
        session_path / "segments.json",
    ]
    for child in session_path.iterdir():
        if not child.is_dir():
            continue
        candidates.extend([child / "subtask.json", child / "segments.json"])
    return next((path for path in candidates if path.exists()), None)


def _segments_from_subtask_json(
    path: pathlib.Path,
    *,
    empty_last_end_policy: str,
) -> list[Segment]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return _segments_from_payload(payload, empty_last_end_policy=empty_last_end_policy)


def _segments_from_payload(payload: dict[str, Any], *, empty_last_end_policy: str) -> list[Segment]:
    if "segments" in payload:
        return _segments_from_segment_list(payload["segments"], empty_last_end_policy=empty_last_end_policy)
    if "subtask_instruction" in payload:
        return _segments_from_instruction_ranges(payload["subtask_instruction"])
    return _segments_from_boundaries(payload)


def _segments_from_segment_list(items: Any, *, empty_last_end_policy: str) -> list[Segment]:
    if not isinstance(items, list):
        return []
    fallback_end = _infer_fallback_end(items)
    segments: list[Segment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("subtask", item.get("current_subtask", ""))).strip()
        if not text:
            continue
        start = int(item.get("start_frame", item.get("start", 0)))
        raw_end = item.get("end_frame", item.get("end"))
        end = _resolve_end_frame(raw_end, fallback_end, start, empty_last_end_policy)
        if end is not None and start < end:
            segments.append((start, end, text))
    return segments


def _infer_fallback_end(items: list[Any]) -> int | None:
    ends: list[int] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_end = item.get("end_frame", item.get("end"))
        if raw_end is None:
            continue
        try:
            ends.append(int(raw_end))
        except (TypeError, ValueError):
            continue
    return max(ends) if ends else None


def _resolve_end_frame(
    raw_end: Any,
    fallback_end: int | None,
    start: int,
    empty_last_end_policy: str,
) -> int | None:
    if raw_end is not None:
        return int(raw_end)
    if fallback_end is not None:
        return fallback_end
    if empty_last_end_policy == "skip":
        return None
    if empty_last_end_policy == "use-start":
        return start + 1
    raise ValueError(f"Could not infer end_frame for segment starting at frame {start}")


def _segments_from_instruction_ranges(items: Any) -> list[Segment]:
    if not isinstance(items, list):
        return []
    segments: list[Segment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for frame_range, subtask in item.items():
            start, end = _parse_frame_range(str(frame_range))
            text = str(subtask).strip()
            if text and start < end:
                segments.append((start, end, text))
    return segments


def _parse_frame_range(frame_range: str) -> tuple[int, int]:
    start, end = frame_range.strip().strip("[]").split(",", maxsplit=1)
    return int(start), int(end) + 1


def _segments_from_boundaries(payload: dict[str, Any]) -> list[Segment]:
    subtasks = [str(item).strip() for item in payload.get("subtask", [])]
    boundaries = [int(item) for item in payload.get("boundaries_frame_indices", [])]
    if not subtasks:
        return []
    starts = [0, *boundaries]
    fallback_end = int(payload.get("num_frames", starts[-1] + 1))
    ends = [*boundaries, fallback_end]
    return [
        (start, end, subtask)
        for start, end, subtask in zip(starts, ends, subtasks)
        if subtask and start < end
    ]


def _segments_to_rows(
    episode_index: int,
    segments: list[Segment],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start, end, subtask in segments:
        rows.append(
            _make_row(
                episode_index=episode_index,
                frame_index=start,
                subtask=subtask,
                event_type="subtask_boundary",
                event_text=f"Started {subtask}.",
                args=args,
            )
        )
        for progress_frame in _progress_sample_frames(
            start,
            end,
            episode_index=episode_index,
            subtask=subtask,
            stride=args.progress_sample_stride,
            fractions=_parse_progress_sample_fractions(args.progress_sample_fractions),
            extra_fractions=_parse_progress_sample_fractions(args.progress_extra_fractions),
            target_frames=args.progress_sample_target_frames,
            min_samples=args.min_progress_samples_per_segment,
            max_samples=args.max_progress_samples_per_segment,
            min_gap=args.progress_min_gap,
            reserved_frames=(start, end - 1) if args.emit_success_events else (start,),
            jitter=args.progress_sample_jitter,
            seed=args.progress_sample_seed,
        ):
            rows.append(
                _make_row(
                    episode_index=episode_index,
                    frame_index=progress_frame,
                    subtask=subtask,
                    event_type="progress",
                    event_text=f"Continuing {subtask}.",
                    args=args,
                )
            )
        if args.emit_success_events:
            rows.append(
                _make_row(
                    episode_index=episode_index,
                    frame_index=end - 1,
                    subtask=subtask,
                    event_type="success",
                    event_text=f"Completed {subtask}.",
                    args=args,
                )
            )
    return _dedupe_rows(rows)


def _parse_progress_sample_fractions(value: str) -> list[float]:
    fractions: list[float] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        fraction = float(stripped)
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"progress sample fraction must be in (0, 1), got {fraction}")
        fractions.append(fraction)
    return fractions


def _progress_sample_frames(
    start: int,
    end: int,
    *,
    episode_index: int,
    subtask: str,
    stride: int,
    fractions: list[float],
    extra_fractions: list[float],
    target_frames: int,
    min_samples: int,
    max_samples: int,
    min_gap: int = 0,
    reserved_frames: tuple[int, ...] = (),
    jitter: float = 0.0,
    seed: int = 0,
) -> list[int]:
    """Choose progress annotation frames inside one segment.

    Recommended mode is dynamic fraction sampling:
    1. `target_frames` controls the number of samples: longer segments get more samples.
    2. sample locations are evenly spaced fractions inside the segment, not raw stride positions.
    3. optional deterministic jitter perturbs those fractions per episode/subtask so labels are not always at fixed
       0.25/0.5/0.75-style positions.
    4. `extra_fractions` can add late-stage samples such as 0.85/0.9/0.95 on top of the primary mode.

    Endpoints are intentionally excluded: `subtask_boundary` covers the start and `--emit-success-events` can cover
    `end - 1`.
    """

    if end - start <= 1 or max_samples <= 0:
        return []
    if not 0.0 <= jitter < 0.5:
        raise ValueError(f"--progress-sample-jitter must be in [0, 0.5), got {jitter}")

    length = end - start
    middle = start + (end - start) // 2
    candidates: list[int] = []
    anchor = middle
    if fractions:
        effective_fractions = _jitter_fractions(
            fractions,
            jitter=jitter,
            seed=seed,
            episode_index=episode_index,
            subtask=subtask,
            start=start,
            end=end,
        )
        candidates.extend(_frames_from_fractions(start, end, effective_fractions))
        anchor = candidates[len(candidates) // 2] if candidates else middle
    elif target_frames > 0:
        dynamic_count = max(length // target_frames, min_samples)
        dynamic_count = max(1, min(dynamic_count, max_samples))
        dynamic_fractions = [(index + 1) / (dynamic_count + 1) for index in range(dynamic_count)]
        effective_fractions = _jitter_fractions(
            dynamic_fractions,
            jitter=jitter,
            seed=seed,
            episode_index=episode_index,
            subtask=subtask,
            start=start,
            end=end,
        )
        candidates.extend(_frames_from_fractions(start, end, effective_fractions))
        anchor = candidates[len(candidates) // 2] if candidates else middle
    else:
        candidates.append(middle)
    if stride > 0 and not fractions and target_frames <= 0:
        candidates.extend(range(start + stride, end, stride))
    if extra_fractions:
        effective_extra_fractions = _jitter_fractions(
            extra_fractions,
            jitter=jitter,
            seed=seed + 1,
            episode_index=episode_index,
            subtask=subtask,
            start=start,
            end=end,
        )
        candidates.extend(_frames_from_fractions(start, end, effective_extra_fractions))

    reserved_set = {int(frame) for frame in reserved_frames}
    raw_unique = sorted({int(frame) for frame in candidates if start < int(frame) < end and int(frame) not in reserved_set})
    unique = _apply_progress_min_gap(raw_unique, anchor=anchor, min_gap=min_gap, reserved_frames=reserved_frames)
    min_budget = min(max(min_samples, 0), max_samples)
    if target_frames > 0 and len(unique) < min_budget:
        for frame in _prioritize_coverage(raw_unique, anchor=anchor):
            if frame not in unique and not _violates_min_gap(frame, unique, min_gap=min_gap, reserved_frames=reserved_frames):
                unique.append(frame)
            if len(unique) >= min_budget:
                break
        unique = sorted(unique)
    if len(unique) <= max_samples:
        return unique
    if anchor in unique:
        remaining = [frame for frame in unique if frame != anchor]
        budget = max_samples - 1
        return sorted([anchor, *_evenly_spaced_subset(remaining, budget)])
    return _evenly_spaced_subset(unique, max_samples)


def _frames_from_fractions(start: int, end: int, fractions: list[float]) -> list[int]:
    length = end - start
    return [int(round(start + length * fraction)) for fraction in fractions]


def _jitter_fractions(
    fractions: list[float],
    *,
    jitter: float,
    seed: int,
    episode_index: int,
    subtask: str,
    start: int,
    end: int,
) -> list[float]:
    if jitter <= 0.0 or len(fractions) == 0:
        return list(fractions)
    digest = hashlib.sha256(f"{seed}|{episode_index}|{subtask}|{start}|{end}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    jittered = [min(max(fraction + rng.uniform(-jitter, jitter), 1e-6), 1.0 - 1e-6) for fraction in fractions]
    return sorted(jittered)


def _apply_progress_min_gap(
    values: list[int],
    *,
    anchor: int,
    min_gap: int,
    reserved_frames: tuple[int, ...] = (),
) -> list[int]:
    if min_gap <= 0:
        return values

    selected: list[int] = []
    if anchor in values and not _violates_min_gap(anchor, selected, min_gap=min_gap, reserved_frames=reserved_frames):
        selected.append(anchor)
    for value in _prioritize_coverage(values, anchor=anchor):
        if value == anchor:
            continue
        if not _violates_min_gap(value, selected, min_gap=min_gap, reserved_frames=reserved_frames):
            selected.append(value)
    return sorted(selected)


def _violates_min_gap(value: int, selected: list[int], *, min_gap: int, reserved_frames: tuple[int, ...]) -> bool:
    if min_gap <= 0:
        return False
    return any(abs(value - existing) < min_gap for existing in selected) or any(
        abs(value - reserved) < min_gap for reserved in reserved_frames
    )


def _prioritize_coverage(values: list[int], *, anchor: int) -> list[int]:
    return sorted(values, key=lambda value: (-abs(value - anchor), value))


def _evenly_spaced_subset(values: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if len(values) <= count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    positions = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[position] for position in positions]


def _make_row(
    *,
    episode_index: int,
    frame_index: int,
    subtask: str,
    event_type: str,
    event_text: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    row: dict[str, object] = {
        "episode_index": int(episode_index),
        "frame_index": int(frame_index),
        "current_subtask": subtask,
        "phase": subtask,
        "event_type": event_type,
        "event_text": event_text,
    }
    for key, value in (
        ("instruction", args.instruction),
        ("target_query", args.target_query),
        ("goal_query", args.goal_query),
    ):
        if value:
            row[key] = value
    return row


def _sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    event_order = {"success": 0, "subtask_boundary": 1, "progress": 2}
    return sorted(
        rows,
        key=lambda row: (
            int(row["episode_index"]),
            int(row["frame_index"]),
            event_order.get(str(row["event_type"]), 2),
            str(row["current_subtask"]),
        ),
    )


def _dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop duplicate labels for the same subtask at the same frame.

    Dense progress fractions can round onto the success frame, especially for
    short segments. Keep the strongest event label for that subtask/frame.
    """

    event_priority = {"progress": 0, "subtask_boundary": 1, "success": 2}
    deduped: dict[tuple[int, int, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row["episode_index"]), int(row["frame_index"]), str(row["current_subtask"]))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue
        current_priority = event_priority.get(str(row["event_type"]), 0)
        existing_priority = event_priority.get(str(existing["event_type"]), 0)
        if current_priority > existing_priority:
            deduped[key] = row
    return _sort_rows(list(deduped.values()))


if __name__ == "__main__":
    main()
