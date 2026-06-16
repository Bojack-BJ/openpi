from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize subtask segmentation text patterns across task directories.")
    parser.add_argument("--subtask-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = summarize(args.subtask_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(_compact_report(result), ensure_ascii=False, indent=2))


def summarize(root: Path) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    transition_counts: Counter[tuple[str, str]] = Counter()
    adjacent_merge: Counter[tuple[str, str]] = Counter()
    task_action_patterns: Counter[str] = Counter()
    lengths: list[int] = []
    episode_segment_counts: list[int] = []
    task_summaries: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []

    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        path = task_dir / "subtask_segments.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        episodes = data.get("episodes", {}) if isinstance(data, dict) else {}
        if not isinstance(episodes, dict) or not episodes:
            continue

        first_episode_segments: list[dict[str, Any]] | None = None
        task_episode_counts: list[int] = []
        for episode_id, episode in sorted(episodes.items(), key=lambda item: _episode_sort_key(item[0])):
            segments = episode.get("segments", []) if isinstance(episode, dict) else []
            if not segments:
                continue
            if first_episode_segments is None:
                first_episode_segments = segments
            task_episode_counts.append(len(segments))
            episode_segment_counts.append(len(segments))
            previous_category: str | None = None
            categories: list[str] = []
            for index, segment in enumerate(segments):
                text = str(segment.get("subtask", "")).strip()
                if not text:
                    continue
                category = classify_subtask(text)
                categories.append(category)
                start = _safe_int(segment.get("start_frame"), default=0)
                end = _safe_int(segment.get("end_frame"), default=start)
                duration = max(0, end - start)
                lengths.append(duration)
                action_counts[category] += 1
                text_counts[normalize_text(text)] += 1
                all_segments.append(
                    {
                        "task_id": task_dir.name,
                        "episode": episode_id,
                        "index": index,
                        "text": text,
                        "category": category,
                        "duration": duration,
                    }
                )
                if previous_category is not None:
                    transition_counts[(previous_category, category)] += 1
                    if _is_likely_merge_pair(previous_category, category):
                        adjacent_merge[(previous_category, category)] += 1
                previous_category = category
            if categories:
                task_action_patterns[" > ".join(categories)] += 1

        if first_episode_segments is not None:
            first_texts = [str(segment.get("subtask", "")).strip() for segment in first_episode_segments]
            task_summaries.append(
                {
                    "task_id": task_dir.name,
                    "num_episodes": len(task_episode_counts),
                    "median_segments_per_episode": statistics.median(task_episode_counts) if task_episode_counts else 0,
                    "first_episode_segments": first_texts,
                    "first_episode_categories": [classify_subtask(text) for text in first_texts],
                }
            )

    return {
        "num_tasks": len(task_summaries),
        "num_segments": len(all_segments),
        "segment_duration_frames": _duration_summary(lengths),
        "segments_per_episode": _duration_summary(episode_segment_counts),
        "action_counts": action_counts.most_common(),
        "top_subtask_texts": text_counts.most_common(80),
        "top_transitions": [([left, right], count) for (left, right), count in transition_counts.most_common(50)],
        "likely_merge_pairs": [([left, right], count) for (left, right), count in adjacent_merge.most_common()],
        "top_task_category_patterns": task_action_patterns.most_common(30),
        "tasks": task_summaries,
    }


def classify_subtask(text: str) -> str:
    normalized = normalize_text(text)
    if any(token in normalized for token in ("move back", "moves back", "return to observation", "returns to observation", "observation region", "observation position", "retract")):
        return "return/retreat"
    if any(token in normalized for token in ("approach", "approaches", "move to right end", "move to left end")):
        return "approach/pregrasp"
    if any(token in normalized for token in ("grasp", "grasps", "grab", "grabs", "pick up", "picks up", "hold", "holds")):
        return "grasp/hold"
    if any(token in normalized for token in ("place", "places", "put ", "puts ", "insert", "inserts", "stack", "drop", "release", "hang", "set ")):
        return "place/release"
    if any(token in normalized for token in ("open", "opens", "close", "closes", "press", "pull", "push", "turn", "rotate", "fold", "unfold")):
        return "operate/articulate"
    if any(token in normalized for token in ("handover", "hand over", "transfer")):
        return "handover"
    if any(token in normalized for token in ("move", "moves", "carry", "bring", "transport", "align", "lift", "raise", "lower")):
        return "transport/align"
    return "other"


def normalize_text(text: str) -> str:
    normalized = text.strip().lower().rstrip(".")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("the ", "")
    return normalized


def _is_likely_merge_pair(left: str, right: str) -> bool:
    return (left, right) in {
        ("approach/pregrasp", "grasp/hold"),
        ("grasp/hold", "transport/align"),
        ("transport/align", "place/release"),
        ("place/release", "return/retreat"),
        ("operate/articulate", "return/retreat"),
    }


def _duration_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "p10": None, "median": 0, "p90": None, "max": 0}
    return {
        "min": min(values),
        "p10": _quantile(values, 0.1),
        "median": statistics.median(values),
        "p90": _quantile(values, 0.9),
        "max": max(values),
    }


def _quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    frac = index - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _compact_report(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_tasks": result["num_tasks"],
        "num_segments": result["num_segments"],
        "segment_duration_frames": result["segment_duration_frames"],
        "segments_per_episode": result["segments_per_episode"],
        "action_counts": result["action_counts"][:12],
        "likely_merge_pairs": result["likely_merge_pairs"],
        "top_task_category_patterns": result["top_task_category_patterns"][:10],
    }


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _episode_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


if __name__ == "__main__":
    main()
