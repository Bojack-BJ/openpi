#!/usr/bin/env python3
"""Audit raw session subtask.json sequence variation.

This checks whether sessions under the same task id share the same ordered
subtask text sequence. Frame timing differences are ignored.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_EXCLUDE_PARTS = (".sam3_mask_stage_work",)


def main() -> None:
    args = parse_args()
    rows = collect_sessions(
        args.root.expanduser().resolve(),
        task_id_depth=args.task_id_depth,
        exclude_parts=tuple(args.exclude_path_part),
    )
    summaries = summarize(rows)
    report = {
        "root": str(args.root),
        "num_sessions": len(rows),
        "num_tasks": len(summaries),
        "num_tasks_with_sequence_variation": sum(1 for row in summaries if row["normalized_sequence_variants"] > 1),
        "num_tasks_with_segment_count_variation": sum(1 for row in summaries if row["segment_count_variants"] > 1),
        "tasks": summaries,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print_human_summary(report, max_examples=args.max_examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Raw segmentation root, e.g. /root/Users/segmentation_data_dtw.")
    parser.add_argument(
        "--task-id-depth",
        type=int,
        default=1,
        help="Use the first N path parts under --root as the grouping key. Default groups by <task_id>.",
    )
    parser.add_argument(
        "--exclude-path-part",
        action="append",
        default=list(DEFAULT_EXCLUDE_PARTS),
        help="Skip any subtask.json whose relative path contains this component. Can be repeated.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=12)
    return parser.parse_args()


def collect_sessions(root: Path, *, task_id_depth: int, exclude_parts: tuple[str, ...]) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("subtask.json")):
        rel = path.relative_to(root)
        if any(part in exclude_parts for part in rel.parts):
            continue
        if len(rel.parts) <= task_id_depth:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {path}") from exc
        segments = data.get("segments")
        if not isinstance(segments, list):
            continue
        texts = [str(seg.get("subtask", "")).strip() for seg in segments if str(seg.get("subtask", "")).strip()]
        if not texts:
            continue
        starts = [seg.get("start_frame") for seg in segments]
        ends = [seg.get("end_frame") for seg in segments]
        rows.append(
            {
                "task_key": "/".join(rel.parts[:task_id_depth]),
                "session_path": str(path.parent),
                "subtask_json": str(path),
                "segment_count": len(texts),
                "sequence": texts,
                "normalized_sequence": [normalize_subtask_text(text) for text in texts],
                "start_frames": starts,
                "end_frames": ends,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_key"]].append(row)

    summaries: list[dict[str, Any]] = []
    for task_key, sessions in sorted(grouped.items()):
        sequence_counter: Counter[tuple[str, ...]] = Counter(tuple(s["normalized_sequence"]) for s in sessions)
        count_counter: Counter[int] = Counter(int(s["segment_count"]) for s in sessions)
        canonical_sequence, canonical_count = sequence_counter.most_common(1)[0]
        variants = []
        for sequence, count in sequence_counter.most_common():
            examples = [
                {
                    "session_path": s["session_path"],
                    "segment_count": s["segment_count"],
                    "sequence": s["sequence"],
                    "start_frames": s["start_frames"],
                    "end_frames": s["end_frames"],
                }
                for s in sessions
                if tuple(s["normalized_sequence"]) == sequence
            ][:3]
            variants.append(
                {
                    "count": count,
                    "fraction": count / len(sessions),
                    "normalized_sequence": list(sequence),
                    "examples": examples,
                }
            )
        noncanonical_sessions = [
            s
            for s in sessions
            if tuple(s["normalized_sequence"]) != canonical_sequence
        ]
        summaries.append(
            {
                "task_key": task_key,
                "num_sessions": len(sessions),
                "segment_count_variants": len(count_counter),
                "segment_count_distribution": dict(sorted(count_counter.items())),
                "normalized_sequence_variants": len(sequence_counter),
                "canonical_sequence_session_count": canonical_count,
                "canonical_sequence_fraction": canonical_count / len(sessions),
                "canonical_normalized_sequence": list(canonical_sequence),
                "num_noncanonical_sessions": len(noncanonical_sessions),
                "noncanonical_session_paths": [s["session_path"] for s in noncanonical_sessions[:10]],
                "variants": variants,
            }
        )
    summaries.sort(
        key=lambda item: (
            item["normalized_sequence_variants"] <= 1,
            -item["normalized_sequence_variants"],
            item["canonical_sequence_fraction"],
            item["task_key"],
        )
    )
    return summaries


def normalize_subtask_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\bthe\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[.。]+$", "", normalized)
    return normalized.strip()


def print_human_summary(report: dict[str, Any], *, max_examples: int) -> None:
    print(f"root: {report['root']}")
    print(f"sessions: {report['num_sessions']}")
    print(f"tasks: {report['num_tasks']}")
    print(f"tasks_with_sequence_variation: {report['num_tasks_with_sequence_variation']}")
    print(f"tasks_with_segment_count_variation: {report['num_tasks_with_segment_count_variation']}")
    print()
    print("Top sequence-variation tasks:")
    shown = 0
    for task in report["tasks"]:
        if task["normalized_sequence_variants"] <= 1:
            continue
        shown += 1
        print(
            f"- {task['task_key']}: sessions={task['num_sessions']} "
            f"seq_variants={task['normalized_sequence_variants']} "
            f"count_variants={task['segment_count_distribution']} "
            f"canonical={task['canonical_sequence_session_count']}/{task['num_sessions']}"
        )
        for variant_index, variant in enumerate(task["variants"][:3], start=1):
            print(
                f"  variant {variant_index}: count={variant['count']} "
                f"fraction={variant['fraction']:.3f} len={len(variant['normalized_sequence'])}"
            )
            print("    " + " | ".join(variant["normalized_sequence"][:10]))
            if len(variant["normalized_sequence"]) > 10:
                print("    ...")
            for example in variant["examples"][:1]:
                print(f"    example={example['session_path']}")
        if shown >= max_examples:
            break
    if shown == 0:
        print("- none")


if __name__ == "__main__":
    main()
