#!/usr/bin/env python3
"""Create coarse HL annotation JSONL files from fine normalized annotations.

The script keeps the original rows intact by default, stores fine labels under
``fine_*`` keys, and replaces train-facing objective fields with coarser labels.
This lets existing HL dataset export and target protocols train on coarser
objectives without changing the core samples schema.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_INPUT_NAME = "hl_annotations_llm_normalized.jsonl"
DEFAULT_OUTPUT_NAME = "hl_annotations_llm_normalized_coarse.jsonl"
MERGE_MODES = ("conservative", "aggressive")


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(args)
    if not jobs:
        raise FileNotFoundError("No input JSONL files found.")

    summaries: list[dict[str, Any]] = []
    for input_jsonl, output_jsonl in jobs:
        if output_jsonl.exists() and not args.overwrite:
            print(f"[Skip] {output_jsonl} exists; pass --overwrite to rewrite.")
            continue
        summary = coarsen_file(
            input_jsonl,
            output_jsonl,
            replace_target_fields=args.replace_target_fields,
            preserve_fine_fields=args.preserve_fine_fields,
            merge_return_into_previous=args.merge_return_into_previous,
            merge_mode=args.merge_mode,
            lookahead_window=args.lookahead_window,
        )
        summaries.append(summary)
        print(
            f"[OK] {input_jsonl} -> {output_jsonl} rows={summary['rows']} "
            f"changed_current={summary['changed_current_objective']} changed_horizon={summary['changed_horizon_current_objective']}"
        )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-jsonl", type=Path, help="Single annotation JSONL to coarsen.")
    group.add_argument("--annotation-root", "--subtask-root", dest="annotation_root", type=Path)
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Output path for single-file mode.")
    parser.add_argument("--input-name", default=DEFAULT_INPUT_NAME, help="Input filename in each task dir.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Output filename in each task dir.")
    parser.add_argument("--task-id-glob", default="*", help="Task directory glob for --annotation-root mode.")
    parser.add_argument("--only-task-id", action="append", nargs="+", default=[])
    parser.add_argument(
        "--replace-target-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace current_objective/current_subtask/horizon_* with coarse labels for direct training.",
    )
    parser.add_argument(
        "--preserve-fine-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write fine_current_objective/fine_current_subtask/fine_horizon_* before replacing fields.",
    )
    parser.add_argument(
        "--merge-return-into-previous",
        action="store_true",
        help=(
            "Attach return/retreat rows to the previous meaningful objective. Off by default because return/reset "
            "is still an executable low-level objective."
        ),
    )
    parser.add_argument(
        "--merge-mode",
        choices=MERGE_MODES,
        default="conservative",
        help=(
            "Lookahead merge policy. conservative merges approach->grasp and handle pregrasp->articulation. "
            "aggressive additionally merges grasp/transport rows toward an immediate place/release objective."
        ),
    )
    parser.add_argument(
        "--lookahead-window",
        type=int,
        default=4,
        help="Number of future distinct objectives considered by rule-based lookahead merges.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def resolve_jobs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.input_jsonl is not None:
        input_jsonl = args.input_jsonl.expanduser().resolve()
        output_jsonl = (
            args.output_jsonl.expanduser().resolve()
            if args.output_jsonl is not None
            else input_jsonl.with_name(DEFAULT_OUTPUT_NAME)
        )
        return [(input_jsonl, output_jsonl)]

    root = args.annotation_root.expanduser().resolve()
    allowed = {task_id for group in args.only_task_id for task_id in group}
    jobs: list[tuple[Path, Path]] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or not task_dir.match(args.task_id_glob):
            continue
        if allowed and task_dir.name not in allowed:
            continue
        input_jsonl = task_dir / args.input_name
        if input_jsonl.is_file():
            jobs.append((input_jsonl, task_dir / args.output_name))
    return jobs


def coarsen_file(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    replace_target_fields: bool,
    preserve_fine_fields: bool,
    merge_return_into_previous: bool,
    merge_mode: str,
    lookahead_window: int,
) -> dict[str, Any]:
    rows = load_jsonl(input_jsonl)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("episode_index", 0))].append(row)

    processed: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    changed_current = 0
    changed_horizon = 0
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda row: (int(row.get("frame_index", 0)), str(row.get("event_type", ""))))
        for index, row in enumerate(episode_rows):
            current = coarsen_row(
                row,
                rows=episode_rows,
                index=index,
                field="current_objective",
                fallback_field="current_subtask",
                merge_return_into_previous=merge_return_into_previous,
                allow_lookahead_merge=True,
                merge_mode=merge_mode,
                lookahead_window=lookahead_window,
            )
            horizon = coarsen_row(
                row,
                rows=episode_rows,
                index=index,
                field="horizon_current_objective",
                fallback_field="horizon_current_subtask",
                merge_return_into_previous=merge_return_into_previous,
                allow_lookahead_merge=False,
                merge_mode=merge_mode,
                lookahead_window=lookahead_window,
            )
            new_row = dict(row)
            if preserve_fine_fields:
                _preserve_fine_fields(new_row)
            new_row["coarse_merge_mode"] = merge_mode
            new_row["coarse_action_type"] = current["action_type"]
            new_row["coarse_current_objective"] = current["objective"]
            new_row["coarse_current_subtask"] = current["objective"]
            new_row["coarse_phase"] = current["objective"]
            new_row["coarse_merge_reason"] = current["merge_reason"]
            new_row["coarse_source_objective"] = current["source_objective"]
            new_row["coarse_horizon_current_objective"] = horizon["objective"]
            new_row["coarse_horizon_current_subtask"] = horizon["objective"]
            new_row["coarse_horizon_phase"] = horizon["objective"]
            new_row["coarse_horizon_action_type"] = horizon["action_type"]
            new_row["coarse_horizon_merge_reason"] = horizon["merge_reason"]
            new_row["coarse_horizon_source_objective"] = horizon["source_objective"]
            if replace_target_fields:
                old_current = str(new_row.get("current_objective") or new_row.get("current_subtask") or "").strip()
                old_horizon = str(
                    new_row.get("horizon_current_objective") or new_row.get("horizon_current_subtask") or ""
                ).strip()
                new_row["current_objective"] = current["objective"]
                new_row["current_subtask"] = current["objective"]
                new_row["phase"] = current["objective"]
                if "horizon_current_objective" in new_row or "horizon_current_subtask" in new_row:
                    new_row["horizon_current_objective"] = horizon["objective"]
                    new_row["horizon_current_subtask"] = horizon["objective"]
                    new_row["horizon_phase"] = horizon["objective"]
                changed_current += int(normalize_text(old_current) != normalize_text(current["objective"]))
                changed_horizon += int(bool(old_horizon) and normalize_text(old_horizon) != normalize_text(horizon["objective"]))
            action_counts[current["action_type"]] += 1
            processed.append(new_row)

    processed.sort(key=lambda row: (int(row.get("episode_index", 0)), int(row.get("frame_index", 0))))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, processed)
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "rows": len(processed),
        "changed_current_objective": changed_current,
        "changed_horizon_current_objective": changed_horizon,
        "merge_mode": merge_mode,
        "lookahead_window": lookahead_window,
        "action_counts": dict(action_counts.most_common()),
    }


def _preserve_fine_fields(row: dict[str, Any]) -> None:
    for key in (
        "current_objective",
        "current_subtask",
        "phase",
        "horizon_current_objective",
        "horizon_current_subtask",
        "horizon_phase",
    ):
        if key in row and f"fine_{key}" not in row:
            row[f"fine_{key}"] = row[key]


def coarsen_row(
    row: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    index: int,
    field: str,
    fallback_field: str,
    merge_return_into_previous: bool,
    allow_lookahead_merge: bool,
    merge_mode: str,
    lookahead_window: int,
) -> dict[str, str]:
    text = str(row.get(field) or row.get(fallback_field) or row.get("current_subtask") or "").strip()
    if not text:
        text = "continue the observed manipulation step"
    action_type = classify_action(text)

    if action_type == "reset_hand" and merge_return_into_previous:
        previous_meaningful = _find_previous_meaningful(rows, index)
        if previous_meaningful:
            return {
                "action_type": "post_action_reset",
                "objective": f"Finish and stabilize after: {previous_meaningful}",
                "merge_reason": "reset_merged_into_previous",
                "source_objective": text,
            }

    if allow_lookahead_merge and action_type in {"approach_object", "grasp_object", "transport_object"}:
        merged = _lookahead_merge_objective(
            text,
            action_type=action_type,
            rows=rows,
            index=index,
            merge_mode=merge_mode,
            lookahead_window=lookahead_window,
        )
        if merged is not None:
            return merged

    return {
        "action_type": action_type,
        "objective": normalize_objective_text(text, action_type=action_type),
        "merge_reason": "classified_without_merge",
        "source_objective": text,
    }


def _lookahead_merge_objective(
    text: str,
    *,
    action_type: str,
    rows: list[dict[str, Any]],
    index: int,
    merge_mode: str,
    lookahead_window: int,
) -> dict[str, str] | None:
    lookahead = _next_distinct_objectives(rows, index, limit=lookahead_window)
    if action_type in {"approach_object", "grasp_object"}:
        for candidate in lookahead:
            candidate_type = classify_action(candidate)
            if candidate_type == "operate_articulated_object" and _mentions_articulated_target(text):
                return {
                    "action_type": "operate_articulated_object",
                    "objective": normalize_objective_text(candidate, action_type=candidate_type),
                    "merge_reason": "articulated_target_lookahead",
                    "source_objective": text,
                }
        if action_type == "approach_object":
            for candidate in lookahead:
                candidate_type = classify_action(candidate)
                if candidate_type == "grasp_object":
                    return {
                        "action_type": "acquire_object",
                        "objective": normalize_objective_text(candidate, action_type="acquire_object"),
                        "merge_reason": "approach_to_acquire_lookahead",
                        "source_objective": text,
                    }
    if merge_mode == "aggressive" and action_type in {"grasp_object", "transport_object"}:
        for candidate in lookahead:
            candidate_type = classify_action(candidate)
            if candidate_type == "place_object":
                return {
                    "action_type": "place_object",
                    "objective": normalize_objective_text(candidate, action_type=candidate_type),
                    "merge_reason": "grasp_or_transport_to_place_lookahead",
                    "source_objective": text,
                }
    return None


def _next_distinct_objectives(rows: list[dict[str, Any]], index: int, *, limit: int) -> list[str]:
    current = _row_objective(rows[index])
    result: list[str] = []
    seen = {normalize_text(current)}
    for next_row in rows[index + 1 :]:
        text = _row_objective(next_row)
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _find_previous_meaningful(rows: list[dict[str, Any]], index: int) -> str:
    for previous in reversed(rows[:index]):
        text = _row_objective(previous)
        action_type = classify_action(text)
        if action_type not in {"reset_hand", "other"}:
            return normalize_objective_text(text, action_type=action_type)
    return ""


def _row_objective(row: dict[str, Any]) -> str:
    return str(row.get("current_objective") or row.get("current_subtask") or row.get("phase") or "").strip()


def classify_action(text: str) -> str:
    normalized = normalize_text(text)
    if any(token in normalized for token in ("move back", "return to", "returns to", "observation region", "observation position", "initial position", "initial region", "retract")):
        return "reset_hand"
    if any(token in normalized for token in ("open", "close", "press", "pull", "push", "turn", "rotate", "twist", "fold", "unfold")):
        return "operate_articulated_object"
    if any(token in normalized for token in ("place", "put ", "insert", "stack", "drop", "release", "hang", "set ")):
        return "place_object"
    if any(token in normalized for token in ("handover", "hand over", "transfer")):
        return "handover_or_transfer"
    if any(token in normalized for token in ("move", "carry", "bring", "transport", "align", "lift", "raise", "lower")):
        return "transport_object"
    if any(token in normalized for token in ("grasp", "grab", "pick up", "hold", "take off", "take out")):
        return "grasp_object"
    if any(token in normalized for token in ("approach", "move to right end", "move to left end")):
        return "approach_object"
    if any(token in normalized for token in ("verify", "check", "observe", "wait")):
        return "wait_or_verify"
    return "other"


def normalize_objective_text(text: str, *, action_type: str) -> str:
    stripped = re.sub(r"\s+", " ", text.strip()).rstrip(".")
    if not stripped:
        return "continue the observed manipulation step"
    if stripped.lower().startswith("the "):
        stripped = stripped[4:]
    if action_type == "approach_object" and "approach" not in stripped.lower():
        return stripped
    return stripped[0].upper() + stripped[1:]


def normalize_text(text: str) -> str:
    normalized = text.strip().lower().rstrip(".")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("the ", "")
    return normalized


def _mentions_articulated_target(text: str) -> bool:
    normalized = normalize_text(text)
    return any(token in normalized for token in ("door", "drawer", "cabinet", "handle", "lid", "cap", "button", "switch"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
