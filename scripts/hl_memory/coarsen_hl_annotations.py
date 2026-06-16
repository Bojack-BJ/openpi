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
DEFAULT_SHORT_RUN_MERGE_MAX_FRAMES = 20


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
            short_run_merge_max_frames=args.short_run_merge_max_frames,
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
    parser.add_argument(
        "--short-run-merge-max-frames",
        type=int,
        default=DEFAULT_SHORT_RUN_MERGE_MAX_FRAMES,
        help=(
            "After row-level coarsening, merge compatible adjacent current-objective runs whose sampled frame span "
            "is <= this threshold. Set to 0 to disable."
        ),
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
    short_run_merge_max_frames: int,
) -> dict[str, Any]:
    rows = load_jsonl(input_jsonl)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("episode_index", 0))].append(row)

    processed: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    changed_current = 0
    changed_horizon = 0
    short_run_merge_counts: Counter[str] = Counter()
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda row: (int(row.get("frame_index", 0)), str(row.get("event_type", ""))))
        episode_processed: list[dict[str, Any]] = []
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
            episode_processed.append(new_row)

        if short_run_merge_max_frames > 0:
            merge_count = merge_short_current_objective_runs(
                episode_processed,
                max_span_frames=short_run_merge_max_frames,
                replace_target_fields=replace_target_fields,
            )
            short_run_merge_counts.update(merge_count)
        processed.extend(episode_processed)

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
        "short_run_merge_max_frames": short_run_merge_max_frames,
        "short_run_merge_counts": dict(short_run_merge_counts.most_common()),
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
            if (
                candidate_type == "operate_articulated_object"
                and _mentions_articulated_target(text)
                and _object_overlap(text, candidate)
            ):
                return {
                    "action_type": "operate_articulated_object",
                    "objective": normalize_objective_text(candidate, action_type=candidate_type),
                    "merge_reason": "articulated_target_lookahead",
                    "source_objective": text,
                }
        if action_type == "approach_object":
            for candidate in lookahead:
                candidate_type = classify_action(candidate)
                if candidate_type == "grasp_object" and _object_overlap(text, candidate):
                    return {
                        "action_type": "acquire_object",
                        "objective": normalize_objective_text(candidate, action_type="acquire_object"),
                        "merge_reason": "approach_to_acquire_lookahead",
                        "source_objective": text,
                    }
    if merge_mode == "aggressive" and action_type in {"grasp_object", "transport_object"}:
        for candidate in lookahead:
            candidate_type = classify_action(candidate)
            if candidate_type == "place_object" and _object_overlap(text, candidate):
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
    if any(token in normalized for token in ("grasp", "grab", "grip", "pick up", "hold", "remove", "take off", "take out")):
        return "grasp_object"
    if any(token in normalized for token in ("approach", "move to right end", "move to left end")):
        return "approach_object"
    if any(token in normalized for token in ("move", "carry", "bring", "transport", "align", "lift", "raise", "lower")):
        return "transport_object"
    if any(token in normalized for token in ("verify", "check", "observe", "wait")):
        return "wait_or_verify"
    return "other"


def normalize_objective_text(text: str, *, action_type: str) -> str:
    stripped = text.strip().replace("_", " ")
    stripped = re.sub(r"\s+", " ", stripped).rstrip(".")
    if not stripped:
        return "continue the observed manipulation step"
    if stripped.lower().startswith("the "):
        stripped = stripped[4:]
    if action_type == "approach_object" and "approach" not in stripped.lower():
        return stripped
    return stripped[0].upper() + stripped[1:]


def normalize_text(text: str) -> str:
    normalized = text.strip().lower().replace("_", " ").rstrip(".")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("the ", "")
    return normalized


def _mentions_articulated_target(text: str) -> bool:
    normalized = normalize_text(text)
    return any(token in normalized for token in ("door", "drawer", "cabinet", "handle", "lid", "cap", "button", "switch"))


def merge_short_current_objective_runs(
    rows: list[dict[str, Any]],
    *,
    max_span_frames: int,
    replace_target_fields: bool,
) -> Counter[str]:
    """Merge very short compatible runs into adjacent runs.

    This is a second-stage cleanup after per-row semantic normalization. It is
    intentionally conservative: reset/return runs are kept, and a short run only
    merges when adjacent action/object semantics are compatible.
    """

    counts: Counter[str] = Counter()
    while True:
        runs = _current_objective_runs(rows)
        merged_any = False
        for run_index, run in enumerate(runs):
            if run["span_frames"] > max_span_frames:
                continue
            if run["action_type"] in {"reset_hand", "post_action_reset", "wait_or_verify"}:
                continue
            target = _choose_short_run_merge_target(runs, run_index)
            if target is None:
                continue
            target_run = runs[target]
            merged_objective = _short_run_merged_objective(run, target_run)
            reason = f"short_run_merged_into_{'next' if target > run_index else 'previous'}"
            for row_index in range(run["start_row"], run["end_row"] + 1):
                _rewrite_current_objective(
                    rows[row_index],
                    objective=merged_objective,
                    action_type=target_run["action_type"],
                    reason=reason,
                    source_objective=run["objective"],
                    replace_target_fields=replace_target_fields,
                )
            if merged_objective != target_run["objective"]:
                for row_index in range(target_run["start_row"], target_run["end_row"] + 1):
                    _rewrite_current_objective(
                        rows[row_index],
                        objective=merged_objective,
                        action_type=target_run["action_type"],
                        reason="short_run_generalized_with_neighbor",
                        source_objective=target_run["objective"],
                        replace_target_fields=replace_target_fields,
                    )
            counts[reason] += 1
            merged_any = True
            break
        if not merged_any:
            break
    return counts


def _current_objective_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row_index, row in enumerate(rows):
        objective = str(row.get("coarse_current_objective") or row.get("current_objective") or row.get("current_subtask") or "")
        key = normalize_text(objective)
        if current is None or current["key"] != key:
            if current is not None:
                _finish_run(current)
                runs.append(current)
            current = {
                "key": key,
                "objective": objective,
                "action_type": str(row.get("coarse_action_type") or classify_action(objective)),
                "start_row": row_index,
                "end_row": row_index,
                "frames": [int(row.get("frame_index", 0))],
                "source_objectives": _row_source_objectives(row),
            }
        else:
            current["end_row"] = row_index
            current["frames"].append(int(row.get("frame_index", 0)))
            current["source_objectives"].extend(_row_source_objectives(row))
    if current is not None:
        _finish_run(current)
        runs.append(current)
    return runs


def _finish_run(run: dict[str, Any]) -> None:
    frames = run["frames"]
    run["start_frame"] = min(frames)
    run["end_frame"] = max(frames)
    run["span_frames"] = max(frames) - min(frames)


def _choose_short_run_merge_target(runs: list[dict[str, Any]], run_index: int) -> int | None:
    candidates: list[tuple[int, int, int]] = []
    current = runs[run_index]
    for neighbor_index in (run_index + 1, run_index - 1):
        if neighbor_index < 0 or neighbor_index >= len(runs):
            continue
        neighbor = runs[neighbor_index]
        if not _compatible_adjacent_runs(current, neighbor):
            continue
        # Prefer next run for preparatory/fragment rows, then the longer neighbor,
        # then the richer natural-language objective.
        direction_score = 1 if neighbor_index > run_index else 0
        rich_score = len(_content_tokens(neighbor["objective"]))
        candidates.append((direction_score, int(neighbor["span_frames"]), rich_score, neighbor_index))
    if not candidates:
        return None
    return max(candidates)[-1]


def _short_run_merged_objective(short_run: dict[str, Any], target_run: dict[str, Any]) -> str:
    """Pick the train-facing label for a compatible short-run merge.

    When two adjacent runs have the same action semantics and equivalent object
    tokens, prefer the shorter wording. This converts labels such as
    "Fold packaging box sides" back to the more general "Fold packaging box"
    without dropping important goal/location tokens from place/transport rows.
    """

    short_objective = str(short_run["objective"])
    target_objective = str(target_run["objective"])
    if short_run["action_type"] != target_run["action_type"]:
        return target_objective
    short_tokens = _content_tokens(short_objective)
    target_tokens = _content_tokens(target_objective)
    if not short_tokens or short_tokens != target_tokens:
        return target_objective
    return min(
        (short_objective, target_objective),
        key=lambda text: (len(_surface_tokens(text)), len(text)),
    )


def _compatible_adjacent_runs(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_type = left["action_type"]
    right_type = right["action_type"]
    if left_type in {"reset_hand", "post_action_reset", "wait_or_verify"}:
        return False
    if right_type in {"reset_hand", "post_action_reset", "wait_or_verify"}:
        return False
    if _has_direction_conflict(left["objective"], right["objective"]):
        return False
    if not _object_overlap(left["objective"], right["objective"]):
        return False
    if _is_delicate_directional_operation(left["objective"]) or _is_delicate_directional_operation(right["objective"]):
        return False
    if left_type in {"grasp_object", "acquire_object", "place_object"} and _is_preparatory_objective(
        right["objective"], right_type
    ):
        return False
    if _run_has_source_action(left, {"grasp_object", "acquire_object", "place_object"}) and _is_preparatory_objective(
        right["objective"], right_type
    ):
        return False
    if {
        left_type,
        right_type,
    }.intersection({"approach_object", "grasp_object"}) and "operate_articulated_object" in {left_type, right_type}:
        return _mentions_articulated_target(left["objective"]) or _mentions_articulated_target(right["objective"])
    if left_type == right_type:
        return True
    return (left_type, right_type) in {
        ("approach_object", "grasp_object"),
        ("approach_object", "acquire_object"),
        ("grasp_object", "acquire_object"),
        ("approach_object", "operate_articulated_object"),
        ("grasp_object", "operate_articulated_object"),
        ("transport_object", "place_object"),
        ("grasp_object", "transport_object"),
    }


def _row_source_objectives(row: dict[str, Any]) -> list[str]:
    values = [
        row.get("fine_current_objective"),
        row.get("fine_current_subtask"),
        row.get("coarse_source_objective"),
        row.get("current_objective"),
        row.get("current_subtask"),
    ]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_text(text)
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _run_has_source_action(run: dict[str, Any], action_types: set[str]) -> bool:
    return any(classify_action(text) in action_types for text in run.get("source_objectives", []))


def _has_direction_conflict(left: str, right: str) -> bool:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    left_clockwise = "clockwise" in left_norm and "counterclockwise" not in left_norm and "anticlockwise" not in left_norm
    right_clockwise = (
        "clockwise" in right_norm and "counterclockwise" not in right_norm and "anticlockwise" not in right_norm
    )
    left_counter = "counterclockwise" in left_norm or "anticlockwise" in left_norm
    right_counter = "counterclockwise" in right_norm or "anticlockwise" in right_norm
    return (left_clockwise and right_counter) or (left_counter and right_clockwise)


def _is_preparatory_objective(objective: str, action_type: str) -> bool:
    normalized = normalize_text(objective)
    if "move right gripper" in normalized or "move left gripper" in normalized:
        return True
    if action_type == "approach_object":
        return True
    return action_type == "transport_object" and "move to" in normalized


def _is_delicate_directional_operation(objective: str) -> bool:
    normalized = normalize_text(objective)
    return any(token in normalized for token in ("rotate", "twist", "clockwise", "counterclockwise", "anticlockwise", "cap", "bottle"))


def _rewrite_current_objective(
    row: dict[str, Any],
    *,
    objective: str,
    action_type: str,
    reason: str,
    source_objective: str,
    replace_target_fields: bool,
) -> None:
    row["coarse_action_type"] = action_type
    row["coarse_current_objective"] = objective
    row["coarse_current_subtask"] = objective
    row["coarse_phase"] = objective
    row["coarse_merge_reason"] = reason
    row["coarse_source_objective"] = source_objective
    if replace_target_fields:
        row["current_objective"] = objective
        row["current_subtask"] = objective
        row["phase"] = objective


def _object_overlap(left: str, right: str) -> bool:
    left_tokens = _object_tokens(left)
    right_tokens = _object_tokens(right)
    return bool(left_tokens and right_tokens and left_tokens.intersection(right_tokens))


def _object_tokens(text: str) -> set[str]:
    return _content_tokens(text)


def _content_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {token for token in re.findall(r"[a-z0-9]+", normalized) if token not in _STOPWORDS and len(token) > 1}


def _surface_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


_STOPWORDS = {
    "a",
    "an",
    "and",
    "both",
    "current",
    "down",
    "for",
    "from",
    "hand",
    "hands",
    "in",
    "into",
    "left",
    "middle",
    "of",
    "on",
    "onto",
    "position",
    "right",
    "side",
    "sides",
    "target",
    "to",
    "up",
    "with",
    # Verbs / phase words.
    "approach",
    "approaches",
    "bring",
    "carry",
    "close",
    "cover",
    "fold",
    "folds",
    "grasp",
    "grasps",
    "hold",
    "insert",
    "lift",
    "move",
    "moves",
    "open",
    "pick",
    "place",
    "places",
    "pull",
    "push",
    "release",
    "return",
    "returns",
    "set",
    "stack",
    "transport",
}


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
