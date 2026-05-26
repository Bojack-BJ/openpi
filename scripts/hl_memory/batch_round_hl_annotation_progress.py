#!/usr/bin/env python3
"""Round subtask_progress values in HL annotation JSONL files."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any


DEFAULT_ANNOTATION_ROOT = pathlib.Path("/root/Users/dataset/lerobot_home/subtask")
DEFAULT_INPUT_NAME = "hl_annotations_llm_normalized.jsonl"
DEFAULT_OUTPUT_NAME = "hl_annotations_llm_normalized.jsonl"


@dataclass(frozen=True)
class RoundResult:
    task_id: str
    input_jsonl: str
    output_jsonl: str
    rows: int
    changed_rows: int
    skipped: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round top-level and llm_gt.subtask_progress values for every task under an annotation root. "
            "By default this rewrites hl_annotations_llm_normalized.jsonl in-place."
        )
    )
    parser.add_argument("--annotation-root", "--subtask-root", dest="annotation_root", type=pathlib.Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=None)
    parser.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--task-id-glob", default="*")
    parser.add_argument("--only-task-id", action="append", nargs="+", default=[])
    parser.add_argument("--quantum", type=float, default=0.05, help="Round to nearest quantum in [0, 1]. Use 0 to only clamp.")
    parser.add_argument(
        "--advance-threshold",
        type=float,
        default=None,
        help="If set, recompute should_advance_objective from rounded progress unless event_type is success.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_root = args.annotation_root.expanduser().resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"--annotation-root is not a directory: {annotation_root}")
    if args.quantum < 0.0:
        raise ValueError("--quantum must be non-negative.")

    results: list[RoundResult] = []
    for task_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir()):
        task_id = task_dir.name
        if args.task_id_glob != "*" and not task_dir.match(args.task_id_glob):
            continue
        allowed = flatten_only_task_ids(args.only_task_id)
        if allowed and task_id not in allowed:
            continue
        input_jsonl = task_dir / args.input_name
        if not input_jsonl.is_file():
            continue
        output_jsonl = resolve_output_path(args, task_id=task_id, task_dir=task_dir)
        result = round_file(input_jsonl, output_jsonl, args=args)
        results.append(result)
        status = "DRY" if args.dry_run else "OK"
        print(
            f"[{status}] {task_id} rows={result.rows} changed={result.changed_rows} -> {result.output_jsonl}",
            flush=True,
        )

    if not results:
        raise FileNotFoundError(f"No task dirs with {args.input_name} under {annotation_root}")
    summary_path = args.summary_json or resolve_default_summary_path(args, annotation_root)
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps([result.__dict__ for result in results], indent=2) + "\n", encoding="utf-8")
    print(f"[Done] tasks={len(results)} changed_rows={sum(result.changed_rows for result in results)} summary={summary_path}")


def round_file(input_jsonl: pathlib.Path, output_jsonl: pathlib.Path, *, args: argparse.Namespace) -> RoundResult:
    rows = read_jsonl(input_jsonl)
    changed = 0
    rounded_rows: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        if round_row_progress(output, quantum=float(args.quantum), advance_threshold=args.advance_threshold):
            changed += 1
        rounded_rows.append(output)

    if not args.dry_run:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if input_jsonl.resolve() == output_jsonl.resolve():
            write_jsonl_atomic(output_jsonl, rounded_rows)
        else:
            write_jsonl(output_jsonl, rounded_rows)
    return RoundResult(
        task_id=input_jsonl.parent.name,
        input_jsonl=str(input_jsonl),
        output_jsonl=str(output_jsonl),
        rows=len(rows),
        changed_rows=changed,
    )


def round_row_progress(row: dict[str, Any], *, quantum: float, advance_threshold: float | None) -> bool:
    changed = False
    llm_gt = row.get("llm_gt")
    rounded = quantize_optional_progress(row.get("subtask_progress"), quantum)
    if rounded is None and isinstance(llm_gt, dict):
        rounded = quantize_optional_progress(llm_gt.get("subtask_progress"), quantum)
    if rounded is not None and row.get("subtask_progress") != rounded:
        row["subtask_progress"] = rounded
        changed = True
    if isinstance(llm_gt, dict):
        if rounded is not None and llm_gt.get("subtask_progress") != rounded:
            llm_gt["subtask_progress"] = rounded
            changed = True
    if advance_threshold is not None:
        if rounded is not None:
            should_advance = bool(str(row.get("event_type", "")) == "success" or rounded >= float(advance_threshold))
            if row.get("should_advance_objective") != should_advance:
                row["should_advance_objective"] = should_advance
                changed = True
            if isinstance(llm_gt, dict) and llm_gt.get("should_advance_objective") != should_advance:
                llm_gt["should_advance_objective"] = should_advance
                changed = True
    return changed


def quantize_optional_progress(value: Any, quantum: float) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    clipped = min(max(numeric, 0.0), 1.0)
    if quantum <= 0.0:
        return float(clipped)
    quantized = round(clipped / quantum) * quantum
    return float(round(min(max(quantized, 0.0), 1.0), 6))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl_atomic(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        tmp_path = pathlib.Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_output_path(args: argparse.Namespace, *, task_id: str, task_dir: pathlib.Path) -> pathlib.Path:
    if args.output_root is not None:
        return (args.output_root / task_id / args.output_name).expanduser().resolve()
    return (task_dir / args.output_name).resolve()


def resolve_default_summary_path(args: argparse.Namespace, annotation_root: pathlib.Path) -> pathlib.Path:
    if args.output_root is not None:
        return (args.output_root / "batch_round_hl_annotation_progress_summary.json").expanduser().resolve()
    return annotation_root / "batch_round_hl_annotation_progress_summary.json"


def flatten_only_task_ids(values: list[list[str]]) -> set[str]:
    return {item for group in values for item in group}


if __name__ == "__main__":
    main()
