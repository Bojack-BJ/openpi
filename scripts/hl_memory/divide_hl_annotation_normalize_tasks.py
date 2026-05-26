#!/usr/bin/env python3
"""Split HL annotation normalization tasks into balanced machine shards."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
from dataclasses import dataclass


DEFAULT_ANNOTATION_ROOT = pathlib.Path("/root/Users/dataset/lerobot_home/subtask")
DEFAULT_INPUT_NAME = "hl_annotations.jsonl"
DEFAULT_OUTPUT_NAME = "hl_annotations_llm_normalized.jsonl"
DEFAULT_OUTPUT_DIR = pathlib.Path("hl_annotation_normalize_shards")
DEFAULT_NORMALIZE_SCRIPT = pathlib.Path("scripts/hl_memory/batch_normalize_hl_annotations_with_llm.py")


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    task_dir: pathlib.Path
    input_jsonl: pathlib.Path
    output_jsonl: pathlib.Path
    input_rows: int
    output_rows: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a subtask/annotation root and split tasks into N balanced shards for "
            "scripts/hl_memory/batch_normalize_hl_annotations_with_llm.py. The output can be copied to "
            "multiple machines; each machine runs one shard's generated command."
        )
    )
    parser.add_argument("--annotation-root", "--subtask-root", dest="annotation_root", type=pathlib.Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shards", type=int, required=True, help="Number of machine shards to generate.")
    parser.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--normalize-output-root",
        type=pathlib.Path,
        default=None,
        help=(
            "The --output-root that will be passed to batch_normalize_hl_annotations_with_llm.py. "
            "Used only to resolve existing outputs for --skip-existing and to generate commands."
        ),
    )
    parser.add_argument("--task-id-glob", default="*")
    parser.add_argument("--only-task-id", action="append", nargs="+", default=[])
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Exclude tasks whose normalized output already has at least as many rows as the input.",
    )
    parser.add_argument("--normalize-script", type=pathlib.Path, default=DEFAULT_NORMALIZE_SCRIPT)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--env-prefix", default="PYTHONPATH=src")
    parser.add_argument("--dry-run", action="store_true", help="Print shard summary without writing files.")
    parser.add_argument(
        "normalize_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra args appended to each generated batch_normalize_hl_annotations_with_llm.py command after '--'. "
            "Do not include --annotation-root, --input-name, --output-name, --output-root, or --only-task-id; "
            "this script fills those from the shard."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shards <= 0:
        raise ValueError("--shards must be positive.")
    annotation_root = args.annotation_root.expanduser().resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"--annotation-root is not a directory: {annotation_root}")

    normalize_args = normalize_passthrough(args.normalize_args)
    tasks = discover_tasks(args, annotation_root=annotation_root)
    if not tasks:
        raise FileNotFoundError(f"No tasks with {args.input_name} under {annotation_root}")
    shards = split_tasks(tasks, shard_count=args.shards)
    commands = [build_command(args, task_ids=[task.task_id for task in shard], normalize_args=normalize_args) for shard in shards]
    payload = build_summary_payload(args, annotation_root=annotation_root, tasks=tasks, shards=shards, commands=commands)

    print_summary(payload)
    if args.dry_run:
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for shard in payload["shards"]:
        shard_index = int(shard["shard_index"])
        task_list_path = output_dir / f"shard_{shard_index:03d}_tasks.txt"
        command_path = output_dir / f"shard_{shard_index:03d}_command.sh"
        task_list_path.write_text("\n".join(shard["task_ids"]) + "\n", encoding="utf-8")
        command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + shard["command"] + "\n", encoding="utf-8")
        command_path.chmod(0o755)
    print(f"[Done] wrote {len(shards)} shards to {output_dir}")


def discover_tasks(args: argparse.Namespace, *, annotation_root: pathlib.Path) -> list[TaskInfo]:
    allowed = flatten_only_task_ids(args.only_task_id)
    tasks: list[TaskInfo] = []
    for task_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir()):
        task_id = task_dir.name
        if args.task_id_glob != "*" and not task_dir.match(args.task_id_glob):
            continue
        if allowed and task_id not in allowed:
            continue
        input_jsonl = task_dir / args.input_name
        if not input_jsonl.is_file():
            continue
        output_jsonl = resolve_normalize_output(args, task_id=task_id, task_dir=task_dir)
        input_rows = count_jsonl_lines(input_jsonl) or 0
        output_rows = count_jsonl_lines(output_jsonl)
        if args.skip_existing and output_rows is not None and output_rows >= input_rows:
            continue
        tasks.append(
            TaskInfo(
                task_id=task_id,
                task_dir=task_dir,
                input_jsonl=input_jsonl,
                output_jsonl=output_jsonl,
                input_rows=input_rows,
                output_rows=output_rows,
            )
        )
    if allowed:
        found = {task.task_id for task in tasks}
        missing = sorted(allowed - found)
        if missing:
            print(f"[Warn] requested task ids with no pending input: {', '.join(missing)}")
    return tasks


def split_tasks(tasks: list[TaskInfo], *, shard_count: int) -> list[list[TaskInfo]]:
    shards: list[list[TaskInfo]] = [[] for _ in range(shard_count)]
    costs = [0] * shard_count
    for task in sorted(tasks, key=lambda item: (item.input_rows, item.task_id), reverse=True):
        shard_index = min(range(shard_count), key=lambda index: (costs[index], len(shards[index]), index))
        shards[shard_index].append(task)
        costs[shard_index] += max(task.input_rows, 1)
    for shard in shards:
        shard.sort(key=lambda item: item.task_id)
    return shards


def build_command(args: argparse.Namespace, *, task_ids: list[str], normalize_args: list[str]) -> str:
    if not task_ids:
        return "echo 'Empty shard; nothing to normalize.'"
    parts: list[str] = []
    if args.env_prefix.strip():
        parts.extend(shlex.split(args.env_prefix.strip()))
    parts.extend(
        [
            args.python_bin,
            str(args.normalize_script),
            "--annotation-root",
            str(args.annotation_root),
            "--input-name",
            args.input_name,
            "--output-name",
            args.output_name,
        ]
    )
    if args.normalize_output_root is not None:
        parts.extend(["--output-root", str(args.normalize_output_root)])
    if task_ids:
        parts.append("--only-task-id")
        parts.extend(task_ids)
    parts.extend(normalize_args)
    return " ".join(shlex.quote(part) for part in parts)


def build_summary_payload(
    args: argparse.Namespace,
    *,
    annotation_root: pathlib.Path,
    tasks: list[TaskInfo],
    shards: list[list[TaskInfo]],
    commands: list[str],
) -> dict[str, object]:
    return {
        "annotation_root": str(annotation_root),
        "input_name": args.input_name,
        "output_name": args.output_name,
        "normalize_output_root": None if args.normalize_output_root is None else str(args.normalize_output_root),
        "task_count": len(tasks),
        "total_input_rows": sum(task.input_rows for task in tasks),
        "shard_count": len(shards),
        "shards": [
            {
                "shard_index": index,
                "task_count": len(shard),
                "input_rows": sum(task.input_rows for task in shard),
                "task_ids": [task.task_id for task in shard],
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "input_rows": task.input_rows,
                        "output_rows": task.output_rows,
                        "input_jsonl": str(task.input_jsonl),
                        "output_jsonl": str(task.output_jsonl),
                    }
                    for task in shard
                ],
                "command": commands[index],
            }
            for index, shard in enumerate(shards)
        ],
    }


def print_summary(payload: dict[str, object]) -> None:
    print(
        "[Info] tasks="
        f"{payload['task_count']} total_input_rows={payload['total_input_rows']} shards={payload['shard_count']}"
    )
    for shard in payload["shards"]:
        assert isinstance(shard, dict)
        print(
            f"[Shard {int(shard['shard_index']):03d}] "
            f"tasks={shard['task_count']} rows={shard['input_rows']}"
        )
        print(str(shard["command"]))


def resolve_normalize_output(args: argparse.Namespace, *, task_id: str, task_dir: pathlib.Path) -> pathlib.Path:
    if args.normalize_output_root is not None:
        return (args.normalize_output_root / task_id / args.output_name).expanduser().resolve()
    return (task_dir / args.output_name).resolve()


def count_jsonl_lines(path: pathlib.Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count


def flatten_only_task_ids(values: list[list[str]]) -> set[str]:
    return {item for group in values for item in group}


def normalize_passthrough(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


if __name__ == "__main__":
    main()
