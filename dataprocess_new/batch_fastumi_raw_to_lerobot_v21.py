#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


SESSION_MARKERS = (
    "RGB_Images",
    "Merged_Trajectory",
    "left_hand",
    "right_hand",
)

DESCRIPTION_KEYS = (
    "human_labeled_description",
    "LLM_labeled_description",
    "task_description",
    "description",
    "task",
    "instruction",
    "prompt",
)

ID_KEYS = (
    "taskid",
    "task_id",
    "id",
    "repo_id",
    "repoid",
    "name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a FastUMI raw root for task ids, look up task descriptions from a JSON file, "
            "and run dataprocess_new/fastumi_raw_to_lerobot_v21.py once per task. "
            "Arguments after '--' are passed through to the v21 converter."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True, help="Root containing one directory per task id.")
    parser.add_argument(
        "--task-json",
        type=Path,
        default=Path("/root/Users/donggaoqi/mmgj/merged_1_12_0424.json"),
        help="JSON file containing taskid -> task description records.",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).with_name("fastumi_raw_to_lerobot_v21.py"),
        help="Path to fastumi_raw_to_lerobot_v21.py.",
    )
    parser.add_argument(
        "--repo-prefix",
        default="",
        help="Optional prefix before each task id for --repo-id. Leave empty to make repo id exactly taskid.",
    )
    parser.add_argument(
        "--task-depth",
        type=int,
        default=1,
        help="Directory depth under raw-root that represents a task id. Default: raw-root/<taskid>.",
    )
    parser.add_argument("--task-id-glob", default="*", help="Glob for task id directories at task depth.")
    parser.add_argument("--only-task-id", action="append", default=[], help="Convert only this task id. Repeatable.")
    parser.add_argument(
        "--skip-missing-description",
        action="store_true",
        help="Skip task dirs missing from task-json instead of failing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help=(
            "Number of task-level conversions to run concurrently. Default 1 keeps tasks serial. "
            "Keep this small because each task can also spawn many --workers internally."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue converting remaining tasks after one conversion fails.",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra args for fastumi_raw_to_lerobot_v21.py. Put them after '--'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"--raw-root does not exist: {raw_root}")
    if args.task_depth <= 0:
        raise ValueError("--task-depth must be positive.")
    if args.task_workers <= 0:
        raise ValueError("--task-workers must be positive.")

    descriptions = load_task_descriptions(args.task_json)
    task_dirs = discover_task_dirs(raw_root, depth=args.task_depth, name_glob=args.task_id_glob)
    if args.only_task_id:
        allowed = set(args.only_task_id)
        task_dirs = [(task_id, path) for task_id, path in task_dirs if task_id in allowed]
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found under {raw_root} at depth {args.task_depth}.")

    passthrough = normalize_passthrough(args.passthrough)
    jobs: list[ConversionJob] = []
    for task_id, task_dir in task_dirs:
        description = descriptions.get(task_id)
        if description is None:
            message = f"No task description found for task_id={task_id!r} in {args.task_json}"
            if args.skip_missing_description:
                print(f"[Skip] {message}")
                continue
            raise KeyError(message)

        repo_id = f"{args.repo_prefix}{task_id}"
        jobs.append(
            ConversionJob(
                task_id=task_id,
                task_dir=task_dir,
                repo_id=repo_id,
                description=description,
                cmd=[
                    sys.executable,
                    str(args.converter),
                    "--raw-dir",
                    str(task_dir),
                    "--repo-id",
                    repo_id,
                    "--task",
                    description,
                    *passthrough,
                ],
            )
        )

    failures = run_jobs(
        jobs,
        task_workers=args.task_workers,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )
    if failures:
        rendered = ", ".join(f"{task_id}:{code}" for task_id, code in failures)
        raise SystemExit(f"Failed tasks: {rendered}")


class ConversionJob:
    def __init__(self, *, task_id: str, task_dir: Path, repo_id: str, description: str, cmd: list[str]):
        self.task_id = task_id
        self.task_dir = task_dir
        self.repo_id = repo_id
        self.description = description
        self.cmd = cmd


_PRINT_LOCK = threading.Lock()


def run_jobs(
    jobs: list[ConversionJob],
    *,
    task_workers: int,
    dry_run: bool,
    continue_on_error: bool,
) -> list[tuple[str, int]]:
    if dry_run or task_workers == 1:
        failures: list[tuple[str, int]] = []
        for job in jobs:
            code = run_one_job(job, dry_run=dry_run)
            if code != 0:
                failures.append((job.task_id, code))
                if not continue_on_error:
                    return failures
        return failures

    failures: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=task_workers) as executor:
        pending = {executor.submit(run_one_job, job, dry_run=False): job for job in jobs}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    code = future.result()
                except Exception as exc:  # noqa: BLE001
                    code = 1
                    with _PRINT_LOCK:
                        print(f"[Error] task_id={job.task_id} raised {type(exc).__name__}: {exc}", file=sys.stderr)
                if code != 0:
                    failures.append((job.task_id, code))
                    if not continue_on_error:
                        for remaining in pending:
                            remaining.cancel()
                        return failures
    return failures


def run_one_job(job: ConversionJob, *, dry_run: bool) -> int:
    with _PRINT_LOCK:
        print(f"[Task] {job.task_id}")
        print(f"[Raw]  {job.task_dir}")
        print(f"[Repo] {job.repo_id}")
        print(f"[Desc] {job.description}")
        print(f"[Cmd]  {shell_join(job.cmd)}")
    if dry_run:
        return 0

    result = subprocess.run(job.cmd, check=False)
    if result.returncode != 0:
        with _PRINT_LOCK:
            print(f"[Error] task_id={job.task_id} failed with code {result.returncode}", file=sys.stderr)
    return result.returncode


def load_task_descriptions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    descriptions = parse_description_payload(payload)
    return {
        str(task_id).strip(): str(description).strip()
        for task_id, description in descriptions.items()
        if str(task_id).strip() and str(description).strip()
    }


def parse_description_payload(payload: Any) -> dict[str, str]:
    if isinstance(payload, dict):
        if all(isinstance(value, str) for value in payload.values()):
            return {str(key): value for key, value in payload.items()}

        for wrapper_key in ("tasks", "data", "items", "annotations"):
            if wrapper_key in payload:
                return parse_description_payload(payload[wrapper_key])

        nested: dict[str, str] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                description = first_present_string(value, DESCRIPTION_KEYS)
                if description:
                    nested[str(key)] = description
        if nested:
            return nested

    if isinstance(payload, list):
        result: dict[str, str] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            task_id = first_present_string(item, ID_KEYS)
            description = first_present_string(item, DESCRIPTION_KEYS)
            if task_id and description:
                result[task_id] = description
        if result:
            return result

    raise ValueError(
        "Unsupported task-json format. Expected {'tasks': [{'taskid': ..., "
        "'human_labeled_description': ...}]}, {taskid: description}, or similar."
    )


def first_present_string(mapping: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    lowered = {key.lower(): value for key, value in mapping.items()}
    for key in keys:
        if key in mapping and str(mapping[key]).strip():
            return str(mapping[key]).strip()
        if key.lower() in lowered and str(lowered[key.lower()]).strip():
            return str(lowered[key.lower()]).strip()
    return None


def discover_task_dirs(raw_root: Path, *, depth: int, name_glob: str) -> list[tuple[str, Path]]:
    if depth == 1:
        candidates = sorted(path for path in raw_root.glob(name_glob) if path.is_dir())
    else:
        pattern = "/".join(["*"] * (depth - 1) + [name_glob])
        candidates = sorted(path for path in raw_root.glob(pattern) if path.is_dir())

    task_dirs: list[tuple[str, Path]] = []
    for path in candidates:
        if looks_like_session_dir(path):
            continue
        if has_session_descendant(path):
            task_dirs.append((path.name, path))
    return task_dirs


def looks_like_session_dir(path: Path) -> bool:
    return any((path / marker).exists() for marker in SESSION_MARKERS)


def has_session_descendant(path: Path) -> bool:
    if looks_like_session_dir(path):
        return True
    for child in path.rglob("*"):
        if child.is_dir() and looks_like_session_dir(child):
            return True
    return False


def normalize_passthrough(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def shell_join(args: list[str]) -> str:
    return " ".join(shell_quote(arg) for arg in args)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=./:,@%")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
