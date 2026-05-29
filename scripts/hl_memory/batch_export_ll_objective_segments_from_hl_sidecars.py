#!/usr/bin/env python3
"""Batch-build LL current-objective segment sidecars for LeRobot task directories."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_SUBTASK_ROOT = Path("/root/Users/dataset/lerobot_home/subtask")
DEFAULT_EXPORT_SCRIPT = Path(__file__).with_name("export_ll_objective_segments_from_hl_sidecar.py")
DEFAULT_OUTPUT_NAME = "ll_current_objective_segments.json"
SUMMARY_NAME = "batch_ll_objective_segments_summary.json"


@dataclass(frozen=True)
class Job:
    task_id: str
    task_dir: Path
    output_json: Path
    cmd: list[str]


@dataclass(frozen=True)
class Result:
    task_id: str
    output_json: str
    returncode: int
    skipped: bool = False
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtask-root", type=Path, default=DEFAULT_SUBTASK_ROOT)
    parser.add_argument("--export-script", type=Path, default=DEFAULT_EXPORT_SCRIPT)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--task-id-glob", default="*")
    parser.add_argument("--only-task-id", action="append", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_dirs = discover_task_dirs(args)
    jobs: list[Job] = []
    results: list[Result] = []
    for task_id, task_dir in task_dirs:
        output_json = task_dir / args.output_name
        if args.skip_existing and not args.overwrite and output_json.exists():
            results.append(Result(task_id=task_id, output_json=str(output_json), returncode=0, skipped=True))
            continue
        cmd = [
            sys.executable,
            str(args.export_script.resolve()),
            "--task-dir",
            str(task_dir.resolve()),
            "--output-json",
            str(output_json.resolve()),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        jobs.append(Job(task_id=task_id, task_dir=task_dir, output_json=output_json, cmd=cmd))

    print(f"[Info] tasks={len(task_dirs)} jobs={len(jobs)} skipped={sum(1 for result in results if result.skipped)}")
    if args.dry_run:
        for job in jobs:
            print("[DryRun]", " ".join(job.cmd))
        results.extend(Result(task_id=job.task_id, output_json=str(job.output_json), returncode=0) for job in jobs)
        write_summary(resolve_summary_path(args), args=args, results=results)
        return

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        for result in executor.map(run_job, jobs):
            results.append(result)
            status = "OK" if result.returncode == 0 else f"FAIL:{result.returncode}"
            print(f"[{status}] {result.task_id} -> {result.output_json}", flush=True)
            if result.returncode != 0 and not args.continue_on_error:
                raise SystemExit(f"Failed task {result.task_id}: {result.error}")

    write_summary(resolve_summary_path(args), args=args, results=results)
    failed = [result for result in results if result.returncode != 0]
    if failed:
        raise SystemExit(f"Failed tasks ({len(failed)}): {', '.join(result.task_id for result in failed)}")
    print(f"[Done] exported={sum(1 for result in results if result.returncode == 0 and not result.skipped)}")


def discover_task_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    root = args.subtask_root.expanduser().resolve()
    allowed = {item for group in args.only_task_id for item in group}
    found: list[tuple[str, Path]] = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_id = task_dir.name
        if args.task_id_glob != "*" and not task_dir.match(args.task_id_glob):
            continue
        if allowed and task_id not in allowed:
            continue
        if (task_dir / "subtask_segments.json").is_file() and (task_dir / "hl_segments_llm_sidecar.json").is_file():
            found.append((task_id, task_dir))
    return found


def run_job(job: Job) -> Result:
    completed = subprocess.run(job.cmd, check=False, text=True, capture_output=True)
    error = completed.stderr.strip() or completed.stdout.strip()
    return Result(
        task_id=job.task_id,
        output_json=str(job.output_json),
        returncode=int(completed.returncode),
        error=error if completed.returncode else "",
    )


def resolve_summary_path(args: argparse.Namespace) -> Path:
    if args.summary_json is not None:
        return args.summary_json.expanduser().resolve()
    return args.subtask_root.expanduser().resolve() / SUMMARY_NAME


def write_summary(path: Path, *, args: argparse.Namespace, results: list[Result]) -> None:
    payload = {
        "subtask_root": str(args.subtask_root),
        "task_count": len(results),
        "exported": sum(1 for result in results if result.returncode == 0 and not result.skipped),
        "skipped_existing": sum(1 for result in results if result.skipped),
        "failed": sum(1 for result in results if result.returncode != 0),
        "tasks": [result.__dict__ for result in sorted(results, key=lambda item: item.task_id)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
