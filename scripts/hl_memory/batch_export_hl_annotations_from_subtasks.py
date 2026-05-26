#!/usr/bin/env python3
"""Batch-export HL annotations.jsonl for every task under a subtask root."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


'''

python /root/Users/donggaoqi/openpi_vlm_finetune/scripts/hl_memory/batch_export_hl_annotations_from_subtasks.py \
  --subtask-root /root/Users/dataset/lerobot_home/subtask \
  --workers 8 \
  --overwrite \
  --continue-on-error 

  --dry-run


'''

DEFAULT_SUBTASK_ROOT = Path("/root/Users/dataset/lerobot_home/subtask")
DEFAULT_EXPORT_SCRIPT = Path(__file__).with_name("export_hl_annotations_from_subtasks.py")
DEFAULT_OUTPUT_NAME = "hl_annotations.jsonl"
SUMMARY_NAME = "batch_export_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run export_hl_annotations_from_subtasks.py for each task directory that contains "
            "subtask_segments.json. Tasks are processed in parallel."
        )
    )
    parser.add_argument(
        "--subtask-root",
        type=Path,
        default=DEFAULT_SUBTASK_ROOT,
        help=f"Root containing one directory per task id (default: {DEFAULT_SUBTASK_ROOT}).",
    )
    parser.add_argument(
        "--export-script",
        type=Path,
        default=DEFAULT_EXPORT_SCRIPT,
        help="Path to export_hl_annotations_from_subtasks.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "If set, write <output-root>/<task_id>/hl_annotations.jsonl. "
            "Otherwise write <task_dir>/hl_annotations.jsonl next to subtask_segments.json."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output JSONL filename (default: {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of tasks to export concurrently (default: 8).",
    )
    parser.add_argument("--task-id-glob", default="*", help="Glob for task directories under --subtask-root.")
    parser.add_argument("--only-task-id", action="append", default=[], help="Export only this task id. Repeatable.")
    parser.add_argument(
        "--instruction",
        default="",
        help="If set, pass the same --instruction to every task. Otherwise read meta/tasks.jsonl per task when present.",
    )
    parser.add_argument("--target-query", default="", help="Optional --target-query copied to every annotation row.")
    parser.add_argument("--goal-query", default="", help="Optional --goal-query copied to every annotation row.")
    parser.add_argument("--emit-success-events", action="store_true", help="Pass --emit-success-events to the exporter.")
    parser.add_argument(
        "--progress-sample-fractions",
        default="",
        help="Pass --progress-sample-fractions to the exporter.",
    )
    parser.add_argument(
        "--progress-extra-fractions",
        default="",
        help="Pass --progress-extra-fractions to the exporter.",
    )
    parser.add_argument(
        "--progress-sample-target-frames",
        type=int,
        default=0,
        help="Pass --progress-sample-target-frames to the exporter.",
    )
    parser.add_argument(
        "--progress-sample-jitter",
        type=float,
        default=0.0,
        help="Pass --progress-sample-jitter to the exporter.",
    )
    parser.add_argument(
        "--progress-sample-seed",
        type=int,
        default=0,
        help="Pass --progress-sample-seed to the exporter.",
    )
    parser.add_argument(
        "--min-progress-samples-per-segment",
        type=int,
        default=0,
        help="Pass --min-progress-samples-per-segment to the exporter.",
    )
    parser.add_argument(
        "--max-progress-samples-per-segment",
        type=int,
        default=None,
        help="Pass --max-progress-samples-per-segment to the exporter.",
    )
    parser.add_argument(
        "--progress-min-gap",
        type=int,
        default=0,
        help="Pass --progress-min-gap to the exporter.",
    )
    parser.add_argument(
        "--short-segment-max-frames",
        type=int,
        default=0,
        help="Pass --short-segment-max-frames to the exporter.",
    )
    parser.add_argument(
        "--short-segment-progress-fractions",
        default="",
        help="Pass --short-segment-progress-fractions to the exporter.",
    )
    parser.add_argument(
        "--short-segment-progress-min-gap",
        type=int,
        default=-1,
        help="Pass --short-segment-progress-min-gap to the exporter.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to the exporter.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose output JSONL already exists (unless --overwrite is set).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue exporting remaining tasks after one task fails.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=f"Write a JSON summary of all tasks (default: <output-root>/{SUMMARY_NAME} or subtask-root/{SUMMARY_NAME}).",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to export_hl_annotations_from_subtasks.py after '--'.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class ExportJob:
    task_id: str
    task_dir: Path
    segments_path: Path
    output_jsonl: Path
    instruction: str
    cmd: list[str]


@dataclass(frozen=True)
class ExportResult:
    task_id: str
    output_jsonl: Path
    returncode: int
    skipped: bool = False
    row_count: int | None = None
    error: str = ""


_PRINT_LOCK = threading.Lock()


def main() -> None:
    args = parse_args()
    subtask_root = args.subtask_root.resolve()
    export_script = args.export_script.resolve()
    if not subtask_root.is_dir():
        raise FileNotFoundError(f"--subtask-root is not a directory: {subtask_root}")
    if not export_script.is_file():
        raise FileNotFoundError(f"--export-script not found: {export_script}")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    passthrough = normalize_passthrough(args.passthrough)
    task_dirs = discover_task_dirs(subtask_root, name_glob=args.task_id_glob)
    if args.only_task_id:
        allowed = set(args.only_task_id)
        task_dirs = [(task_id, path) for task_id, path in task_dirs if task_id in allowed]
    if not task_dirs:
        if args.only_task_id:
            requested = ", ".join(sorted(set(args.only_task_id)))
            raise FileNotFoundError(
                f"No matching task directories under {subtask_root} for --only-task-id: {requested}"
            )
        raise FileNotFoundError(f"No task directories with subtask_segments.json under {subtask_root}")

    jobs: list[ExportJob] = []
    skipped_existing: list[ExportResult] = []
    for task_id, task_dir in task_dirs:
        segments_path = task_dir / "subtask_segments.json"
        output_jsonl = resolve_output_path(args, task_id=task_id, task_dir=task_dir)
        if args.skip_existing and not args.overwrite and output_jsonl.exists():
            skipped_existing.append(
                ExportResult(
                    task_id=task_id,
                    output_jsonl=output_jsonl,
                    returncode=0,
                    skipped=True,
                    row_count=_count_jsonl_lines(output_jsonl),
                )
            )
            continue

        instruction = args.instruction.strip() or load_task_instruction(task_dir)
        cmd = build_export_command(
            export_script=export_script,
            segments_path=segments_path,
            output_jsonl=output_jsonl,
            instruction=instruction,
            target_query=args.target_query,
            goal_query=args.goal_query,
            emit_success_events=args.emit_success_events,
            progress_sample_fractions=args.progress_sample_fractions,
            progress_extra_fractions=args.progress_extra_fractions,
            progress_sample_target_frames=args.progress_sample_target_frames,
            progress_sample_jitter=args.progress_sample_jitter,
            progress_sample_seed=args.progress_sample_seed,
            min_progress_samples_per_segment=args.min_progress_samples_per_segment,
            max_progress_samples_per_segment=args.max_progress_samples_per_segment,
            progress_min_gap=args.progress_min_gap,
            short_segment_max_frames=args.short_segment_max_frames,
            short_segment_progress_fractions=args.short_segment_progress_fractions,
            short_segment_progress_min_gap=args.short_segment_progress_min_gap,
            overwrite=args.overwrite,
            passthrough=passthrough,
        )
        jobs.append(
            ExportJob(
                task_id=task_id,
                task_dir=task_dir,
                segments_path=segments_path,
                output_jsonl=output_jsonl,
                instruction=instruction,
                cmd=cmd,
            )
        )

    print(f"[Info] subtask_root={subtask_root}")
    print(f"[Info] tasks_with_segments={len(task_dirs)} export_jobs={len(jobs)} skipped_existing={len(skipped_existing)}")

    job_failures, completed_ids, cancelled_ids = run_jobs(
        jobs,
        workers=args.workers,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )

    results: list[ExportResult] = list(skipped_existing)
    for job, code in job_failures:
        row_count = _count_jsonl_lines(job.output_jsonl) if job.output_jsonl.exists() else None
        results.append(
            ExportResult(
                task_id=job.task_id,
                output_jsonl=job.output_jsonl,
                returncode=code,
                row_count=row_count,
                error=f"exporter exited with code {code}",
            )
        )

    failed_ids = {item.task_id for item, _ in job_failures}
    for job in jobs:
        if job.task_id in failed_ids:
            continue
        if job.task_id in cancelled_ids:
            results.append(
                ExportResult(
                    task_id=job.task_id,
                    output_jsonl=job.output_jsonl,
                    returncode=-1,
                    error="cancelled before export finished",
                )
            )
            continue
        if job.task_id not in completed_ids:
            continue
        if args.dry_run:
            results.append(
                ExportResult(
                    task_id=job.task_id,
                    output_jsonl=job.output_jsonl,
                    returncode=0,
                    row_count=None,
                )
            )
            continue
        results.append(
            ExportResult(
                task_id=job.task_id,
                output_jsonl=job.output_jsonl,
                returncode=0,
                row_count=_count_jsonl_lines(job.output_jsonl),
            )
        )

    summary_path = resolve_summary_path(args, subtask_root=subtask_root)
    if not args.dry_run:
        write_summary(summary_path, subtask_root=subtask_root, results=results)

    failed = [result for result in results if result.returncode != 0]
    if failed:
        rendered = ", ".join(f"{result.task_id}:{result.returncode}" for result in failed)
        raise SystemExit(f"Failed or cancelled tasks ({len(failed)}): {rendered}")

    ok_count = sum(1 for result in results if result.returncode == 0 and not result.skipped)
    skip_count = sum(1 for result in results if result.skipped)
    print(f"[Done] exported={ok_count} skipped_existing={skip_count} summary={summary_path}")


def discover_task_dirs(subtask_root: Path, *, name_glob: str) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for child in sorted(subtask_root.iterdir()):
        if not child.is_dir():
            continue
        if name_glob != "*" and not child.match(name_glob):
            continue
        segments_path = child / "subtask_segments.json"
        if segments_path.is_file():
            found.append((child.name, child))
    return found


def resolve_output_path(args: argparse.Namespace, *, task_id: str, task_dir: Path) -> Path:
    if args.output_root is not None:
        return (args.output_root / task_id / args.output_name).resolve()
    return (task_dir / args.output_name).resolve()


def resolve_summary_path(args: argparse.Namespace, *, subtask_root: Path) -> Path:
    if args.summary_json is not None:
        return args.summary_json.resolve()
    if args.output_root is not None:
        return (args.output_root / SUMMARY_NAME).resolve()
    return (subtask_root / SUMMARY_NAME).resolve()


def load_task_instruction(task_dir: Path) -> str:
    tasks_path = task_dir / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        return ""
    with tasks_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            for key in ("task", "instruction", "prompt", "description"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def build_export_command(
    *,
    export_script: Path,
    segments_path: Path,
    output_jsonl: Path,
    instruction: str,
    target_query: str,
    goal_query: str,
    emit_success_events: bool,
    progress_sample_fractions: str,
    progress_extra_fractions: str,
    progress_sample_target_frames: int,
    progress_sample_jitter: float,
    progress_sample_seed: int,
    min_progress_samples_per_segment: int,
    max_progress_samples_per_segment: int | None,
    progress_min_gap: int,
    short_segment_max_frames: int,
    short_segment_progress_fractions: str,
    short_segment_progress_min_gap: int,
    overwrite: bool,
    passthrough: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(export_script),
        "--subtask-segments-json",
        str(segments_path),
        "--output-jsonl",
        str(output_jsonl),
    ]
    if instruction:
        cmd.extend(["--instruction", instruction])
    if target_query:
        cmd.extend(["--target-query", target_query])
    if goal_query:
        cmd.extend(["--goal-query", goal_query])
    if emit_success_events:
        cmd.append("--emit-success-events")
    if progress_sample_fractions:
        cmd.extend(["--progress-sample-fractions", progress_sample_fractions])
    if progress_extra_fractions:
        cmd.extend(["--progress-extra-fractions", progress_extra_fractions])
    if progress_sample_target_frames > 0:
        cmd.extend(["--progress-sample-target-frames", str(progress_sample_target_frames)])
    if progress_sample_jitter > 0:
        cmd.extend(["--progress-sample-jitter", str(progress_sample_jitter)])
    if progress_sample_seed:
        cmd.extend(["--progress-sample-seed", str(progress_sample_seed)])
    if min_progress_samples_per_segment > 0:
        cmd.extend(["--min-progress-samples-per-segment", str(min_progress_samples_per_segment)])
    if max_progress_samples_per_segment is not None:
        cmd.extend(["--max-progress-samples-per-segment", str(max_progress_samples_per_segment)])
    if progress_min_gap > 0:
        cmd.extend(["--progress-min-gap", str(progress_min_gap)])
    if short_segment_max_frames > 0:
        cmd.extend(["--short-segment-max-frames", str(short_segment_max_frames)])
    if short_segment_progress_fractions:
        cmd.extend(["--short-segment-progress-fractions", short_segment_progress_fractions])
    if short_segment_progress_min_gap != -1:
        cmd.extend(["--short-segment-progress-min-gap", str(short_segment_progress_min_gap)])
    if overwrite:
        cmd.append("--overwrite")
    cmd.extend(passthrough)
    return cmd


def run_jobs(
    jobs: list[ExportJob],
    *,
    workers: int,
    dry_run: bool,
    continue_on_error: bool,
) -> tuple[list[tuple[ExportJob, int]], set[str], set[str]]:
    if not jobs:
        return [], set(), set()

    if dry_run:
        return [], {job.task_id for job in jobs}, set()

    if workers == 1:
        failures: list[tuple[ExportJob, int]] = []
        completed: set[str] = set()
        for index, job in enumerate(jobs):
            code = run_one_job(job, dry_run=False)
            if code == 0:
                completed.add(job.task_id)
                continue
            failures.append((job, code))
            if not continue_on_error:
                cancelled = {pending.task_id for pending in jobs[index + 1 :]}
                return failures, completed, cancelled
        return failures, completed, set()

    failures: list[tuple[ExportJob, int]] = []
    completed: set[str] = set()
    cancelled: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
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
                        print(
                            f"[Error] task_id={job.task_id} raised {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                if code == 0:
                    completed.add(job.task_id)
                    continue
                failures.append((job, code))
                if not continue_on_error:
                    for remaining_future in pending:
                        remaining_job = pending[remaining_future]
                        cancelled.add(remaining_job.task_id)
                        remaining_future.cancel()
                    return failures, completed, cancelled
    return failures, completed, set()


def run_one_job(job: ExportJob, *, dry_run: bool) -> int:
    with _PRINT_LOCK:
        print(f"[Task] {job.task_id}")
        print(f"[In]   {job.segments_path}")
        print(f"[Out]  {job.output_jsonl}")
        if job.instruction:
            preview = job.instruction if len(job.instruction) <= 120 else job.instruction[:117] + "..."
            print(f"[Inst] {preview}")
        print(f"[Cmd]  {shell_join(job.cmd)}")
    if dry_run:
        return 0

    job.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(job.cmd, check=False)
    if result.returncode != 0:
        with _PRINT_LOCK:
            print(f"[Error] task_id={job.task_id} failed with code {result.returncode}", file=sys.stderr)
        return result.returncode

    row_count = _count_jsonl_lines(job.output_jsonl)
    with _PRINT_LOCK:
        print(f"[OK]   task_id={job.task_id} rows={row_count} -> {job.output_jsonl}")
    return 0


def write_summary(summary_path: Path, *, subtask_root: Path, results: list[ExportResult]) -> None:
    payload = {
        "subtask_root": str(subtask_root),
        "task_count": len(results),
        "exported": sum(1 for item in results if item.returncode == 0 and not item.skipped),
        "skipped_existing": sum(1 for item in results if item.skipped),
        "failed": sum(1 for item in results if item.returncode > 0),
        "cancelled": sum(1 for item in results if item.returncode < 0),
        "tasks": [
            {
                "task_id": item.task_id,
                "output_jsonl": str(item.output_jsonl),
                "returncode": item.returncode,
                "skipped": item.skipped,
                "row_count": item.row_count,
                "error": item.error,
            }
            for item in sorted(results, key=lambda row: row.task_id)
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _count_jsonl_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count


def normalize_passthrough(values: list[str]) -> list[str]:
    if not values:
        return []
    if values[0] == "--":
        return values[1:]
    return values


def shell_join(parts: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in parts)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(char not in ' \t\n\r"\'\\$' for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
