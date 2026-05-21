#!/usr/bin/env python3
"""Batch convert LeRobot subtask repos plus annotations into HL-memory train data."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path



'''
cd /root/Users/donggaoqi/openpi_vlm_finetune

PYTHONPATH=src /root/Users/miniconda3/envs/pi0_dgq/bin/python \
  scripts/hl_memory/batch_export_hl_memory_dataset_from_subtasks.py \
  --source-config-name toy_block_placement_Ba_qwen3_5_2b_400m_guided \
  --subtask-root /root/Users/dataset/lerobot_home/subtask \
  --output-root /root/Users/dataset/hl_memory/subtask \
  --repo-prefix subtask/ \
  --workers 4 \
  --auto-export-annotations \
  --overwrite \
  --continue-on-error

'''

DEFAULT_SUBTASK_ROOT = Path("/root/Users/dataset/lerobot_home/subtask")
DEFAULT_OUTPUT_ROOT = Path("/root/Users/dataset/hl_memory/subtask")
DEFAULT_EXPORT_SCRIPT = Path(__file__).with_name("export_hl_memory_dataset.py")
DEFAULT_ANNOTATION_SCRIPT = Path(__file__).with_name("export_hl_annotations_from_subtasks.py")
DEFAULT_ANNOTATIONS_NAME = "hl_annotations.jsonl"
SUMMARY_NAME = "batch_hl_memory_export_summary.json"


@dataclass(frozen=True)
class Job:
    task_id: str
    task_dir: Path
    repo_id: str
    annotations_jsonl: Path
    train_dir: Path
    val_dir: Path
    log_path: Path
    cmd: list[str]
    annotation_cmd: list[str] | None


@dataclass(frozen=True)
class Result:
    task_id: str
    repo_id: str
    returncode: int
    train_dir: str
    val_dir: str
    annotations_jsonl: str
    log_path: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch export HL-memory train/val datasets from HF_LEROBOT_HOME/subtask/<task_id>."
    )
    parser.add_argument("--source-config-name", required=True, help="Base OpenPI training config to reuse transforms/model shape.")
    parser.add_argument("--subtask-root", type=Path, default=DEFAULT_SUBTASK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-prefix", default="subtask/", help="Repo id prefix. Default makes repo_id=subtask/<task_id>.")
    parser.add_argument("--task-id-glob", default="*")
    parser.add_argument("--only-task-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--visual-mode", choices=("raw", "config"), default="raw")
    parser.add_argument("--missing-episode-policy", choices=("error", "skip"), default="error")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--status-interval-s", type=float, default=60.0, help="Print batch heartbeat while jobs are still running.")
    parser.add_argument("--stream-output", action="store_true", help="Also stream child process output to the terminal with task prefixes.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--annotations-name",
        default=DEFAULT_ANNOTATIONS_NAME,
        help=(
            f"Annotation JSONL filename under each task directory. Default: {DEFAULT_ANNOTATIONS_NAME}. "
            "Use hl_annotations_llm_normalized.jsonl after running batch_normalize_hl_annotations_with_llm.py."
        ),
    )
    parser.add_argument("--auto-export-annotations", action="store_true", help="Create hl_annotations.jsonl when missing.")
    parser.add_argument("--emit-success-events", action="store_true", help="Forward to annotation exporter when auto-exporting.")
    parser.add_argument("--progress-sample-stride", type=int, default=0, help="Forward to annotation exporter.")
    parser.add_argument("--max-progress-samples-per-segment", type=int, default=1, help="Forward to annotation exporter.")
    parser.add_argument("--export-script", type=Path, default=DEFAULT_EXPORT_SCRIPT)
    parser.add_argument("--annotation-script", type=Path, default=DEFAULT_ANNOTATION_SCRIPT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Extra args for export_hl_memory_dataset.py after --.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    passthrough = normalize_passthrough(args.passthrough)
    jobs = build_jobs(args, passthrough=passthrough)
    if not jobs:
        raise FileNotFoundError(f"No task dirs with subtask_segments.json under {args.subtask_root}")
    print(f"[Info] tasks={len(jobs)} workers={args.workers} output_root={args.output_root}")

    results: list[Result] = []
    if args.dry_run:
        for job in jobs:
            print(render_job(job))
            results.append(result_from_job(job, returncode=0))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_job = {}
            for job in jobs:
                print(f"[Queue] {job.task_id} repo_id={job.repo_id} log={job.log_path}", flush=True)
                future_to_job[pool.submit(run_job, job, stream_output=args.stream_output)] = job
            pending = set(future_to_job)
            while pending:
                done, pending = wait(pending, timeout=args.status_interval_s, return_when=FIRST_COMPLETED)
                if not done:
                    running = [future_to_job[future].task_id for future in pending]
                    preview = ", ".join(running[: min(8, len(running))])
                    suffix = "" if len(running) <= 8 else f", ... +{len(running) - 8}"
                    print(
                        f"[Status] done={len(results)}/{len(jobs)} running={len(running)} "
                        f"tasks=[{preview}{suffix}]",
                        flush=True,
                    )
                    continue
                for future in done:
                    job = future_to_job[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - CLI diagnostic path
                        result = result_from_job(job, returncode=1, error=str(exc))
                    results.append(result)
                    status = "OK" if result.returncode == 0 else f"FAIL:{result.returncode}"
                    print(f"[{status}] {job.task_id} repo_id={job.repo_id} log={job.log_path}", flush=True)
                    if result.returncode != 0 and not args.continue_on_error:
                        raise SystemExit(f"Task {job.task_id} failed: {result.error}\nLog: {job.log_path}")

    summary_path = args.summary_json or (args.output_root / SUMMARY_NAME)
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps([r.__dict__ for r in sorted(results, key=lambda x: x.task_id)], indent=2) + "\n")
    failed = [r for r in results if r.returncode != 0]
    if failed:
        raise SystemExit(f"Failed tasks: {', '.join(r.task_id for r in failed)}")
    print(f"[Done] exported={len(results)} summary={summary_path}")


def build_jobs(args: argparse.Namespace, *, passthrough: list[str]) -> list[Job]:
    subtask_root = args.subtask_root.resolve()
    task_dirs: list[Path] = []
    allowed = set(args.only_task_id)
    for child in sorted(subtask_root.iterdir()):
        if not child.is_dir() or not child.match(args.task_id_glob):
            continue
        if allowed and child.name not in allowed:
            continue
        if (child / "subtask_segments.json").is_file():
            task_dirs.append(child)

    jobs: list[Job] = []
    for task_dir in task_dirs:
        task_id = task_dir.name
        repo_id = f"{args.repo_prefix}{task_id}"
        annotations_jsonl = task_dir / args.annotations_name
        raw_annotations_jsonl = task_dir / DEFAULT_ANNOTATIONS_NAME
        if not annotations_jsonl.exists() and not args.auto_export_annotations:
            print(f"[Skip] {task_id}: missing {annotations_jsonl}; pass --auto-export-annotations to create raw annotations")
            continue
        train_dir = args.output_root / task_id / "train"
        val_dir = args.output_root / task_id / "val"
        log_path = args.output_root / task_id / "batch_export.log"
        if args.skip_existing and not args.overwrite and (train_dir / "samples.jsonl").exists() and (val_dir / "samples.jsonl").exists():
            print(f"[Skip] {task_id}: train/val samples already exist")
            continue
        annotation_cmd = None
        if args.auto_export_annotations and args.annotations_name != DEFAULT_ANNOTATIONS_NAME and not annotations_jsonl.exists():
            print(
                f"[Skip] {task_id}: missing normalized annotations {annotations_jsonl}; "
                f"run batch_normalize_hl_annotations_with_llm.py first"
            )
            continue
        if (
            args.auto_export_annotations
            and args.annotations_name == DEFAULT_ANNOTATIONS_NAME
            and (args.overwrite or not raw_annotations_jsonl.exists())
        ):
            annotation_cmd = [
                sys.executable,
                str(args.annotation_script.resolve()),
                "--subtask-segments-json",
                str(task_dir / "subtask_segments.json"),
                "--output-jsonl",
                str(raw_annotations_jsonl),
                "--overwrite",
            ]
            if args.emit_success_events:
                annotation_cmd.append("--emit-success-events")
            if args.progress_sample_stride > 0:
                annotation_cmd.extend(["--progress-sample-stride", str(args.progress_sample_stride)])
                annotation_cmd.extend(["--max-progress-samples-per-segment", str(args.max_progress_samples_per_segment)])
        cmd = [
            sys.executable,
            str(args.export_script.resolve()),
            "--source-config-name",
            args.source_config_name,
            "--annotations-jsonl",
            str(annotations_jsonl),
            "--output-train-dir",
            str(train_dir),
            "--output-val-dir",
            str(val_dir),
            "--repo-id-override",
            repo_id,
            "--asset-id-override",
            repo_id,
            "--subtask-segments-path-override",
            "subtask_segments.json",
            "--visual-mode",
            args.visual_mode,
            "--val-ratio",
            str(args.val_ratio),
            "--split-seed",
            str(args.split_seed),
            "--missing-episode-policy",
            args.missing_episode_policy,
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        cmd.extend(passthrough)
        jobs.append(Job(task_id, task_dir, repo_id, annotations_jsonl, train_dir, val_dir, log_path, cmd, annotation_cmd))
    return jobs


def run_job(job: Job, *, stream_output: bool) -> Result:
    env = os.environ.copy()
    env.setdefault("HF_LEROBOT_HOME", str(job.task_dir.parent.parent))
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Running] {job.task_id} repo_id={job.repo_id} log={job.log_path}", flush=True)
    with job.log_path.open("w", encoding="utf-8") as log:
        log.write(f"[Task] {job.task_id}\n[Repo] {job.repo_id}\n")
        log.write(f"[HF_LEROBOT_HOME] {env.get('HF_LEROBOT_HOME', '')}\n")
        if job.annotation_cmd is not None:
            returncode, tail = run_command(
                job.annotation_cmd,
                env=env,
                log=log,
                task_id=job.task_id,
                stage="annotations",
                stream_output=stream_output,
            )
            if returncode != 0:
                return result_from_job(job, returncode, tail)
        returncode, tail = run_command(
            job.cmd,
            env=env,
            log=log,
            task_id=job.task_id,
            stage="export",
            stream_output=stream_output,
        )
    if returncode != 0:
        return result_from_job(job, returncode, tail)
    return result_from_job(job, 0)


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    log,
    task_id: str,
    stage: str,
    stream_output: bool,
) -> tuple[int, str]:
    log.write(f"\n[Command:{stage}] {' '.join(cmd)}\n")
    log.flush()
    tail: deque[str] = deque(maxlen=120)
    start_time = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log.write(line)
        tail.append(line.rstrip("\n"))
        if stream_output:
            print(f"[{task_id}:{stage}] {line}", end="", flush=True)
    returncode = proc.wait()
    elapsed_s = time.perf_counter() - start_time
    log.write(f"\n[Exit:{stage}] returncode={returncode} elapsed_s={elapsed_s:.1f}\n")
    log.flush()
    return returncode, "\n".join(tail)


def result_from_job(job: Job, returncode: int, error: str = "") -> Result:
    return Result(
        task_id=job.task_id,
        repo_id=job.repo_id,
        returncode=returncode,
        train_dir=str(job.train_dir),
        val_dir=str(job.val_dir),
        annotations_jsonl=str(job.annotations_jsonl),
        log_path=str(job.log_path),
        error=error[-4000:],
    )


def render_job(job: Job) -> str:
    lines = [f"[Task] {job.task_id}", f"[Repo] {job.repo_id}", f"[Log] {job.log_path}"]
    if job.annotation_cmd is not None:
        lines.append("[AnnotationCmd] " + " ".join(job.annotation_cmd))
    lines.append("[ExportCmd] " + " ".join(job.cmd))
    return "\n".join(lines)


def normalize_passthrough(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


if __name__ == "__main__":
    main()
