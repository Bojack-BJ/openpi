#!/usr/bin/env python3
"""Batch convert LeRobot subtask repos plus annotations into HL-memory train data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
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
SUMMARY_NAME = "batch_hl_memory_export_summary.json"


@dataclass(frozen=True)
class Job:
    task_id: str
    task_dir: Path
    repo_id: str
    annotations_jsonl: Path
    train_dir: Path
    val_dir: Path
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
    parser.add_argument("--dry-run", action="store_true")
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
            future_to_job = {pool.submit(run_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - CLI diagnostic path
                    result = result_from_job(job, returncode=1, error=str(exc))
                results.append(result)
                status = "OK" if result.returncode == 0 else f"FAIL:{result.returncode}"
                print(f"[{status}] {job.task_id} repo_id={job.repo_id}")
                if result.returncode != 0 and not args.continue_on_error:
                    raise SystemExit(f"Task {job.task_id} failed: {result.error}")

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
        annotations_jsonl = task_dir / "hl_annotations.jsonl"
        if not annotations_jsonl.exists() and not args.auto_export_annotations:
            print(f"[Skip] {task_id}: missing {annotations_jsonl}; pass --auto-export-annotations to create it")
            continue
        train_dir = args.output_root / task_id / "train"
        val_dir = args.output_root / task_id / "val"
        if args.skip_existing and not args.overwrite and (train_dir / "samples.jsonl").exists() and (val_dir / "samples.jsonl").exists():
            print(f"[Skip] {task_id}: train/val samples already exist")
            continue
        annotation_cmd = None
        if args.auto_export_annotations and (args.overwrite or not annotations_jsonl.exists()):
            annotation_cmd = [
                sys.executable,
                str(args.annotation_script.resolve()),
                "--subtask-segments-json",
                str(task_dir / "subtask_segments.json"),
                "--output-jsonl",
                str(annotations_jsonl),
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
        jobs.append(Job(task_id, task_dir, repo_id, annotations_jsonl, train_dir, val_dir, cmd, annotation_cmd))
    return jobs


def run_job(job: Job) -> Result:
    env = os.environ.copy()
    env.setdefault("HF_LEROBOT_HOME", str(job.task_dir.parent.parent))
    if job.annotation_cmd is not None:
        annotation = subprocess.run(job.annotation_cmd, text=True, capture_output=True, check=False, env=env)
        if annotation.returncode != 0:
            return result_from_job(job, annotation.returncode, annotation.stderr or annotation.stdout)
    proc = subprocess.run(job.cmd, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        return result_from_job(job, proc.returncode, proc.stderr or proc.stdout)
    return result_from_job(job, 0)


def result_from_job(job: Job, returncode: int, error: str = "") -> Result:
    return Result(
        task_id=job.task_id,
        repo_id=job.repo_id,
        returncode=returncode,
        train_dir=str(job.train_dir),
        val_dir=str(job.val_dir),
        annotations_jsonl=str(job.annotations_jsonl),
        error=error[-4000:],
    )


def render_job(job: Job) -> str:
    lines = [f"[Task] {job.task_id}", f"[Repo] {job.repo_id}"]
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
