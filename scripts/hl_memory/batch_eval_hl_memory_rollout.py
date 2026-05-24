#!/usr/bin/env python3
"""Batch run HL-memory rollout eval over multiple exported task datasets."""

from __future__ import annotations

import argparse
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
from typing import Any


DEFAULT_EVAL_SCRIPT = Path(__file__).with_name("eval_hl_memory_rollout.py")
SUMMARY_NAME = "batch_hl_memory_eval_summary.json"


@dataclass(frozen=True)
class Job:
    task_id: str
    dataset_dir: Path
    output_json: Path
    log_path: Path
    cmd: list[str]
    gpu_id: str | None = None


@dataclass(frozen=True)
class Result:
    task_id: str
    dataset_dir: str
    output_json: str
    log_path: str
    gpu_id: str | None
    returncode: int
    elapsed_s: float
    metrics: dict[str, Any] | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/hl_memory/eval_hl_memory_rollout.py for every exported task split under a root and "
            "aggregate per-task ablation metrics. Extra eval args go after `--`."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-glob", default="*/val", help="Glob under dataset-root. Default: */val.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id-glob", default="*")
    parser.add_argument("--only-task-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1, help="Parallel eval processes. Use 1 for one GPU/model copy.")
    parser.add_argument(
        "--gpu-ids",
        default="",
        help=(
            "Comma-separated physical CUDA ids assigned round-robin to eval subprocesses, e.g. 0,1,2,3. "
            "Each subprocess sees one GPU via CUDA_VISIBLE_DEVICES, so pass --device cuda to the eval script."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--status-interval-s", type=float, default=60.0)
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Extra args for eval_hl_memory_rollout.py after --.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    passthrough = _normalize_passthrough(args.passthrough)
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    jobs = _build_jobs(args, passthrough=passthrough, gpu_ids=gpu_ids)
    if not jobs:
        raise FileNotFoundError(f"No dataset dirs found under {args.dataset_root} matching {args.dataset_glob!r}")
    gpu_info = f" gpu_ids={','.join(gpu_ids)}" if gpu_ids else ""
    if gpu_ids and args.workers > len(gpu_ids):
        print(
            f"[Warn] workers={args.workers} > gpu_ids={len(gpu_ids)}; multiple eval jobs may share a GPU.",
            flush=True,
        )
    print(f"[Info] tasks={len(jobs)} workers={args.workers}{gpu_info} output_root={args.output_root}", flush=True)

    results: list[Result] = []
    if args.dry_run:
        for job in jobs:
            print(_render_job(job))
            results.append(_result_from_job(job, returncode=0, elapsed_s=0.0, metrics={}))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_job = {}
            for job in jobs:
                gpu_text = f" gpu={job.gpu_id}" if job.gpu_id is not None else ""
                print(f"[Queue] {job.task_id}{gpu_text} dataset={job.dataset_dir} log={job.log_path}", flush=True)
                future_to_job[pool.submit(_run_job, job, stream_output=args.stream_output)] = job
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
                        result = _result_from_job(job, returncode=1, elapsed_s=0.0, error=str(exc))
                    results.append(result)
                    status = "OK" if result.returncode == 0 else f"FAIL:{result.returncode}"
                    print(
                        f"[{status}] {job.task_id} elapsed={result.elapsed_s:.1f}s output={job.output_json}",
                        flush=True,
                    )
                    if result.returncode != 0 and not args.continue_on_error:
                        raise SystemExit(f"Task {job.task_id} failed: {result.error}\nLog: {job.log_path}")

    summary = {
        "dataset_root": str(args.dataset_root),
        "dataset_glob": args.dataset_glob,
        "eval_script": str(args.eval_script),
        "passthrough": passthrough,
        "aggregate": _aggregate_metrics(results),
        "results": [_result_to_dict(result) for result in sorted(results, key=lambda item: item.task_id)],
    }
    summary_path = args.summary_json or (args.output_root / SUMMARY_NAME)
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    failed = [result for result in results if result.returncode != 0]
    if failed:
        raise SystemExit(f"Failed tasks: {', '.join(result.task_id for result in failed)}")
    print(f"[Done] evaluated={len(results)} summary={summary_path}", flush=True)


def _build_jobs(args: argparse.Namespace, *, passthrough: list[str], gpu_ids: list[str]) -> list[Job]:
    dataset_root = args.dataset_root.resolve()
    allowed = set(args.only_task_id)
    dataset_dirs = sorted(path for path in dataset_root.glob(args.dataset_glob) if (path / "samples.jsonl").is_file())
    jobs: list[Job] = []
    for dataset_dir in dataset_dirs:
        task_id = _task_id_from_dataset_dir(dataset_root, dataset_dir)
        if allowed and task_id not in allowed:
            continue
        if not Path(task_id).match(args.task_id_glob):
            continue
        output_dir = args.output_root / task_id
        output_json = output_dir / "eval_metrics.json"
        log_path = output_dir / "eval.log"
        if args.skip_existing and not args.overwrite and output_json.exists():
            print(f"[Skip] {task_id}: output exists at {output_json}", flush=True)
            continue
        cmd = [
            sys.executable,
            str(args.eval_script.resolve()),
            "--dataset-dir",
            str(dataset_dir),
            "--output-json",
            str(output_json),
            *passthrough,
        ]
        gpu_id = gpu_ids[len(jobs) % len(gpu_ids)] if gpu_ids else None
        jobs.append(
            Job(
                task_id=task_id,
                dataset_dir=dataset_dir,
                output_json=output_json,
                log_path=log_path,
                cmd=cmd,
                gpu_id=gpu_id,
            )
        )
    return jobs


def _run_job(job: Job, *, stream_output: bool) -> Result:
    started_at = time.perf_counter()
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_json.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("w", encoding="utf-8") as log:
        log.write(_render_job(job) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            job.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_job_env(job),
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if stream_output:
                print(f"[{job.task_id}] {line}", end="", flush=True)
        returncode = process.wait()
    elapsed_s = time.perf_counter() - started_at
    metrics: dict[str, Any] | None = None
    error = ""
    if returncode == 0:
        try:
            metrics = json.loads(job.output_json.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            returncode = 1
            error = f"Failed to read metrics JSON: {type(exc).__name__}: {exc}"
    else:
        error = f"eval command exited with {returncode}"
    return _result_from_job(job, returncode=returncode, elapsed_s=elapsed_s, metrics=metrics, error=error)


def _aggregate_metrics(results: list[Result]) -> dict[str, dict[str, float]]:
    successful = [result for result in results if result.returncode == 0 and isinstance(result.metrics, dict)]
    per_mode_values: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for result in successful:
        assert result.metrics is not None
        for mode, metrics in result.metrics.items():
            if not isinstance(metrics, dict):
                continue
            num_steps = float(metrics.get("num_steps", 0.0) or 0.0)
            num_episodes = float(metrics.get("num_episodes", 0.0) or 0.0)
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                if key in {"num_steps", "num_generate_batches"}:
                    weight = 1.0
                    weighted_value = float(value)
                elif key == "num_episodes":
                    weight = 1.0
                    weighted_value = float(value)
                else:
                    weight = num_episodes if key == "episode_sequence_accuracy" else num_steps
                    weighted_value = float(value)
                per_mode_values.setdefault(mode, {}).setdefault(key, []).append((weighted_value, weight))

    aggregate: dict[str, dict[str, float]] = {}
    for mode, metrics in per_mode_values.items():
        aggregate[mode] = {}
        for key, values in metrics.items():
            if key in {"num_steps", "num_episodes", "num_generate_batches"}:
                aggregate[mode][key] = sum(value for value, _ in values)
                continue
            weighted_total = sum(value * weight for value, weight in values)
            weight_total = sum(weight for _, weight in values)
            aggregate[mode][key] = weighted_total / weight_total if weight_total > 0 else 0.0
    return aggregate


def _task_id_from_dataset_dir(dataset_root: Path, dataset_dir: Path) -> str:
    relative = dataset_dir.relative_to(dataset_root)
    parts = relative.parts
    if len(parts) >= 2 and parts[-1] in {"train", "val", "test"}:
        return "/".join(parts[:-1])
    return "/".join(parts)


def _normalize_passthrough(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _parse_gpu_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _job_env(job: Job) -> dict[str, str] | None:
    if job.gpu_id is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = job.gpu_id
    return env


def _render_job(job: Job) -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={job.gpu_id} " if job.gpu_id is not None else ""
    return prefix + " ".join(job.cmd)


def _result_from_job(
    job: Job,
    *,
    returncode: int,
    elapsed_s: float,
    metrics: dict[str, Any] | None = None,
    error: str = "",
) -> Result:
    return Result(
        task_id=job.task_id,
        dataset_dir=str(job.dataset_dir),
        output_json=str(job.output_json),
        log_path=str(job.log_path),
        gpu_id=job.gpu_id,
        returncode=returncode,
        elapsed_s=elapsed_s,
        metrics=metrics,
        error=error,
    )


def _result_to_dict(result: Result) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "dataset_dir": result.dataset_dir,
        "output_json": result.output_json,
        "log_path": result.log_path,
        "gpu_id": result.gpu_id,
        "returncode": result.returncode,
        "elapsed_s": result.elapsed_s,
        "metrics": result.metrics,
        "error": result.error,
    }


if __name__ == "__main__":
    main()
