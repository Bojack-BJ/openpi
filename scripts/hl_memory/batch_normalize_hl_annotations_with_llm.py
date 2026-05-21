#!/usr/bin/env python3
"""Batch-normalize HL annotations JSONL files with one offline LLM load."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
import pathlib
import sys
from typing import Any

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - lightweight local help/dry-run environments.
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable


DEFAULT_ANNOTATION_ROOT = pathlib.Path("/root/Users/dataset/lerobot_home/subtask")
DEFAULT_INPUT_NAME = "hl_annotations.jsonl"
DEFAULT_OUTPUT_NAME = "hl_annotations_llm_normalized.jsonl"
DEFAULT_SIDECAR_NAME = "hl_segments_llm_sidecar.json"
SUMMARY_NAME = "batch_hl_annotation_normalize_summary.json"


@dataclass(frozen=True)
class NormalizeJob:
    task_id: str
    task_dir: pathlib.Path
    input_jsonl: pathlib.Path
    output_jsonl: pathlib.Path
    sidecar_json: pathlib.Path


@dataclass(frozen=True)
class NormalizeResult:
    task_id: str
    input_jsonl: str
    output_jsonl: str
    returncode: int
    input_rows: int | None = None
    output_rows: int | None = None
    skipped: bool = False
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run normalize_hl_annotations_with_llm.py for every task directory under an annotation root. "
            "The model is loaded once and reused across tasks."
        )
    )
    parser.add_argument(
        "--annotation-root",
        "--subtask-root",
        dest="annotation_root",
        type=pathlib.Path,
        default=DEFAULT_ANNOTATION_ROOT,
        help=f"Root containing one directory per task id (default: {DEFAULT_ANNOTATION_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=None,
        help=(
            "If set, write <output-root>/<task_id>/<output-name>. "
            "Otherwise write normalized JSONL next to each input JSONL."
        ),
    )
    parser.add_argument("--input-name", default=DEFAULT_INPUT_NAME, help=f"Input filename (default: {DEFAULT_INPUT_NAME}).")
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output filename (default: {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--sidecar-name",
        default=DEFAULT_SIDECAR_NAME,
        help=f"Task-level segment sidecar filename (default: {DEFAULT_SIDECAR_NAME}).",
    )
    parser.add_argument("--task-id-glob", default="*", help="Glob for task directories under --annotation-root.")
    parser.add_argument(
        "--only-task-id",
        action="append",
        nargs="+",
        default=[],
        help="Normalize only these task ids. Can be repeated or given multiple ids after one flag.",
    )
    parser.add_argument("--model-path", default="/root/Users/lixiaotong/Qwen3.5-27B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--granularity", choices=["task", "segment", "row"], default="task")
    parser.add_argument("--memory-summary-mode", choices=["llm", "code"], default="llm")
    parser.add_argument("--advance-threshold", type=float, default=0.85)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite each output JSONL from scratch. Equivalent to --no-resume for per-task writes.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose output JSONL already exists and has at least as many rows as input.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without loading the model.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after one task fails.")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "Number of model-loading worker processes. With N > 1 and no --worker-gpu-groups, "
            "workers are assigned to the first N visible GPUs."
        ),
    )
    parser.add_argument(
        "--worker-gpu-groups",
        default="",
        help=(
            "Optional CUDA_VISIBLE_DEVICES groups for workers, separated by semicolon. "
            "Examples: '0;1;2;3' for one GPU per worker, or '0,1;2,3' for two sharded workers. "
            "When set, it overrides --parallel-workers."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=pathlib.Path,
        default=None,
        help=f"Write a JSON summary (default: <output-root>/{SUMMARY_NAME} or annotation-root/{SUMMARY_NAME}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_root = args.annotation_root.expanduser().resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"--annotation-root is not a directory: {annotation_root}")

    jobs, skipped = build_jobs(args, annotation_root=annotation_root)
    if not jobs and not skipped:
        raise FileNotFoundError(f"No task dirs with {args.input_name} under {annotation_root}")

    print(f"[Info] annotation_root={annotation_root}")
    print(f"[Info] jobs={len(jobs)} skipped_existing={len(skipped)} output_root={args.output_root or '<next to input>'}")
    for job in jobs:
        print(render_job(job))

    if args.dry_run:
        results = [
            NormalizeResult(
                task_id=job.task_id,
                input_jsonl=str(job.input_jsonl),
                output_jsonl=str(job.output_jsonl),
                returncode=0,
                input_rows=count_jsonl_lines(job.input_jsonl),
                output_rows=None,
            )
            for job in jobs
        ]
        results.extend(skipped)
        write_summary(resolve_summary_path(args, annotation_root), annotation_root=annotation_root, results=results)
        print(f"[DryRun] planned={len(jobs)} skipped_existing={len(skipped)}")
        return

    results: list[NormalizeResult] = list(skipped)
    if should_run_parallel(args, jobs):
        results.extend(run_jobs_parallel(jobs, args=args))
    else:
        results.extend(run_jobs_serial(jobs, args=args))

    summary_path = resolve_summary_path(args, annotation_root)
    write_summary(summary_path, annotation_root=annotation_root, results=results)
    failed = [result for result in results if result.returncode != 0]
    if failed:
        rendered = ", ".join(f"{result.task_id}:{result.returncode}" for result in failed)
        raise SystemExit(f"Failed tasks ({len(failed)}): {rendered}")
    print(f"[Done] normalized={sum(1 for r in results if r.returncode == 0 and not r.skipped)} summary={summary_path}")


def should_run_parallel(args: argparse.Namespace, jobs: list[NormalizeJob]) -> bool:
    return len(jobs) > 1 and len(resolve_worker_gpu_groups(args)) > 1


def run_jobs_serial(jobs: list[NormalizeJob], *, args: argparse.Namespace) -> list[NormalizeResult]:
    normalizer_module = load_normalizer_module()
    tokenizer, model = normalizer_module._load_model(args)  # pylint: disable=protected-access
    results: list[NormalizeResult] = []
    for job in jobs:
        result = run_one_job_safely(job, args=args, tokenizer=tokenizer, model=model, normalizer_module=normalizer_module)
        results.append(result)
        print_result(result, job=job)
        if result.returncode != 0 and not args.continue_on_error:
            break
    return results


def run_jobs_parallel(jobs: list[NormalizeJob], *, args: argparse.Namespace) -> list[NormalizeResult]:
    worker_gpu_groups = resolve_worker_gpu_groups(args)
    shards = shard_jobs(jobs, shard_count=len(worker_gpu_groups))
    print(
        "[Info] parallel_workers="
        f"{len(worker_gpu_groups)} gpu_groups={','.join(group or '<inherit>' for group in worker_gpu_groups)}"
    )
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []
    for worker_index, (gpu_group, worker_jobs) in enumerate(zip(worker_gpu_groups, shards, strict=True)):
        if not worker_jobs:
            continue
        process = ctx.Process(
            target=worker_main,
            kwargs={
                "worker_index": worker_index,
                "gpu_group": gpu_group,
                "jobs": worker_jobs,
                "args": args,
                "queue": queue,
            },
        )
        process.start()
        processes.append(process)

    results: list[NormalizeResult] = []
    alive = len(processes)
    while alive:
        message = queue.get()
        if isinstance(message, dict) and message.get("type") == "result":
            result = result_from_message(message)
            results.append(result)
            print_result(result, job=None)
            if result.returncode != 0 and not args.continue_on_error:
                print("[Warn] A worker failed; waiting for existing workers to stop.", file=sys.stderr)
        elif isinstance(message, dict) and message.get("type") == "done":
            alive -= 1
        elif isinstance(message, dict) and message.get("type") == "log":
            print(str(message.get("text", "")), flush=True)

    for process in processes:
        process.join()
        if process.exitcode not in {0, None}:
            results.append(
                NormalizeResult(
                    task_id=f"worker_{process.pid}",
                    input_jsonl="",
                    output_jsonl="",
                    returncode=int(process.exitcode or 1),
                    error=f"worker process exited with code {process.exitcode}",
                )
            )
    return results


def worker_main(
    *,
    worker_index: int,
    gpu_group: str,
    jobs: list[NormalizeJob],
    args: argparse.Namespace,
    queue: Any,
) -> None:
    try:
        if gpu_group:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
        queue.put(
            {
                "type": "log",
                "text": (
                    f"[Worker {worker_index}] jobs={len(jobs)} "
                    f"CUDA_VISIBLE_DEVICES={gpu_group or os.environ.get('CUDA_VISIBLE_DEVICES', '<inherit>')}"
                ),
            }
        )
        normalizer_module = load_normalizer_module()
        tokenizer, model = normalizer_module._load_model(args)  # pylint: disable=protected-access
        for job in jobs:
            result = run_one_job_safely(job, args=args, tokenizer=tokenizer, model=model, normalizer_module=normalizer_module)
            queue.put(result_to_message(result))
            if result.returncode != 0 and not args.continue_on_error:
                break
    except Exception as exc:  # noqa: BLE001
        queue.put(result_to_message(
            NormalizeResult(
                task_id=f"worker_{worker_index}",
                input_jsonl="",
                output_jsonl="",
                returncode=1,
                error=f"{type(exc).__name__}: {exc}",
            )
        ))
    finally:
        queue.put({"type": "done", "worker_index": worker_index})


def run_one_job_safely(
    job: NormalizeJob,
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    normalizer_module: Any,
) -> NormalizeResult:
    try:
        return run_one_job(job, args=args, tokenizer=tokenizer, model=model, normalizer_module=normalizer_module)
    except Exception as exc:  # noqa: BLE001
        return NormalizeResult(
            task_id=job.task_id,
            input_jsonl=str(job.input_jsonl),
            output_jsonl=str(job.output_jsonl),
            returncode=1,
            input_rows=count_jsonl_lines(job.input_jsonl),
            output_rows=count_jsonl_lines(job.output_jsonl),
            error=f"{type(exc).__name__}: {exc}",
        )


def print_result(result: NormalizeResult, *, job: NormalizeJob | None) -> None:
    output_jsonl = job.output_jsonl if job is not None else result.output_jsonl
    status = "OK" if result.returncode == 0 else f"FAIL:{result.returncode}"
    print(f"[{status}] task_id={result.task_id} rows={result.output_rows} -> {output_jsonl}", flush=True)
    if result.error:
        print(f"[Error] {result.task_id}: {result.error}", file=sys.stderr, flush=True)


def result_to_message(result: NormalizeResult) -> dict[str, Any]:
    return {
        "type": "result",
        "task_id": result.task_id,
        "input_jsonl": result.input_jsonl,
        "output_jsonl": result.output_jsonl,
        "returncode": result.returncode,
        "input_rows": result.input_rows,
        "output_rows": result.output_rows,
        "skipped": result.skipped,
        "error": result.error,
    }


def result_from_message(message: dict[str, Any]) -> NormalizeResult:
    return NormalizeResult(
        task_id=str(message.get("task_id", "")),
        input_jsonl=str(message.get("input_jsonl", "")),
        output_jsonl=str(message.get("output_jsonl", "")),
        returncode=int(message.get("returncode", 1)),
        input_rows=message.get("input_rows"),
        output_rows=message.get("output_rows"),
        skipped=bool(message.get("skipped", False)),
        error=str(message.get("error", "")),
    )


def resolve_worker_gpu_groups(args: argparse.Namespace) -> list[str]:
    if args.worker_gpu_groups.strip():
        return [group.strip() for group in args.worker_gpu_groups.split(";") if group.strip()]
    parallel_workers = max(int(args.parallel_workers), 1)
    if parallel_workers == 1:
        return [""]
    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not visible_devices:
        visible_devices = [str(index) for index in range(parallel_workers)]
    return [visible_devices[index % len(visible_devices)] for index in range(parallel_workers)]


def shard_jobs(jobs: list[NormalizeJob], *, shard_count: int) -> list[list[NormalizeJob]]:
    shards: list[list[NormalizeJob]] = [[] for _ in range(shard_count)]
    costs = [0] * shard_count
    sorted_jobs = sorted(jobs, key=lambda job: count_jsonl_lines(job.input_jsonl) or 0, reverse=True)
    for job in sorted_jobs:
        shard_index = min(range(shard_count), key=lambda index: costs[index])
        shards[shard_index].append(job)
        costs[shard_index] += count_jsonl_lines(job.input_jsonl) or 1
    return shards


def build_jobs(args: argparse.Namespace, *, annotation_root: pathlib.Path) -> tuple[list[NormalizeJob], list[NormalizeResult]]:
    allowed = flatten_only_task_ids(args.only_task_id)
    jobs: list[NormalizeJob] = []
    skipped: list[NormalizeResult] = []
    for task_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir()):
        task_id = task_dir.name
        if args.task_id_glob != "*" and not task_dir.match(args.task_id_glob):
            continue
        if allowed and task_id not in allowed:
            continue
        input_jsonl = task_dir / args.input_name
        if not input_jsonl.is_file():
            continue
        output_jsonl = resolve_output_path(args, task_id=task_id, task_dir=task_dir)
        sidecar_json = resolve_sidecar_path(args, task_id=task_id, task_dir=task_dir)
        input_rows = count_jsonl_lines(input_jsonl)
        output_rows = count_jsonl_lines(output_jsonl)
        if (
            args.skip_existing
            and not args.overwrite
            and input_rows is not None
            and output_rows is not None
            and output_rows >= input_rows
        ):
            skipped.append(
                NormalizeResult(
                    task_id=task_id,
                    input_jsonl=str(input_jsonl),
                    output_jsonl=str(output_jsonl),
                    returncode=0,
                    input_rows=input_rows,
                    output_rows=output_rows,
                    skipped=True,
                )
            )
            continue
        jobs.append(
            NormalizeJob(
                task_id=task_id,
                task_dir=task_dir,
                input_jsonl=input_jsonl,
                output_jsonl=output_jsonl,
                sidecar_json=sidecar_json,
            )
        )
    if allowed:
        found = {job.task_id for job in jobs} | {result.task_id for result in skipped}
        missing = sorted(allowed - found)
        if missing:
            print(f"[Warn] requested task ids with no input JSONL: {', '.join(missing)}", file=sys.stderr)
    return jobs, skipped


def run_one_job(
    job: NormalizeJob,
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    normalizer_module: Any,
) -> NormalizeResult:
    rows = normalizer_module._read_jsonl(job.input_jsonl)  # pylint: disable=protected-access
    if args.limit_per_task is not None:
        rows = rows[: args.limit_per_task]
    job_args = argparse.Namespace(**vars(args))
    job_args.input_jsonl = job.input_jsonl
    job_args.output_jsonl = job.output_jsonl
    job_args.sidecar_json = job.sidecar_json
    resume = bool(job_args.resume and not job_args.overwrite)
    done = normalizer_module._read_done(job.output_jsonl) if resume else set()  # pylint: disable=protected-access
    job.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"

    with job.output_jsonl.open(mode, encoding="utf-8") as stream:
        normalizer_module.normalize_rows(  # pylint: disable=protected-access
            rows,
            args=job_args,
            tokenizer=tokenizer,
            model=model,
            done=done,
            stream=stream,
        )

    return NormalizeResult(
        task_id=job.task_id,
        input_jsonl=str(job.input_jsonl),
        output_jsonl=str(job.output_jsonl),
        returncode=0,
        input_rows=len(rows),
        output_rows=count_jsonl_lines(job.output_jsonl),
    )


def resolve_output_path(args: argparse.Namespace, *, task_id: str, task_dir: pathlib.Path) -> pathlib.Path:
    if args.output_root is not None:
        return (args.output_root / task_id / args.output_name).expanduser().resolve()
    return (task_dir / args.output_name).resolve()


def resolve_sidecar_path(args: argparse.Namespace, *, task_id: str, task_dir: pathlib.Path) -> pathlib.Path:
    if args.output_root is not None:
        return (args.output_root / task_id / args.sidecar_name).expanduser().resolve()
    return (task_dir / args.sidecar_name).resolve()


def load_normalizer_module() -> Any:
    script_dir = pathlib.Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import normalize_hl_annotations_with_llm  # pylint: disable=import-outside-toplevel

    return normalize_hl_annotations_with_llm


def resolve_summary_path(args: argparse.Namespace, annotation_root: pathlib.Path) -> pathlib.Path:
    if args.summary_json is not None:
        return args.summary_json.expanduser().resolve()
    if args.output_root is not None:
        return (args.output_root / SUMMARY_NAME).expanduser().resolve()
    return annotation_root / SUMMARY_NAME


def flatten_only_task_ids(values: list[list[str]]) -> set[str]:
    return {item for group in values for item in group}


def count_jsonl_lines(path: pathlib.Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count


def render_job(job: NormalizeJob) -> str:
    return "\n".join(
        [
            f"[Task] {job.task_id}",
            f"[In]   {job.input_jsonl}",
            f"[Out]  {job.output_jsonl}",
            f"[Side] {job.sidecar_json}",
        ]
    )


def write_summary(summary_path: pathlib.Path, *, annotation_root: pathlib.Path, results: list[NormalizeResult]) -> None:
    payload = {
        "annotation_root": str(annotation_root),
        "task_count": len(results),
        "normalized": sum(1 for item in results if item.returncode == 0 and not item.skipped),
        "skipped_existing": sum(1 for item in results if item.skipped),
        "failed": sum(1 for item in results if item.returncode != 0),
        "tasks": [
            {
                "task_id": item.task_id,
                "input_jsonl": item.input_jsonl,
                "output_jsonl": item.output_jsonl,
                "returncode": item.returncode,
                "input_rows": item.input_rows,
                "output_rows": item.output_rows,
                "skipped": item.skipped,
                "error": item.error,
            }
            for item in sorted(results, key=lambda row: row.task_id)
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
