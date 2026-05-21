#!/usr/bin/env python3
"""Batch-normalize HL annotations JSONL files with one offline LLM load."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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
SUMMARY_NAME = "batch_hl_annotation_normalize_summary.json"


@dataclass(frozen=True)
class NormalizeJob:
    task_id: str
    task_dir: pathlib.Path
    input_jsonl: pathlib.Path
    output_jsonl: pathlib.Path


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
    parser.add_argument("--granularity", choices=["segment", "row"], default="segment")
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

    normalizer_module = load_normalizer_module()
    tokenizer, model = normalizer_module._load_model(args)  # pylint: disable=protected-access
    results: list[NormalizeResult] = list(skipped)
    for job in jobs:
        try:
            result = run_one_job(job, args=args, tokenizer=tokenizer, model=model, normalizer_module=normalizer_module)
        except Exception as exc:  # noqa: BLE001
            result = NormalizeResult(
                task_id=job.task_id,
                input_jsonl=str(job.input_jsonl),
                output_jsonl=str(job.output_jsonl),
                returncode=1,
                input_rows=count_jsonl_lines(job.input_jsonl),
                output_rows=count_jsonl_lines(job.output_jsonl),
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        status = "OK" if result.returncode == 0 else f"FAIL:{result.returncode}"
        print(f"[{status}] task_id={job.task_id} rows={result.output_rows} -> {job.output_jsonl}")
        if result.returncode != 0 and not args.continue_on_error:
            break

    summary_path = resolve_summary_path(args, annotation_root)
    write_summary(summary_path, annotation_root=annotation_root, results=results)
    failed = [result for result in results if result.returncode != 0]
    if failed:
        rendered = ", ".join(f"{result.task_id}:{result.returncode}" for result in failed)
        raise SystemExit(f"Failed tasks ({len(failed)}): {rendered}")
    print(f"[Done] normalized={sum(1 for r in results if r.returncode == 0 and not r.skipped)} summary={summary_path}")


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
        jobs.append(NormalizeJob(task_id=task_id, task_dir=task_dir, input_jsonl=input_jsonl, output_jsonl=output_jsonl))
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
    resume = bool(args.resume and not args.overwrite)
    done = normalizer_module._read_done(job.output_jsonl) if resume else set()  # pylint: disable=protected-access
    job.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"

    with job.output_jsonl.open(mode, encoding="utf-8") as stream:
        normalizer_module.normalize_rows(  # pylint: disable=protected-access
            rows,
            args=args,
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
