from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import dataclasses
import json
import pathlib
import random

import tyro

from openpi.hl_memory.crosstask import CrossTaskVideoRecord
from openpi.hl_memory.crosstask import build_coverage_report
from openpi.hl_memory.crosstask import index_local_videos
from openpi.hl_memory.crosstask import probe_video_index_decodability
from openpi.hl_memory.crosstask import read_task_info
from openpi.hl_memory.crosstask import read_video_records
from openpi.hl_memory.crosstask import write_video_records


@dataclasses.dataclass
class SplitArgs:
    crosstask_release_dir: pathlib.Path
    videos_root: pathlib.Path
    output_dir: pathlib.Path
    tasks_file: str = "tasks_primary.txt"
    train_videos_csv: str = "videos.csv"
    val_videos_csv: str = "videos_val.csv"
    annotations_dir: str = "annotations"
    val_ratio: float = 0.2
    seed: int = 0
    verify_decodable: bool = True
    num_probe_positions: int = 5


def main(args: SplitArgs) -> None:
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_task_info(args.crosstask_release_dir / args.tasks_file)
    train_records = read_video_records(args.crosstask_release_dir / args.train_videos_csv)
    val_records = read_video_records(args.crosstask_release_dir / args.val_videos_csv)
    seen = set()
    records: list[CrossTaskVideoRecord] = []
    for record in [*train_records, *val_records]:
        key = (record.task_id, record.video_id)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)

    video_index = index_local_videos(args.videos_root)
    coverage = build_coverage_report(
        records,
        tasks=tasks,
        crosstask_release_dir=args.crosstask_release_dir,
        annotations_dir=args.annotations_dir,
        video_index=video_index,
    )
    matched_records = list(coverage.matched_records)
    decodable_report = None
    corrupt_matched_records: list[CrossTaskVideoRecord] = []
    if args.verify_decodable:
        decodable_report = probe_video_index_decodability(
            video_index,
            num_probe_positions=args.num_probe_positions,
        )
        decodable_ids = set(decodable_report.decodable_video_index)
        corrupt_ids = set(decodable_report.corrupt_video_index)
        corrupt_matched_records = [record for record in matched_records if record.video_id in corrupt_ids]
        matched_records = [record for record in matched_records if record.video_id in decodable_ids]

    train_split, val_split = _split_records_by_task(
        matched_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_csv = args.output_dir / "train_records.csv"
    val_csv = args.output_dir / "val_records.csv"
    write_video_records(train_csv, train_split)
    write_video_records(val_csv, val_split)

    summary = {
        "candidate_records": coverage.total_records,
        "matched_records": coverage.matched_count,
        "decodable_matched_records": len(matched_records),
        "corrupt_matched_records": len(corrupt_matched_records),
        "missing_annotations": coverage.missing_annotations,
        "missing_local_videos": coverage.missing_local_videos,
        "indexed_decodable_video_files": None if decodable_report is None else decodable_report.decodable_count,
        "indexed_corrupt_video_files": None if decodable_report is None else decodable_report.corrupt_count,
        "train_records": len(train_split),
        "val_records": len(val_split),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "verify_decodable": args.verify_decodable,
        "num_probe_positions": args.num_probe_positions,
        "tasks_in_train": len({record.task_id for record in train_split}),
        "tasks_in_val": len({record.task_id for record in val_split}),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")

    print("CrossTask matched split")
    print(f"  matched_records:      {summary['matched_records']}")
    if args.verify_decodable:
        print(f"  decodable_matched:   {summary['decodable_matched_records']}")
        print(f"  corrupt_matched:     {summary['corrupt_matched_records']}")
        print(f"  indexed_decodable:   {summary['indexed_decodable_video_files']}")
        print(f"  indexed_corrupt:     {summary['indexed_corrupt_video_files']}")
    print(f"  missing_annotations: {summary['missing_annotations']}")
    print(f"  missing_local_videos:{summary['missing_local_videos']}")
    print(f"  train_records:       {summary['train_records']}")
    print(f"  val_records:         {summary['val_records']}")
    print(f"  tasks_in_train:      {summary['tasks_in_train']}")
    print(f"  tasks_in_val:        {summary['tasks_in_val']}")
    print(f"  train_csv:           {train_csv}")
    print(f"  val_csv:             {val_csv}")


def _split_records_by_task(
    records: Sequence[CrossTaskVideoRecord],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[CrossTaskVideoRecord], list[CrossTaskVideoRecord]]:
    grouped: dict[str, list[CrossTaskVideoRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task_id].append(record)

    rng = random.Random(seed)
    train_split: list[CrossTaskVideoRecord] = []
    val_split: list[CrossTaskVideoRecord] = []

    for task_id in sorted(grouped):
        task_records = list(grouped[task_id])
        rng.shuffle(task_records)
        if len(task_records) == 1:
            train_split.extend(task_records)
            continue

        val_count = int(round(len(task_records) * val_ratio))
        val_count = max(1, val_count)
        val_count = min(len(task_records) - 1, val_count)

        val_split.extend(task_records[:val_count])
        train_split.extend(task_records[val_count:])

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    return train_split, val_split


if __name__ == "__main__":
    main(tyro.cli(SplitArgs))
