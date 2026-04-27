from __future__ import annotations

import dataclasses
import json
import pathlib

import tyro

from openpi.hl_memory.crosstask import build_coverage_report
from openpi.hl_memory.crosstask import index_local_videos
from openpi.hl_memory.crosstask import read_task_info
from openpi.hl_memory.crosstask import read_video_records


@dataclasses.dataclass
class CoverageArgs:
    crosstask_release_dir: pathlib.Path
    videos_root: pathlib.Path
    split: str = "all"
    tasks_file: str = "tasks_primary.txt"
    train_videos_csv: str = "videos.csv"
    val_videos_csv: str = "videos_val.csv"
    annotations_dir: str = "annotations"
    max_show_missing: int = 20
    output_json: pathlib.Path | None = None
    dump_missing_video_ids: pathlib.Path | None = None
    dump_matched_video_ids: pathlib.Path | None = None


def main(args: CoverageArgs) -> None:
    if args.split not in {"train", "val", "all"}:
        raise ValueError("split must be one of: train, val, all")

    tasks = read_task_info(args.crosstask_release_dir / args.tasks_file)
    train_records = read_video_records(args.crosstask_release_dir / args.train_videos_csv)
    val_records = read_video_records(args.crosstask_release_dir / args.val_videos_csv)

    if args.split == "train":
        records = train_records
    elif args.split == "val":
        records = val_records
    else:
        seen = set()
        records = []
        for record in [*train_records, *val_records]:
            key = (record.task_id, record.video_id)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)

    video_index = index_local_videos(args.videos_root)
    report = build_coverage_report(
        records,
        tasks=tasks,
        crosstask_release_dir=args.crosstask_release_dir,
        annotations_dir=args.annotations_dir,
        video_index=video_index,
    )

    payload = {
        "split": args.split,
        "candidate_records": report.total_records,
        "matched_local_records": report.matched_count,
        "missing_annotations": report.missing_annotations,
        "missing_local_videos": report.missing_local_videos,
        "indexed_local_video_files": len(video_index),
        "matched_video_ids": [record.video_id for record in report.matched_records],
        "missing_video_ids": [record.video_id for record in report.missing_local_video_records],
    }

    print(f"CrossTask {args.split} coverage")
    print(f"  indexed_local_video_files: {payload['indexed_local_video_files']}")
    print(f"  candidate_records:         {payload['candidate_records']}")
    print(f"  matched_local_records:     {payload['matched_local_records']}")
    print(f"  missing_annotations:       {payload['missing_annotations']}")
    print(f"  missing_local_videos:      {payload['missing_local_videos']}")

    if report.missing_local_video_records and args.max_show_missing > 0:
        print("  first_missing_video_ids:")
        for record in report.missing_local_video_records[: args.max_show_missing]:
            print(f"    - {record.task_id}/{record.video_id}")

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    if args.dump_missing_video_ids is not None:
        args.dump_missing_video_ids.write_text(
            "\n".join(record.video_id for record in report.missing_local_video_records) + "\n"
        )
    if args.dump_matched_video_ids is not None:
        args.dump_matched_video_ids.write_text("\n".join(record.video_id for record in report.matched_records) + "\n")


if __name__ == "__main__":
    main(tyro.cli(CoverageArgs))
