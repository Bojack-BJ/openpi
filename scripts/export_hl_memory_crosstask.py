from __future__ import annotations

from collections import defaultdict
import dataclasses
import json
import logging
import pathlib
from typing import Any

import cv2
from PIL import Image
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.crosstask import build_subtask_annotations
from openpi.hl_memory.crosstask import read_segments
from openpi.hl_memory.crosstask import read_task_info
from openpi.hl_memory.crosstask import read_video_records
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.labels import DEFAULT_LANGUAGE_MEMORY
from openpi.hl_memory.labels import TaskProgressState
from openpi.hl_memory.labels import derive_keyframe_positions
from openpi.hl_memory.labels import render_language_memory
from openpi.hl_memory.labels import update_progress_state
from openpi.hl_memory.memory import EpisodicKeyframeMemory
from openpi.hl_memory.memory import build_recent_context_indices
from openpi.hl_memory.memory import map_relative_positions_to_absolute


@dataclasses.dataclass
class ExportCrossTaskArgs:
    crosstask_release_dir: pathlib.Path
    videos_root: pathlib.Path
    output_dir: pathlib.Path
    split: str = "train"
    tasks_file: str = "tasks_primary.txt"
    train_videos_csv: str = "videos.csv"
    val_videos_csv: str = "videos_val.csv"
    annotations_dir: str = "annotations"
    recent_frames_length: int = 8
    frame_subsample: int = 1
    memory_length: int = 8
    merge_distance: int = 1
    frame_height: int = 224
    frame_width: int = 224
    max_videos: int | None = None
    max_seconds_per_video: int | None = None
    overwrite: bool = False


def main(args: ExportCrossTaskArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.split not in {"train", "val", "all"}:
        raise ValueError("split must be one of: train, val, all")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists and is not empty. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frames").mkdir(exist_ok=True)

    hl_config = HLMemoryConfig(
        recent_frames_length=args.recent_frames_length,
        frame_subsample=args.frame_subsample,
        memory_length=args.memory_length,
        merge_distance=args.merge_distance,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
    )

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

    records = [record for record in records if record.task_id in tasks]
    if args.max_videos is not None:
        records = records[: args.max_videos]

    video_index = _index_video_files(args.videos_root)
    frame_cache: dict[tuple[str, int], str] = {}
    samples: list[ExportedHLMemorySample] = []

    for episode_index, record in enumerate(records):
        segment_path = args.crosstask_release_dir / args.annotations_dir / f"{record.task_id}_{record.video_id}.csv"
        if not segment_path.exists():
            logging.warning("Skipping %s/%s because annotation file is missing: %s", record.task_id, record.video_id, segment_path)
            continue
        video_path = video_index.get(record.video_id)
        if video_path is None:
            logging.warning("Skipping %s/%s because no local video file matched stem `%s`.", record.task_id, record.video_id, record.video_id)
            continue

        segments = read_segments(segment_path)
        annotations = build_subtask_annotations(
            episode_index=episode_index,
            task=tasks[record.task_id],
            segments=segments,
        )
        if args.max_seconds_per_video is not None:
            annotations = [annotation for annotation in annotations if annotation.frame_index < args.max_seconds_per_video]
        if not annotations:
            continue

        samples.extend(
            _export_episode(
                episode_index=episode_index,
                video_id=record.video_id,
                video_path=video_path,
                annotations=annotations,
                output_dir=args.output_dir,
                frame_cache=frame_cache,
                hl_config=hl_config,
            )
        )

    _write_jsonl(args.output_dir / "samples.jsonl", [sample.to_dict() for sample in samples])
    metadata = {
        "schema_version": "hl_memory_v1_crosstask",
        "source": "crosstask",
        "split": args.split,
        "num_samples": len(samples),
        "num_videos": len({sample.episode_index for sample in samples}),
        "hl_memory_config": dataclasses.asdict(hl_config),
        "crosstask_release_dir": str(args.crosstask_release_dir),
        "videos_root": str(args.videos_root),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")
    logging.info("Exported %d CrossTask HL samples to %s", len(samples), args.output_dir)


def _export_episode(
    *,
    episode_index: int,
    video_id: str,
    video_path: pathlib.Path,
    annotations,
    output_dir: pathlib.Path,
    frame_cache: dict[tuple[str, int], str],
    hl_config: HLMemoryConfig,
) -> list[ExportedHLMemorySample]:
    progress_state = TaskProgressState()
    keyframe_memory = EpisodicKeyframeMemory(
        memory_length=hl_config.memory_length,
        merge_distance=hl_config.merge_distance,
    )
    samples: list[ExportedHLMemorySample] = []
    video_reader = _VideoFrameReader(video_path)
    try:
        for step_index, annotation in enumerate(annotations):
            recent_indices = build_recent_context_indices(
                timestep=annotation.frame_index,
                frame_subsample=hl_config.frame_subsample,
                recent_frames_length=hl_config.recent_frames_length,
            )
            recent_frame_paths = tuple(
                _ensure_frame_saved(
                    video_id,
                    second_index,
                    video_reader,
                    output_dir=output_dir,
                    cache=frame_cache,
                    hl_config=hl_config,
                )
                for second_index in recent_indices
            )
            memory_frame_paths = tuple(
                frame_cache[(video_id, second_index)]
                for second_index in keyframe_memory.visible_indices(recent_indices)
                if (video_id, second_index) in frame_cache
            )

            current_language_memory = render_language_memory(progress_state)
            next_progress_state = update_progress_state(progress_state, annotation)
            updated_language_memory = render_language_memory(next_progress_state)
            keyframe_positions = derive_keyframe_positions(annotations, step_index, recent_indices)

            sample = ExportedHLMemorySample(
                sample_id=f"crosstask_{video_id}_step_{step_index:06d}",
                episode_index=episode_index,
                step_index=step_index,
                frame_index=annotation.frame_index,
                instruction=annotation.instruction,
                language_memory=current_language_memory or DEFAULT_LANGUAGE_MEMORY,
                updated_language_memory=updated_language_memory,
                current_subtask=annotation.current_subtask,
                phase=annotation.phase or annotation.current_subtask,
                target_query=annotation.target_query,
                goal_query=annotation.goal_query,
                keyframe_positions=keyframe_positions,
                memory_frame_paths=memory_frame_paths,
                recent_frame_paths=recent_frame_paths,
                recent_frame_indices=tuple(recent_indices),
                event_type=annotation.event_type,
                event_text=annotation.event_text,
            )
            samples.append(sample)

            absolute_keyframes, _ = map_relative_positions_to_absolute(keyframe_positions, recent_indices)
            keyframe_memory.add_candidates(absolute_keyframes)
            progress_state = next_progress_state
    finally:
        video_reader.close()
    return samples


def _index_video_files(videos_root: pathlib.Path) -> dict[str, pathlib.Path]:
    index: dict[str, pathlib.Path] = {}
    for path in videos_root.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem
        if stem not in index:
            index[stem] = path
    return index


def _ensure_frame_saved(
    video_id: str,
    second_index: int,
    video_reader: "_VideoFrameReader",
    *,
    output_dir: pathlib.Path,
    cache: dict[tuple[str, int], str],
    hl_config: HLMemoryConfig,
) -> str:
    cache_key = (video_id, second_index)
    if cache_key in cache:
        return cache[cache_key]
    image = video_reader.read(second_index)
    image = image.resize((hl_config.frame_width, hl_config.frame_height), Image.Resampling.BILINEAR)
    relative_path = pathlib.Path("frames") / f"{video_id}_{second_index:06d}.png"
    image.save(output_dir / relative_path)
    cache[cache_key] = str(relative_path)
    return cache[cache_key]


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


class _VideoFrameReader:
    def __init__(self, path: pathlib.Path):
        self._path = path
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open video file: {path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self._fps = fps if fps > 0.0 else 25.0
        self._duration_sec = frame_count / self._fps if frame_count > 0.0 else 0.0

    def read(self, second_index: int) -> Image.Image:
        target_sec = float(second_index)
        if self._duration_sec > 0.0:
            max_seek_sec = max(self._duration_sec - (1.0 / self._fps), 0.0)
            target_sec = min(target_sec, max_seek_sec)

        for candidate_sec in (target_sec, max(target_sec - 0.25, 0.0), max(target_sec - 0.5, 0.0)):
            self._capture.set(cv2.CAP_PROP_POS_MSEC, candidate_sec * 1000.0)
            ok, frame = self._capture.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)

        raise ValueError(f"Failed to decode frame near second {second_index} from {self._path}")

    def close(self) -> None:
        self._capture.release()


if __name__ == "__main__":
    main(tyro.cli(ExportCrossTaskArgs))
