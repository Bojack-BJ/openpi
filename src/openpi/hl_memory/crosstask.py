from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import math
import pathlib

from openpi.hl_memory.labels import SubtaskAnnotation


@dataclasses.dataclass(frozen=True)
class CrossTaskTaskInfo:
    task_id: str
    title: str
    url: str
    steps: tuple[str, ...]

    @property
    def num_steps(self) -> int:
        return len(self.steps)


@dataclasses.dataclass(frozen=True)
class CrossTaskVideoRecord:
    task_id: str
    video_id: str
    url: str


@dataclasses.dataclass(frozen=True)
class CrossTaskSegment:
    step_index: int
    start_sec: float
    end_sec: float


@dataclasses.dataclass(frozen=True)
class CrossTaskCoverageReport:
    total_records: int
    matched_records: tuple[CrossTaskVideoRecord, ...]
    missing_annotation_records: tuple[CrossTaskVideoRecord, ...]
    missing_local_video_records: tuple[CrossTaskVideoRecord, ...]

    @property
    def matched_count(self) -> int:
        return len(self.matched_records)

    @property
    def missing_annotations(self) -> int:
        return len(self.missing_annotation_records)

    @property
    def missing_local_videos(self) -> int:
        return len(self.missing_local_video_records)


def read_task_info(path: pathlib.Path | str) -> dict[str, CrossTaskTaskInfo]:
    path = pathlib.Path(path)
    tasks: dict[str, CrossTaskTaskInfo] = {}
    with path.open() as handle:
        while True:
            task_id = handle.readline()
            if task_id == "":
                break
            task_id = task_id.strip()
            if not task_id:
                continue
            title = handle.readline().strip()
            url = handle.readline().strip()
            num_steps = int(handle.readline().strip())
            steps = tuple(step.strip() for step in handle.readline().strip().split(",") if step.strip())
            if len(steps) != num_steps:
                raise ValueError(f"Task {task_id} declares {num_steps} steps but listed {len(steps)} in {path}.")
            tasks[task_id] = CrossTaskTaskInfo(task_id=task_id, title=title, url=url, steps=steps)
            # Consume blank separator line when present.
            _ = handle.readline()
    return tasks


def read_video_records(path: pathlib.Path | str) -> list[CrossTaskVideoRecord]:
    path = pathlib.Path(path)
    records: list[CrossTaskVideoRecord] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",", maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f"Expected `task,video_id,url` on line {line_number} in {path}.")
            task_id, video_id, url = parts
            records.append(CrossTaskVideoRecord(task_id=task_id, video_id=video_id, url=url))
    return records


def write_video_records(path: pathlib.Path | str, records: Iterable[CrossTaskVideoRecord]) -> None:
    path = pathlib.Path(path)
    with path.open("w") as handle:
        for record in records:
            handle.write(f"{record.task_id},{record.video_id},{record.url}\n")


def read_segments(path: pathlib.Path | str) -> list[CrossTaskSegment]:
    path = pathlib.Path(path)
    segments: list[CrossTaskSegment] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            if len(parts) != 3:
                raise ValueError(f"Expected `step,start,end` on line {line_number} in {path}.")
            raw_step, raw_start, raw_end = parts
            step_index = int(raw_step) - 1
            start_sec = float(raw_start)
            end_sec = float(raw_end)
            if end_sec < start_sec:
                raise ValueError(f"Invalid segment on line {line_number} in {path}: end < start.")
            segments.append(CrossTaskSegment(step_index=step_index, start_sec=start_sec, end_sec=end_sec))
    segments.sort(key=lambda item: (item.start_sec, item.end_sec, item.step_index))
    return segments


def build_subtask_annotations(
    *,
    episode_index: int,
    task: CrossTaskTaskInfo,
    segments: Iterable[CrossTaskSegment],
) -> list[SubtaskAnnotation]:
    annotations: list[SubtaskAnnotation] = []
    for segment in segments:
        if segment.step_index < 0 or segment.step_index >= len(task.steps):
            raise ValueError(
                f"Segment step_index={segment.step_index} is out of range for task {task.task_id} with {len(task.steps)} steps."
            )

        step_text = task.steps[segment.step_index]
        start_index = int(math.floor(segment.start_sec))
        end_index = max(start_index + 1, int(math.ceil(segment.end_sec)))

        for second in range(start_index, end_index):
            event_type = "none"
            event_text = ""
            is_first = second == start_index
            is_last = second == end_index - 1
            if is_first and is_last:
                event_type = "success"
                event_text = f"Started and completed {step_text}."
            elif is_first:
                event_type = "subtask_boundary"
                event_text = f"Started {step_text}."
            elif is_last:
                event_type = "success"
                event_text = f"Completed {step_text}."

            annotations.append(
                SubtaskAnnotation(
                    episode_index=episode_index,
                    frame_index=second,
                    instruction=task.title,
                    current_subtask=step_text,
                    phase=step_text,
                    target_query="",
                    goal_query="",
                    event_type=event_type,
                    event_text=event_text,
                )
            )

    annotations.sort(key=lambda item: item.frame_index)
    return annotations


def index_local_videos(videos_root: pathlib.Path | str) -> dict[str, pathlib.Path]:
    root = pathlib.Path(videos_root)
    index: dict[str, pathlib.Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem
        if stem not in index:
            index[stem] = path
    return index


def build_coverage_report(
    records: Iterable[CrossTaskVideoRecord],
    *,
    tasks: dict[str, CrossTaskTaskInfo],
    crosstask_release_dir: pathlib.Path | str,
    annotations_dir: str = "annotations",
    video_index: dict[str, pathlib.Path],
) -> CrossTaskCoverageReport:
    release_dir = pathlib.Path(crosstask_release_dir)
    filtered_records = [record for record in records if record.task_id in tasks]

    matched: list[CrossTaskVideoRecord] = []
    missing_annotations: list[CrossTaskVideoRecord] = []
    missing_local_videos: list[CrossTaskVideoRecord] = []

    for record in filtered_records:
        segment_path = release_dir / annotations_dir / f"{record.task_id}_{record.video_id}.csv"
        if not segment_path.exists():
            missing_annotations.append(record)
            continue
        if record.video_id not in video_index:
            missing_local_videos.append(record)
            continue
        matched.append(record)

    return CrossTaskCoverageReport(
        total_records=len(filtered_records),
        matched_records=tuple(matched),
        missing_annotation_records=tuple(missing_annotations),
        missing_local_video_records=tuple(missing_local_videos),
    )
