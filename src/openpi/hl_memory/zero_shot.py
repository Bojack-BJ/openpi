from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import pathlib
import re

from PIL import Image

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import LoadedVideoClips
from openpi.hl_memory.data import build_loaded_video_clips_from_frames
from openpi.hl_memory.schema import HLMemoryPrediction


@dataclasses.dataclass(frozen=True)
class ZeroShotClipSelection:
    video_path: pathlib.Path
    duration_sec: float | None
    memory_seconds: tuple[float, ...]
    recent_seconds: tuple[float, ...]


_GENERIC_MEMORY_STRINGS = {
    "",
    "task started",
    "task started.",
    "no progress has been recorded yet",
    "no progress has been recorded yet.",
}


def parse_seconds_argument(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return []
    parsed: list[float] = []
    for chunk in value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        parsed.append(float(stripped))
    return _sorted_unique_seconds(parsed)


def build_recent_seconds(
    duration_sec: float | None,
    *,
    clip_length: int,
    recent_end_sec: float | None = None,
    recent_step_sec: float = 1.0,
    explicit_seconds: Iterable[float] | None = None,
) -> list[float]:
    if clip_length <= 0:
        raise ValueError("clip_length must be positive.")
    if recent_step_sec <= 0:
        raise ValueError("recent_step_sec must be positive.")
    if explicit_seconds is not None:
        explicit = _sorted_unique_seconds(explicit_seconds)
        if not explicit:
            return []
        return explicit[-clip_length:]

    if duration_sec is None or duration_sec <= 0.0:
        resolved_end_sec = max(recent_end_sec or 0.0, 0.0)
    else:
        resolved_end_sec = duration_sec if recent_end_sec is None else min(max(recent_end_sec, 0.0), duration_sec)

    seconds = [max(resolved_end_sec - recent_step_sec * index, 0.0) for index in range(clip_length)]
    return _sorted_unique_seconds(seconds)


def build_auto_memory_seconds(
    duration_sec: float | None,
    *,
    recent_seconds: Iterable[float],
    clip_length: int,
) -> list[float]:
    if clip_length <= 0:
        raise ValueError("clip_length must be positive.")
    recent_seconds = list(recent_seconds)
    if not recent_seconds:
        prefix_end_sec = max(duration_sec or 0.0, 0.0)
    else:
        prefix_end_sec = max(recent_seconds[0] - 1e-3, 0.0)
        if duration_sec is not None and duration_sec > 0.0:
            prefix_end_sec = min(prefix_end_sec, duration_sec)
    if prefix_end_sec <= 0.0:
        return []
    if clip_length == 1:
        return [prefix_end_sec / 2.0]

    seconds = [prefix_end_sec * index / (clip_length - 1) for index in range(clip_length)]
    return _sorted_unique_seconds(seconds)


def build_zero_shot_clips_from_video(
    video_path: pathlib.Path | str,
    *,
    config: HLMemoryConfig,
    recent_end_sec: float | None = None,
    recent_step_sec: float = 1.0,
    recent_seconds: Iterable[float] | None = None,
    memory_seconds: Iterable[float] | None = None,
    auto_memory: bool = True,
) -> tuple[LoadedVideoClips, ZeroShotClipSelection]:
    reader = VideoFrameReader(video_path)
    try:
        resolved_recent_seconds = build_recent_seconds(
            reader.duration_sec,
            clip_length=config.recent_frames_length,
            recent_end_sec=recent_end_sec,
            recent_step_sec=recent_step_sec,
            explicit_seconds=recent_seconds,
        )
        if memory_seconds is not None:
            resolved_memory_seconds = _sorted_unique_seconds(memory_seconds)
        elif auto_memory:
            resolved_memory_seconds = build_auto_memory_seconds(
                reader.duration_sec,
                recent_seconds=resolved_recent_seconds,
                clip_length=config.memory_length,
            )
        else:
            resolved_memory_seconds = []
        resolved_memory_seconds = _filter_visible_memory_seconds(resolved_memory_seconds, resolved_recent_seconds)

        memory_frames = [reader.read(second) for second in resolved_memory_seconds]
        recent_frames = [reader.read(second) for second in resolved_recent_seconds]
    finally:
        reader.close()

    clips = build_loaded_video_clips_from_frames(
        memory_frames,
        recent_frames,
        config=config,
        memory_valid_length=len(memory_frames),
        recent_valid_length=len(recent_frames),
    )
    selection = ZeroShotClipSelection(
        video_path=pathlib.Path(video_path),
        duration_sec=reader.duration_sec,
        memory_seconds=tuple(resolved_memory_seconds),
        recent_seconds=tuple(resolved_recent_seconds),
    )
    return clips, selection


def read_video_duration_sec(video_path: pathlib.Path | str) -> float | None:
    reader = VideoFrameReader(video_path)
    try:
        return reader.duration_sec
    finally:
        reader.close()


def build_rollout_end_seconds(
    duration_sec: float | None,
    *,
    interval_sec: float,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[float]:
    if interval_sec <= 0.0:
        raise ValueError("interval_sec must be positive.")
    resolved_start = max(float(start_sec), 0.0)
    if end_sec is None:
        if duration_sec is None:
            raise ValueError("end_sec must be set when video duration is unavailable.")
        resolved_end = duration_sec
    else:
        resolved_end = max(float(end_sec), 0.0)
        if duration_sec is not None and duration_sec > 0.0:
            resolved_end = min(resolved_end, duration_sec)
    if resolved_end < resolved_start:
        return [resolved_start]

    seconds: list[float] = []
    current = resolved_start
    while current <= resolved_end + 1e-6:
        seconds.append(round(current, 6))
        current += interval_sec
    if not seconds or seconds[-1] < resolved_end - 1e-6:
        seconds.append(round(resolved_end, 6))
    return seconds


def keyframe_candidate_seconds(
    prediction: HLMemoryPrediction,
    selection: ZeroShotClipSelection,
    *,
    recent_valid_length: int,
) -> tuple[float, ...]:
    seconds: list[float] = []
    for position in prediction.keyframe_candidate_positions:
        if position <= 0 or position > recent_valid_length:
            continue
        if position > len(selection.recent_seconds):
            continue
        second = selection.recent_seconds[position - 1]
        if second not in seconds:
            seconds.append(second)
    return tuple(seconds)


def update_rollout_memory_seconds(
    memory_seconds: Iterable[float],
    candidate_seconds: Iterable[float],
    *,
    memory_length: int,
    merge_distance_sec: float | None = None,
) -> tuple[float, ...]:
    if memory_length <= 0:
        raise ValueError("memory_length must be positive.")
    seconds = _sorted_unique_seconds([*memory_seconds, *candidate_seconds])
    if merge_distance_sec is not None and merge_distance_sec > 0.0:
        seconds = _cluster_keyframe_seconds(seconds, merge_distance_sec=merge_distance_sec)
    return tuple(seconds[-memory_length:])


def apply_rollout_language_memory_rule(
    prediction: HLMemoryPrediction,
    *,
    previous_memory: str,
    recent_end_sec: float,
    max_chars: int = 420,
) -> tuple[HLMemoryPrediction, bool]:
    """Ensures rollout memory stays useful for the downstream LL VLM.

    The memory is not a debug trace. It is a compact context string consumed by
    a low-level VLM policy, so the fallback rewrites generic or log-like memory
    into stable action-useful fields.
    """
    model_memory = prediction.updated_language_memory.strip()
    previous_memory = previous_memory.strip()
    should_replace = (
        _is_generic_memory(model_memory)
        or _normalize_text(model_memory) == _normalize_text(previous_memory)
        or _looks_like_instruction_echo(model_memory, prediction.current_subtask)
        or _looks_like_debug_log_memory(model_memory)
        or not _has_ll_memory_fields(model_memory)
    )
    if not should_replace:
        compacted = _compact_ll_memory(model_memory, max_chars=max_chars)
        return dataclasses.replace(prediction, updated_language_memory=compacted), compacted != model_memory

    state = _parse_ll_memory(previous_memory)
    updated_memory = _render_ll_memory(state, prediction)
    updated_memory = _compact_ll_memory(updated_memory, max_chars=max_chars)
    return dataclasses.replace(prediction, updated_language_memory=updated_memory), True


def build_zero_shot_sample(
    *,
    video_path: pathlib.Path | str,
    instruction: str,
    language_memory: str = "",
    memory_seconds: Iterable[float] = (),
    recent_seconds: Iterable[float] = (),
) -> ExportedHLMemorySample:
    recent_seconds = list(recent_seconds)
    memory_seconds = list(memory_seconds)
    return ExportedHLMemorySample(
        sample_id=f"zero_shot::{pathlib.Path(video_path).stem}",
        episode_index=0,
        step_index=0,
        frame_index=int(recent_seconds[-1]) if recent_seconds else 0,
        instruction=instruction,
        language_memory=language_memory,
        updated_language_memory=language_memory or "No progress has been recorded yet.",
        current_subtask="placeholder",
        phase="placeholder",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=tuple(int(round(second)) for second in memory_seconds),
        memory_valid_length=len(memory_seconds),
        recent_frame_paths=(),
        recent_frame_indices=tuple(int(round(second)) for second in recent_seconds),
        recent_valid_length=len(recent_seconds),
        event_type="none",
        event_text="",
    )


def save_zero_shot_debug_frames(
    debug_dir: pathlib.Path | str,
    *,
    clips: LoadedVideoClips,
    selection: ZeroShotClipSelection,
) -> None:
    debug_dir = pathlib.Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(clips.memory_frames[: clips.memory_valid_length]):
        second = selection.memory_seconds[index]
        frame.save(debug_dir / f"memory_{index:02d}_{_format_second(second)}.png")
    for index, frame in enumerate(clips.recent_frames[: clips.recent_valid_length]):
        second = selection.recent_seconds[index]
        frame.save(debug_dir / f"recent_{index:02d}_{_format_second(second)}.png")


def save_keyframe_candidate_frames(
    debug_dir: pathlib.Path | str,
    *,
    clips: LoadedVideoClips,
    selection: ZeroShotClipSelection,
    positions: Iterable[int],
) -> list[pathlib.Path]:
    debug_dir = pathlib.Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[pathlib.Path] = []
    for position in positions:
        if position <= 0 or position > clips.recent_valid_length or position > len(selection.recent_seconds):
            continue
        second = selection.recent_seconds[position - 1]
        output_path = debug_dir / f"keyframe_candidate_pos{position:02d}_{_format_second(second)}.png"
        clips.recent_frames[position - 1].save(output_path)
        saved_paths.append(output_path)
    return saved_paths


class VideoFrameReader:
    def __init__(self, path: pathlib.Path | str):
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Zero-shot HL video loading requires `opencv-python`.") from exc

        self._cv2 = cv2
        self._path = pathlib.Path(path)
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise ValueError(f"Failed to open video file: {self._path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self._fps = fps if fps > 0.0 else 25.0
        self._duration_sec = frame_count / self._fps if frame_count > 0.0 else None

    @property
    def duration_sec(self) -> float | None:
        return self._duration_sec

    def read(self, second: float) -> Image.Image:
        target_sec = max(float(second), 0.0)
        if self._duration_sec is not None and self._duration_sec > 0.0:
            max_seek_sec = max(self._duration_sec - (1.0 / self._fps), 0.0)
            target_sec = min(target_sec, max_seek_sec)

        for candidate_sec in (target_sec, max(target_sec - 0.25, 0.0), max(target_sec - 0.5, 0.0)):
            self._capture.set(self._cv2.CAP_PROP_POS_MSEC, candidate_sec * 1000.0)
            ok, frame = self._capture.read()
            if ok and frame is not None:
                rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
        raise ValueError(f"Failed to decode frame near second {second} from {self._path}")

    def close(self) -> None:
        self._capture.release()


def _sorted_unique_seconds(seconds: Iterable[float]) -> list[float]:
    resolved = sorted(max(float(second), 0.0) for second in seconds)
    deduped: list[float] = []
    for second in resolved:
        if not deduped or abs(second - deduped[-1]) > 1e-6:
            deduped.append(second)
    return deduped


def _filter_visible_memory_seconds(memory_seconds: Iterable[float], recent_seconds: Iterable[float]) -> list[float]:
    recent_seconds = list(recent_seconds)
    if not recent_seconds:
        return _sorted_unique_seconds(memory_seconds)
    earliest_recent = min(recent_seconds)
    visible = set(round(second, 6) for second in recent_seconds)
    return [
        second
        for second in _sorted_unique_seconds(memory_seconds)
        if round(second, 6) not in visible and second < earliest_recent
    ]


def _cluster_keyframe_seconds(seconds: list[float], *, merge_distance_sec: float) -> list[float]:
    if not seconds:
        return []
    clusters: list[list[float]] = [[seconds[0]]]
    for second in seconds[1:]:
        if second - clusters[-1][-1] <= merge_distance_sec:
            clusters[-1].append(second)
        else:
            clusters.append([second])
    return [cluster[(len(cluster) - 1) // 2] for cluster in clusters]


def _format_second(second: float) -> str:
    return f"{second:08.3f}s".replace(".", "p")


def _is_generic_memory(text: str) -> bool:
    return _normalize_text(text) in _GENERIC_MEMORY_STRINGS


def _looks_like_instruction_echo(memory: str, current_subtask: str) -> bool:
    normalized_memory = _normalize_text(memory)
    normalized_subtask = _normalize_text(current_subtask)
    if not normalized_memory or not normalized_subtask:
        return False
    return normalized_memory == normalized_subtask


def _looks_like_debug_log_memory(memory: str) -> bool:
    normalized = _normalize_text(memory)
    return (
        normalized.startswith("progress memory:")
        or "t=" in normalized
        or "target=" in normalized
        or "goal=" in normalized
    )


def _has_ll_memory_fields(memory: str) -> bool:
    keys = {line.split(":", 1)[0].strip().lower() for line in memory.splitlines() if ":" in line}
    return {"task progress", "current objective", "relevant objects", "notes"}.issubset(keys)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_ll_memory(memory: str) -> dict[str, str]:
    fields = {
        "task progress": "",
        "current objective": "",
        "relevant objects": "",
        "notes": "",
    }
    for line in memory.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key in fields:
            fields[normalized_key] = value.strip()
    return fields


def _render_ll_memory(fields: dict[str, str], prediction: HLMemoryPrediction) -> str:
    current_subtask = _clean_memory_field(prediction.current_subtask) or "continue the task"
    previous_progress = _clean_memory_field(fields.get("task progress", ""))
    progress = _progress_sentence(previous_progress, current_subtask)
    relevant_objects = _merge_relevant_objects(
        fields.get("relevant objects", ""),
        prediction.target_query,
        prediction.goal_query,
    )
    notes = _clean_memory_field(fields.get("notes", "")) or "none"
    return "\n".join(
        [
            f"Task progress: {progress}",
            f"Current objective: {current_subtask}",
            f"Relevant objects: {relevant_objects or 'none'}",
            f"Notes: {notes}",
        ]
    )


def _progress_sentence(previous_progress: str, current_subtask: str) -> str:
    if not previous_progress or _is_generic_memory(previous_progress):
        return f"The robot is working on: {current_subtask}."
    normalized_progress = _normalize_text(previous_progress)
    normalized_subtask = _normalize_text(current_subtask)
    if normalized_subtask and normalized_subtask in normalized_progress:
        return previous_progress
    return f"{previous_progress.rstrip('.')} The current focus is: {current_subtask}."


def _merge_relevant_objects(*values: str) -> str:
    objects: list[str] = []
    for value in values:
        for chunk in re.split(r"[,;/]", value):
            cleaned = _clean_memory_field(chunk)
            if not cleaned or cleaned.lower() == "none":
                continue
            if _normalize_text(cleaned) not in {_normalize_text(item) for item in objects}:
                objects.append(cleaned)
    return ", ".join(objects[:6])


def _clean_memory_field(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip())
    cleaned = re.sub(r"\bt=\d+(?:\.\d+)?s\b", "", cleaned)
    cleaned = cleaned.replace("target=", "").replace("goal=", "")
    cleaned = cleaned.strip(" ;-")
    return cleaned


def _compact_ll_memory(memory: str, *, max_chars: int) -> str:
    lines = []
    for line in memory.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = _clean_memory_field(value)
        lines.append(f"{key.strip()}: {value or 'none'}")
    compacted = "\n".join(lines)
    if len(compacted) <= max_chars:
        return compacted
    truncated: list[str] = []
    per_line_budget = max(max_chars // max(len(lines), 1) - 8, 24)
    for line in lines:
        if len(line) > per_line_budget:
            line = line[: per_line_budget - 3].rstrip() + "..."
        truncated.append(line)
    return "\n".join(truncated)
