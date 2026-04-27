from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import pathlib

from PIL import Image

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import LoadedVideoClips
from openpi.hl_memory.data import build_loaded_video_clips_from_frames


@dataclasses.dataclass(frozen=True)
class ZeroShotClipSelection:
    video_path: pathlib.Path
    duration_sec: float | None
    memory_seconds: tuple[float, ...]
    recent_seconds: tuple[float, ...]


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


def _format_second(second: float) -> str:
    return f"{second:08.3f}s".replace(".", "p")
