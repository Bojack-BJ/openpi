from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Iterator
import dataclasses
import json
import pathlib

from PIL import Image

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.frame_composer import compose_context_panel
from openpi.hl_memory.labels import SubtaskAnnotation
from openpi.hl_memory.schema import HLMemoryPrediction


@dataclasses.dataclass(frozen=True)
class ExportedHLMemorySample:
    sample_id: str
    episode_index: int
    step_index: int
    frame_index: int
    instruction: str
    language_memory: str
    updated_language_memory: str
    current_subtask: str
    phase: str
    target_query: str
    goal_query: str
    keyframe_positions: tuple[int, ...]
    memory_frame_paths: tuple[str, ...]
    recent_frame_paths: tuple[str, ...]
    recent_frame_indices: tuple[int, ...]
    event_type: str = "none"
    event_text: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ExportedHLMemorySample":
        return cls(
            sample_id=str(data["sample_id"]),
            episode_index=int(data["episode_index"]),
            step_index=int(data["step_index"]),
            frame_index=int(data["frame_index"]),
            instruction=str(data["instruction"]),
            language_memory=str(data["language_memory"]),
            updated_language_memory=str(data["updated_language_memory"]),
            current_subtask=str(data["current_subtask"]),
            phase=str(data["phase"]),
            target_query=str(data["target_query"]),
            goal_query=str(data["goal_query"]),
            keyframe_positions=tuple(int(value) for value in data["keyframe_positions"]),
            memory_frame_paths=tuple(str(value) for value in data["memory_frame_paths"]),
            recent_frame_paths=tuple(str(value) for value in data["recent_frame_paths"]),
            recent_frame_indices=tuple(int(value) for value in data["recent_frame_indices"]),
            event_type=str(data.get("event_type", "none")),
            event_text=str(data.get("event_text", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "episode_index": self.episode_index,
            "step_index": self.step_index,
            "frame_index": self.frame_index,
            "instruction": self.instruction,
            "language_memory": self.language_memory,
            "updated_language_memory": self.updated_language_memory,
            "current_subtask": self.current_subtask,
            "phase": self.phase,
            "target_query": self.target_query,
            "goal_query": self.goal_query,
            "keyframe_positions": list(self.keyframe_positions),
            "memory_frame_paths": list(self.memory_frame_paths),
            "recent_frame_paths": list(self.recent_frame_paths),
            "recent_frame_indices": list(self.recent_frame_indices),
            "event_type": self.event_type,
            "event_text": self.event_text,
        }

    def target_prediction(self) -> HLMemoryPrediction:
        return HLMemoryPrediction(
            updated_language_memory=self.updated_language_memory,
            current_subtask=self.current_subtask,
            keyframe_positions=self.keyframe_positions,
            phase=self.phase,
            target_query=self.target_query,
            goal_query=self.goal_query,
        )

    def with_runtime_context(
        self,
        *,
        language_memory: str,
        memory_frame_paths: Iterable[str],
    ) -> "ExportedHLMemorySample":
        return dataclasses.replace(
            self,
            language_memory=language_memory,
            memory_frame_paths=tuple(memory_frame_paths),
        )

    def resolve_memory_frame_paths(self, dataset_dir: pathlib.Path) -> list[pathlib.Path]:
        return [(dataset_dir / path).resolve() for path in self.memory_frame_paths]

    def resolve_recent_frame_paths(self, dataset_dir: pathlib.Path) -> list[pathlib.Path]:
        return [(dataset_dir / path).resolve() for path in self.recent_frame_paths]


def load_annotations_jsonl(path: pathlib.Path | str) -> list[SubtaskAnnotation]:
    path = pathlib.Path(path)
    annotations: list[SubtaskAnnotation] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            annotations.append(SubtaskAnnotation.from_dict(payload))
    annotations.sort(key=lambda item: (item.episode_index, item.frame_index))
    return annotations


def load_exported_samples(dataset_dir: pathlib.Path | str) -> list[ExportedHLMemorySample]:
    dataset_dir = pathlib.Path(dataset_dir)
    sample_path = dataset_dir / "samples.jsonl"
    samples: list[ExportedHLMemorySample] = []
    with sample_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {sample_path}") from exc
            samples.append(ExportedHLMemorySample.from_dict(payload))
    samples.sort(key=lambda item: (item.episode_index, item.step_index))
    return samples


def group_annotations_by_episode(
    annotations: Iterable[SubtaskAnnotation],
) -> dict[int, list[SubtaskAnnotation]]:
    grouped: dict[int, list[SubtaskAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.episode_index].append(annotation)
    return dict(grouped)


def group_samples_by_episode(
    samples: Iterable[ExportedHLMemorySample],
) -> dict[int, list[ExportedHLMemorySample]]:
    grouped: dict[int, list[ExportedHLMemorySample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.episode_index].append(sample)
    return dict(grouped)


class ExportedHLMemoryDataset:
    def __init__(self, samples: list[ExportedHLMemorySample]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ExportedHLMemorySample:
        return self._samples[index]

    def __iter__(self) -> Iterator[ExportedHLMemorySample]:
        return iter(self._samples)


def build_context_panel_for_sample(
    sample: ExportedHLMemorySample,
    dataset_dir: pathlib.Path | str,
    config: HLMemoryConfig,
) -> Image.Image:
    dataset_dir = pathlib.Path(dataset_dir)
    memory_frames = [_load_rgb_image(path) for path in sample.resolve_memory_frame_paths(dataset_dir)]
    recent_frames = [_load_rgb_image(path) for path in sample.resolve_recent_frame_paths(dataset_dir)]
    return compose_context_panel(
        memory_frames,
        recent_frames,
        frame_height=config.frame_height,
        frame_width=config.frame_width,
        columns=config.panel_columns,
        gap=config.panel_gap,
    )


def _load_rgb_image(path: pathlib.Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()
