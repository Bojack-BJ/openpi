from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import difflib
from typing import Literal

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.labels import DEFAULT_LANGUAGE_MEMORY
from openpi.hl_memory.memory import EpisodicKeyframeMemory
from openpi.hl_memory.memory import map_relative_positions_to_absolute
from openpi.hl_memory.schema import HLMemoryPrediction


AblationMode = Literal["no_memory", "language_memory_only", "keyframe_memory_only", "full"]


def group_samples_by_episode(
    samples: Sequence[ExportedHLMemorySample],
) -> dict[int, list[ExportedHLMemorySample]]:
    grouped: dict[int, list[ExportedHLMemorySample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.episode_index].append(sample)
    return dict(grouped)


def compute_prediction_metrics(
    prediction: HLMemoryPrediction,
    sample: ExportedHLMemorySample,
) -> dict[str, float]:
    expected = sample.target_prediction()
    expected_positions = set(expected.keyframe_candidate_positions)
    predicted_positions = set(prediction.keyframe_candidate_positions)
    overlap = expected_positions & predicted_positions

    precision = len(overlap) / len(predicted_positions) if predicted_positions else float(not expected_positions)
    recall = len(overlap) / len(expected_positions) if expected_positions else float(not predicted_positions)
    memory_similarity = difflib.SequenceMatcher(
        None,
        _normalize_text(prediction.updated_language_memory),
        _normalize_text(expected.updated_language_memory),
    ).ratio()
    return {
        "subtask_exact_match": float(prediction.current_subtask == expected.current_subtask),
        "subtask_normalized_match": float(
            _normalize_text(prediction.current_subtask) == _normalize_text(expected.current_subtask)
        ),
        "phase_accuracy": float(_normalize_text(prediction.phase) == _normalize_text(expected.phase)),
        "target_query_accuracy": float(_normalize_text(prediction.target_query) == _normalize_text(expected.target_query)),
        "goal_query_accuracy": float(_normalize_text(prediction.goal_query) == _normalize_text(expected.goal_query)),
        "keyframe_precision": precision,
        "keyframe_recall": recall,
        "language_memory_similarity": memory_similarity,
        "memory_drift": 1.0 - memory_similarity,
        "event_accuracy": _event_accuracy(prediction, sample),
    }


def evaluate_ablation_modes(
    samples: Sequence[ExportedHLMemorySample],
    hl_config: HLMemoryConfig,
    predict_fn: Callable[[ExportedHLMemorySample], HLMemoryPrediction],
) -> dict[AblationMode, dict[str, float]]:
    grouped = group_samples_by_episode(samples)
    return {
        mode: run_offline_rollout(grouped, hl_config, predict_fn, mode=mode)
        for mode in ("no_memory", "language_memory_only", "keyframe_memory_only", "full")
    }


def run_offline_rollout(
    samples_by_episode: Mapping[int, Sequence[ExportedHLMemorySample]],
    hl_config: HLMemoryConfig,
    predict_fn: Callable[[ExportedHLMemorySample], HLMemoryPrediction],
    *,
    mode: AblationMode,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    total_steps = 0
    exact_sequence_episodes = 0

    for episode_samples in samples_by_episode.values():
        memory = EpisodicKeyframeMemory(
            memory_length=hl_config.memory_length,
            merge_distance=hl_config.merge_distance,
        )
        frame_lookup = {
            frame_index: frame_path
            for sample in episode_samples
            for frame_index, frame_path in zip(sample.recent_frame_indices, sample.recent_frame_paths)
        }
        language_memory = DEFAULT_LANGUAGE_MEMORY
        sequence_ok = True

        for sample in episode_samples:
            runtime_sample = _with_ablation_context(
                sample,
                mode=mode,
                memory=memory,
                frame_lookup=frame_lookup,
                language_memory=language_memory,
            )
            prediction = predict_fn(runtime_sample)
            metrics = compute_prediction_metrics(prediction, sample)
            for key, value in metrics.items():
                totals[key] += value
            sequence_ok &= _normalize_text(prediction.current_subtask) == _normalize_text(sample.current_subtask)
            total_steps += 1

            if mode in ("language_memory_only", "full"):
                language_memory = prediction.updated_language_memory
            if mode in ("keyframe_memory_only", "full"):
                absolute_positions, _ = map_relative_positions_to_absolute(
                    prediction.keyframe_candidate_positions,
                    sample.recent_frame_indices,
                )
                memory.add_candidates(absolute_positions)

        exact_sequence_episodes += int(sequence_ok)

    if total_steps == 0:
        return {}

    metrics = {key: value / total_steps for key, value in totals.items()}
    metrics["episode_sequence_accuracy"] = exact_sequence_episodes / max(len(samples_by_episode), 1)
    metrics["num_steps"] = float(total_steps)
    metrics["num_episodes"] = float(len(samples_by_episode))
    return dict(metrics)


def _with_ablation_context(
    sample: ExportedHLMemorySample,
    *,
    mode: AblationMode,
    memory: EpisodicKeyframeMemory,
    frame_lookup: Mapping[int, str],
    language_memory: str,
) -> ExportedHLMemorySample:
    runtime_language_memory = language_memory if mode in ("language_memory_only", "full") else DEFAULT_LANGUAGE_MEMORY
    if mode in ("keyframe_memory_only", "full"):
        memory_indices = tuple(index for index in memory.visible_indices(sample.recent_frame_indices) if index in frame_lookup)
        memory_paths = tuple(frame_lookup[index] for index in memory_indices)
    else:
        memory_indices = ()
        memory_paths = ()
    return sample.with_runtime_context(
        language_memory=runtime_language_memory,
        memory_frame_paths=memory_paths,
        memory_frame_indices=memory_indices,
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _event_accuracy(prediction: HLMemoryPrediction, sample: ExportedHLMemorySample) -> float:
    if not sample.event_text.strip():
        return 1.0
    return float(_normalize_text(sample.event_text) in _normalize_text(prediction.updated_language_memory))
