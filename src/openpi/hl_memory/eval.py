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
        "objective_exact_match": float(prediction.current_objective == expected.current_objective),
        "objective_normalized_match": float(
            _normalize_text(prediction.current_objective) == _normalize_text(expected.current_objective)
        ),
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
            sequence_ok &= _normalize_text(prediction.current_objective) == _normalize_text(sample.target_prediction().current_objective)
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


def run_offline_rollout_batched(
    samples_by_episode: Mapping[int, Sequence[ExportedHLMemorySample]],
    hl_config: HLMemoryConfig,
    predict_batch_fn: Callable[[Sequence[ExportedHLMemorySample]], Sequence[HLMemoryPrediction]],
    *,
    mode: AblationMode,
    batch_size: int,
    on_sample_done: Callable[[ExportedHLMemorySample], None] | None = None,
    on_batch_start: Callable[[int], None] | None = None,
) -> dict[str, float]:
    if batch_size <= 1:
        sample_count = sum(len(episode_samples) for episode_samples in samples_by_episode.values())

        def predict_one(sample: ExportedHLMemorySample) -> HLMemoryPrediction:
            if on_batch_start is not None:
                on_batch_start(1)
            prediction = predict_batch_fn([sample])[0]
            if on_sample_done is not None:
                on_sample_done(sample)
            return prediction

        metrics = run_offline_rollout(
            samples_by_episode,
            hl_config,
            predict_one,
            mode=mode,
        )
        if metrics:
            metrics["eval_batch_size_requested"] = float(batch_size)
            metrics["actual_eval_batch_size_mean"] = 1.0
            metrics["actual_eval_batch_size_max"] = 1.0
            metrics["num_generate_batches"] = float(sample_count)
        return metrics

    totals: dict[str, float] = defaultdict(float)
    total_steps = 0
    actual_batch_sizes: list[int] = []
    states = {
        episode_index: _EpisodeRolloutState(episode_samples, hl_config)
        for episode_index, episode_samples in samples_by_episode.items()
        if episode_samples
    }

    while True:
        pending_episode_indices = [
            episode_index for episode_index, state in states.items() if state.has_next_sample()
        ]
        if not pending_episode_indices:
            break

        batch_episode_indices = pending_episode_indices[:batch_size]
        batch_samples = [
            states[episode_index].runtime_sample(mode=mode)
            for episode_index in batch_episode_indices
        ]
        actual_batch_sizes.append(len(batch_samples))
        if on_batch_start is not None:
            on_batch_start(len(batch_samples))
        predictions = list(predict_batch_fn(batch_samples))
        if len(predictions) != len(batch_samples):
            raise ValueError(f"Expected {len(batch_samples)} predictions, got {len(predictions)}.")

        for episode_index, runtime_sample, prediction in zip(
            batch_episode_indices,
            batch_samples,
            predictions,
            strict=True,
        ):
            state = states[episode_index]
            source_sample = state.current_sample
            metrics = compute_prediction_metrics(prediction, source_sample)
            for key, value in metrics.items():
                totals[key] += value
            state.advance(prediction, mode=mode)
            total_steps += 1
            if on_sample_done is not None:
                on_sample_done(runtime_sample)

    if total_steps == 0:
        return {}

    metrics = {key: value / total_steps for key, value in totals.items()}
    episode_count = len(states)
    exact_sequence_episodes = sum(int(state.sequence_ok) for state in states.values())
    metrics["episode_sequence_accuracy"] = exact_sequence_episodes / max(episode_count, 1)
    metrics["num_steps"] = float(total_steps)
    metrics["num_episodes"] = float(episode_count)
    metrics["eval_batch_size_requested"] = float(batch_size)
    metrics["actual_eval_batch_size_mean"] = (
        sum(actual_batch_sizes) / len(actual_batch_sizes) if actual_batch_sizes else 0.0
    )
    metrics["actual_eval_batch_size_max"] = float(max(actual_batch_sizes, default=0))
    metrics["num_generate_batches"] = float(len(actual_batch_sizes))
    return dict(metrics)


class _EpisodeRolloutState:
    def __init__(self, episode_samples: Sequence[ExportedHLMemorySample], hl_config: HLMemoryConfig):
        self.samples = list(episode_samples)
        self.memory = EpisodicKeyframeMemory(
            memory_length=hl_config.memory_length,
            merge_distance=hl_config.merge_distance,
        )
        self.frame_lookup = {
            frame_index: frame_path
            for sample in self.samples
            for frame_index, frame_path in zip(sample.recent_frame_indices, sample.recent_frame_paths)
        }
        self.language_memory = DEFAULT_LANGUAGE_MEMORY
        self.sample_index = 0
        self.sequence_ok = True

    @property
    def current_sample(self) -> ExportedHLMemorySample:
        return self.samples[self.sample_index]

    def has_next_sample(self) -> bool:
        return self.sample_index < len(self.samples)

    def runtime_sample(self, *, mode: AblationMode) -> ExportedHLMemorySample:
        return _with_ablation_context(
            self.current_sample,
            mode=mode,
            memory=self.memory,
            frame_lookup=self.frame_lookup,
            language_memory=self.language_memory,
        )

    def advance(self, prediction: HLMemoryPrediction, *, mode: AblationMode) -> None:
        source_sample = self.current_sample
        self.sequence_ok &= (
            _normalize_text(prediction.current_objective)
            == _normalize_text(source_sample.target_prediction().current_objective)
        )
        if mode in ("language_memory_only", "full"):
            self.language_memory = prediction.updated_language_memory
        if mode in ("keyframe_memory_only", "full"):
            absolute_positions, _ = map_relative_positions_to_absolute(
                prediction.keyframe_candidate_positions,
                source_sample.recent_frame_indices,
            )
            self.memory.add_candidates(absolute_positions)
        self.sample_index += 1


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
