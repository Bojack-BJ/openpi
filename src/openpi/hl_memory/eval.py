from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import difflib
from typing import Literal

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.labels import DEFAULT_LANGUAGE_MEMORY
from openpi.hl_memory.memory import EpisodicKeyframeMemory
from openpi.hl_memory.memory import map_relative_positions_to_absolute
from openpi.hl_memory.schema import HLMemoryPrediction
from openpi.hl_memory.schema import render_language_memory_fields
from openpi.hl_memory.zero_shot import apply_rollout_language_memory_rule


AblationMode = Literal["no_memory", "language_memory_only", "keyframe_memory_only", "full"]
PredictionCallback = Callable[
    [AblationMode, ExportedHLMemorySample, ExportedHLMemorySample, HLMemoryPrediction, dict[str, float]],
    None,
]


@dataclasses.dataclass(frozen=True)
class EvalRolloutOptions:
    apply_rollout_memory_rule: bool = False
    known_prior_eval: bool = False
    known_prior_advance_threshold: float = 0.65
    known_prior_match_threshold: float = 0.62
    known_prior_max_advance_steps: int = 3


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
    *,
    target_protocol: str = "hl_v1",
    keyframe_candidate_label_mode: str = "canonical",
) -> dict[str, float]:
    expected = sample.target_prediction(
        target_protocol=target_protocol,
        keyframe_candidate_label_mode=keyframe_candidate_label_mode,
    )
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
    expected_completed = _normalize_text(expected.completed_objective)
    predicted_completed = _normalize_text(prediction.completed_objective)
    expected_has_completed = bool(expected_completed)
    predicted_has_completed = bool(predicted_completed)
    return {
        "objective_exact_match": float(prediction.current_objective == expected.current_objective),
        "objective_normalized_match": float(
            _normalize_text(prediction.current_objective) == _normalize_text(expected.current_objective)
        ),
        "horizon_objective_normalized_match": float(
            _normalize_text(prediction.horizon_current_objective) == _normalize_text(expected.horizon_current_objective)
        ),
        "subtask_progress_mae": _optional_float_mae(prediction.subtask_progress, expected.subtask_progress),
        "subtask_progress_accuracy_0_1": _optional_float_within(
            prediction.subtask_progress,
            expected.subtask_progress,
            tolerance=0.1,
        ),
        "should_advance_accuracy": _optional_bool_accuracy(
            prediction.should_advance_objective,
            expected.should_advance_objective,
        ),
        "active_hand_accuracy": _optional_text_accuracy(prediction.active_hand, expected.active_hand),
        "target_query_accuracy": float(_normalize_text(prediction.target_query) == _normalize_text(expected.target_query)),
        "goal_query_accuracy": float(_normalize_text(prediction.goal_query) == _normalize_text(expected.goal_query)),
        "target_object_accuracy": float(
            _normalize_text(prediction.target_object) == _normalize_text(expected.target_object)
        ),
        "target_slot_accuracy": float(_normalize_text(prediction.target_slot) == _normalize_text(expected.target_slot)),
        "keyframe_precision": precision,
        "keyframe_recall": recall,
        "keyframe_f1": (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "keyframe_event_recall": _keyframe_event_recall(prediction, sample),
        "keyframe_event_duplicate_predictions": _keyframe_event_duplicate_predictions(prediction, sample),
        "keyframe_event_timing_error": _keyframe_event_timing_error(prediction, sample),
        "completed_objective_precision": float(
            (not predicted_has_completed) or (expected_has_completed and predicted_completed == expected_completed)
        ),
        "completed_objective_recall": float(
            (not expected_has_completed) or (predicted_has_completed and predicted_completed == expected_completed)
        ),
        "language_memory_similarity": memory_similarity,
        "memory_drift": 1.0 - memory_similarity,
        "legacy_subtask_exact_match": float(prediction.current_subtask == expected.current_subtask),
        "legacy_subtask_normalized_match": float(
            _normalize_text(prediction.current_subtask) == _normalize_text(expected.current_subtask)
        ),
        "legacy_phase_accuracy": float(_normalize_text(prediction.phase) == _normalize_text(expected.phase)),
        "legacy_event_accuracy": _event_accuracy(prediction, sample),
}


def _keyframe_event_recall(prediction: HLMemoryPrediction, sample: ExportedHLMemorySample) -> float:
    if not sample.keyframe_event_ids:
        return 1.0
    predicted = set(prediction.keyframe_candidate_positions)
    expected = set(sample.keyframe_candidate_positions)
    return float(bool(predicted & expected))


def _keyframe_event_duplicate_predictions(prediction: HLMemoryPrediction, sample: ExportedHLMemorySample) -> float:
    if not sample.keyframe_event_ids:
        return 0.0
    predicted_hits = set(prediction.keyframe_candidate_positions) & set(sample.keyframe_candidate_positions)
    return float(max(len(predicted_hits) - 1, 0))


def _keyframe_event_timing_error(prediction: HLMemoryPrediction, sample: ExportedHLMemorySample) -> float:
    if not sample.keyframe_event_frame_indices:
        return 0.0
    position_lookup = {position: int(frame_index) for position, frame_index in enumerate(sample.recent_frame_indices, start=1)}
    predicted_frames = [
        position_lookup[position]
        for position in prediction.keyframe_candidate_positions
        if position in position_lookup
    ]
    if not predicted_frames:
        return -1.0
    canonical = int(sample.keyframe_event_frame_indices[0])
    return float(min(abs(frame_index - canonical) for frame_index in predicted_frames))


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
    rollout_options: EvalRolloutOptions | None = None,
    on_prediction: PredictionCallback | None = None,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    total_steps = 0
    exact_sequence_episodes = 0

    for episode_samples in samples_by_episode.values():
        state = _EpisodeRolloutState(episode_samples, hl_config, rollout_options=rollout_options)

        for sample in episode_samples:
            runtime_sample = state.runtime_sample(mode=mode)
            prediction = predict_fn(runtime_sample)
            prediction = state.postprocess_prediction(prediction, mode=mode)
            metrics = compute_prediction_metrics(
                prediction,
                sample,
                target_protocol=hl_config.target_protocol,
                keyframe_candidate_label_mode=hl_config.keyframe_candidate_label_mode,
            )
            for key, value in metrics.items():
                totals[key] += value
            if on_prediction is not None:
                on_prediction(mode, runtime_sample, sample, prediction, metrics)
            state.advance(prediction, mode=mode)
            total_steps += 1
        exact_sequence_episodes += int(state.sequence_ok)

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
    rollout_options: EvalRolloutOptions | None = None,
    on_prediction: PredictionCallback | None = None,
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
            rollout_options=rollout_options,
            on_prediction=on_prediction,
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
        episode_index: _EpisodeRolloutState(episode_samples, hl_config, rollout_options=rollout_options)
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
            prediction = state.postprocess_prediction(prediction, mode=mode)
            metrics = compute_prediction_metrics(
                prediction,
                source_sample,
                target_protocol=hl_config.target_protocol,
                keyframe_candidate_label_mode=hl_config.keyframe_candidate_label_mode,
            )
            for key, value in metrics.items():
                totals[key] += value
            if on_prediction is not None:
                on_prediction(mode, runtime_sample, source_sample, prediction, metrics)
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
    def __init__(
        self,
        episode_samples: Sequence[ExportedHLMemorySample],
        hl_config: HLMemoryConfig,
        *,
        rollout_options: EvalRolloutOptions | None = None,
    ):
        self.samples = list(episode_samples)
        self.rollout_options = rollout_options or EvalRolloutOptions()
        self.memory = EpisodicKeyframeMemory(
            memory_length=hl_config.memory_length,
            merge_distance=hl_config.merge_distance,
        )
        self.frame_lookup = {
            frame_index: frame_path
            for sample in self.samples
            for frame_index, frame_path in zip(sample.recent_frame_indices, sample.recent_frame_paths)
        }
        self.known_prior_steps = self.samples[0].step_prior if self.samples else ()
        self.known_prior_index = 0
        self.target_protocol = hl_config.target_protocol
        self.keyframe_candidate_label_mode = hl_config.keyframe_candidate_label_mode
        self.language_memory = self._initial_language_memory()
        self.sample_index = 0
        self.sequence_ok = True

    def _initial_language_memory(self) -> str:
        if self.rollout_options.known_prior_eval and self.known_prior_steps:
            return _known_prior_language_memory(
                self.known_prior_steps,
                self.known_prior_index,
                task_progress="No completed subtask yet.",
                notes="Known-prior eval initialized.",
            )
        return DEFAULT_LANGUAGE_MEMORY

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

    def postprocess_prediction(self, prediction: HLMemoryPrediction, *, mode: AblationMode) -> HLMemoryPrediction:
        if self.rollout_options.apply_rollout_memory_rule and mode in ("language_memory_only", "full"):
            prediction, _ = apply_rollout_language_memory_rule(
                prediction,
                previous_memory=self.language_memory,
                recent_end_sec=float(self.current_sample.frame_index),
            )
        if self.rollout_options.known_prior_eval and self.known_prior_steps:
            prediction, self.known_prior_index = _apply_known_prior_eval_state(
                prediction,
                known_prior_steps=self.known_prior_steps,
                current_index=self.known_prior_index,
                advance_threshold=self.rollout_options.known_prior_advance_threshold,
                match_threshold=self.rollout_options.known_prior_match_threshold,
                max_advance_steps=self.rollout_options.known_prior_max_advance_steps,
            )
        return prediction

    def advance(self, prediction: HLMemoryPrediction, *, mode: AblationMode) -> None:
        source_sample = self.current_sample
        self.sequence_ok &= (
            _normalize_text(prediction.current_objective)
            == _normalize_text(
                source_sample.target_prediction(
                    target_protocol=self.target_protocol,
                    keyframe_candidate_label_mode=self.keyframe_candidate_label_mode,
                ).current_objective
            )
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
        memory_indices = tuple(index for index in memory.selected_indices() if index in frame_lookup)
        memory_paths = tuple(frame_lookup[index] for index in memory_indices)
    else:
        memory_indices = ()
        memory_paths = ()
    return sample.with_runtime_context(
        language_memory=runtime_language_memory,
        memory_frame_paths=memory_paths,
        memory_frame_indices=memory_indices,
    )


def _apply_known_prior_eval_state(
    prediction: HLMemoryPrediction,
    *,
    known_prior_steps: tuple[str, ...],
    current_index: int,
    advance_threshold: float,
    match_threshold: float,
    max_advance_steps: int,
) -> tuple[HLMemoryPrediction, int]:
    current_index = max(0, min(current_index, len(known_prior_steps) - 1))
    match = _match_prediction_to_prior_step(
        prediction,
        known_prior_steps=known_prior_steps,
        current_index=current_index,
        max_advance_steps=max_advance_steps,
    )
    should_advance = _known_prior_should_advance(prediction, threshold=advance_threshold)
    next_index = current_index
    if int(match["index"]) > current_index and float(match["score"]) >= match_threshold:
        next_index = int(match["index"])
    elif should_advance and current_index + 1 < len(known_prior_steps):
        next_index = current_index + 1

    current_objective = known_prior_steps[next_index]
    progress = 0.0 if next_index != current_index else prediction.subtask_progress
    advance_flag = False if next_index != current_index else prediction.should_advance_objective
    task_progress = _known_prior_task_progress(known_prior_steps, next_index)
    relevant_objects = prediction.relevant_objects or tuple(
        item for item in (prediction.target_query, prediction.goal_query) if item and item.lower() != "none"
    )
    updated_language_memory = render_language_memory_fields(
        task_progress=task_progress,
        current_objective=current_objective,
        relevant_objects=relevant_objects,
        notes=prediction.notes or "Known-prior eval output.",
    )
    return (
        dataclasses.replace(
            prediction,
            task_progress=task_progress,
            current_objective=current_objective,
            current_subtask=current_objective,
            phase=current_objective,
            updated_language_memory=updated_language_memory,
            subtask_progress=progress,
            should_advance_objective=advance_flag,
        ),
        next_index,
    )


def _known_prior_should_advance(prediction: HLMemoryPrediction, *, threshold: float) -> bool:
    if prediction.should_advance_objective:
        return True
    if prediction.subtask_progress is None:
        return False
    return float(prediction.subtask_progress) >= threshold


def _match_prediction_to_prior_step(
    prediction: HLMemoryPrediction,
    *,
    known_prior_steps: tuple[str, ...],
    current_index: int,
    max_advance_steps: int,
) -> dict[str, object]:
    max_advance_steps = max(0, int(max_advance_steps))
    last_index = min(len(known_prior_steps) - 1, current_index + max_advance_steps)
    candidate_texts = [
        prediction.current_objective,
        prediction.current_subtask,
        prediction.phase,
    ]
    best: dict[str, object] = {
        "index": current_index,
        "score": 0.0,
        "source": "",
        "matched_step": known_prior_steps[current_index],
    }
    for source in candidate_texts:
        source_text = str(source or "").strip()
        if not source_text:
            continue
        for index in range(current_index, last_index + 1):
            score = _prior_step_similarity(source_text, known_prior_steps[index])
            if score > float(best["score"]):
                best = {
                    "index": index,
                    "score": score,
                    "source": source_text,
                    "matched_step": known_prior_steps[index],
                }
    return best


def _prior_step_similarity(left: str, right: str) -> float:
    left_norm = _normalize_prior_step_text(left)
    right_norm = _normalize_prior_step_text(right)
    if not left_norm or not right_norm:
        return 0.0
    sequence_score = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if not left_tokens or not right_tokens:
        return sequence_score
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(sequence_score, token_score)


def _normalize_prior_step_text(text: str) -> str:
    lowered = text.lower()
    normalized_chars = [char if char.isalnum() else " " for char in lowered]
    return " ".join("".join(normalized_chars).split())


def _known_prior_task_progress(steps: tuple[str, ...], current_index: int) -> str:
    completed = [step for step in steps[:current_index] if step.strip()]
    if not completed:
        return "No completed subtask yet."
    rendered = "; ".join(completed[-4:])
    prefix = "Completed known-prior subtasks"
    if len(completed) > 4:
        return f"{prefix}: ...; {rendered}."
    return f"{prefix}: {rendered}."


def _known_prior_language_memory(
    steps: tuple[str, ...],
    current_index: int,
    *,
    task_progress: str,
    notes: str,
) -> str:
    current_index = max(0, min(current_index, len(steps) - 1))
    return render_language_memory_fields(
        task_progress=task_progress,
        current_objective=steps[current_index],
        relevant_objects=(),
        notes=notes,
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _optional_float_mae(predicted: float | None, expected: float | None) -> float:
    if predicted is None and expected is None:
        return 0.0
    if predicted is None or expected is None:
        return 1.0
    return abs(float(predicted) - float(expected))


def _optional_float_within(predicted: float | None, expected: float | None, *, tolerance: float) -> float:
    return float(_optional_float_mae(predicted, expected) <= tolerance)


def _optional_bool_accuracy(predicted: bool | None, expected: bool | None) -> float:
    return float(predicted is expected)


def _optional_text_accuracy(predicted: str, expected: str) -> float:
    return float(_normalize_text(predicted) == _normalize_text(expected))


def _event_accuracy(prediction: HLMemoryPrediction, sample: ExportedHLMemorySample) -> float:
    if not sample.event_text.strip():
        return 1.0
    return float(_normalize_text(sample.event_text) in _normalize_text(prediction.updated_language_memory))
