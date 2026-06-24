from __future__ import annotations

from collections.abc import Callable, Sequence
import random
from typing import TypeVar

from openpi.hl_memory.data import ExportedHLMemorySample


T = TypeVar("T")


def sample_keyframe_stratified(
    items: Sequence[T],
    batch_size: int,
    rng: random.Random,
    *,
    sample_from_item: Callable[[T], ExportedHLMemorySample],
    keyframe_positive_sample_ratio: float,
    keyframe_confirm_positive_sample_ratio: float,
    target_protocol: str,
    keyframe_candidate_label_mode: str,
) -> list[T]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not items:
        raise ValueError("Cannot sample from an empty sequence.")
    if not 0.0 <= keyframe_positive_sample_ratio <= 1.0:
        raise ValueError("--keyframe-positive-sample-ratio must be in [0, 1].")
    if not 0.0 <= keyframe_confirm_positive_sample_ratio <= 1.0:
        raise ValueError("--keyframe-confirm-positive-sample-ratio must be in [0, 1].")
    if keyframe_confirm_positive_sample_ratio > keyframe_positive_sample_ratio:
        raise ValueError(
            "--keyframe-confirm-positive-sample-ratio cannot exceed --keyframe-positive-sample-ratio."
        )
    if keyframe_confirm_positive_sample_ratio > 0.0 and target_protocol not in {
        "keyframe_gated_memory_two_pass",
        "memer_film_progress_two_pass",
    }:
        raise ValueError(
            "--keyframe-confirm-positive-sample-ratio is only supported by "
            "--target-protocol keyframe_gated_memory_two_pass or memer_film_progress_two_pass."
        )

    if keyframe_positive_sample_ratio <= 0.0:
        return _uniform_sample(items, batch_size, rng)

    confirm_positives: list[T] = []
    proposal_only_positives: list[T] = []
    negatives: list[T] = []
    for item in items:
        prediction = sample_from_item(item).target_prediction(
            target_protocol=target_protocol,
            keyframe_candidate_label_mode=keyframe_candidate_label_mode,
        )
        if prediction.completed_objective:
            confirm_positives.append(item)
        elif prediction.keyframe_candidate_positions:
            proposal_only_positives.append(item)
        else:
            negatives.append(item)

    pools = (
        (confirm_positives, keyframe_confirm_positive_sample_ratio),
        (
            proposal_only_positives,
            keyframe_positive_sample_ratio - keyframe_confirm_positive_sample_ratio,
        ),
        (negatives, 1.0 - keyframe_positive_sample_ratio),
    )
    nonempty_pools = [(pool, weight) for pool, weight in pools if pool and weight > 0.0]
    if not nonempty_pools:
        return _uniform_sample(items, batch_size, rng)

    total_weight = sum(weight for _, weight in nonempty_pools)
    batch: list[T] = []
    for _ in range(batch_size):
        threshold = rng.random() * total_weight
        cumulative = 0.0
        selected_pool = nonempty_pools[-1][0]
        for pool, weight in nonempty_pools:
            cumulative += weight
            if threshold < cumulative:
                selected_pool = pool
                break
        batch.append(rng.choice(selected_pool))
    return batch


def _uniform_sample(items: Sequence[T], batch_size: int, rng: random.Random) -> list[T]:
    if len(items) >= batch_size:
        return rng.sample(list(items), batch_size)
    return [rng.choice(items) for _ in range(batch_size)]
