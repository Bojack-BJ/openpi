from collections import Counter
import random

import pytest

from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.sampling import sample_keyframe_stratified


def _sample(name: str, *, proposal: bool, confirm: bool) -> ExportedHLMemorySample:
    return ExportedHLMemorySample(
        sample_id=name,
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="",
        updated_language_memory="",
        current_subtask=name,
        phase=name,
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(1,) if proposal else (),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective=name,
        keyframe_label=confirm,
    )


def test_batch_size_one_uses_probabilistic_three_way_sampling():
    items = [
        _sample("confirm", proposal=True, confirm=True),
        _sample("proposal", proposal=True, confirm=False),
        _sample("negative", proposal=False, confirm=False),
    ]
    rng = random.Random(7)
    counts = Counter(
        sample_keyframe_stratified(
            items,
            1,
            rng,
            sample_from_item=lambda item: item,
            keyframe_positive_sample_ratio=0.4,
            keyframe_confirm_positive_sample_ratio=0.2,
            target_protocol="keyframe_gated_memory_two_pass",
            keyframe_candidate_label_mode="event_band",
        )[0].sample_id
        for _ in range(20_000)
    )

    assert counts["confirm"] / 20_000 == pytest.approx(0.2, abs=0.015)
    assert counts["proposal"] / 20_000 == pytest.approx(0.2, abs=0.015)
    assert counts["negative"] / 20_000 == pytest.approx(0.6, abs=0.015)


def test_confirm_ratio_requires_two_pass_and_cannot_exceed_total_positive_ratio():
    item = _sample("confirm", proposal=True, confirm=True)
    common = {
        "items": [item],
        "batch_size": 1,
        "rng": random.Random(0),
        "sample_from_item": lambda value: value,
        "keyframe_candidate_label_mode": "event_band",
    }
    with pytest.raises(ValueError, match="only supported"):
        sample_keyframe_stratified(
            **common,
            keyframe_positive_sample_ratio=0.4,
            keyframe_confirm_positive_sample_ratio=0.2,
            target_protocol="memer_objective",
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        sample_keyframe_stratified(
            **common,
            keyframe_positive_sample_ratio=0.2,
            keyframe_confirm_positive_sample_ratio=0.4,
            target_protocol="keyframe_gated_memory_two_pass",
        )
