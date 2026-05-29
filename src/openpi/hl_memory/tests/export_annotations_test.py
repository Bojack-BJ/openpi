import argparse

from scripts.hl_memory.export_hl_annotations_from_subtasks import _segments_to_rows


def _args(**overrides) -> argparse.Namespace:
    values = dict(
        sampling_mode="fraction-rules",
        dense_sample_stride_frames=5,
        prediction_horizon_steps=2,
        keyframe_label_mode="event_boundary",
        keyframe_rule_path=None,
        instruction="task",
        target_query="",
        goal_query="",
        emit_success_events=False,
        progress_sample_fractions="",
        progress_extra_fractions="",
        short_segment_progress_fractions="",
        progress_sample_stride=0,
        progress_sample_target_frames=0,
        min_progress_samples_per_segment=0,
        max_progress_samples_per_segment=20,
        progress_min_gap=0,
        short_segment_max_frames=0,
        short_segment_progress_min_gap=-1,
        progress_sample_jitter=0.0,
        progress_sample_seed=0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_dense_stride_creates_expected_frames_and_horizon_labels():
    rows = _segments_to_rows(
        0,
        [(0, 12, "approach cup"), (12, 24, "place cup")],
        _args(sampling_mode="dense-stride", dense_sample_stride_frames=5, prediction_horizon_steps=2),
    )

    assert [row["frame_index"] for row in rows] == [0, 5, 10, 12, 17, 22]
    assert rows[0]["horizon_frame_index"] == 10
    assert rows[0]["horizon_current_objective"] == "approach cup"
    assert rows[2]["horizon_frame_index"] == 20
    assert rows[2]["horizon_current_objective"] == "place cup"
    assert rows[-1]["horizon_frame_index"] == 23
    assert rows[-1]["horizon_current_objective"] == "place cup"


def test_memer_rules_label_last_state_changing_keyframe_only():
    rows = _segments_to_rows(
        0,
        [(0, 10, "approach cup"), (10, 20, "place cup")],
        _args(
            sampling_mode="dense-stride",
            dense_sample_stride_frames=5,
            prediction_horizon_steps=0,
            keyframe_label_mode="memer_rules",
        ),
    )

    labels = {int(row["frame_index"]): row.get("keyframe_label") for row in rows}
    assert labels[0] is False
    assert labels[5] is False
    assert labels[19] is True
