from openpi.hl_memory.zero_shot import build_auto_memory_seconds
from openpi.hl_memory.zero_shot import build_recent_seconds
from openpi.hl_memory.zero_shot import build_rollout_end_seconds
from openpi.hl_memory.schema import HLMemoryPrediction
from openpi.hl_memory.zero_shot import apply_rollout_language_memory_rule
from openpi.hl_memory.zero_shot import parse_seconds_argument
from openpi.hl_memory.zero_shot import update_rollout_memory_seconds


def test_parse_seconds_argument_sorts_and_dedupes():
    assert parse_seconds_argument("3,1,1,2.5") == [1.0, 2.5, 3.0]


def test_build_recent_seconds_defaults_from_video_end():
    assert build_recent_seconds(10.0, clip_length=4, recent_step_sec=2.0) == [4.0, 6.0, 8.0, 10.0]


def test_build_auto_memory_seconds_uses_prefix_before_recent():
    assert build_auto_memory_seconds(20.0, recent_seconds=[12.0, 13.0], clip_length=3) == [0.0, 5.9995, 11.999]


def test_build_rollout_end_seconds_includes_final_video_end():
    assert build_rollout_end_seconds(10.0, interval_sec=4.0, start_sec=2.0) == [2.0, 6.0, 10.0]


def test_update_rollout_memory_seconds_keeps_latest_unique_seconds():
    assert update_rollout_memory_seconds([1.0, 2.0], [2.0, 5.0, 7.0], memory_length=3) == (2.0, 5.0, 7.0)


def test_update_rollout_memory_seconds_clusters_temporally_close_keyframes():
    assert update_rollout_memory_seconds(
        [5.0],
        [7.0, 9.0, 13.0],
        memory_length=8,
        merge_distance_sec=2.0,
    ) == (7.0, 13.0)


def test_rollout_language_memory_rule_advances_generic_memory():
    prediction = HLMemoryPrediction(
        updated_language_memory="Task started.",
        current_subtask="Move the shoebox to the center of the table.",
        keyframe_candidate_positions=(),
        phase="preparation",
        target_query="shoebox",
        goal_query="center of table",
    )

    updated, changed = apply_rollout_language_memory_rule(
        prediction,
        previous_memory="Task started.",
        recent_end_sec=8.0,
    )

    assert changed
    assert "Move the shoebox to the center of the table." in updated.updated_language_memory
    assert "t=8.0s" in updated.updated_language_memory


def test_rollout_language_memory_rule_merges_repeated_subtask():
    previous_memory = (
        "Progress memory:\n"
        "- t=8.0s; [preparation]; Move the shoebox to the center of the table.; "
        "target=shoebox; goal=center of table\n"
        "Current subtask: Move the shoebox to the center of the table."
    )
    prediction = HLMemoryPrediction(
        updated_language_memory=previous_memory,
        current_subtask="Move the shoebox to the center of the table.",
        keyframe_candidate_positions=(),
        phase="preparation",
        target_query="shoebox",
        goal_query="center of table",
    )

    updated, changed = apply_rollout_language_memory_rule(
        prediction,
        previous_memory=previous_memory,
        recent_end_sec=10.0,
    )

    assert changed
    assert updated.updated_language_memory.count("- ") == 1
    assert "t=10.0s" in updated.updated_language_memory
