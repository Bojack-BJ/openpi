import pathlib

from PIL import Image

from openpi.hl_memory.data import LoadedVideoClips
from openpi.hl_memory.schema import HLMemoryPrediction
from openpi.hl_memory.zero_shot import ZeroShotClipSelection
from openpi.hl_memory.zero_shot import apply_rollout_language_memory_rule
from openpi.hl_memory.zero_shot import build_auto_memory_seconds
from openpi.hl_memory.zero_shot import build_recent_seconds
from openpi.hl_memory.zero_shot import build_rollout_end_seconds
from openpi.hl_memory.zero_shot import parse_seconds_argument
from openpi.hl_memory.zero_shot import parse_video_paths_argument
from openpi.hl_memory.zero_shot import save_prediction_debug_panel
from openpi.hl_memory.zero_shot import update_rollout_memory_seconds


def test_parse_seconds_argument_sorts_and_dedupes():
    assert parse_seconds_argument("3,1,1,2.5") == [1.0, 2.5, 3.0]


def test_parse_video_paths_argument_supports_csv_mapping():
    assert parse_video_paths_argument("robot_0=/tmp/left.mp4,robot_1=/tmp/right.mp4") == {
        "robot_0": pathlib.Path("/tmp/left.mp4"),
        "robot_1": pathlib.Path("/tmp/right.mp4"),
    }


def test_parse_video_paths_argument_supports_json_mapping():
    assert parse_video_paths_argument('{"front": "/tmp/front.mp4", "robot_0": "/tmp/left.mp4"}') == {
        "front": pathlib.Path("/tmp/front.mp4"),
        "robot_0": pathlib.Path("/tmp/left.mp4"),
    }


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
    assert "Task progress:" in updated.updated_language_memory
    assert "Current objective: Move the shoebox to the center of the table." in updated.updated_language_memory
    assert "Relevant objects: shoebox, center of table" in updated.updated_language_memory
    assert "t=8.0s" not in updated.updated_language_memory


def test_rollout_language_memory_rule_keeps_ll_friendly_shape_for_repeated_subtask():
    previous_memory = (
        "Task progress: The robot is working on: Move the shoebox to the center of the table.\n"
        "Current objective: Move the shoebox to the center of the table.\n"
        "Relevant objects: shoebox, center of table\n"
        "Notes: none"
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
    assert updated.updated_language_memory.count("\n") == 3
    assert "Current objective: Move the shoebox to the center of the table." in updated.updated_language_memory
    assert "t=10.0s" not in updated.updated_language_memory


def test_rollout_language_memory_rule_rewrites_debug_log_memory():
    prediction = HLMemoryPrediction(
        updated_language_memory=(
            "Progress memory:\n"
            "- t=8.0s; [preparation]; Move the shoebox; target=shoebox; goal=center"
        ),
        current_subtask="Fold the shoebox",
        keyframe_candidate_positions=(),
        phase="action",
        target_query="shoebox",
        goal_query="folded shoebox",
    )

    updated, changed = apply_rollout_language_memory_rule(
        prediction,
        previous_memory="Task started.",
        recent_end_sec=10.0,
    )

    assert changed
    assert updated.updated_language_memory.startswith("Task progress:")
    assert "Progress memory" not in updated.updated_language_memory
    assert "target=" not in updated.updated_language_memory


def test_save_prediction_debug_panel_writes_current_frame_summary(tmp_path):
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=(
            Image.new("RGB", (32, 24), color=(255, 0, 0)),
            Image.new("RGB", (32, 24), color=(0, 255, 0)),
        ),
        memory_valid_length=0,
        recent_valid_length=2,
    )
    selection = ZeroShotClipSelection(
        video_path=pathlib.Path("demo.mp4"),
        duration_sec=2.0,
        memory_seconds=(),
        recent_seconds=(0.0, 1.0),
    )
    prediction = HLMemoryPrediction(
        updated_language_memory=(
            "Task progress: crescent block placed.\n"
            "Current objective: return right hand.\n"
            "Relevant objects: crescent block, slot\n"
            "Notes: none"
        ),
        current_subtask="return right hand after crescent placement",
        keyframe_candidate_positions=(2,),
        phase="retreat",
        target_query="right hand",
        goal_query="home position",
    )

    output_path = save_prediction_debug_panel(
        tmp_path / "debug_panel.png",
        clips=clips,
        selection=selection,
        prediction=prediction,
        step_index=1,
        recent_end_sec=1.0,
        language_memory_before="Task started.",
        language_memory_after=prediction.updated_language_memory,
        keyframe_candidate_seconds=(1.0,),
    )

    assert output_path.exists()
    assert Image.open(output_path).size == (1400, 820)
