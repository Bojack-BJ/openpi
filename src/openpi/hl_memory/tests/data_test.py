from __future__ import annotations

import pathlib
import tempfile
import json

from PIL import Image

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import build_loaded_video_clips_from_frames
from openpi.hl_memory.data import load_exported_samples
from openpi.hl_memory.data import load_video_clips_for_sample


def _sample(**overrides) -> ExportedHLMemorySample:
    values = dict(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="memory",
        updated_language_memory="updated",
        current_subtask="current step",
        phase="current phase",
        target_query="target",
        goal_query="goal",
        keyframe_candidate_positions=(1, 3),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("frames/recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
    )
    values.update(overrides)
    return ExportedHLMemorySample(**values)


def test_memer_objective_target_uses_current_and_horizon_objectives():
    sample = _sample(
        current_objective="current objective",
        horizon_frame_index=10,
        horizon_current_objective="horizon objective",
        horizon_current_subtask="horizon subtask",
        horizon_phase="horizon phase",
    )

    prediction = sample.target_prediction(target_protocol="memer_objective")

    assert prediction.current_objective == "current objective"
    assert prediction.horizon_current_objective == "horizon objective"
    assert prediction.current_subtask == "current objective"
    assert prediction.phase == "current objective"
    assert prediction.keyframe_candidate_positions == (1, 3)
    assert prediction.target_query == ""
    assert prediction.goal_query == ""


def test_memer_objective_target_falls_back_to_current_objective():
    sample = _sample(current_objective="current objective")

    prediction = sample.target_prediction(target_protocol="memer_objective")

    assert prediction.current_objective == "current objective"
    assert prediction.horizon_current_objective == "current objective"


def test_subtask_keyframe_target_uses_current_objective_and_keyframes():
    sample = _sample(
        current_objective="current objective",
        current_subtask="current subtask",
        horizon_frame_index=10,
        horizon_current_objective="horizon objective",
        horizon_current_subtask="horizon subtask",
        horizon_phase="horizon phase",
    )

    prediction = sample.target_prediction(target_protocol="subtask_keyframe")

    assert prediction.current_subtask == "current objective"
    assert prediction.current_objective == "current objective"
    assert prediction.phase == "current objective"
    assert prediction.keyframe_candidate_positions == (1, 3)
    assert prediction.subtask_progress is None
    assert prediction.should_advance_objective is None


def test_target_prediction_can_use_canonical_keyframe_candidates():
    sample = _sample(
        current_objective="current objective",
        keyframe_candidate_positions=(1, 2, 3),
        keyframe_event_frame_indices=(5,),
        recent_frame_indices=(0, 5, 10),
        recent_valid_length=3,
    )

    prediction = sample.target_prediction(
        target_protocol="memer_objective",
        keyframe_candidate_label_mode="canonical",
    )

    assert prediction.keyframe_candidate_positions == (2,)


def test_keyframe_gated_memory_target_sets_completed_objective_from_keyframe_labels():
    sample = _sample(
        current_objective="place toast on plate",
        horizon_current_objective="grasp steak",
        keyframe_candidate_positions=(2,),
        keyframe_label=True,
    )

    prediction = sample.target_prediction(target_protocol="keyframe_gated_memory")

    assert prediction.current_objective == "place toast on plate"
    assert prediction.horizon_current_objective == "grasp steak"
    assert prediction.keyframe_candidate_positions == (2,)
    assert prediction.completed_objective == "place toast on plate"
    assert prediction.updated_language_memory


def test_keyframe_gated_memory_target_leaves_completed_objective_empty_without_keyframes():
    sample = _sample(
        current_objective="place toast on plate",
        keyframe_candidate_positions=(2,),
        keyframe_label=False,
    )

    prediction = sample.target_prediction(target_protocol="keyframe_gated_memory")

    assert prediction.completed_objective == ""


def test_known_prior_tracker_target_uses_progress_and_advance_labels():
    sample = _sample(
        current_objective="insert into top-left hole",
        task_progress="Completed subtasks: Grasp the stick.",
        subtask_progress=0.75,
        should_advance_objective=False,
    )

    prediction = sample.target_prediction(target_protocol="known_prior_tracker")

    assert prediction.current_objective == "insert into top-left hole"
    assert prediction.task_progress == "Completed subtasks: Grasp the stick."
    assert prediction.subtask_progress == 0.75
    assert prediction.should_advance_objective is False


def test_state_context_objective_protocols_use_memer_objective_target_without_progress_target():
    sample = _sample(
        current_objective="insert into top-left hole",
        horizon_current_objective="insert into top-right hole",
        task_progress="Completed subtasks: Grasp the stick.",
        last_objective="grasp stick",
        previous_stage_objective="grasp stick",
        subtask_progress=0.75,
        should_advance_objective=False,
    )

    for protocol in ("objective_memory_state", "objective_last_objective", "objective_prev_stage"):
        prediction = sample.target_prediction(target_protocol=protocol)

        assert prediction.current_objective == "insert into top-left hole"
        assert prediction.horizon_current_objective == "insert into top-right hole"
        assert prediction.keyframe_candidate_positions == (1, 3)
        if protocol == "objective_memory_state":
            assert prediction.updated_language_memory == sample.updated_language_memory
        else:
            assert prediction.task_progress == "No completed subtask yet."
        if protocol == "objective_last_objective":
            assert prediction.last_objective == "grasp stick"
        if protocol == "objective_prev_stage":
            assert prediction.previous_stage_objective == "grasp stick"
        assert prediction.subtask_progress is None
        assert prediction.should_advance_objective is None


def test_load_exported_samples_backfills_last_and_previous_stage_objectives():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = pathlib.Path(tmp_dir)
        samples_path = root / "samples.jsonl"
        samples = [
            _sample(sample_id="s0", step_index=0, current_objective="grasp stick"),
            _sample(sample_id="s1", step_index=1, current_objective="grasp stick"),
            _sample(sample_id="s2", step_index=2, current_objective="insert top-left"),
            _sample(sample_id="s3", step_index=3, current_objective="insert top-left"),
            _sample(sample_id="s4", step_index=4, current_objective="insert top-right"),
        ]
        with samples_path.open("w") as handle:
            for sample in samples:
                payload = sample.to_dict()
                payload.pop("last_objective", None)
                payload.pop("previous_stage_objective", None)
                handle.write(json.dumps(payload) + "\n")

        loaded = load_exported_samples(root)

        assert [sample.last_objective for sample in loaded] == [
            "",
            "grasp stick",
            "grasp stick",
            "insert top-left",
            "insert top-left",
        ]
        assert [sample.previous_stage_objective for sample in loaded] == [
            "",
            "",
            "grasp stick",
            "grasp stick",
            "insert top-left",
        ]


def test_sample_round_trips_proprio_fields():
    sample = _sample(
        recent_robot_states=((0.1, 0.2), (0.3, 0.4)),
        recent_robot_state_masks=((1.0, 0.0), (1.0, 1.0)),
        robot_state_dim_names=("joint_0", "joint_1"),
    )

    loaded = ExportedHLMemorySample.from_dict(sample.to_dict())

    assert loaded.recent_robot_states == ((0.1, 0.2), (0.3, 0.4))
    assert loaded.recent_robot_state_masks == ((1.0, 0.0), (1.0, 1.0))
    assert loaded.robot_state_dim_names == ("joint_0", "joint_1")


def test_load_video_clips_for_sample_pads_and_tracks_valid_lengths():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = pathlib.Path(tmp_dir)
        frames_dir = root / "frames"
        frames_dir.mkdir()
        Image.new("RGB", (10, 10), color=(255, 0, 0)).save(frames_dir / "mem.png")
        Image.new("RGB", (10, 10), color=(0, 255, 0)).save(frames_dir / "recent0.png")
        Image.new("RGB", (10, 10), color=(0, 0, 255)).save(frames_dir / "recent1.png")

        sample = ExportedHLMemorySample(
            sample_id="sample",
            episode_index=0,
            step_index=0,
            frame_index=0,
            instruction="task",
            language_memory="memory",
            updated_language_memory="updated",
            current_subtask="step",
            phase="step",
            target_query="",
            goal_query="",
            keyframe_candidate_positions=(1,),
            memory_frame_paths=("frames/mem.png",),
            memory_frame_indices=(4,),
            memory_valid_length=1,
            recent_frame_paths=("frames/recent0.png", "frames/recent1.png"),
            recent_frame_indices=(8, 9),
            recent_valid_length=2,
        )
        config = HLMemoryConfig(
            recent_frames_length=4,
            memory_length=3,
            frame_height=8,
            frame_width=8,
        )

        clips = load_video_clips_for_sample(sample, root, config)

        assert clips.memory_valid_length == 1
        assert clips.recent_valid_length == 2
        assert len(clips.memory_frames) == 3
        assert len(clips.recent_frames) == 4
        assert clips.memory_frames[0].size == (8, 8)
        assert clips.recent_frames[0].size == (8, 8)


def test_load_video_clips_for_sample_keeps_exported_canvas_size():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = pathlib.Path(tmp_dir)
        frames_dir = root / "frames"
        frames_dir.mkdir()
        Image.new("RGB", (456, 224), color=(255, 0, 0)).save(frames_dir / "recent.png")

        sample = ExportedHLMemorySample(
            sample_id="sample",
            episode_index=0,
            step_index=0,
            frame_index=0,
            instruction="task",
            language_memory="memory",
            updated_language_memory="updated",
            current_subtask="step",
            phase="step",
            target_query="",
            goal_query="",
            keyframe_candidate_positions=(),
            memory_frame_paths=(),
            memory_frame_indices=(),
            memory_valid_length=0,
            recent_frame_paths=("frames/recent.png",),
            recent_frame_indices=(0,),
            recent_valid_length=1,
        )
        config = HLMemoryConfig(recent_frames_length=1, memory_length=1)

        clips = load_video_clips_for_sample(sample, root, config)

        assert clips.recent_frames[0].size == (456, 224)


def test_build_loaded_video_clips_preserves_aspect_ratio_with_padding():
    config = HLMemoryConfig(
        recent_frames_length=1,
        memory_length=1,
        frame_height=10,
        frame_width=10,
    )
    wide_frame = Image.new("RGB", (20, 10), color=(255, 0, 0))

    clips = build_loaded_video_clips_from_frames(
        memory_frames=(),
        recent_frames=(wide_frame,),
        config=config,
    )

    assert clips.recent_frames[0].size == (10, 10)
    assert clips.recent_frames[0].getpixel((5, 5)) == (255, 0, 0)
    assert clips.recent_frames[0].getpixel((5, 0)) == (0, 0, 0)


def test_build_loaded_video_clips_can_preserve_input_frame_size():
    config = HLMemoryConfig(
        recent_frames_length=1,
        memory_length=1,
        frame_height=10,
        frame_width=10,
    )
    wide_frame = Image.new("RGB", (20, 10), color=(255, 0, 0))

    clips = build_loaded_video_clips_from_frames(
        memory_frames=(),
        recent_frames=(wide_frame,),
        config=config,
        preserve_input_size=True,
    )

    assert clips.recent_frames[0].size == (20, 10)
