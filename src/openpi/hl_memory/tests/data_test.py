from __future__ import annotations

import pathlib
import tempfile

from PIL import Image

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import build_loaded_video_clips_from_frames
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


def test_memer_objective_target_uses_horizon_objective_when_available():
    sample = _sample(
        current_objective="current objective",
        horizon_frame_index=10,
        horizon_current_objective="horizon objective",
        horizon_current_subtask="horizon subtask",
        horizon_phase="horizon phase",
    )

    prediction = sample.target_prediction(target_protocol="memer_objective")

    assert prediction.current_objective == "horizon objective"
    assert prediction.current_subtask == "horizon subtask"
    assert prediction.phase == "horizon phase"
    assert prediction.keyframe_candidate_positions == (1, 3)
    assert prediction.target_query == ""
    assert prediction.goal_query == ""


def test_memer_objective_target_falls_back_to_current_objective():
    sample = _sample(current_objective="current objective")

    prediction = sample.target_prediction(target_protocol="memer_objective")

    assert prediction.current_objective == "current objective"


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
