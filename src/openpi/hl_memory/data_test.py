from __future__ import annotations

import pathlib
import tempfile

from PIL import Image

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import build_loaded_video_clips_from_frames
from openpi.hl_memory.data import load_video_clips_for_sample


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
