import json

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.hf_adapter import Qwen25HLAdapter


def test_qwen_video_metadata_uses_configured_effective_fps():
    adapter = Qwen25HLAdapter(HLMemoryConfig(training_fps=20.0, frame_subsample=5))

    metadata = adapter._prepare_video_metadata([[object(), object(), object()]])

    assert metadata == [
        {
            "total_num_frames": 3,
            "fps": 4.0,
            "duration": 0.5,
            "frames_indices": [0, 1, 2],
        }
    ]


def test_memer_objective_target_text_is_minimal_and_uses_horizon_label():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="memer_objective"))
    sample = ExportedHLMemorySample(
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
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="current objective",
        horizon_current_objective="horizon objective",
    )

    payload = json.loads(adapter.build_target_text(sample))

    assert payload == {
        "current_objective": "horizon objective",
        "keyframe_candidate_positions": [2],
    }
