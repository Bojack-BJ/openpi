import json

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.hf_adapter import Qwen25HLAdapter
from openpi.hl_memory.proprio import PROPRIO_FRAME_TOKEN
from openpi.hl_memory.proprio import PROPRIO_SUMMARY_TOKEN
from openpi.hl_memory.proprio import build_proprio_batch
from openpi.hl_memory.proprio import render_proprio_token_text


def test_qwen_video_metadata_uses_configured_recent_sample_rate():
    adapter = Qwen25HLAdapter(HLMemoryConfig(training_fps=20.0, frame_subsample=5, recent_sample_hz=4.0))

    metadata = adapter._prepare_video_metadata([[object(), object(), object()]])

    assert metadata == [
        {
            "total_num_frames": 3,
            "fps": 4.0,
            "duration": 0.5,
            "frames_indices": [0, 1, 2],
        }
    ]


def test_memer_objective_target_text_is_minimal_and_uses_current_and_horizon_labels():
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
        "current_objective": "current objective",
        "horizon_current_objective": "horizon objective",
        "keyframe_candidate_positions": [2],
    }


def test_subtask_keyframe_target_text_is_minimal_and_uses_current_objective():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="subtask_keyframe"))
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
        horizon_current_subtask="horizon subtask",
    )

    payload = json.loads(adapter.build_target_text(sample))

    assert payload == {
        "current_objective": "current objective",
        "keyframe_candidate_positions": [2],
    }


def test_proprio_per_frame_plus_summary_renders_expected_token_count():
    config = HLMemoryConfig(proprio_enabled=True, proprio_token_mode="per_frame_plus_summary")
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
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent0.png", "recent1.png"),
        recent_frame_indices=(0, 1),
        recent_valid_length=2,
        recent_robot_states=((0.0,) * 14, (1.0,) * 14),
        recent_robot_state_masks=((1.0,) * 14, (1.0,) * 14),
    )

    rendered = render_proprio_token_text(sample, config)

    assert rendered.count(PROPRIO_SUMMARY_TOKEN) == 1
    assert rendered.count(PROPRIO_FRAME_TOKEN) == 2


def test_proprio_batch_uses_sample_recent_state_shape():
    import torch

    config = HLMemoryConfig(proprio_enabled=True, proprio_token_mode="per_frame", proprio_state_dim=14)
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
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent0.png", "recent1.png"),
        recent_frame_indices=(0, 1),
        recent_valid_length=2,
        recent_robot_states=((0.0,) * 14, (1.0,) * 14),
        recent_robot_state_masks=((1.0,) * 7 + (0.0,) * 7, (1.0,) * 14),
    )

    states, masks = build_proprio_batch([sample], config, device=torch.device("cpu"))

    assert tuple(states.shape) == (1, 2, 14)
    assert tuple(masks.shape) == (1, 2, 14)
    assert masks[0, 0, 7:].sum().item() == 0.0
