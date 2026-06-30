import torch

from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.proprio import apply_proprio_ablation
from openpi.hl_memory.proprio import ProprioTokenProjector


def _sample_with_proprio() -> ExportedHLMemorySample:
    return ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="do task",
        language_memory="",
        updated_language_memory="",
        current_subtask="move",
        phase="move",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("a.png", "b.png"),
        recent_frame_indices=(0, 1),
        recent_valid_length=2,
        recent_robot_states=(
            tuple(float(i) for i in range(14)),
            tuple(float(i + 100) for i in range(14)),
        ),
        recent_robot_state_masks=(
            tuple(1.0 for _ in range(14)),
            tuple(1.0 for _ in range(14)),
        ),
    )


def test_apply_proprio_ablation_gripper_only_zeros_non_gripper_channels():
    ablated = apply_proprio_ablation(_sample_with_proprio(), mode="gripper_only")

    for row in ablated.recent_robot_states:
        assert row[6] != 0.0
        assert row[13] != 0.0
        assert all(row[index] == 0.0 for index in range(14) if index not in {6, 13})


def test_apply_proprio_ablation_reverse_time_reverses_sequence():
    sample = _sample_with_proprio()
    ablated = apply_proprio_ablation(sample, mode="reverse_time")

    assert ablated.recent_robot_states[0] == sample.recent_robot_states[1]
    assert ablated.recent_robot_states[1] == sample.recent_robot_states[0]


def test_split_proprio_projector_preserves_expected_shapes():
    projector = ProprioTokenProjector(
        state_dim=14,
        hidden_dim=32,
        output_dim=24,
        dropout=0.0,
        noise_std=0.0,
        projector_mode="split",
    )
    states = torch.randn(2, 4, 14)
    masks = torch.ones_like(states)

    frame_tokens, summary = projector(states, masks)

    assert frame_tokens.shape == (2, 4, 24)
    assert summary.shape == (2, 24)
