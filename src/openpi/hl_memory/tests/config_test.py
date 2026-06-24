import pytest

from openpi.hl_memory.config import HLMemoryConfig


def test_qwen3_5_defaults_to_2b_model_id():
    config = HLMemoryConfig(vlm_backend="qwen3_5_vl")

    assert config.vlm_variant == "qwen3_5_2b"
    assert config.resolved_model_id == "Qwen/Qwen3.5-2B"
    assert config.supports_runtime_backend
    assert not config.enable_thinking
    assert config.thinking_budget_tokens == 128
    assert config.thinking_max_new_tokens == 1024
    assert config.parallel_mode == "none"
    assert config.device_map == "auto"
    assert config.tensor_parallel_plan == "auto"


def test_qwen3_5_4b_variant_sets_model_id():
    config = HLMemoryConfig(vlm_backend="qwen3_5_vl", vlm_variant="4b")

    assert config.vlm_variant == "qwen3_5_4b"
    assert config.resolved_model_id == "Qwen/Qwen3.5-4B"


def test_qwen3_vl_defaults_to_4b_model_id():
    config = HLMemoryConfig(vlm_backend="qwen3_vl")

    assert config.vlm_variant == "qwen3_vl_4b"
    assert config.resolved_model_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert config.supports_runtime_backend


def test_qwen3_vl_4b_alias_sets_model_id():
    config = HLMemoryConfig(vlm_backend="qwen3_vl", vlm_variant="4b")

    assert config.vlm_variant == "qwen3_vl_4b"
    assert config.resolved_model_id == "Qwen/Qwen3-VL-4B-Instruct"


def test_qwen3_5_27b_variant_sets_model_id():
    config = HLMemoryConfig(vlm_backend="qwen3_5_vl", vlm_variant="27b")

    assert config.vlm_variant == "qwen3_5_27b"
    assert config.resolved_model_id == "Qwen/Qwen3.5-27B"


def test_qwen3_5_rejects_unknown_variant():
    with pytest.raises(ValueError, match="qwen3_5_vl"):
        HLMemoryConfig(vlm_backend="qwen3_5_vl", vlm_variant="qwen3_5_9b")


def test_qwen3_vl_rejects_unknown_variant():
    with pytest.raises(ValueError, match="qwen3_vl"):
        HLMemoryConfig(vlm_backend="qwen3_vl", vlm_variant="qwen3_vl_8b")


def test_thinking_budget_must_be_positive():
    with pytest.raises(ValueError, match="thinking_budget_tokens"):
        HLMemoryConfig(thinking_budget_tokens=0)


def test_parallel_mode_accepts_device_map():
    config = HLMemoryConfig(parallel_mode="device_map", device_map="balanced")

    assert config.parallel_mode == "device_map"
    assert config.device_map == "balanced"


def test_parallel_mode_accepts_tensor_parallel():
    config = HLMemoryConfig(parallel_mode="tensor_parallel", tensor_parallel_plan="auto")

    assert config.parallel_mode == "tensor_parallel"
    assert config.tensor_parallel_plan == "auto"


def test_parallel_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="parallel_mode"):
        HLMemoryConfig(parallel_mode="fsdp")  # type: ignore[arg-type]


def test_hl_memory_config_defaults_to_fixed_recent_sample_rate():
    config = HLMemoryConfig(training_fps=20.0, frame_subsample=5)

    assert config.recent_sample_hz == 2.0
    assert config.recent_step_sec == 0.5
    assert config.recent_window_sec == 3.5
    assert config.video_fps == 2.0
    assert config.frame_width == 456
    assert config.frame_height == 224


def test_hl_memory_config_derives_window_from_recent_sample_rate():
    config = HLMemoryConfig(training_fps=20.0, frame_subsample=2, recent_frames_length=9, recent_sample_hz=2.0)

    assert config.recent_step_sec == 0.5
    assert config.recent_window_sec == 4.0
    assert config.video_fps == 2.0


def test_target_protocol_accepts_memer_objective():
    config = HLMemoryConfig(target_protocol="memer_objective")

    assert config.target_protocol == "memer_objective"


def test_target_protocol_accepts_subtask_keyframe():
    config = HLMemoryConfig(target_protocol="subtask_keyframe")

    assert config.target_protocol == "subtask_keyframe"


def test_target_protocol_accepts_known_prior_tracker():
    config = HLMemoryConfig(target_protocol="known_prior_tracker")

    assert config.target_protocol == "known_prior_tracker"


def test_target_protocol_accepts_state_context_objective_protocols():
    for protocol in ("objective_memory_state", "objective_last_objective", "objective_prev_stage"):
        config = HLMemoryConfig(target_protocol=protocol)

        assert config.target_protocol == protocol


def test_target_protocol_accepts_keyframe_gated_memory():
    config = HLMemoryConfig(target_protocol="keyframe_gated_memory")

    assert config.target_protocol == "keyframe_gated_memory"


def test_target_protocol_accepts_keyframe_gated_memory_two_pass():
    config = HLMemoryConfig(target_protocol="keyframe_gated_memory_two_pass")

    assert config.target_protocol == "keyframe_gated_memory_two_pass"


def test_target_protocol_accepts_memer_film_progress_two_pass():
    config = HLMemoryConfig(
        target_protocol="memer_film_progress_two_pass",
        progress_condition_enabled=True,
        state_condition_enabled=True,
    )

    assert config.target_protocol == "memer_film_progress_two_pass"
    assert config.progress_condition_input_mode == "completed_only"
    assert config.state_condition_mode == "film"


def test_typed_mask_protocol_requires_qwen25():
    config = HLMemoryConfig(
        vlm_backend="qwen2_5_vl",
        target_protocol="keyframe_gated_memory_typed_mask",
    )
    assert config.target_protocol == "keyframe_gated_memory_typed_mask"

    for backend in ("qwen3_5_vl", "qwen3_vl"):
        with pytest.raises(ValueError, match="only supported"):
            HLMemoryConfig(
                vlm_backend=backend,
                target_protocol="keyframe_gated_memory_typed_mask",
            )


def test_two_pass_training_proposal_noise_probability_is_validated():
    config = HLMemoryConfig(two_pass_training_proposal_noise_probability=0.25)

    assert config.two_pass_training_proposal_noise_probability == 0.25
    with pytest.raises(ValueError, match="two_pass_training_proposal_noise_probability"):
        HLMemoryConfig(two_pass_training_proposal_noise_probability=1.1)


def test_target_protocol_rejects_unknown_value():
    with pytest.raises(ValueError, match="target_protocol"):
        HLMemoryConfig(target_protocol="unknown")  # type: ignore[arg-type]


def test_proprio_config_accepts_soft_token_modes():
    config = HLMemoryConfig(proprio_enabled=True, proprio_token_mode="per_frame_plus_summary")

    assert config.proprio_enabled is True
    assert config.proprio_state_dim == 14
    assert config.proprio_hidden_dim == 512


def test_proprio_config_rejects_unknown_token_mode():
    with pytest.raises(ValueError, match="proprio_token_mode"):
        HLMemoryConfig(proprio_enabled=True, proprio_token_mode="none")  # type: ignore[arg-type]


def test_keyframe_auxiliary_config_validates_dimensions_and_timing_sigma():
    config = HLMemoryConfig(
        keyframe_aux_enabled=True,
        keyframe_aux_hidden_dim=256,
        keyframe_aux_timing_sigma_sec=0.25,
    )

    assert config.keyframe_aux_enabled is True
    assert config.keyframe_aux_hidden_dim == 256
    assert config.keyframe_aux_timing_sigma_sec == 0.25

    with pytest.raises(ValueError, match="keyframe_aux_hidden_dim"):
        HLMemoryConfig(keyframe_aux_hidden_dim=0)
    with pytest.raises(ValueError, match="keyframe_aux_timing_sigma_sec"):
        HLMemoryConfig(keyframe_aux_timing_sigma_sec=0.0)
