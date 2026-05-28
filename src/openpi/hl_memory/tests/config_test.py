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


def test_qwen3_5_27b_variant_sets_model_id():
    config = HLMemoryConfig(vlm_backend="qwen3_5_vl", vlm_variant="27b")

    assert config.vlm_variant == "qwen3_5_27b"
    assert config.resolved_model_id == "Qwen/Qwen3.5-27B"


def test_qwen3_5_rejects_unknown_variant():
    with pytest.raises(ValueError, match="qwen3_5_vl"):
        HLMemoryConfig(vlm_backend="qwen3_5_vl", vlm_variant="qwen3_5_9b")


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


def test_hl_memory_config_derives_video_fps_from_training_rate_and_subsample():
    config = HLMemoryConfig(training_fps=20.0, frame_subsample=5)

    assert config.video_fps == 4.0
    assert config.frame_width == 456
    assert config.frame_height == 224


def test_target_protocol_accepts_memer_objective():
    config = HLMemoryConfig(target_protocol="memer_objective")

    assert config.target_protocol == "memer_objective"


def test_target_protocol_rejects_unknown_value():
    with pytest.raises(ValueError, match="target_protocol"):
        HLMemoryConfig(target_protocol="unknown")  # type: ignore[arg-type]
