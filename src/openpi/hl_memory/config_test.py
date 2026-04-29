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


def test_qwen3_5_4b_variant_sets_model_id():
    config = HLMemoryConfig(vlm_backend="qwen3_5_vl", vlm_variant="4b")

    assert config.vlm_variant == "qwen3_5_4b"
    assert config.resolved_model_id == "Qwen/Qwen3.5-4B"


def test_qwen3_5_rejects_unknown_variant():
    with pytest.raises(ValueError, match="qwen3_5_vl"):
        HLMemoryConfig(vlm_backend="qwen3_5_vl", vlm_variant="qwen3_5_9b")


def test_thinking_budget_must_be_positive():
    with pytest.raises(ValueError, match="thinking_budget_tokens"):
        HLMemoryConfig(thinking_budget_tokens=0)
