import flax.nnx as nnx
import jax
import pytest

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_full_finetune_explicit_paligemma_backend():
    config = _pi0_config.Pi0Config(vlm_backend="paligemma")
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_uses_neutral_vlm_runtime_handle():
    config = _pi0_config.Pi0Config(vlm_backend="paligemma")
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    assert hasattr(abstract_model, "vlm_with_expert")
    assert not hasattr(abstract_model, "PaliGemma")


def test_pi0_load_remaps_legacy_paligemma_root():
    config = _pi0_config.Pi0Config(vlm_backend="paligemma")
    model = config.create(jax.random.key(0))
    params = nnx.state(model, nnx.Param).to_pure_dict()
    legacy_params = dict(params)
    legacy_params["PaliGemma"] = legacy_params.pop("vlm_with_expert")

    restored = config.load(legacy_params, remove_extra_params=False)
    flat_state = nnx.state(restored, nnx.Param).flat_state()
    assert any("vlm_with_expert" in str(path) for path in flat_state)


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(vlm_backbone_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(vlm_backbone_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_pi0_qwen_backend_scaffold_raises_on_image_embedding():
    config = _pi0_config.Pi0Config(
        vlm_backend="qwen2_5_vl",
        vlm_backbone_variant="qwen2_5_3b",
        action_expert_variant="qwen2_5_3b",
    )
    model = config.create(jax.random.key(0))
    with pytest.raises(NotImplementedError, match="JAX Qwen image embedding is not implemented yet"):
        model.embed_prefix(config.fake_obs())
