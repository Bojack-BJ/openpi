import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.pi0_config as _pi0_config
import openpi.models.qwen2_5.text as _qwen2_5_text
import openpi.models.vlm_backbone_config as _vlm_backbone_config


def _make_qwen_config(action_expert_variant: _vlm_backbone_config.Variant = "qwen2_5_3b") -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        vlm_backend="qwen2_5_vl",
        vlm_backbone_variant="qwen2_5_3b",
        action_expert_variant=action_expert_variant,
    )


def test_qwen_jax_scaffold_embed_language_tokens_shape():
    config = _make_qwen_config()
    model = config.create(jax.random.key(0))

    tokens = jnp.ones((2, 4), dtype=jnp.int32)
    embeds = model.vlm_with_expert.embed_language_tokens(tokens)

    assert embeds.shape == (2, 4, _vlm_backbone_config.get_config("qwen2_5_3b").width)


def test_qwen_jax_scaffold_text_forward_shapes():
    config = _make_qwen_config()
    model = config.create(jax.random.key(0))
    qwen_cfg = _vlm_backbone_config.get_config("qwen2_5_3b")

    prefix = jnp.ones((2, 3, qwen_cfg.width), dtype=jnp.bfloat16)
    suffix = jnp.ones((2, 2, qwen_cfg.width), dtype=jnp.bfloat16)
    mask = jnp.ones((2, 5, 5), dtype=bool)
    positions = jnp.broadcast_to(jnp.arange(5, dtype=jnp.int32)[None, :], (2, 5))

    (prefix_out, suffix_out), kv_cache = model.vlm_with_expert.forward(
        [prefix, suffix],
        mask=mask,
        positions=positions,
    )

    assert prefix_out.shape == prefix.shape
    assert suffix_out.shape == suffix.shape
    assert kv_cache is not None


def test_qwen_jax_config_vocab_matches_text_module():
    qwen_cfg = _vlm_backbone_config.get_config("qwen2_5_3b")
    assert qwen_cfg.vocab_size == _qwen2_5_text.QWEN2_5_VL_VOCAB_SIZE


def test_qwen2_5_small_action_experts_preserve_attention_interface():
    vlm_3b = _vlm_backbone_config.get_config("qwen2_5_3b")
    expert_700m = _vlm_backbone_config.get_config("qwen2_5_3b_action_700m")
    expert_400m = _vlm_backbone_config.get_config("qwen2_5_3b_action_400m")
    vlm_7b = _vlm_backbone_config.get_config("qwen2_5_7b")
    expert_1b = _vlm_backbone_config.get_config("qwen2_5_7b_action_1b")

    for vlm_cfg, expert_cfg in (
        (vlm_3b, expert_700m),
        (vlm_3b, expert_400m),
        (vlm_7b, expert_1b),
    ):
        assert expert_cfg.depth == vlm_cfg.depth
        assert expert_cfg.num_heads == vlm_cfg.num_heads
        assert expert_cfg.num_kv_heads == vlm_cfg.num_kv_heads
        assert expert_cfg.head_dim == vlm_cfg.head_dim
        assert expert_cfg.width < vlm_cfg.width
        assert expert_cfg.mlp_dim < vlm_cfg.mlp_dim


def test_qwen2_5_small_action_expert_model_builds():
    config = _make_qwen_config(action_expert_variant="qwen2_5_3b_action_400m")
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    flat_state = nnx.state(abstract_model, nnx.Param).flat_state()
    flat_paths = ["/".join(str(part) for part in path) for path in flat_state]

    assert any("action_in_proj/kernel" in path for path in flat_paths)
    assert any("vlm_with_expert/llm/layers_0/pre_attention_norm_1" in path for path in flat_paths)


def test_qwen_jax_scaffold_embed_image_and_multimodal_positions():
    config = _make_qwen_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)

    prefix_tokens, prefix_mask, _, prefix_layout = model.embed_prefix(obs)
    positions = model.vlm_with_expert.build_prefix_positions(prefix_mask, prefix_layout=prefix_layout)

    assert prefix_tokens.shape[0] == 2
    assert positions.shape == (3, 2, prefix_mask.shape[1])
    assert prefix_layout.image_token_lengths
    assert prefix_layout.image_grid_thw
