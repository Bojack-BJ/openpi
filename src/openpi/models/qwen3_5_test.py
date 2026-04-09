import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config
import openpi.models.qwen3_5.rotary as _qwen3_5_rotary
import openpi.models.vlm_backbone_config as _vlm_backbone_config


def _make_qwen3_5_config() -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        vlm_backend="qwen3_5_vl",
        vlm_backbone_variant="qwen3_5_2b",
        action_expert_variant="qwen3_5_2b",
    )


def test_qwen3_5_jax_embed_language_tokens_shape():
    config = _make_qwen3_5_config()
    model = config.create(jax.random.key(0))

    embeds = model.vlm_with_expert.embed_language_tokens(jax.numpy.ones((2, 4), dtype=jax.numpy.int32))
    assert embeds.shape == (2, 4, _vlm_backbone_config.get_config("qwen3_5_2b").width)


def test_qwen3_5_config_uses_official_hybrid_layout():
    config = _vlm_backbone_config.get_config("qwen3_5_2b")

    assert config.layer_types is not None
    assert config.layer_types[:4] == ("linear_attention", "linear_attention", "linear_attention", "full_attention")
    assert config.linear_num_key_heads == 16
    assert config.linear_num_value_heads == 16
    assert config.vision_temporal_patch_size == 2
    assert config.vision_spatial_merge_size == 2


def test_qwen3_5_4b_config_matches_official_geometry():
    config = _vlm_backbone_config.get_config("qwen3_5_4b")

    assert config.width == 2560
    assert config.depth == 32
    assert config.mlp_dim == 9216
    assert config.num_heads == 16
    assert config.num_kv_heads == 4
    assert config.head_dim == 256
    assert config.layer_types == ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 8
    assert config.linear_num_key_heads == 16
    assert config.linear_num_value_heads == 32
    assert config.linear_key_head_dim == 128
    assert config.linear_value_head_dim == 128
    assert config.linear_conv_kernel_dim == 4
    assert config.partial_rotary_factor == 0.25
    assert config.mrope_section == (11, 11, 10)
    assert config.vocab_size == 248320
    assert config.rope_theta == 10_000_000.0
    assert config.vision_hidden_size == 1024
    assert config.vision_depth == 24
    assert config.vision_mlp_dim == 4096
    assert config.vision_num_heads == 16
    assert config.vision_patch_size == 16
    assert config.vision_temporal_patch_size == 2
    assert config.vision_num_positions == 2304
    assert config.vision_spatial_merge_size == 2
    assert config.vision_merger_dim == 4096


def test_qwen3_5_small_action_experts_preserve_attention_interface():
    vlm_2b = _vlm_backbone_config.get_config("qwen3_5_2b")
    expert_700m = _vlm_backbone_config.get_config("qwen3_5_2b_action_700m")
    expert_400m = _vlm_backbone_config.get_config("qwen3_5_2b_action_400m")
    vlm_4b = _vlm_backbone_config.get_config("qwen3_5_4b")
    expert_1b = _vlm_backbone_config.get_config("qwen3_5_4b_action_1b")
    expert_4b_700m = _vlm_backbone_config.get_config("qwen3_5_4b_action_700m")
    expert_4b_400m = _vlm_backbone_config.get_config("qwen3_5_4b_action_400m")

    for vlm_cfg, expert_cfg in (
        (vlm_2b, expert_700m),
        (vlm_2b, expert_400m),
        (vlm_4b, expert_1b),
        (vlm_4b, expert_4b_700m),
        (vlm_4b, expert_4b_400m),
    ):
        assert expert_cfg.depth == vlm_cfg.depth
        assert expert_cfg.num_heads == vlm_cfg.num_heads
        assert expert_cfg.num_kv_heads == vlm_cfg.num_kv_heads
        assert expert_cfg.head_dim == vlm_cfg.head_dim
        assert expert_cfg.layer_types == vlm_cfg.layer_types
        assert expert_cfg.linear_num_key_heads == vlm_cfg.linear_num_key_heads
        assert expert_cfg.linear_num_value_heads == vlm_cfg.linear_num_value_heads
        assert expert_cfg.linear_key_head_dim == vlm_cfg.linear_key_head_dim
        assert expert_cfg.linear_value_head_dim == vlm_cfg.linear_value_head_dim
        assert expert_cfg.linear_conv_kernel_dim == vlm_cfg.linear_conv_kernel_dim
        assert expert_cfg.width < vlm_cfg.width
        assert expert_cfg.mlp_dim < vlm_cfg.mlp_dim


def test_qwen3_5_small_action_expert_model_builds():
    config = _pi0_config.Pi0Config(
        vlm_backend="qwen3_5_vl",
        vlm_backbone_variant="qwen3_5_2b",
        action_expert_variant="qwen3_5_2b_action_400m",
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    flat_state = nnx.state(abstract_model, nnx.Param).flat_state()
    flat_paths = ["/".join(str(part) for part in path) for path in flat_state]

    assert any("action_in_proj/kernel" in path for path in flat_paths)
    assert any(
        "vlm_with_expert/llm/layers/0/layers_0/pre_attention_norm_1" in path
        or "vlm_with_expert/llm/layers_0/pre_attention_norm_1" in path
        for path in flat_paths
    )


def test_qwen3_5_resolves_mrope_sections_for_mixed_rotary_widths():
    assert _qwen3_5_rotary.resolve_mrope_section(64, (11, 11, 10)) == (11, 11, 10)
    assert _qwen3_5_rotary.resolve_mrope_section(32, (11, 11, 10)) == (6, 5, 5)


def test_qwen3_5_jax_embed_image_shape():
    config = _make_qwen3_5_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)

    image_tokens, metadata = model.vlm_with_expert.embed_image_with_metadata(obs.images["base_0_rgb"])

    assert image_tokens.shape[0] == 2
    assert image_tokens.shape[-1] == _vlm_backbone_config.get_config("qwen3_5_2b").width
    assert metadata is not None
    assert metadata["grid_thw"] == (1, 7, 7)


def test_qwen3_5_jax_embed_prefix_and_positions():
    config = _make_qwen3_5_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)

    prefix_tokens, prefix_mask, prefix_ar_mask, prefix_layout = model.embed_prefix(obs)
    positions = model.vlm_with_expert.build_prefix_positions(prefix_mask, prefix_layout=prefix_layout)

    assert prefix_tokens.shape[0] == 2
    assert positions.shape == (3, 2, prefix_mask.shape[1])
    assert prefix_layout.image_token_lengths
    assert bool(jax.numpy.all(prefix_ar_mask))


def test_qwen3_5_text_param_layout_matches_current_official_key_assumptions():
    config = _make_qwen3_5_config()
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    flat_state = nnx.state(abstract_model, nnx.Param).flat_state()
    flat_paths = ["/".join(str(part) for part in path) for path in flat_state]

    assert any(
        "vlm_with_expert/llm/layers/0/layers_3/self_attn/qg_einsum" in path
        or "vlm_with_expert/llm/layers_3/self_attn/qg_einsum" in path
        for path in flat_paths
    )
    assert not any(
        "vlm_with_expert/llm/layers/0/layers_3/self_attn/q_einsum" in path
        or "vlm_with_expert/llm/layers_3/self_attn/q_einsum" in path
        for path in flat_paths
    )
    assert any(
        "vlm_with_expert/llm/layers/0/layers_0/self_attn/norm" in path
        or "vlm_with_expert/llm/layers_0/self_attn/norm" in path
        for path in flat_paths
    )
    assert not any(
        "vlm_with_expert/llm/layers/0/layers_0/self_attn/q_norm" in path
        or "vlm_with_expert/llm/layers_0/self_attn/q_norm" in path
        for path in flat_paths
    )
    assert not any(
        "vlm_with_expert/llm/layers/0/layers_0/self_attn/k_norm" in path
        or "vlm_with_expert/llm/layers_0/self_attn/k_norm" in path
        for path in flat_paths
    )
