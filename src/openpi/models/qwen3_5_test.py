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
