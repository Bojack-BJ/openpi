import jax
import jax.numpy as jnp

import openpi.models.pi0_config as _pi0_config
import openpi.models.qwen2_5.text as _qwen2_5_text
import openpi.models.vlm_backbone_config as _vlm_backbone_config


def _make_qwen_config() -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        vlm_backend="qwen2_5_vl",
        vlm_backbone_variant="qwen2_5_3b",
        action_expert_variant="qwen2_5_3b",
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
