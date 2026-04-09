import dataclasses
from typing import Literal

import openpi.models.lora as lora


@dataclasses.dataclass
class Config:
    """Geometry for one expert branch in the shared-attention stack.

    `width` and `mlp_dim` are expert-local; different experts may use different values.
    Joint attention only requires the paired experts to agree on the attention interface
    (`depth`, `num_heads`, `num_kv_heads`, `head_dim`) for the implementation that mixes them.
    """

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    layer_types: tuple[str, ...] | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    partial_rotary_factor: float | None = None
    mrope_section: tuple[int, int, int] | None = None
    vocab_size: int | None = None
    rope_theta: float | None = None
    vision_hidden_size: int | None = None
    vision_depth: int | None = None
    vision_mlp_dim: int | None = None
    vision_num_heads: int | None = None
    vision_patch_size: int | None = None
    vision_temporal_patch_size: int | None = None
    vision_num_positions: int | None = None
    vision_spatial_merge_size: int | None = None
    vision_merger_dim: int | None = None
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal[
    "dummy",
    "gemma_300m",
    "gemma_300m_lora",
    "gemma_2b",
    "gemma_2b_lora",
    "qwen2_5_3b",
    "qwen2_5_7b",
    "qwen2_5_3b_action_700m",
    "qwen2_5_3b_action_400m",
    "qwen2_5_7b_action_1b",
    "qwen3_5_2b",
    "qwen3_5_4b",
    "qwen3_5_2b_action_700m",
    "qwen3_5_2b_action_400m",
    "qwen3_5_4b_action_1b",
    "qwen3_5_4b_action_700m",
    "qwen3_5_4b_action_400m",
]


def get_config(variant: Variant) -> Config:
    """Returns geometry and optional LoRA config for the selected VLM backbone."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    if variant == "qwen2_5_7b":
        return Config(
            width=3584,
            depth=28,
            mlp_dim=18_944,
            num_heads=28,
            num_kv_heads=4,
            head_dim=128,
            vocab_size=151_936,
        )
    if variant == "qwen2_5_7b_action_1b":
        return Config(
            width=1536,
            depth=28,
            mlp_dim=6144,
            num_heads=28,
            num_kv_heads=4,
            head_dim=128,
            vocab_size=151_936,
        )
    if variant == "qwen2_5_3b":
        return Config(
            width=2048,
            depth=36,
            mlp_dim=11_008,
            num_heads=16,
            num_kv_heads=2,
            head_dim=128,
            vocab_size=151_936,
        )
    if variant == "qwen2_5_3b_action_700m":
        return Config(
            width=1024,
            depth=36,
            mlp_dim=4096,
            num_heads=16,
            num_kv_heads=2,
            head_dim=128,
            vocab_size=151_936,
        )
    if variant == "qwen2_5_3b_action_400m":
        return Config(
            width=768,
            depth=36,
            mlp_dim=3072,
            num_heads=16,
            num_kv_heads=2,
            head_dim=128,
            vocab_size=151_936,
        )
    if variant == "qwen3_5_2b":
        return Config(
            width=2048,
            depth=24,
            mlp_dim=6144,
            num_heads=8,
            num_kv_heads=2,
            head_dim=256,
            layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 6,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
            vocab_size=248_320,
            rope_theta=10_000_000.0,
            vision_hidden_size=1024,
            vision_depth=24,
            vision_mlp_dim=4096,
            vision_num_heads=16,
            vision_patch_size=16,
            vision_temporal_patch_size=2,
            vision_num_positions=2304,
            vision_spatial_merge_size=2,
            vision_merger_dim=4096,
        )
    if variant == "qwen3_5_2b_action_700m":
        return Config(
            width=1280,
            depth=24,
            mlp_dim=5120,
            num_heads=8,
            num_kv_heads=2,
            head_dim=256,
            layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 6,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
            vocab_size=248_320,
            rope_theta=10_000_000.0,
            vision_hidden_size=1024,
            vision_depth=24,
            vision_mlp_dim=4096,
            vision_num_heads=16,
            vision_patch_size=16,
            vision_temporal_patch_size=2,
            vision_num_positions=2304,
            vision_spatial_merge_size=2,
            vision_merger_dim=4096,
        )
    if variant == "qwen3_5_2b_action_400m":
        return Config(
            width=768,
            depth=24,
            mlp_dim=3072,
            num_heads=8,
            num_kv_heads=2,
            head_dim=256,
            layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 6,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
            vocab_size=248_320,
            rope_theta=10_000_000.0,
            vision_hidden_size=1024,
            vision_depth=24,
            vision_mlp_dim=4096,
            vision_num_heads=16,
            vision_patch_size=16,
            vision_temporal_patch_size=2,
            vision_num_positions=2304,
            vision_spatial_merge_size=2,
            vision_merger_dim=4096,
        )
    if variant == "qwen3_5_4b":
        return Config(
            width=2560,
            depth=32,
            mlp_dim=9216,
            num_heads=16,
            num_kv_heads=4,
            head_dim=256,
            layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 8,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
            vocab_size=248_320,
            rope_theta=10_000_000.0,
            vision_hidden_size=1024,
            vision_depth=24,
            vision_mlp_dim=4096,
            vision_num_heads=16,
            vision_patch_size=16,
            vision_temporal_patch_size=2,
            vision_num_positions=2304,
            vision_spatial_merge_size=2,
            vision_merger_dim=4096,
        )
    if variant == "qwen3_5_4b_action_1b":
        return Config(
            width=1536,
            depth=32,
            mlp_dim=6144,
            num_heads=16,
            num_kv_heads=4,
            head_dim=256,
            layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 8,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
            vocab_size=248_320,
            rope_theta=10_000_000.0,
            vision_hidden_size=1024,
            vision_depth=24,
            vision_mlp_dim=4096,
            vision_num_heads=16,
            vision_patch_size=16,
            vision_temporal_patch_size=2,
            vision_num_positions=2304,
            vision_spatial_merge_size=2,
            vision_merger_dim=4096,
        )
    if variant == "qwen3_5_4b_action_700m":
        return Config(
            width=1280,
            depth=32,
            mlp_dim=5120,
            num_heads=16,
            num_kv_heads=4,
            head_dim=256,
            layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 8,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
            vocab_size=248_320,
            rope_theta=10_000_000.0,
            vision_hidden_size=1024,
            vision_depth=24,
            vision_mlp_dim=4096,
            vision_num_heads=16,
            vision_patch_size=16,
            vision_temporal_patch_size=2,
            vision_num_positions=2304,
            vision_spatial_merge_size=2,
            vision_merger_dim=4096,
        )
    if variant == "qwen3_5_4b_action_400m":
        return Config(
            width=1024,
            depth=32,
            mlp_dim=4096,
            num_heads=16,
            num_kv_heads=4,
            head_dim=256,
            layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 8,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
            vocab_size=248_320,
            rope_theta=10_000_000.0,
            vision_hidden_size=1024,
            vision_depth=24,
            vision_mlp_dim=4096,
            vision_num_heads=16,
            vision_patch_size=16,
            vision_temporal_patch_size=2,
            vision_num_positions=2304,
            vision_spatial_merge_size=2,
            vision_merger_dim=4096,
        )
    raise ValueError(f"Unknown variant: {variant}")
