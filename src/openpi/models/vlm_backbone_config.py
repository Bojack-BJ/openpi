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
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "qwen2_5_7b"]


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
        )
    raise ValueError(f"Unknown variant: {variant}")
