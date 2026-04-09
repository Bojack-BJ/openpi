import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge

import openpi.models.qwen2_5.adapter as _qwen2_5_adapter
import openpi.models.qwen3_5.rotary as _qwen3_5_rotary
import openpi.models.qwen3_5.text as _qwen3_5_text
import openpi.models.qwen3_5.vision as _qwen3_5_vision


QWEN3_5_VOCAB_SIZE = 248_320
QWEN3_5_ROPE_THETA = 1_000_000.0


class Qwen3_5_VLWithExpertModel(_qwen2_5_adapter.Qwen2_5_VLWithExpertModel):
    """JAX Qwen3.5 adapter.

    This path now uses official-style Qwen3.5 backbones:
    - 3x `linear_attention` + 1x `full_attention` repeating layer layout
    - `GatedDeltaNet` linear-attention blocks
    - `GatedAttention` full-attention blocks
    - a dedicated Conv3D-patch vision tower plus learned embeddings and patch merger
    """

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        *,
        use_adarms,
        precision: str,
        image_example,
        rngs: nnx.Rngs,
        hf_model_id: str | None = None,
    ):
        self.hf_model_id = hf_model_id
        if any(use_adarms):
            raise NotImplementedError("JAX Qwen3.5 adapter does not support pi05/AdaRMS conditioning yet.")
        if vlm_config.depth != action_expert_config.depth:
            raise ValueError(
                "JAX Qwen3.5 adapter currently requires matching prefix/expert depth: "
                f"{vlm_config.depth} != {action_expert_config.depth}"
            )
        if (
            vlm_config.num_heads != action_expert_config.num_heads
            or vlm_config.num_kv_heads != action_expert_config.num_kv_heads
            or vlm_config.head_dim != action_expert_config.head_dim
            or vlm_config.layer_types != action_expert_config.layer_types
            or vlm_config.linear_num_key_heads != action_expert_config.linear_num_key_heads
            or vlm_config.linear_num_value_heads != action_expert_config.linear_num_value_heads
            or vlm_config.linear_key_head_dim != action_expert_config.linear_key_head_dim
            or vlm_config.linear_value_head_dim != action_expert_config.linear_value_head_dim
            or vlm_config.linear_conv_kernel_dim != action_expert_config.linear_conv_kernel_dim
        ):
            raise ValueError(
                "JAX Qwen3.5 adapter currently requires matching full/linear attention geometry for prefix and expert."
            )

        mrope_section = vlm_config.mrope_section or _qwen3_5_rotary.default_mrope_section(
            _qwen3_5_rotary.rotary_dim(
                vlm_config.head_dim,
                vlm_config.partial_rotary_factor or _qwen3_5_rotary.QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR,
            )
        )
        llm = nnx_bridge.ToNNX(
            _qwen3_5_text.Module(
                configs=[vlm_config, action_expert_config],
                embed_dtype=precision,
                vocab_size=vlm_config.vocab_size or QWEN3_5_VOCAB_SIZE,
                rope_theta=vlm_config.rope_theta or QWEN3_5_ROPE_THETA,
                partial_rotary_factor=vlm_config.partial_rotary_factor
                or _qwen3_5_rotary.QWEN3_5_DEFAULT_PARTIAL_ROTARY_FACTOR,
                mrope_section=mrope_section,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
        self.llm = llm
        self.vision = _qwen3_5_vision.Qwen3_5VisionTower(
            output_width=vlm_config.width,
            precision=precision,
            image_example=image_example,
            rngs=rngs,
            hidden_size=vlm_config.vision_hidden_size or _qwen3_5_vision.QWEN3_5_VISION_HIDDEN_SIZE,
            depth=vlm_config.vision_depth or _qwen3_5_vision.QWEN3_5_VISION_DEPTH,
            mlp_dim=vlm_config.vision_mlp_dim or _qwen3_5_vision.QWEN3_5_VISION_MLP_DIM,
            num_heads=vlm_config.vision_num_heads or _qwen3_5_vision.QWEN3_5_VISION_NUM_HEADS,
            patch_size=vlm_config.vision_patch_size or _qwen3_5_vision.QWEN3_5_VISION_PATCH_SIZE,
            temporal_patch_size=vlm_config.vision_temporal_patch_size
            or _qwen3_5_vision.QWEN3_5_TEMPORAL_PATCH_SIZE,
            num_positions=vlm_config.vision_num_positions or _qwen3_5_vision.QWEN3_5_VISION_NUM_POSITIONS,
            spatial_merge_size=vlm_config.vision_spatial_merge_size or _qwen3_5_vision.QWEN3_5_SPATIAL_MERGE_SIZE,
            merger_dim=vlm_config.vision_merger_dim or _qwen3_5_vision.QWEN3_5_VISION_MERGER_DIM,
        )
