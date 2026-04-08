import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge

import openpi.models.qwen2_5.text as _qwen_text
import openpi.models.qwen2_5.vision as _qwen_vision


class Qwen2_5_VLWithExpertModel(nnx.Module):
    """Initial JAX Qwen2.5-VL adapter scaffold.

    This wires a real Qwen-style text/expert transformer stack into the JAX backend factory.
    The image tower and multimodal projector are still TODO, so full VLA training is not yet
    runnable on JAX.
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
        del image_example
        self.hf_model_id = hf_model_id
        if any(use_adarms):
            raise NotImplementedError("JAX Qwen adapter does not support pi05/AdaRMS conditioning yet.")
        if vlm_config.depth != action_expert_config.depth:
            raise ValueError(
                "JAX Qwen adapter currently requires matching prefix/expert depth: "
                f"{vlm_config.depth} != {action_expert_config.depth}"
            )
        if (
            vlm_config.num_heads != action_expert_config.num_heads
            or vlm_config.num_kv_heads != action_expert_config.num_kv_heads
            or vlm_config.head_dim != action_expert_config.head_dim
        ):
            raise ValueError(
                "JAX Qwen adapter currently requires matching attention geometry for prefix and expert."
            )

        llm = nnx_bridge.ToNNX(
            _qwen_text.Module(
                configs=[vlm_config, action_expert_config],
                embed_dtype=precision,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
        self.llm = llm

    def embed_image(self, image):
        del image
        _qwen_vision.raise_qwen_vision_not_implemented()

    def embed_language_tokens(self, tokens):
        return self.llm(tokens, method="embed")

    def forward(self, inputs_embeds, mask, positions, kv_cache=None, adarms_cond=None):
        if adarms_cond is None:
            adarms_cond = [None, None]
        return self.llm(inputs_embeds, mask=mask, positions=positions, kv_cache=kv_cache, adarms_cond=adarms_cond)
