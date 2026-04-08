from collections.abc import Mapping
from typing import Any
from typing import Literal
from typing import Protocol

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from typing_extensions import runtime_checkable

import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip

VLMBackend = Literal["paligemma", "qwen2_vl", "qwen2_5_vl", "internvl3"]
LEGACY_VLM_CHECKPOINT_ROOT = "PaliGemma"
RUNTIME_VLM_ROOT = "vlm_with_expert"


@runtime_checkable
class VLMWithExpertModel(Protocol):
    def embed_image(self, image): ...

    def embed_language_tokens(self, tokens): ...

    def forward(self, inputs_embeds, mask, positions, kv_cache=None, adarms_cond=None): ...


class PaliGemmaWithExpertModel(nnx.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        *,
        use_adarms,
        precision: str,
        image_example,
        rngs: nnx.Rngs,
    ):
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[vlm_config, action_expert_config],
                embed_dtype=precision,
                adarms=use_adarms[1],
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=vlm_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=precision,
            )
        )
        img.lazy_init(image_example, train=False, rngs=rngs)

        self.llm = llm
        self.img = img

    def embed_image(self, image):
        image_tokens, _ = self.img(image, train=False)
        return image_tokens

    def embed_language_tokens(self, tokens):
        return self.llm(tokens, method="embed")

    def forward(self, inputs_embeds, mask, positions, kv_cache=None, adarms_cond=None):
        if adarms_cond is None:
            adarms_cond = [None, None]
        return self.llm(inputs_embeds, mask=mask, positions=positions, kv_cache=kv_cache, adarms_cond=adarms_cond)


def remap_legacy_vlm_checkpoint_root(
    params: Mapping[str, Any], *, target_root: str = RUNTIME_VLM_ROOT
) -> dict[str, Any] | Mapping[str, Any]:
    """Maps legacy JAX checkpoint roots onto the runtime VLM handle when needed."""
    if target_root in params or LEGACY_VLM_CHECKPOINT_ROOT not in params:
        return params
    remapped = dict(params)
    remapped[target_root] = remapped.pop(LEGACY_VLM_CHECKPOINT_ROOT)
    return remapped


def remap_legacy_vlm_checkpoint_root_for_reference(
    params: Mapping[str, Any], reference_params: Mapping[str, Any]
) -> dict[str, Any] | Mapping[str, Any]:
    if RUNTIME_VLM_ROOT in reference_params:
        return remap_legacy_vlm_checkpoint_root(params, target_root=RUNTIME_VLM_ROOT)
    return params


def create_vlm_with_expert_model(
    vlm_backend: VLMBackend,
    vlm_backbone_config,
    action_expert_config,
    *,
    use_adarms,
    precision: str,
    image_example,
    rngs: nnx.Rngs,
    hf_model_id: str | None = None,
) -> VLMWithExpertModel:
    del hf_model_id

    if vlm_backend == "paligemma":
        return PaliGemmaWithExpertModel(
            vlm_backbone_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=precision,
            image_example=image_example,
            rngs=rngs,
        )

    if vlm_backend in ("qwen2_vl", "qwen2_5_vl", "internvl3"):
        raise NotImplementedError(
            f"JAX Pi0 recognizes `vlm_backend={vlm_backend}` but does not implement it yet. "
            "Use `vlm_backend='paligemma'` for JAX, or use the PyTorch runtime for Qwen-based training."
        )

    raise ValueError(f"Unsupported vlm_backend: {vlm_backend}")
