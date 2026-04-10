from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import Literal
from typing import Protocol

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax.numpy as jnp
from typing_extensions import runtime_checkable

import openpi.models.gemma as _gemma
from openpi.models.qwen2_5.adapter import Qwen2_5_VLWithExpertModel
from openpi.models.qwen3_5.adapter import Qwen3_5_VLWithExpertModel
import openpi.models.siglip as _siglip

VLMBackend = Literal["paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_5_vl", "internvl3"]
LEGACY_VLM_CHECKPOINT_ROOT = "PaliGemma"
RUNTIME_VLM_ROOT = "vlm_with_expert"


@dataclasses.dataclass(frozen=True)
class PrefixPositionLayout:
    image_token_lengths: tuple[int, ...] = ()
    image_grid_thw: tuple[tuple[int, int, int] | None, ...] = ()
    language_token_length: int = 0


@runtime_checkable
class VLMWithExpertModel(Protocol):
    def embed_image(self, image): ...

    def embed_image_with_metadata(self, image): ...

    def embed_language_tokens(self, tokens): ...

    def forward(self, inputs_embeds, mask, positions, kv_cache=None, adarms_cond=None): ...

    def build_prefix_positions(self, prefix_pad_mask, *, prefix_layout: PrefixPositionLayout): ...

    def build_joint_positions(self, prefix_pad_mask, suffix_pad_mask, *, prefix_layout: PrefixPositionLayout): ...

    def build_decode_positions(self, prefix_pad_mask, suffix_pad_mask, *, prefix_layout: PrefixPositionLayout): ...


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

    def embed_image_with_metadata(self, image):
        return self.embed_image(image), None

    def embed_language_tokens(self, tokens):
        return self.llm(tokens, method="embed")

    def forward(self, inputs_embeds, mask, positions, kv_cache=None, adarms_cond=None):
        if adarms_cond is None:
            adarms_cond = [None, None]
        return self.llm(inputs_embeds, mask=mask, positions=positions, kv_cache=kv_cache, adarms_cond=adarms_cond)

    def build_prefix_positions(self, prefix_pad_mask, *, prefix_layout: PrefixPositionLayout):
        del prefix_layout
        return jnp.cumsum(prefix_pad_mask, axis=1) - 1

    def build_joint_positions(self, prefix_pad_mask, suffix_pad_mask, *, prefix_layout: PrefixPositionLayout):
        del prefix_layout
        input_mask = jnp.concatenate([prefix_pad_mask, suffix_pad_mask], axis=1)
        return jnp.cumsum(input_mask, axis=1) - 1

    def build_decode_positions(self, prefix_pad_mask, suffix_pad_mask, *, prefix_layout: PrefixPositionLayout):
        del prefix_layout
        return jnp.sum(prefix_pad_mask, axis=-1)[:, None] + jnp.cumsum(suffix_pad_mask, axis=-1) - 1


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
    use_remat: bool = True,
) -> VLMWithExpertModel:
    if vlm_backend == "paligemma":
        return PaliGemmaWithExpertModel(
            vlm_backbone_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=precision,
            image_example=image_example,
            rngs=rngs,
        )

    if vlm_backend in ("qwen2_vl", "qwen2_5_vl"):
        return Qwen2_5_VLWithExpertModel(
            vlm_backbone_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=precision,
            image_example=image_example,
            rngs=rngs,
            hf_model_id=hf_model_id,
        )
    if vlm_backend == "qwen3_5_vl":
        return Qwen3_5_VLWithExpertModel(
            vlm_backbone_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=precision,
            image_example=image_example,
            rngs=rngs,
            hf_model_id=hf_model_id,
            use_remat=use_remat,
        )

    if vlm_backend == "internvl3":
        raise NotImplementedError(
            "JAX Pi0 recognizes `vlm_backend=internvl3` but does not implement it yet. "
            "Use `vlm_backend='paligemma'` for JAX today, or use the PyTorch runtime for non-PaliGemma backends."
        )

    raise ValueError(f"Unsupported vlm_backend: {vlm_backend}")
