import dataclasses
from typing import TYPE_CHECKING
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.vlm_backbone as _vlm_backbone
import openpi.models.vlm_backbone_config as _vlm_backbone_config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


VLMBackend = Literal["paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_5_vl", "internvl3"]


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    # Backend selection now routes prompt-tokenizer selection everywhere and dispatches the JAX/PyTorch
    # Pi0 runtime through backend factories. JAX fully supports `paligemma`; the JAX `qwen2_5_vl`
    # path includes a native vision tower, projector, and backend-owned multimodal positions; and
    # `qwen3_5_vl` now uses an official-style hybrid Qwen3.5 text backbone, a dedicated JAX Qwen3.5
    # vision tower, and a backend-specific Hugging Face weight loader. Legacy JAX checkpoints saved
    # under the historical top-level `PaliGemma` subtree are remapped automatically at load time.
    vlm_backend: VLMBackend = "paligemma"
    vlm_hf_model_id: str | None = None
    vlm_backbone_variant: _vlm_backbone_config.Variant = "gemma_2b"
    action_expert_variant: _vlm_backbone_config.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.vlm_backend not in ("paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_5_vl", "internvl3"):
            raise ValueError(f"Unsupported vlm_backend: {self.vlm_backend}")
        if self.vlm_backend in ("qwen2_vl", "qwen2_5_vl") and self.vlm_hf_model_id is None:
            default_model_id = (
                "Qwen/Qwen2.5-VL-3B-Instruct"
                if self.vlm_backbone_variant == "qwen2_5_3b"
                else "Qwen/Qwen2.5-VL-7B-Instruct"
            )
            object.__setattr__(self, "vlm_hf_model_id", default_model_id)
        if self.vlm_backend == "qwen3_5_vl" and self.vlm_hf_model_id is None:
            default_model_id = "Qwen/Qwen3.5-2B" if self.vlm_backbone_variant == "qwen3_5_2b" else "Qwen/Qwen3.5-4B"
            object.__setattr__(self, "vlm_hf_model_id", default_model_id)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True):
        params = _vlm_backbone.remap_legacy_vlm_checkpoint_root(params)
        return super().load(params, remove_extra_params=remove_extra_params)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        vlm_backbone_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.vlm_backbone_variant:
            filters.append(
                vlm_backbone_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
