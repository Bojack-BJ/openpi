from typing import Literal

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.qwen2_vl_pytorch import Qwen2_5_VLWithExpertModel

VLMBackend = Literal["paligemma", "qwen2_vl", "qwen2_5_vl", "internvl3"]


def create_vlm_with_expert_model(
    vlm_backend: VLMBackend,
    vlm_config,
    action_expert_config,
    *,
    use_adarms,
    precision: str,
    hf_model_id: str | None = None,
):
    if vlm_backend == "paligemma":
        return PaliGemmaWithExpertModel(
            vlm_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=precision,
            hf_model_id=hf_model_id,
        )
    if vlm_backend in ("qwen2_vl", "qwen2_5_vl"):
        return Qwen2_5_VLWithExpertModel(
            vlm_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=precision,
            hf_model_id=hf_model_id,
        )

    # TODO: add an InternVL3 adapter that implements the same model-facing contract as the
    # PaliGemma/Qwen backends: prompt tokenization, image embedding, joint prefix/suffix attention,
    # and checkpoint loading.
    raise NotImplementedError(
        f"`vlm_backend={vlm_backend}` is not implemented yet. "
        "The current codebase can now route through a generic VLM backend factory, "
        "but this backend still needs a dedicated adapter for processor/tokenizer, "
        "vision-language embedding, shared prefix/suffix attention, and checkpoint loading."
    )
