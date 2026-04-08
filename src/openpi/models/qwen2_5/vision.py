def raise_qwen_vision_not_implemented() -> None:
    raise NotImplementedError(
        "JAX Qwen image embedding is not implemented yet. "
        "The text/expert transformer scaffold now lives under `openpi.models.qwen2_5`, "
        "but the Qwen2.5-VL vision tower, projector, and multimodal THW position path are still TODO."
    )
