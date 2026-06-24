from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from PIL import Image
import torch

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import LoadedVideoClips
from openpi.hl_memory.hf_adapter import _compute_qwen25_position_ids
from openpi.hl_memory.hf_adapter import _model_compute_dtype
from openpi.hl_memory.hf_adapter import create_hf_adapter
from openpi.hl_memory.typed_attention import build_qwen25_typed_attention_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether an HF Qwen-VL backend accepts a 4D additive causal attention mask."
    )
    parser.add_argument("--model-path", default=None, help="Local checkpoint path or HF model id.")
    parser.add_argument("--vlm-backend", default="qwen3_vl", choices=["qwen2_5_vl", "qwen3_5_vl", "qwen3_vl"])
    parser.add_argument("--vlm-variant", default="qwen3_vl_4b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--frame-width", type=int, default=456)
    parser.add_argument("--frame-height", type=int, default=224)
    parser.add_argument("--recent-frames-length", type=int, default=8)
    parser.add_argument("--memory-length", type=int, default=8)
    parser.add_argument("--recent-sample-hz", type=float, default=2.0)
    parser.add_argument("--trust-2d-only", action="store_true", help="Exit 0 when 2D succeeds but 4D fails.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _black_frames(count: int, *, width: int, height: int) -> tuple[Image.Image, ...]:
    return tuple(Image.new("RGB", (width, height), color=(0, 0, 0)) for _ in range(count))


def _sample() -> ExportedHLMemorySample:
    return ExportedHLMemorySample(
        sample_id="probe",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="Move the object to the target location.",
        language_memory="No accepted completed event yet.",
        updated_language_memory="",
        current_subtask="Move to the object",
        phase="Move to the object",
        target_query="object",
        goal_query="target",
        keyframe_candidate_positions=(1,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=(),
        recent_frame_indices=(),
        recent_valid_length=8,
        current_objective="Move to the object",
        horizon_current_objective="Grasp the object",
    )


def _clips(*, memory_length: int, recent_length: int, width: int, height: int) -> LoadedVideoClips:
    return LoadedVideoClips(
        memory_frames=_black_frames(memory_length, width=width, height=height),
        recent_frames=_black_frames(recent_length, width=width, height=height),
        memory_valid_length=memory_length,
        recent_valid_length=recent_length,
    )


def _run_forward(model: Any, inputs: dict[str, Any]) -> None:
    with torch.inference_mode():
        try:
            _ = model(**inputs, use_cache=False)
        except TypeError as exc:
            if "use_cache" not in str(exc):
                raise
            _ = model(**inputs)


def _forward_inputs(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in tensors.items()
        if isinstance(value, torch.Tensor) and key != "labels" and not key.startswith("_hl_")
    }


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_variant=args.vlm_variant,
        precision=args.precision,
        target_protocol="keyframe_gated_memory",
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        recent_frames_length=args.recent_frames_length,
        memory_length=args.memory_length,
        recent_sample_hz=args.recent_sample_hz,
    )
    adapter = create_hf_adapter(config)
    loaded = adapter.load(model_path=args.model_path, device=args.device)
    device = adapter._resolve_input_device(loaded.model, args.device)  # noqa: SLF001
    clips = _clips(
        memory_length=args.memory_length,
        recent_length=args.recent_frames_length,
        width=args.frame_width,
        height=args.frame_height,
    )
    sample = _sample()
    training_inputs = dict(adapter.prepare_training_inputs(loaded, sample, clips, device=device))
    inputs_2d = _forward_inputs(training_inputs)
    attention_mask_2d = inputs_2d.get("attention_mask")
    if not isinstance(attention_mask_2d, torch.Tensor):
        raise ValueError("Processor did not return a tensor `attention_mask`; cannot probe 4D mask support.")
    labels = training_inputs.get("labels")
    if not isinstance(labels, torch.Tensor):
        raise ValueError("Training input preparation did not return tensor `labels`; cannot locate typed spans.")

    result: dict[str, Any] = {
        "backend": args.vlm_backend,
        "variant": config.vlm_variant,
        "model": args.model_path or config.resolved_model_id,
        "sequence_length": int(attention_mask_2d.shape[-1]),
        "supervised_tokens": int((labels != -100).sum().item()),
        "two_d_forward": False,
        "four_d_forward": False,
        "four_d_error": None,
        "typed_spans_found": False,
    }

    _run_forward(loaded.model, inputs_2d)
    result["two_d_forward"] = True

    inputs_4d = dict(inputs_2d)
    tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
    position_ids = _compute_qwen25_position_ids(
        loaded.model,
        input_ids=inputs_2d["input_ids"],
        attention_mask=attention_mask_2d,
        model_inputs=inputs_2d,
    )
    additive_mask, spans = build_qwen25_typed_attention_mask(
        input_ids=inputs_2d["input_ids"],
        attention_mask=attention_mask_2d,
        labels=labels,
        tokenizer=tokenizer,
        dtype=_model_compute_dtype(loaded.model),
    )
    result["typed_spans_found"] = any(span is not None for span in spans)
    inputs_4d["position_ids"] = position_ids
    inputs_4d["attention_mask"] = additive_mask
    try:
        _run_forward(loaded.model, inputs_4d)
    except Exception as exc:  # noqa: BLE001
        result["four_d_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.trust_2d_only:
            return
        raise SystemExit(1) from exc

    result["four_d_forward"] = True
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
