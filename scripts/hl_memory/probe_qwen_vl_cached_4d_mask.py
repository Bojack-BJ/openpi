from __future__ import annotations

import argparse
import copy
import json
from typing import Any

from PIL import Image
import torch

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import LoadedVideoClips
from openpi.hl_memory.hf_adapter import _compute_qwen25_position_ids
from openpi.hl_memory.hf_adapter import _extend_qwen25_sequence_inputs
from openpi.hl_memory.hf_adapter import _model_compute_dtype
from openpi.hl_memory.hf_adapter import create_hf_adapter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe cached Qwen-VL decoding with a 4D prefill mask and per-token 4D row mask."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vlm-backend", choices=["qwen2_5_vl", "qwen3_vl"], required=True)
    parser.add_argument("--vlm-variant", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    return parser.parse_args()


def _sample() -> ExportedHLMemorySample:
    return ExportedHLMemorySample(
        sample_id="cached-mask-probe",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="Move the object to the target.",
        language_memory="No accepted completed event yet.",
        updated_language_memory="",
        current_subtask="Move to the object",
        phase="Move to the object",
        target_query="object",
        goal_query="target",
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=(),
        recent_frame_indices=(),
        recent_valid_length=2,
    )


def _clips() -> LoadedVideoClips:
    frames = tuple(Image.new("RGB", (456, 224), color=(0, 0, 0)) for _ in range(2))
    return LoadedVideoClips(
        memory_frames=frames,
        recent_frames=frames,
        memory_valid_length=2,
        recent_valid_length=2,
    )


def _causal_additive_mask(attention_mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    length = attention_mask.shape[-1]
    positions = torch.arange(length, device=attention_mask.device)
    causal = positions[:, None] >= positions[None, :]
    valid = attention_mask.to(torch.bool)
    allow = causal[None, :, :] & valid[:, :, None] & valid[:, None, :]
    additive = torch.zeros(
        (attention_mask.shape[0], 1, length, length),
        dtype=dtype,
        device=attention_mask.device,
    )
    additive.masked_fill_(~allow[:, None, :, :], torch.finfo(dtype).min)
    return additive


def _decode_row_mask(*, batch: int, key_length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.zeros((batch, 1, 1, key_length), dtype=dtype, device=device)


def _tensor_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor) and key not in {"position_ids", "rope_deltas"}
    }


def main() -> None:
    args = _parse_args()
    config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_variant=args.vlm_variant,
        precision=args.precision,
        target_protocol="keyframe_gated_memory",
        recent_frames_length=2,
        memory_length=2,
    )
    adapter = create_hf_adapter(config)
    loaded = adapter.load(model_path=args.model_path, device=args.device)
    device = adapter._resolve_input_device(loaded.model, args.device)  # noqa: SLF001
    encoded = _tensor_inputs(adapter._encode_prompt_only(loaded.processor, _sample(), _clips()), device)  # noqa: SLF001
    input_ids = encoded.pop("input_ids")
    attention_mask_2d = encoded.pop("attention_mask")
    dtype = _model_compute_dtype(loaded.model)
    position_ids = _compute_qwen25_position_ids(
        loaded.model,
        input_ids=input_ids,
        attention_mask=attention_mask_2d,
        model_inputs=encoded,
    )
    prefill_mask = _causal_additive_mask(attention_mask_2d, dtype=dtype)

    result: dict[str, Any] = {
        "backend": args.vlm_backend,
        "model": args.model_path,
        "prompt_length": int(input_ids.shape[-1]),
        "prefill_4d_cache": False,
        "decode_row_4d_cache": False,
        "decode_row_mask_effect_max_abs": None,
        "error": None,
    }
    try:
        with torch.inference_mode():
            prefill = loaded.model(
                input_ids=input_ids,
                attention_mask=prefill_mask,
                position_ids=position_ids,
                use_cache=True,
                **encoded,
            )
            result["prefill_4d_cache"] = prefill.past_key_values is not None
            next_token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
            full_ids = torch.cat([input_ids, next_token], dim=-1)
            full_attention_mask = torch.cat([attention_mask_2d, torch.ones_like(next_token)], dim=-1)
            extended_inputs = _extend_qwen25_sequence_inputs(
                encoded,
                prompt_length=input_ids.shape[-1],
                sequence_length=full_ids.shape[-1],
            )
            full_position_ids = _compute_qwen25_position_ids(
                loaded.model,
                input_ids=full_ids,
                attention_mask=full_attention_mask,
                model_inputs=extended_inputs,
            )
            allow_all_row = _decode_row_mask(
                batch=input_ids.shape[0],
                key_length=full_ids.shape[-1],
                dtype=dtype,
                device=device,
            )
            self_only_row = torch.full_like(allow_all_row, torch.finfo(dtype).min)
            self_only_row[..., -1] = 0
            allow_all = loaded.model(
                input_ids=next_token,
                attention_mask=allow_all_row,
                position_ids=full_position_ids[..., -1:],
                past_key_values=copy.deepcopy(prefill.past_key_values),
                cache_position=torch.tensor([input_ids.shape[-1]], dtype=torch.long, device=device),
                use_cache=False,
            )
            self_only = loaded.model(
                input_ids=next_token,
                attention_mask=self_only_row,
                position_ids=full_position_ids[..., -1:],
                past_key_values=copy.deepcopy(prefill.past_key_values),
                cache_position=torch.tensor([input_ids.shape[-1]], dtype=torch.long, device=device),
                use_cache=False,
            )
            result["decode_row_mask_effect_max_abs"] = float(
                (allow_all.logits - self_only.logits).abs().max().item()
            )
            decode = loaded.model(
                input_ids=next_token,
                attention_mask=allow_all_row,
                position_ids=full_position_ids[..., -1:],
                past_key_values=copy.deepcopy(prefill.past_key_values),
                cache_position=torch.tensor([input_ids.shape[-1]], dtype=torch.long, device=device),
                use_cache=True,
            )
            result["decode_row_4d_cache"] = decode.logits.shape[-2] == 1
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["decode_row_4d_cache"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
