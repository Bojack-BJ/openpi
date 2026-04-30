from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Literal

from PIL import Image
import torch
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.config_io import resolve_cli_args_with_yaml
from openpi.hl_memory.hf_adapter import create_hf_adapter
from openpi.hl_memory.target_grounding import CandidateMask
from openpi.hl_memory.target_grounding import MaskSelectionPrediction
from openpi.hl_memory.target_grounding import TargetPointPrediction
from openpi.hl_memory.target_grounding import build_mask_selection_prompt
from openpi.hl_memory.target_grounding import build_target_point_prompt
from openpi.hl_memory.target_grounding import generate_sam_candidate_masks
from openpi.hl_memory.target_grounding import load_candidate_masks
from openpi.hl_memory.target_grounding import overlay_mask_candidates
from openpi.hl_memory.target_grounding import overlay_point
from openpi.hl_memory.target_grounding import save_mask
from openpi.hl_memory.target_grounding import segment_from_point_with_sam


TargetGroundingMode = Literal["point", "mask_select", "both"]


@dataclasses.dataclass
class TargetGroundingArgs:
    image_path: pathlib.Path
    instruction: str
    output_dir: pathlib.Path = pathlib.Path("hl_target_grounding_debug")
    config_yaml: pathlib.Path | None = None
    mode: TargetGroundingMode = "point"
    current_subtask: str = ""
    language_memory: str = ""
    target_query: str = ""
    goal_query: str = ""
    model_path: str | None = None
    local_vlm_ckpt_path: pathlib.Path | None = None
    vlm_backend: str = "qwen2_5_vl"
    vlm_variant: str | None = None
    vlm_hf_model_id: str | None = None
    precision: str = "bfloat16"
    max_new_tokens: int = 192
    enable_thinking: bool = False
    thinking_budget_tokens: int = 128
    thinking_max_new_tokens: int = 1024
    candidate_mask_dir: pathlib.Path | None = None
    sam_checkpoint: pathlib.Path | None = None
    sam_model_type: str = "vit_h"
    max_candidates: int = 16
    min_mask_area: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: TargetGroundingArgs) -> None:
    image = Image.open(args.image_path).convert("RGB")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_variant=args.vlm_variant,
        vlm_hf_model_id=args.vlm_hf_model_id,
        precision=args.precision,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
        thinking_max_new_tokens=args.thinking_max_new_tokens,
    )
    adapter = create_hf_adapter(config)
    resolved_model_path = args.model_path
    if args.local_vlm_ckpt_path is not None:
        resolved_model_path = str(args.local_vlm_ckpt_path)
    loaded = adapter.load(model_path=resolved_model_path, device=args.device)

    payload: dict[str, object] = {
        "image_path": str(args.image_path),
        "instruction": args.instruction,
        "current_subtask": args.current_subtask,
        "language_memory": args.language_memory,
        "target_query": args.target_query,
        "goal_query": args.goal_query,
        "resolved_model_id": config.resolved_model_id if resolved_model_path is None else resolved_model_path,
        "mode": args.mode,
    }

    if args.mode in ("point", "both"):
        payload["point_first"] = _run_point_first(args, adapter=adapter, loaded=loaded, image=image)

    if args.mode in ("mask_select", "both"):
        payload["mask_select"] = _run_mask_select(args, adapter=adapter, loaded=loaded, image=image)

    output_json = args.output_dir / "target_grounding_result.json"
    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    print(rendered)
    output_json.write_text(rendered + "\n")


def _run_point_first(args: TargetGroundingArgs, *, adapter, loaded, image: Image.Image) -> dict[str, object]:
    prompt = build_target_point_prompt(
        instruction=args.instruction,
        current_subtask=args.current_subtask,
        language_memory=args.language_memory,
        target_query=args.target_query,
        goal_query=args.goal_query,
        image_size=image.size,
    )
    raw_output = adapter.generate_image_text_response(
        loaded,
        system_prompt=_system_prompt(),
        user_text=prompt,
        images=[image],
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    prediction = TargetPointPrediction.from_json(raw_output, image_size=image.size)
    point_overlay_path = args.output_dir / "point_first_overlay.png"
    overlay_point(image, prediction.point_xy, label=prediction.target_query or "target").save(point_overlay_path)

    result: dict[str, object] = {
        "raw_model_output": raw_output,
        "prediction": prediction.to_dict(),
        "point_overlay_path": str(point_overlay_path),
    }

    if args.sam_checkpoint is not None:
        mask = segment_from_point_with_sam(
            image,
            prediction.point_xy,
            checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.device,
        )
        mask_path = save_mask(mask, args.output_dir / "point_first_sam_mask.png")
        selected_overlay_path = args.output_dir / "point_first_sam_overlay.png"
        overlay_mask_candidates(
            image,
            [CandidateMask(mask_id=1, mask=mask)],
            selected_mask_id=1,
        ).save(selected_overlay_path)
        result["sam_mask_path"] = str(mask_path)
        result["sam_overlay_path"] = str(selected_overlay_path)

    return result


def _run_mask_select(args: TargetGroundingArgs, *, adapter, loaded, image: Image.Image) -> dict[str, object]:
    candidates = _load_or_generate_candidates(args, image=image)
    if not candidates:
        raise ValueError("No candidate masks were available for mask selection.")

    candidate_overlay = overlay_mask_candidates(image, candidates)
    candidate_overlay_path = args.output_dir / "mask_select_candidates.png"
    candidate_overlay.save(candidate_overlay_path)

    prompt = build_mask_selection_prompt(
        instruction=args.instruction,
        current_subtask=args.current_subtask,
        language_memory=args.language_memory,
        target_query=args.target_query,
        goal_query=args.goal_query,
        candidate_count=len(candidates),
    )
    raw_output = adapter.generate_image_text_response(
        loaded,
        system_prompt=_system_prompt(),
        user_text=prompt,
        images=[image, candidate_overlay],
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    prediction = MaskSelectionPrediction.from_json(raw_output, max_mask_id=len(candidates))
    selected = candidates[prediction.selected_mask_id - 1]
    selected_mask_path = save_mask(selected.mask, args.output_dir / "mask_select_selected_mask.png")
    selected_overlay_path = args.output_dir / "mask_select_selected_overlay.png"
    overlay_mask_candidates(
        image,
        candidates,
        selected_mask_id=prediction.selected_mask_id,
    ).save(selected_overlay_path)

    return {
        "raw_model_output": raw_output,
        "prediction": prediction.to_dict(),
        "candidate_count": len(candidates),
        "candidate_overlay_path": str(candidate_overlay_path),
        "selected_mask_path": str(selected_mask_path),
        "selected_overlay_path": str(selected_overlay_path),
        "selected_candidate": _candidate_payload(selected),
    }


def _load_or_generate_candidates(args: TargetGroundingArgs, *, image: Image.Image) -> list[CandidateMask]:
    if args.candidate_mask_dir is not None:
        return load_candidate_masks(args.candidate_mask_dir, min_area=args.min_mask_area)
    if args.sam_checkpoint is None:
        raise ValueError("Mask selection requires either `--candidate-mask-dir` or `--sam-checkpoint`.")
    return generate_sam_candidate_masks(
        image,
        checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.device,
        max_candidates=args.max_candidates,
        min_area=args.min_mask_area,
    )


def _candidate_payload(candidate: CandidateMask) -> dict[str, object]:
    payload: dict[str, object] = {
        "mask_id": candidate.mask_id,
        "area": candidate.area,
        "bbox_xyxy": list(candidate.bbox_xyxy),
    }
    if candidate.source is not None:
        payload["source"] = str(candidate.source)
    if candidate.score is not None:
        payload["score"] = candidate.score
    return payload


def _system_prompt() -> str:
    return (
        "You are a robot visual grounding policy. Ground the target object for the current manipulation step. "
        "Use only the provided visual evidence and task context. Output only valid JSON."
    )


if __name__ == "__main__":
    main(resolve_cli_args_with_yaml(TargetGroundingArgs, tyro))
