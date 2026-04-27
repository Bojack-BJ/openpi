from __future__ import annotations

import dataclasses
import json
import pathlib

import torch
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.hf_adapter import create_hf_adapter
from openpi.hl_memory.zero_shot import build_zero_shot_clips_from_video
from openpi.hl_memory.zero_shot import build_zero_shot_sample
from openpi.hl_memory.zero_shot import parse_seconds_argument
from openpi.hl_memory.zero_shot import save_zero_shot_debug_frames


@dataclasses.dataclass
class ZeroShotArgs:
    video_path: pathlib.Path
    instruction: str
    model_path: str | None = None
    output_json: pathlib.Path | None = None
    debug_dir: pathlib.Path | None = None
    language_memory: str = ""
    memory_seconds: str | None = None
    recent_seconds: str | None = None
    recent_end_sec: float | None = None
    recent_step_sec: float = 1.0
    auto_memory: bool = True
    vlm_backend: str = "qwen2_5_vl"
    vlm_hf_model_id: str | None = None
    precision: str = "bfloat16"
    recent_frames_length: int = 8
    memory_length: int = 8
    frame_height: int = 224
    frame_width: int = 224
    allow_single_frame_fallback: bool = True
    max_new_tokens: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: ZeroShotArgs) -> None:
    config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_hf_model_id=args.vlm_hf_model_id,
        precision=args.precision,
        recent_frames_length=args.recent_frames_length,
        memory_length=args.memory_length,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        allow_single_frame_fallback=args.allow_single_frame_fallback,
        max_new_tokens=args.max_new_tokens,
    )
    adapter = create_hf_adapter(config)
    loaded = adapter.load(model_path=args.model_path, device=args.device)

    clips, selection = build_zero_shot_clips_from_video(
        args.video_path,
        config=config,
        recent_end_sec=args.recent_end_sec,
        recent_step_sec=args.recent_step_sec,
        recent_seconds=parse_seconds_argument(args.recent_seconds) or None,
        memory_seconds=parse_seconds_argument(args.memory_seconds) or None,
        auto_memory=args.auto_memory,
    )
    sample = build_zero_shot_sample(
        video_path=args.video_path,
        instruction=args.instruction,
        language_memory=args.language_memory,
        memory_seconds=selection.memory_seconds,
        recent_seconds=selection.recent_seconds,
    )
    prediction = adapter.predict(loaded, sample, clips, device=args.device)

    payload = {
        "video_path": str(args.video_path),
        "model_path": args.model_path,
        "resolved_model_id": config.resolved_model_id if args.model_path is None else args.model_path,
        "instruction": args.instruction,
        "language_memory": args.language_memory,
        "duration_sec": selection.duration_sec,
        "memory_seconds": list(selection.memory_seconds),
        "recent_seconds": list(selection.recent_seconds),
        "memory_valid_length": clips.memory_valid_length,
        "recent_valid_length": clips.recent_valid_length,
        "prediction": prediction.to_dict(),
    }

    if args.debug_dir is not None:
        save_zero_shot_debug_frames(args.debug_dir, clips=clips, selection=selection)
        payload["debug_dir"] = str(args.debug_dir)

    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main(tyro.cli(ZeroShotArgs))
