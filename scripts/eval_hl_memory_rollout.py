from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import torch
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.config_io import resolve_cli_args_with_yaml
from openpi.hl_memory.data import load_video_clips_for_sample
from openpi.hl_memory.data import load_exported_samples
from openpi.hl_memory.eval import evaluate_ablation_modes
from openpi.hl_memory.hf_adapter import create_hf_adapter


@dataclasses.dataclass
class EvalArgs:
    dataset_dir: pathlib.Path
    model_path: pathlib.Path | None = None
    config_yaml: pathlib.Path | None = None
    vlm_backend: str = "qwen2_5_vl"
    vlm_hf_model_id: str | None = None
    local_vlm_ckpt_path: pathlib.Path | None = None
    precision: str = "bfloat16"
    output_json: pathlib.Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: EvalArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = load_exported_samples(args.dataset_dir)
    resolved_model_path = args.model_path or args.local_vlm_ckpt_path
    if resolved_model_path is None:
        raise ValueError("Set either `model_path` or `local_vlm_ckpt_path`.")
    hl_config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_hf_model_id=args.vlm_hf_model_id or str(resolved_model_path),
        precision=args.precision,
    )
    adapter = create_hf_adapter(hl_config)
    loaded = adapter.load(model_path=str(resolved_model_path), device=args.device)

    def predict(sample):
        clips = load_video_clips_for_sample(sample, args.dataset_dir, hl_config)
        return adapter.predict(loaded, sample, clips, device=args.device)

    metrics = evaluate_ablation_modes(samples, hl_config, predict)
    rendered = json.dumps(metrics, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main(resolve_cli_args_with_yaml(EvalArgs, tyro))
