from __future__ import annotations

import json
import logging
import pathlib

import torch
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import build_context_panel_for_sample
from openpi.hl_memory.data import load_exported_samples
from openpi.hl_memory.eval import evaluate_ablation_modes
from openpi.hl_memory.hf_adapter import create_hf_adapter


@dataclasses.dataclass
class EvalArgs:
    dataset_dir: pathlib.Path
    model_path: pathlib.Path
    vlm_backend: str = "qwen2_5_vl"
    vlm_hf_model_id: str | None = None
    precision: str = "bfloat16"
    output_json: pathlib.Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: EvalArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = load_exported_samples(args.dataset_dir)
    hl_config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_hf_model_id=args.vlm_hf_model_id or str(args.model_path),
        precision=args.precision,
    )
    adapter = create_hf_adapter(hl_config)
    loaded = adapter.load(model_path=str(args.model_path), device=args.device)

    def predict(sample):
        panel = build_context_panel_for_sample(sample, args.dataset_dir, hl_config)
        return adapter.predict(loaded, sample, panel, device=args.device)

    metrics = evaluate_ablation_modes(samples, hl_config, predict)
    rendered = json.dumps(metrics, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
