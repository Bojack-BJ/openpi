from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import random

import torch
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.config_io import resolve_cli_args_with_yaml
from openpi.hl_memory.data import load_video_clips_for_sample
from openpi.hl_memory.data import load_exported_samples
from openpi.hl_memory.hf_adapter import create_hf_adapter


@dataclasses.dataclass
class TrainArgs:
    dataset_dir: pathlib.Path
    output_dir: pathlib.Path
    config_yaml: pathlib.Path | None = None
    vlm_backend: str = "qwen2_5_vl"
    vlm_hf_model_id: str | None = None
    local_vlm_ckpt_path: pathlib.Path | None = None
    precision: str = "bfloat16"
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 1
    grad_accum_steps: int = 1
    num_train_steps: int = 100
    save_interval: int = 50
    log_interval: int = 10
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: TrainArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = load_exported_samples(args.dataset_dir)
    if not samples:
        raise ValueError(f"No exported HL memory samples found in {args.dataset_dir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hl_config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_hf_model_id=args.vlm_hf_model_id,
        precision=args.precision,
    )
    adapter = create_hf_adapter(hl_config)
    loaded = adapter.load(
        model_path=None if args.local_vlm_ckpt_path is None else str(args.local_vlm_ckpt_path),
        device=args.device,
    )
    loaded.model.train()
    optimizer = torch.optim.AdamW(
        loaded.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)
    running_loss = 0.0

    for step in range(1, args.num_train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum_steps):
            batch = _sample_batch(samples, args.batch_size, rng)
            micro_loss = 0.0
            for sample in batch:
                clips = load_video_clips_for_sample(sample, args.dataset_dir, hl_config)
                inputs = adapter.prepare_training_inputs(loaded, sample, clips, device=args.device)
                outputs = loaded.model(**inputs)
                loss = outputs.loss / max(len(batch), 1)
                micro_loss += float(loss.detach().cpu())
                (loss / args.grad_accum_steps).backward()
            step_loss += micro_loss

        torch.nn.utils.clip_grad_norm_(loaded.model.parameters(), args.max_grad_norm)
        optimizer.step()
        running_loss += step_loss

        if step % args.log_interval == 0:
            logging.info("step=%d loss=%.6f", step, running_loss / args.log_interval)
            running_loss = 0.0
        if step % args.save_interval == 0 or step == args.num_train_steps:
            _save_checkpoint(
                output_dir=args.output_dir / f"checkpoint-step-{step:06d}",
                loaded=loaded,
                hl_config=hl_config,
                train_args=args,
            )


def _sample_batch(samples, batch_size: int, rng: random.Random):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(samples) >= batch_size:
        return rng.sample(samples, batch_size)
    return [rng.choice(samples) for _ in range(batch_size)]


def _save_checkpoint(*, output_dir: pathlib.Path, loaded, hl_config: HLMemoryConfig, train_args: TrainArgs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded.model.save_pretrained(output_dir)
    loaded.processor.save_pretrained(output_dir)
    metadata = {
        "hl_memory_config": dataclasses.asdict(hl_config),
        "train_args": {
            key: str(value) if isinstance(value, pathlib.Path) else value
            for key, value in dataclasses.asdict(train_args).items()
        },
    }
    (output_dir / "hl_memory_train_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main(resolve_cli_args_with_yaml(TrainArgs, tyro))
