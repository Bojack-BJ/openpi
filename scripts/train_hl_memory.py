from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import pathlib
import random

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
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
    vlm_variant: str | None = None
    vlm_hf_model_id: str | None = None
    local_vlm_ckpt_path: pathlib.Path | None = None
    precision: str = "bfloat16"
    enable_thinking: bool = False
    thinking_budget_tokens: int = 128
    thinking_max_new_tokens: int = 1024
    parallel_mode: str = "none"
    device_map: str = "auto"
    tensor_parallel_plan: str = "auto"
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
    ddp_backend: str = "nccl"


def main(args: TrainArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    distributed = _init_distributed(args)
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    is_main = rank == 0
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    samples = load_exported_samples(args.dataset_dir)
    if not samples:
        raise ValueError(f"No exported HL memory samples found in {args.dataset_dir}.")
    if distributed and args.parallel_mode != "none":
        raise ValueError("DDP training requires --parallel-mode none. Do not combine DDP with device_map/tensor_parallel.")

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Loaded %d HL samples. world_size=%d batch_size_per_rank=%d grad_accum_steps=%d",
            len(samples),
            world_size,
            args.batch_size,
            args.grad_accum_steps,
        )
    hl_config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_variant=args.vlm_variant,
        vlm_hf_model_id=args.vlm_hf_model_id,
        precision=args.precision,
        enable_thinking=args.enable_thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
        thinking_max_new_tokens=args.thinking_max_new_tokens,
        parallel_mode=args.parallel_mode,
        device_map=args.device_map,
        tensor_parallel_plan=args.tensor_parallel_plan,
    )
    adapter = create_hf_adapter(hl_config)
    device = _resolve_training_device(args, distributed=distributed)
    loaded = adapter.load(
        model_path=None if args.local_vlm_ckpt_path is None else str(args.local_vlm_ckpt_path),
        device=device,
    )
    loaded.model.train()
    if distributed:
        loaded = dataclasses.replace(
            loaded,
            model=torch.nn.parallel.DistributedDataParallel(
                loaded.model,
                device_ids=[device.index] if device.type == "cuda" else None,
            ),
        )
    optimizer = torch.optim.AdamW(
        loaded.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed + rank)
    running_loss = 0.0

    progress = tqdm(
        range(1, args.num_train_steps + 1),
        desc="HL train",
        disable=not is_main,
        dynamic_ncols=True,
        unit="it",
    )
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum_steps):
            batch = _sample_batch(samples, args.batch_size, rng)
            micro_loss = 0.0
            for sample in batch:
                clips = load_video_clips_for_sample(sample, args.dataset_dir, hl_config)
                inputs = adapter.prepare_training_inputs(loaded, sample, clips, device=device)
                outputs = loaded.model(**inputs)
                loss = outputs.loss / max(len(batch), 1)
                micro_loss += float(loss.detach().cpu())
                (loss / args.grad_accum_steps).backward()
            step_loss += micro_loss

        torch.nn.utils.clip_grad_norm_(loaded.model.parameters(), args.max_grad_norm)
        optimizer.step()
        logged_step_loss = _mean_across_ranks(step_loss, device=device) if distributed else step_loss
        running_loss += logged_step_loss

        if is_main and step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            progress.set_postfix(loss=f"{avg_loss:.6f}")
            logging.info("step=%d/%d loss=%.6f", step, args.num_train_steps, avg_loss)
            running_loss = 0.0
        save_due = step % args.save_interval == 0 or step == args.num_train_steps
        if is_main and save_due:
            _save_checkpoint(
                output_dir=args.output_dir / f"checkpoint-step-{step:06d}",
                loaded=loaded,
                hl_config=hl_config,
                train_args=args,
            )
        if distributed and save_due:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


def _init_distributed(args: TrainArgs) -> bool:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    backend = args.ddp_backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=30))
    return True


def _resolve_training_device(args: TrainArgs, *, distributed: bool) -> torch.device:
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
    return torch.device(args.device)


def _mean_across_ranks(value: float, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.detach().cpu())


def _sample_batch(samples, batch_size: int, rng: random.Random):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(samples) >= batch_size:
        return rng.sample(samples, batch_size)
    return [rng.choice(samples) for _ in range(batch_size)]


def _save_checkpoint(*, output_dir: pathlib.Path, loaded, hl_config: HLMemoryConfig, train_args: TrainArgs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = loaded.model.module if isinstance(loaded.model, torch.nn.parallel.DistributedDataParallel) else loaded.model
    model.save_pretrained(output_dir)
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
