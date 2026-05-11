from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import math
import os
import pathlib
import random
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.config_io import resolve_cli_args_with_yaml
from openpi.hl_memory.data import FrameCache
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
    lora_enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    language_memory_dropout: float = 0.0
    language_memory_dropout_value: str = "Task started."
    batch_size: int = 1
    grad_accum_steps: int = 1
    num_train_steps: int = 100
    save_interval: int = 50
    log_interval: int = 10
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ddp_backend: str = "nccl"
    frame_cache_size: int = 4096
    use_grad_scaler: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "openpi-hl-memory"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None


def main(args: TrainArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    distributed = _init_distributed(args)
    try:
        _train(args, distributed=distributed)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(args: TrainArgs, *, distributed: bool) -> None:
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
    wandb_run = _init_wandb(args, sample_count=len(samples), world_size=world_size) if is_main else None
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
    if args.lora_enabled:
        loaded = dataclasses.replace(loaded, model=_apply_lora(loaded.model, args=args, is_main=is_main))
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
        (parameter for parameter in loaded.model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler_enabled = args.use_grad_scaler and device.type == "cuda" and not _has_fp16_trainable_parameters(loaded.model)
    if is_main and args.use_grad_scaler and not scaler_enabled:
        logging.info("Disabling GradScaler because trainable parameters are already fp16/bf16.")
    grad_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    frame_cache = FrameCache(args.frame_cache_size)
    rng = random.Random(args.seed + rank)
    running_loss = 0.0
    running_data_time = 0.0
    running_step_time = 0.0

    progress = tqdm(
        range(1, args.num_train_steps + 1),
        desc="HL train",
        disable=not is_main,
        dynamic_ncols=True,
        unit="it",
    )
    try:
        for step in progress:
            step_start_time = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_data_time = 0.0
            for accum_index in range(args.grad_accum_steps):
                batch = _sample_batch(samples, args.batch_size, rng)
                micro_loss = 0.0
                sync_gradients = accum_index == args.grad_accum_steps - 1
                sync_context = (
                    loaded.model.no_sync()
                    if distributed and isinstance(loaded.model, torch.nn.parallel.DistributedDataParallel) and not sync_gradients
                    else nullcontext()
                )
                with sync_context:
                    for sample in batch:
                        sample = _maybe_drop_language_memory(sample, args=args, rng=rng)
                        data_start_time = time.perf_counter()
                        clips = load_video_clips_for_sample(sample, args.dataset_dir, hl_config, frame_cache=frame_cache)
                        inputs = adapter.prepare_training_inputs(loaded, sample, clips, device=device)
                        supervised_tokens = int((inputs["labels"] != -100).sum().detach().cpu().item())
                        if supervised_tokens <= 0:
                            raise ValueError(
                                f"HL sample {sample.sample_id} produced zero supervised target tokens after label masking. "
                                "This would make the language-model loss NaN."
                            )
                        step_data_time += time.perf_counter() - data_start_time
                        outputs = loaded.model(**inputs)
                        loss = outputs.loss / max(len(batch), 1)
                        loss_value = float(loss.detach().cpu())
                        if not math.isfinite(loss_value):
                            raise FloatingPointError(
                                f"Non-finite HL loss for sample_id={sample.sample_id}: loss={loss_value} "
                                f"supervised_tokens={supervised_tokens} input_shape={tuple(inputs['input_ids'].shape)}. "
                                "Try --precision bfloat16 on bf16-capable GPUs, or reduce learning rate."
                            )
                        micro_loss += loss_value
                        grad_scaler.scale(loss / args.grad_accum_steps).backward()
                step_loss += micro_loss

            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(loaded.model.parameters(), args.max_grad_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            step_elapsed = time.perf_counter() - step_start_time
            logged_step_loss = _mean_across_ranks(step_loss, device=device) if distributed else step_loss
            running_loss += logged_step_loss
            logged_data_time = _mean_across_ranks(step_data_time, device=device) if distributed else step_data_time
            logged_step_time = _mean_across_ranks(step_elapsed, device=device) if distributed else step_elapsed
            running_data_time += logged_data_time
            running_step_time += logged_step_time

            if is_main and step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                avg_data_time = running_data_time / args.log_interval
                avg_step_time = running_step_time / args.log_interval
                data_fraction = avg_data_time / max(avg_step_time, 1e-6)
                progress.set_postfix(
                    loss=f"{avg_loss:.6f}",
                    data_s=f"{avg_data_time:.2f}",
                    step_s=f"{avg_step_time:.2f}",
                )
                logging.info(
                    "step=%d/%d loss=%.6f data_s/it=%.2f step_s/it=%.2f",
                    step,
                    args.num_train_steps,
                    avg_loss,
                    avg_data_time,
                    avg_step_time,
                )
                _wandb_log(
                    wandb_run,
                    {
                        "train/loss": avg_loss,
                        "time/data_s_per_it": avg_data_time,
                        "time/step_s_per_it": avg_step_time,
                        "time/data_fraction": data_fraction,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/global_batch_size": args.batch_size * args.grad_accum_steps * world_size,
                    },
                    step=step,
                )
                running_loss = 0.0
                running_data_time = 0.0
                running_step_time = 0.0
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
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _apply_lora(model: torch.nn.Module, *, args: TrainArgs, is_main: bool) -> torch.nn.Module:
    try:
        from peft import LoraConfig
        from peft import TaskType
        from peft import get_peft_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LoRA training requires `peft`. Install it or run without --lora-enabled.") from exc

    target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    if not target_modules:
        raise ValueError("--lora-target-modules must contain at least one module name.")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()
    return model


def _maybe_drop_language_memory(sample, *, args: TrainArgs, rng: random.Random):
    if args.language_memory_dropout <= 0.0:
        return sample
    if not 0.0 <= args.language_memory_dropout <= 1.0:
        raise ValueError(f"--language-memory-dropout must be in [0, 1], got {args.language_memory_dropout}")
    if rng.random() >= args.language_memory_dropout:
        return sample
    return dataclasses.replace(sample, language_memory=args.language_memory_dropout_value)


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


def _init_wandb(args: TrainArgs, *, sample_count: int, world_size: int):
    if not args.wandb_enabled:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            **{
                key: str(value) if isinstance(value, pathlib.Path) else value
                for key, value in dataclasses.asdict(args).items()
            },
            "sample_count": sample_count,
            "world_size": world_size,
            "effective_global_batch_size": args.batch_size * args.grad_accum_steps * world_size,
        },
    )


def _wandb_log(wandb_run, payload: dict[str, float | int], *, step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(payload, step=step)


def _has_fp16_trainable_parameters(model: torch.nn.Module) -> bool:
    return any(parameter.requires_grad and parameter.dtype in {torch.float16, torch.bfloat16} for parameter in model.parameters())


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
