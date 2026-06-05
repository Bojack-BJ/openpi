from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import logging
import math
import os
import pathlib
import random
import time
from contextlib import nullcontext
from typing import NamedTuple

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






'''

cd /root/Users/donggaoqi/openpi_vlm_finetune

PYTHONPATH=src /root/Users/miniconda3/envs/hl_qwen35/bin/python \
  scripts/hl_memory/train_hl_memory_multitask.py \
  --dataset-root /root/Users/dataset/hl_memory/subtask \
  --dataset-glob '*/train' \
  --output-dir /root/Users/donggaoqi/openpi_vlm_finetune/hl_memory_ckpts/subtask_multitask_qwen3_5_2b_lora \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-2B \
  --lora-enabled \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-train-steps 100000 \
  --save-interval 15000


'''




@dataclasses.dataclass
class TrainArgs:
    output_dir: pathlib.Path
    dataset_dir: pathlib.Path | None = None
    dataset_dirs: tuple[pathlib.Path, ...] = ()
    dataset_dirs_json: pathlib.Path | None = None
    dataset_root: pathlib.Path | None = None
    dataset_glob: str = "*/train"
    val_dataset_dir: pathlib.Path | None = None
    val_dataset_dirs: tuple[pathlib.Path, ...] = ()
    val_dataset_dirs_json: pathlib.Path | None = None
    val_dataset_root: pathlib.Path | None = None
    val_dataset_glob: str = "*/val"
    val_interval: int = 0
    val_batches: int = 10
    val_seed: int = 12345
    config_yaml: pathlib.Path | None = None
    vlm_backend: str = "qwen2_5_vl"
    vlm_variant: str | None = None
    vlm_hf_model_id: str | None = None
    local_vlm_ckpt_path: pathlib.Path | None = None
    precision: str = "bfloat16"
    enable_thinking: bool = False
    thinking_budget_tokens: int = 128
    thinking_max_new_tokens: int = 1024
    training_fps: float = 20.0
    frame_subsample: int = 5
    frame_height: int = 224
    frame_width: int = 456
    parallel_mode: str = "none"
    device_map: str = "auto"
    tensor_parallel_plan: str = "auto"
    target_protocol: str = "hl_v1"
    learning_rate: float = 5e-6
    weight_decay: float = 1e-4
    lora_enabled: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    vision_tower_train_mode: str = "frozen"
    vision_tower_unfreeze_last_n_layers: int = 0
    language_memory_dropout: float = 0.3
    language_memory_dropout_value: str = "No progress has been recorded yet."
    step_prior_dropout: float = 0.3
    batch_size: int = 1
    grad_accum_steps: int = 1
    num_train_steps: int = 100
    save_interval: int = 50
    log_interval: int = 10
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ddp_backend: str = "nccl"
    distributed_strategy: str = "ddp"
    fsdp_min_num_params: int = 20_000_000
    fsdp_cpu_offload: bool = False
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

    dataset_dirs = _resolve_dataset_dirs(args)
    samples = _load_multitask_samples(dataset_dirs)
    if not samples:
        raise ValueError(f"No exported HL memory samples found in: {dataset_dirs}.")
    val_samples: list[RuntimeSample] = []
    if args.val_interval > 0:
        val_dataset_dirs = _resolve_val_dataset_dirs(args)
        val_samples = _load_multitask_samples(val_dataset_dirs)
        if not val_samples:
            raise ValueError(f"No exported HL validation samples found in: {val_dataset_dirs}.")
    if distributed and args.parallel_mode != "none":
        raise ValueError(
            "Distributed training requires --parallel-mode none. Do not combine DDP/FSDP with device_map/tensor_parallel."
        )
    if args.distributed_strategy not in {"ddp", "fsdp"}:
        raise ValueError(f"--distributed-strategy must be 'ddp' or 'fsdp', got {args.distributed_strategy!r}")
    if args.distributed_strategy == "fsdp" and not distributed:
        raise ValueError("--distributed-strategy fsdp requires launching with torchrun.")

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Loaded %d HL samples. world_size=%d batch_size_per_rank=%d grad_accum_steps=%d",
            len(samples),
            world_size,
            args.batch_size,
            args.grad_accum_steps,
        )
        if val_samples:
            logging.info(
                "Loaded %d HL validation samples. val_interval=%d val_batches_per_rank=%d",
                len(val_samples),
                args.val_interval,
                args.val_batches,
            )
    wandb_run = (
        _init_wandb(args, sample_count=len(samples), val_sample_count=len(val_samples), world_size=world_size)
        if is_main
        else None
    )
    hl_config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_variant=args.vlm_variant,
        vlm_hf_model_id=args.vlm_hf_model_id,
        precision=args.precision,
        training_fps=args.training_fps,
        frame_subsample=args.frame_subsample,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        enable_thinking=args.enable_thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
        thinking_max_new_tokens=args.thinking_max_new_tokens,
        parallel_mode=args.parallel_mode,
        device_map=args.device_map,
        tensor_parallel_plan=args.tensor_parallel_plan,
        target_protocol=args.target_protocol,
    )
    adapter = create_hf_adapter(hl_config)
    device = _resolve_training_device(args, distributed=distributed)
    loaded = adapter.load(
        model_path=None if args.local_vlm_ckpt_path is None else str(args.local_vlm_ckpt_path),
        device=device,
    )
    if args.lora_enabled:
        loaded = dataclasses.replace(loaded, model=_apply_lora(loaded.model, args=args, is_main=is_main))
    _configure_vision_tower_training(loaded.model, args=args, is_main=is_main)
    if distributed and args.distributed_strategy == "fsdp":
        _align_fsdp_parameter_dtypes(loaded.model, args=args, is_main=is_main)
    loaded.model.train()
    if distributed:
        if args.distributed_strategy == "fsdp":
            fsdp_report = _build_fsdp_report(loaded.model, args=args, world_size=world_size)
            if is_main:
                _log_fsdp_report(fsdp_report)
            loaded = dataclasses.replace(
                loaded,
                model=_wrap_fsdp(
                    loaded.model,
                    args=args,
                    device=device,
                    ignored_modules=fsdp_report.ignored_modules,
                ),
            )
        else:
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
    scaler_enabled = (
        args.use_grad_scaler
        and args.precision == "float16"
        and device.type == "cuda"
        and not _has_fp16_trainable_parameters(loaded.model)
    )
    if is_main and args.use_grad_scaler and not scaler_enabled:
        logging.info("Disabling GradScaler; it is only used for float16 training with fp32 trainable parameters.")
    grad_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    frame_cache = FrameCache(args.frame_cache_size)
    rng = random.Random(args.seed + rank)
    val_rng = random.Random(args.val_seed + rank)
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
                    if _should_use_no_sync(loaded.model, sync_gradients=sync_gradients)
                    else nullcontext()
                )
                with sync_context:
                    batch_samples = [_maybe_apply_training_dropouts(item.sample, args=args, rng=rng) for item in batch]
                    data_start_time = time.perf_counter()
                    batch_clips = [
                        load_video_clips_for_sample(sample, item.dataset_dir, hl_config, frame_cache=frame_cache)
                        for item, sample in zip(batch, batch_samples, strict=True)
                    ]
                    inputs = adapter.prepare_training_batch_inputs(loaded, batch_samples, batch_clips, device=device)
                    supervised_tokens = (inputs["labels"] != -100).sum(dim=1).detach().cpu().tolist()
                    bad_indices = [index for index, count in enumerate(supervised_tokens) if int(count) <= 0]
                    if bad_indices:
                        bad_sample = batch_samples[bad_indices[0]]
                        raise ValueError(
                            f"HL sample {bad_sample.sample_id} produced zero supervised target tokens after label masking. "
                            "This would make the language-model loss NaN."
                        )
                    step_data_time += time.perf_counter() - data_start_time
                    loss = _compute_target_loss(loaded.model, inputs)
                    loss_value = float(loss.detach().cpu())
                    if not math.isfinite(loss_value):
                        sample_ids = ",".join(sample.sample_id for sample in batch_samples[:4])
                        raise FloatingPointError(
                            f"Non-finite HL loss for batch sample_ids={sample_ids}: loss={loss_value} "
                            f"input_shape={tuple(inputs['input_ids'].shape)}. "
                            "Try --precision bfloat16 on bf16-capable GPUs, or reduce learning rate."
                        )
                    micro_loss += loss_value / args.grad_accum_steps
                    grad_scaler.scale(loss / args.grad_accum_steps).backward()
                step_loss += micro_loss

            grad_scaler.unscale_(optimizer)
            _clip_grad_norm(loaded.model, args.max_grad_norm)
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
            if val_samples and step % args.val_interval == 0:
                val_started_at = time.perf_counter()
                val_loss = _evaluate_loss(
                    loaded=loaded,
                    adapter=adapter,
                    samples=val_samples,
                    batch_size=args.batch_size,
                    num_batches=args.val_batches,
                    rng=val_rng,
                    hl_config=hl_config,
                    frame_cache=frame_cache,
                    device=device,
                )
                logged_val_loss = _mean_across_ranks(val_loss, device=device) if distributed else val_loss
                val_elapsed = time.perf_counter() - val_started_at
                logged_val_elapsed = _mean_across_ranks(val_elapsed, device=device) if distributed else val_elapsed
                if is_main:
                    logging.info(
                        "step=%d/%d val_loss=%.6f val_batches_per_rank=%d val_time_s=%.1f",
                        step,
                        args.num_train_steps,
                        logged_val_loss,
                        args.val_batches,
                        logged_val_elapsed,
                    )
                    _wandb_log(
                        wandb_run,
                        {
                            "val/loss": logged_val_loss,
                            "val/time_s": logged_val_elapsed,
                            "val/batches_per_rank": args.val_batches,
                            "val/effective_samples": args.val_batches * args.batch_size * world_size,
                        },
                        step=step,
                    )
            save_due = step % args.save_interval == 0 or step == args.num_train_steps
            if save_due and (is_main or _is_fsdp_model(loaded.model)):
                _save_checkpoint(
                    output_dir=args.output_dir / f"checkpoint-step-{step:06d}",
                    loaded=loaded,
                    hl_config=hl_config,
                    train_args=args,
                    is_main=is_main,
                )
            if distributed and save_due:
                dist.barrier()
    finally:
        if wandb_run is not None:
            wandb_run.finish()


class RuntimeSample(NamedTuple):
    dataset_dir: pathlib.Path
    sample: object


class FSDPReport(NamedTuple):
    total_params: int
    trainable_params: int
    ignored_params: int
    estimated_sharded_params: int
    world_size: int
    min_num_params: int
    ignored_module_names: tuple[str, ...]
    ignored_modules: tuple[torch.nn.Module, ...]
    candidate_wrap_count: int
    candidate_wrap_params: int


def _resolve_dataset_dirs(args: TrainArgs) -> list[pathlib.Path]:
    dirs: list[pathlib.Path] = []
    if args.dataset_dir is not None:
        dirs.append(args.dataset_dir)
    dirs.extend(args.dataset_dirs)
    if args.dataset_dirs_json is not None:
        payload = json.loads(args.dataset_dirs_json.read_text())
        if not isinstance(payload, list):
            raise ValueError("--dataset-dirs-json must contain a JSON list of dataset directories.")
        dirs.extend(pathlib.Path(str(item)) for item in payload)
    if args.dataset_root is not None:
        dirs.extend(sorted(path for path in args.dataset_root.glob(args.dataset_glob) if (path / "samples.jsonl").is_file()))
    unique: list[pathlib.Path] = []
    seen: set[str] = set()
    for dataset_dir in dirs:
        resolved = dataset_dir.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if not (resolved / "samples.jsonl").is_file():
            raise FileNotFoundError(f"Missing samples.jsonl under {resolved}")
        unique.append(resolved)
    if not unique:
        raise ValueError("Set --dataset-dir, --dataset-dirs, --dataset-dirs-json, or --dataset-root.")
    return unique


def _resolve_val_dataset_dirs(args: TrainArgs) -> list[pathlib.Path]:
    dirs: list[pathlib.Path] = []
    if args.val_dataset_dir is not None:
        dirs.append(args.val_dataset_dir)
    dirs.extend(args.val_dataset_dirs)
    if args.val_dataset_dirs_json is not None:
        payload = json.loads(args.val_dataset_dirs_json.read_text())
        if not isinstance(payload, list):
            raise ValueError("--val-dataset-dirs-json must contain a JSON list of dataset directories.")
        dirs.extend(pathlib.Path(str(item)) for item in payload)
    val_root = args.val_dataset_root
    if val_root is None and not dirs:
        val_root = args.dataset_root
    if val_root is not None:
        dirs.extend(sorted(path for path in val_root.glob(args.val_dataset_glob) if (path / "samples.jsonl").is_file()))
    unique: list[pathlib.Path] = []
    seen: set[str] = set()
    for dataset_dir in dirs:
        resolved = dataset_dir.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if not (resolved / "samples.jsonl").is_file():
            raise FileNotFoundError(f"Missing validation samples.jsonl under {resolved}")
        unique.append(resolved)
    if not unique:
        raise ValueError(
            "Set --val-dataset-dir, --val-dataset-dirs, --val-dataset-dirs-json, or "
            "--val-dataset-root when --val-interval > 0."
        )
    return unique


def _load_multitask_samples(dataset_dirs: list[pathlib.Path]) -> list[RuntimeSample]:
    items: list[RuntimeSample] = []
    for dataset_dir in dataset_dirs:
        loaded = load_exported_samples(dataset_dir)
        items.extend(RuntimeSample(dataset_dir=dataset_dir, sample=sample) for sample in loaded)
        logging.info("Loaded %d HL samples from %s", len(loaded), dataset_dir)
    return items


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


def _configure_vision_tower_training(model: torch.nn.Module, *, args: TrainArgs, is_main: bool) -> None:
    mode = args.vision_tower_train_mode
    if mode not in {"frozen", "last_n", "full"}:
        raise ValueError(f"--vision-tower-train-mode must be frozen|last_n|full, got {mode!r}")
    if mode == "frozen":
        if args.vision_tower_unfreeze_last_n_layers:
            raise ValueError("--vision-tower-unfreeze-last-n-layers is only valid with --vision-tower-train-mode last_n")

    vision_named_params = [(name, parameter) for name, parameter in model.named_parameters() if _is_vision_module_name(name)]
    if not vision_named_params:
        if mode == "frozen":
            return
        raise ValueError(
            f"--vision-tower-train-mode {mode!r} was set, but no vision tower parameters were found. "
            "Inspect model.named_parameters() for this VLM backend."
        )

    for _, parameter in vision_named_params:
        parameter.requires_grad = False

    trainable_param_ids: set[int]
    trainable_module_names: list[str]
    if mode == "frozen":
        trainable_param_ids = set()
        trainable_module_names = ["<none>"]
    elif mode == "full":
        trainable_param_ids = {id(parameter) for _, parameter in vision_named_params}
        trainable_module_names = ["<entire vision tower>"]
    else:
        if args.vision_tower_unfreeze_last_n_layers <= 0:
            raise ValueError("--vision-tower-unfreeze-last-n-layers must be > 0 with --vision-tower-train-mode last_n")
        vision_blocks = _find_vision_block_modules(model)
        if not vision_blocks:
            raise ValueError(
                "--vision-tower-train-mode last_n was set, but no vision block ModuleList/Sequential was found. "
                "Use --vision-tower-train-mode full or update the block-name heuristics for this backend."
            )
        selected_blocks = vision_blocks[-args.vision_tower_unfreeze_last_n_layers :]
        selected_tail_modules = _find_vision_tail_modules(model)
        trainable_param_ids = set()
        trainable_module_names = []
        for name, module in selected_blocks + selected_tail_modules:
            trainable_module_names.append(name)
            trainable_param_ids.update(id(parameter) for parameter in module.parameters(recurse=True))

    for _, parameter in vision_named_params:
        parameter.requires_grad = id(parameter) in trainable_param_ids

    if is_main:
        trainable_params = sum(parameter.numel() for _, parameter in vision_named_params if parameter.requires_grad)
        total_params = sum(parameter.numel() for _, parameter in vision_named_params)
        preview = ", ".join(trainable_module_names[:8])
        suffix = "" if len(trainable_module_names) <= 8 else f", ... +{len(trainable_module_names) - 8}"
        logging.info(
            "Vision tower train mode=%s trainable=%s/%s params modules=%s%s",
            mode,
            _format_count(trainable_params),
            _format_count(total_params),
            preview,
            suffix,
        )


def _find_vision_block_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    block_containers: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if not _is_vision_module_name(name):
            continue
        if not isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
            continue
        if name.split(".")[-1] in {"blocks", "layers", "layer", "resblocks"}:
            block_containers.append((name, module))
    if not block_containers:
        return []
    container_name, container = max(block_containers, key=lambda item: len(item[1]))
    return [(f"{container_name}.{index}", child) for index, child in enumerate(container)]


def _find_vision_tail_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    tail_modules: list[tuple[str, torch.nn.Module]] = []
    tail_names = {"merger", "patch_merger"}
    for name, module in model.named_modules():
        if not _is_vision_module_name(name):
            continue
        if name.split(".")[-1] in tail_names:
            tail_modules.append((name, module))
    return tail_modules


def _is_vision_module_name(name: str) -> bool:
    components = name.split(".")
    vision_components = {"visual", "vision_tower", "vision_model", "vision_encoder"}
    if any(component in vision_components for component in components):
        return True
    return any(component.startswith("visual") or component.startswith("vision") for component in components)


def _align_fsdp_parameter_dtypes(model: torch.nn.Module, *, args: TrainArgs, is_main: bool) -> None:
    target_dtype = _training_torch_dtype(args)
    if target_dtype is None:
        return
    converted = 0
    seen: set[int] = set()
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        if parameter.is_floating_point() and parameter.dtype != target_dtype:
            parameter.data = parameter.data.to(dtype=target_dtype)
            converted += 1
    if is_main and converted:
        logging.info("Converted %d floating parameters to %s before FSDP wrapping.", converted, target_dtype)


def _training_torch_dtype(args: TrainArgs) -> torch.dtype | None:
    if args.precision == "bfloat16":
        return torch.bfloat16
    if args.precision == "float16":
        return torch.float16
    if args.precision == "float32":
        return torch.float32
    return None


def _wrap_fsdp(
    model: torch.nn.Module,
    *,
    args: TrainArgs,
    device: torch.device,
    ignored_modules: tuple[torch.nn.Module, ...],
) -> torch.nn.Module:
    try:
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    except ImportError as exc:
        raise ImportError("FSDP requires a PyTorch build with torch.distributed.fsdp support.") from exc

    mixed_precision = None
    target_dtype = _training_torch_dtype(args)
    if target_dtype in {torch.bfloat16, torch.float16}:
        mixed_precision = MixedPrecision(param_dtype=target_dtype, reduce_dtype=target_dtype, buffer_dtype=target_dtype)
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=args.fsdp_min_num_params,
    )
    cpu_offload = CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None
    return FullyShardedDataParallel(
        model,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload,
        device_id=device if device.type == "cuda" else None,
        ignored_modules=ignored_modules or None,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )


def _build_fsdp_report(model: torch.nn.Module, *, args: TrainArgs, world_size: int) -> FSDPReport:
    ignored_named_modules = _fsdp_ignored_named_modules(model)
    ignored_modules = tuple(module for _, module in ignored_named_modules)
    total_params = _unique_param_count(model.parameters())
    trainable_params = _unique_param_count(parameter for parameter in model.parameters() if parameter.requires_grad)
    ignored_params = _unique_module_param_count(ignored_modules)
    candidate_wrap_count = 0
    candidate_wrap_params = 0
    ignored_ids = {id(module) for module in ignored_modules}
    for module in model.modules():
        if id(module) in ignored_ids:
            continue
        module_params = _unique_param_count(module.parameters(recurse=False))
        if module_params >= args.fsdp_min_num_params:
            candidate_wrap_count += 1
            candidate_wrap_params += module_params
    return FSDPReport(
        total_params=total_params,
        trainable_params=trainable_params,
        ignored_params=ignored_params,
        estimated_sharded_params=max(total_params - ignored_params, 0),
        world_size=world_size,
        min_num_params=args.fsdp_min_num_params,
        ignored_module_names=tuple(name for name, _ in ignored_named_modules),
        ignored_modules=ignored_modules,
        candidate_wrap_count=candidate_wrap_count,
        candidate_wrap_params=candidate_wrap_params,
    )


def _log_fsdp_report(report: FSDPReport) -> None:
    estimated_per_rank_params = report.ignored_params + math.ceil(report.estimated_sharded_params / max(report.world_size, 1))
    logging.info(
        "FSDP report: total_params=%s trainable_params=%s ignored_replicated_params=%s "
        "estimated_sharded_params=%s estimated_params_per_rank=%s world_size=%d fsdp_min_num_params=%d "
        "candidate_direct_wrap_modules=%d candidate_direct_wrap_params=%s",
        _format_count(report.total_params),
        _format_count(report.trainable_params),
        _format_count(report.ignored_params),
        _format_count(report.estimated_sharded_params),
        _format_count(estimated_per_rank_params),
        report.world_size,
        report.min_num_params,
        report.candidate_wrap_count,
        _format_count(report.candidate_wrap_params),
    )
    logging.info("FSDP ignored replicated modules: %s", ", ".join(report.ignored_module_names) or "none")


def _fsdp_ignored_named_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    # PEFT/LoRA freezes embeddings and lm_head by default. Keeping those
    # frozen modules out of root FSDP avoids tied/flat parameter shape issues
    # such as lm_head.weight becoming a 1D flat parameter at F.linear().
    ignored: list[tuple[str, torch.nn.Module]] = []
    seen: set[int] = set()
    module_names = {id(module): name for name, module in model.named_modules()}
    for accessor_name in ("get_input_embeddings", "get_output_embeddings"):
        accessor = getattr(model, accessor_name, None)
        if not callable(accessor):
            continue
        module = accessor()
        if module is None or id(module) in seen:
            continue
        parameters = list(module.parameters(recurse=True))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            ignored.append((module_names.get(id(module), accessor_name), module))
            seen.add(id(module))
    for name, module in model.named_modules():
        if not name.endswith(("lm_head", "embed_tokens")) or id(module) in seen:
            continue
        parameters = list(module.parameters(recurse=True))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            ignored.append((name, module))
            seen.add(id(module))
    return ignored


def _unique_module_param_count(modules: tuple[torch.nn.Module, ...]) -> int:
    return _unique_param_count(parameter for module in modules for parameter in module.parameters(recurse=True))


def _unique_param_count(parameters) -> int:
    total = 0
    seen: set[int] = set()
    for parameter in parameters:
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        total += parameter.numel()
    return total


def _format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def _maybe_apply_training_dropouts(sample, *, args: TrainArgs, rng: random.Random):
    sample = _maybe_drop_language_memory(sample, args=args, rng=rng)
    return _maybe_drop_step_prior(sample, args=args, rng=rng)


def _maybe_drop_language_memory(sample, *, args: TrainArgs, rng: random.Random):
    if args.language_memory_dropout <= 0.0:
        return sample
    if not 0.0 <= args.language_memory_dropout <= 1.0:
        raise ValueError(f"--language-memory-dropout must be in [0, 1], got {args.language_memory_dropout}")
    if rng.random() >= args.language_memory_dropout:
        return sample
    return dataclasses.replace(sample, language_memory=args.language_memory_dropout_value)


def _maybe_drop_step_prior(sample, *, args: TrainArgs, rng: random.Random):
    if args.step_prior_dropout <= 0.0:
        return sample
    if not 0.0 <= args.step_prior_dropout <= 1.0:
        raise ValueError(f"--step-prior-dropout must be in [0, 1], got {args.step_prior_dropout}")
    if rng.random() >= args.step_prior_dropout:
        return sample
    return dataclasses.replace(sample, step_prior=())


def _compute_target_loss(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = inputs["labels"]
    logits_to_keep = _target_logit_positions(labels)
    if logits_to_keep is None:
        outputs = model(**inputs)
        return outputs.loss
    model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
    try:
        outputs = model(**model_inputs, logits_to_keep=logits_to_keep)
    except TypeError:
        outputs = model(**inputs)
        return outputs.loss
    logits = outputs.logits
    shifted_labels = torch.nn.functional.pad(labels, (0, 1), value=-100).index_select(1, logits_to_keep + 1)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )


def _target_logit_positions(labels: torch.Tensor) -> torch.Tensor | None:
    # A logit at position i predicts the label at position i + 1. Since all
    # prompt/padding labels are -100, only positions before supervised target
    # tokens need vocab projection.
    if labels.shape[1] < 2:
        return None
    supervised_next_token = labels[:, 1:] != -100
    positions = torch.nonzero(supervised_next_token.any(dim=0), as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    return positions.to(device=labels.device, dtype=torch.long)


def _evaluate_loss(
    *,
    loaded,
    adapter,
    samples: list[RuntimeSample],
    batch_size: int,
    num_batches: int,
    rng: random.Random,
    hl_config: HLMemoryConfig,
    frame_cache: FrameCache,
    device: torch.device,
) -> float:
    if num_batches <= 0:
        raise ValueError("--val-batches must be positive when validation is enabled.")
    was_training = loaded.model.training
    loaded.model.eval()
    total_loss = 0.0
    try:
        with torch.no_grad():
            for _ in range(num_batches):
                batch = _sample_batch(samples, batch_size, rng)
                batch_samples = [item.sample for item in batch]
                batch_clips = [
                    load_video_clips_for_sample(sample, item.dataset_dir, hl_config, frame_cache=frame_cache)
                    for item, sample in zip(batch, batch_samples, strict=True)
                ]
                inputs = adapter.prepare_training_batch_inputs(loaded, batch_samples, batch_clips, device=device)
                supervised_tokens = (inputs["labels"] != -100).sum(dim=1).detach().cpu().tolist()
                bad_indices = [index for index, count in enumerate(supervised_tokens) if int(count) <= 0]
                if bad_indices:
                    bad_sample = batch_samples[bad_indices[0]]
                    raise ValueError(
                        f"HL validation sample {bad_sample.sample_id} produced zero supervised target tokens "
                        "after label masking."
                    )
                loss = _compute_target_loss(loaded.model, inputs)
                loss_value = float(loss.detach().cpu())
                if not math.isfinite(loss_value):
                    sample_ids = ",".join(sample.sample_id for sample in batch_samples[:4])
                    raise FloatingPointError(
                        f"Non-finite HL validation loss for batch sample_ids={sample_ids}: loss={loss_value} "
                        f"input_shape={tuple(inputs['input_ids'].shape)}."
                    )
                total_loss += loss_value
    finally:
        if was_training:
            loaded.model.train()
    return total_loss / num_batches


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


def _init_wandb(args: TrainArgs, *, sample_count: int, val_sample_count: int, world_size: int):
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
            "val_sample_count": val_sample_count,
            "world_size": world_size,
            "effective_global_batch_size": args.batch_size * args.grad_accum_steps * world_size,
        },
    )


def _wandb_log(wandb_run, payload: dict[str, float | int], *, step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(payload, step=step)


def _has_fp16_trainable_parameters(model: torch.nn.Module) -> bool:
    return any(parameter.requires_grad and parameter.dtype in {torch.float16, torch.bfloat16} for parameter in model.parameters())


def _clip_grad_norm(model: torch.nn.Module, max_norm: float) -> None:
    if _is_fsdp_model(model):
        model.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _sample_batch(samples: list[RuntimeSample], batch_size: int, rng: random.Random):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(samples) >= batch_size:
        return rng.sample(samples, batch_size)
    return [rng.choice(samples) for _ in range(batch_size)]


def _save_checkpoint(*, output_dir: pathlib.Path, loaded, hl_config: HLMemoryConfig, train_args: TrainArgs, is_main: bool) -> None:
    if _is_fsdp_model(loaded.model):
        try:
            from torch.distributed.fsdp import FullStateDictConfig
            from torch.distributed.fsdp import FullyShardedDataParallel
            from torch.distributed.fsdp import StateDictType
        except ImportError as exc:
            raise ImportError("Saving FSDP checkpoints requires torch.distributed.fsdp support.") from exc
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(loaded.model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = loaded.model.state_dict()
        model = loaded.model.module
        if not is_main:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        if not is_main:
            return
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


def _is_fsdp_model(model: torch.nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return isinstance(model, FullyShardedDataParallel)


def _should_use_no_sync(model: torch.nn.Module, *, sync_gradients: bool) -> bool:
    if sync_gradients:
        return False
    # FSDP no_sync keeps accumulated gradients unsharded and can increase peak memory.
    # For this script FSDP is mainly a memory-saving path, so only DDP uses no_sync.
    return isinstance(model, torch.nn.parallel.DistributedDataParallel)


if __name__ == "__main__":
    main(resolve_cli_args_with_yaml(TrainArgs, tyro))
