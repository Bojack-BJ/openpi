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
from openpi.hl_memory.proprio import save_proprio_state_if_available
from openpi.hl_memory.proprio import temporarily_unwrap_proprio_embeddings
from openpi.hl_memory.sampling import sample_keyframe_stratified
from openpi.hl_memory.training_loss import compute_hl_target_loss


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
    training_fps: float = 20.0
    frame_subsample: int = 5
    recent_sample_hz: float = 2.0
    frame_height: int = 224
    frame_width: int = 456
    parallel_mode: str = "none"
    device_map: str = "auto"
    tensor_parallel_plan: str = "auto"
    target_protocol: str = "hl_v1"
    typed_mask_suppress_language_memory: bool = False
    proprio_enabled: bool = False
    proprio_token_mode: str = "per_frame_plus_summary"
    proprio_state_dim: int = 14
    proprio_hidden_dim: int = 512
    proprio_dropout: float = 0.0
    proprio_noise_std: float = 0.0
    keyframe_event_band_before_sec: float = 1.0
    keyframe_event_band_after_sec: float = 0.5
    keyframe_candidate_label_mode: str = "event_band"
    keyframe_positive_sample_ratio: float = 0.0
    keyframe_confirm_positive_sample_ratio: float = 0.0
    two_pass_training_proposal_noise_probability: float = 0.25
    two_pass_predict_loss_weight: float = 1.0
    two_pass_confirm_loss_weight: float = 1.0
    learning_rate: float = 5e-6
    vision_tower_learning_rate: float | None = None
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
        training_fps=args.training_fps,
        frame_subsample=args.frame_subsample,
        recent_sample_hz=args.recent_sample_hz,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        enable_thinking=args.enable_thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
        thinking_max_new_tokens=args.thinking_max_new_tokens,
        parallel_mode=args.parallel_mode,
        device_map=args.device_map,
        tensor_parallel_plan=args.tensor_parallel_plan,
        target_protocol=args.target_protocol,
        typed_mask_suppress_language_memory=args.typed_mask_suppress_language_memory,
        proprio_enabled=args.proprio_enabled,
        proprio_token_mode=args.proprio_token_mode,
        proprio_state_dim=args.proprio_state_dim,
        proprio_hidden_dim=args.proprio_hidden_dim,
        proprio_dropout=args.proprio_dropout,
        proprio_noise_std=args.proprio_noise_std,
        keyframe_event_band_before_sec=args.keyframe_event_band_before_sec,
        keyframe_event_band_after_sec=args.keyframe_event_band_after_sec,
        keyframe_candidate_label_mode=args.keyframe_candidate_label_mode,
        two_pass_training_proposal_noise_probability=args.two_pass_training_proposal_noise_probability,
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
    loaded.model.train()
    if distributed:
        loaded = dataclasses.replace(
            loaded,
            model=torch.nn.parallel.DistributedDataParallel(
                loaded.model,
                device_ids=[device.index] if device.type == "cuda" else None,
            ),
        )
    optimizer = _build_optimizer(loaded.model, args=args, is_main=is_main)
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
                batch = _sample_batch(
                    samples,
                    args.batch_size,
                    rng,
                    keyframe_positive_sample_ratio=args.keyframe_positive_sample_ratio,
                    keyframe_confirm_positive_sample_ratio=args.keyframe_confirm_positive_sample_ratio,
                    target_protocol=hl_config.target_protocol,
                    keyframe_candidate_label_mode=hl_config.keyframe_candidate_label_mode,
                )
                micro_loss = 0.0
                sync_gradients = accum_index == args.grad_accum_steps - 1
                sync_context = (
                    loaded.model.no_sync()
                    if distributed and isinstance(loaded.model, torch.nn.parallel.DistributedDataParallel) and not sync_gradients
                    else nullcontext()
                )
                with sync_context:
                    batch_samples = [_maybe_apply_training_dropouts(sample, args=args, rng=rng) for sample in batch]
                    data_start_time = time.perf_counter()
                    batch_clips = [
                        load_video_clips_for_sample(sample, args.dataset_dir, hl_config, frame_cache=frame_cache)
                        for sample in batch_samples
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
                    loss = compute_hl_target_loss(
                        loaded.model,
                        inputs,
                        two_pass_predict_weight=args.two_pass_predict_loss_weight,
                        two_pass_confirm_weight=args.two_pass_confirm_loss_weight,
                    )
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
                        **_optimizer_lr_metrics(optimizer),
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


def _build_optimizer(model: torch.nn.Module, *, args: TrainArgs, is_main: bool) -> torch.optim.Optimizer:
    if args.vision_tower_learning_rate is None:
        return torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    vision_params: list[torch.nn.Parameter] = []
    base_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        is_vision_parameter = bool(getattr(parameter, "_hl_vision_tower_param", False)) or _is_vision_module_name(name)
        if is_vision_parameter:
            vision_params.append(parameter)
        else:
            base_params.append(parameter)

    param_groups: list[dict[str, object]] = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.learning_rate, "name": "base"})
    if vision_params:
        param_groups.append({"params": vision_params, "lr": args.vision_tower_learning_rate, "name": "vision_tower"})
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")

    if is_main:
        logging.info(
            "Optimizer param groups: base_params=%s lr=%g vision_params=%s lr=%g",
            _format_count(sum(parameter.numel() for parameter in base_params)),
            args.learning_rate,
            _format_count(sum(parameter.numel() for parameter in vision_params)),
            args.vision_tower_learning_rate,
        )
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def _optimizer_lr_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        lr = float(group["lr"])
        if name == "base":
            metrics["train/lr"] = lr
        elif name == "vision_tower":
            metrics["train/vision_lr"] = lr
        else:
            metrics[f"train/lr_{name}"] = lr
    if "train/lr" not in metrics and optimizer.param_groups:
        metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
    return metrics


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
        parameter._hl_vision_tower_param = parameter.requires_grad

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


def _sample_batch(
    samples,
    batch_size: int,
    rng: random.Random,
    *,
    keyframe_positive_sample_ratio: float = 0.0,
    keyframe_confirm_positive_sample_ratio: float = 0.0,
    target_protocol: str = "hl_v1",
    keyframe_candidate_label_mode: str = "event_band",
):
    return sample_keyframe_stratified(
        samples,
        batch_size,
        rng,
        sample_from_item=lambda sample: sample,
        keyframe_positive_sample_ratio=keyframe_positive_sample_ratio,
        keyframe_confirm_positive_sample_ratio=keyframe_confirm_positive_sample_ratio,
        target_protocol=target_protocol,
        keyframe_candidate_label_mode=keyframe_candidate_label_mode,
    )


def _sample_with_replacement(samples, count: int, rng: random.Random):
    if count <= 0:
        return []
    if len(samples) >= count:
        return rng.sample(samples, count)
    return [rng.choice(samples) for _ in range(count)]


def _save_checkpoint(*, output_dir: pathlib.Path, loaded, hl_config: HLMemoryConfig, train_args: TrainArgs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = loaded.model.module if isinstance(loaded.model, torch.nn.parallel.DistributedDataParallel) else loaded.model
    with temporarily_unwrap_proprio_embeddings(model):
        model.save_pretrained(output_dir)
    save_proprio_state_if_available(model, output_dir, hl_config)
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
