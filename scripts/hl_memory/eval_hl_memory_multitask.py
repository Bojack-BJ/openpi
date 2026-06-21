from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import dataclasses
import json
import logging
import pathlib
import time
from typing import NamedTuple

import torch
from tqdm.auto import tqdm
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.config_io import resolve_cli_args_with_yaml
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import FrameCache
from openpi.hl_memory.data import load_exported_samples
from openpi.hl_memory.data import load_video_clips_for_sample
from openpi.hl_memory.eval import AblationMode
from openpi.hl_memory.eval import compute_prediction_metrics
from openpi.hl_memory.labels import DEFAULT_LANGUAGE_MEMORY
from openpi.hl_memory.memory import EpisodicKeyframeMemory
from openpi.hl_memory.memory import map_relative_positions_to_absolute
from openpi.hl_memory.hf_adapter import create_hf_adapter


class RuntimeSample(NamedTuple):
    dataset_dir: pathlib.Path
    sample: ExportedHLMemorySample


@dataclasses.dataclass
class EvalArgs:
    dataset_root: pathlib.Path
    model_path: pathlib.Path
    dataset_glob: str = "*/val"
    output_json: pathlib.Path | None = None
    predictions_jsonl: pathlib.Path | None = None
    config_yaml: pathlib.Path | None = None
    vlm_backend: str = "qwen2_5_vl"
    vlm_variant: str | None = None
    vlm_hf_model_id: str | None = None
    precision: str = "bfloat16"
    enable_thinking: bool = False
    thinking_budget_tokens: int = 128
    thinking_max_new_tokens: int = 1024
    training_fps: float = 20.0
    frame_subsample: int = 5
    recent_sample_hz: float = 2.0
    parallel_mode: str = "none"
    device_map: str = "auto"
    tensor_parallel_plan: str = "auto"
    target_protocol: str = "hl_v1"
    proprio_enabled: bool = False
    proprio_token_mode: str = "per_frame_plus_summary"
    proprio_state_dim: int = 14
    proprio_hidden_dim: int = 512
    proprio_dropout: float = 0.0
    proprio_noise_std: float = 0.0
    keyframe_event_band_before_sec: float = 1.0
    keyframe_event_band_after_sec: float = 0.5
    keyframe_candidate_label_mode: str = "event_band"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    frame_cache_size: int = 512
    eval_modes: str = "sample_context"
    max_tasks: int | None = None
    max_episodes_per_task: int | None = None
    max_samples: int | None = None
    progress: bool = True


def main(args: EvalArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    started_at = time.perf_counter()
    dataset_dirs = _resolve_dataset_dirs(args)
    items = _load_runtime_samples(dataset_dirs, args)
    if not items:
        raise ValueError("No samples selected for eval.")
    logging.info("[stage] selected %d samples from %d dataset dirs in %.1fs", len(items), len(dataset_dirs), time.perf_counter() - started_at)

    hl_config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_variant=args.vlm_variant,
        vlm_hf_model_id=args.vlm_hf_model_id or str(args.model_path),
        precision=args.precision,
        training_fps=args.training_fps,
        frame_subsample=args.frame_subsample,
        recent_sample_hz=args.recent_sample_hz,
        enable_thinking=args.enable_thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
        thinking_max_new_tokens=args.thinking_max_new_tokens,
        parallel_mode=args.parallel_mode,
        device_map=args.device_map,
        tensor_parallel_plan=args.tensor_parallel_plan,
        target_protocol=args.target_protocol,
        proprio_enabled=args.proprio_enabled,
        proprio_token_mode=args.proprio_token_mode,
        proprio_state_dim=args.proprio_state_dim,
        proprio_hidden_dim=args.proprio_hidden_dim,
        proprio_dropout=args.proprio_dropout,
        proprio_noise_std=args.proprio_noise_std,
        keyframe_event_band_before_sec=args.keyframe_event_band_before_sec,
        keyframe_event_band_after_sec=args.keyframe_event_band_after_sec,
        keyframe_candidate_label_mode=args.keyframe_candidate_label_mode,
    )
    adapter = create_hf_adapter(hl_config)
    logging.info("[stage] loading model from %s", args.model_path)
    loaded = adapter.load(model_path=str(args.model_path), device=args.device)
    frame_cache = FrameCache(args.frame_cache_size)
    grouped = _group_runtime_samples(items)
    modes = _parse_eval_modes(args.eval_modes)
    prediction_rows: list[dict[str, object]] = []

    def predict(item: RuntimeSample):
        clips = load_video_clips_for_sample(item.sample, item.dataset_dir, hl_config, frame_cache=frame_cache)
        return adapter.predict(loaded, item.sample, clips, device=args.device)

    metrics = {}
    for mode in modes:
        if mode == "sample_context":
            metrics[mode] = _run_sample_context_eval(
                items,
                hl_config,
                predict,
                prediction_rows=prediction_rows if mode == modes[-1] else None,
                show_progress=args.progress,
            )
        else:
            metrics[mode] = _run_multitask_rollout(
                grouped,
                hl_config,
                predict,
                mode=mode,
                prediction_rows=prediction_rows if mode == modes[-1] else None,
                show_progress=args.progress,
            )

    result = {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "dataset_glob": args.dataset_glob,
        "target_protocol": hl_config.target_protocol,
        "num_dataset_dirs": len(dataset_dirs),
        "num_samples": len(items),
        "metrics": metrics,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
        logging.info("[stage] wrote metrics to %s", args.output_json)
    if args.predictions_jsonl is not None:
        args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_jsonl.open("w") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        logging.info("[stage] wrote predictions to %s", args.predictions_jsonl)


def _resolve_dataset_dirs(args: EvalArgs) -> list[pathlib.Path]:
    root = args.dataset_root.expanduser()
    dirs = sorted(path for path in root.glob(args.dataset_glob) if (path / "samples.jsonl").exists())
    if args.max_tasks is not None:
        dirs = dirs[: args.max_tasks]
    if not dirs:
        raise ValueError(f"No eval dirs matched {root / args.dataset_glob}")
    return [path.resolve() for path in dirs]


def _load_runtime_samples(dataset_dirs: Sequence[pathlib.Path], args: EvalArgs) -> list[RuntimeSample]:
    items: list[RuntimeSample] = []
    for dataset_dir in dataset_dirs:
        samples = load_exported_samples(dataset_dir)
        if args.max_episodes_per_task is not None:
            allowed = []
            seen = set()
            for sample in samples:
                if sample.episode_index in seen:
                    continue
                seen.add(sample.episode_index)
                allowed.append(sample.episode_index)
                if len(allowed) >= args.max_episodes_per_task:
                    break
            allowed_set = set(allowed)
            samples = [sample for sample in samples if sample.episode_index in allowed_set]
        for sample in samples:
            items.append(RuntimeSample(dataset_dir=dataset_dir, sample=sample))
            if args.max_samples is not None and len(items) >= args.max_samples:
                return items
    return items


def _group_runtime_samples(items: Sequence[RuntimeSample]) -> dict[tuple[str, int], list[RuntimeSample]]:
    grouped: dict[tuple[str, int], list[RuntimeSample]] = defaultdict(list)
    for item in items:
        grouped[(str(item.dataset_dir), item.sample.episode_index)].append(item)
    return dict(grouped)


def _run_multitask_rollout(
    grouped: Mapping[tuple[str, int], Sequence[RuntimeSample]],
    hl_config: HLMemoryConfig,
    predict_fn,
    *,
    mode: AblationMode,
    prediction_rows: list[dict[str, object]] | None,
    show_progress: bool,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    total_steps = 0
    exact_sequence_episodes = 0
    progress = tqdm(total=sum(len(items) for items in grouped.values()), desc=f"HL eval {mode}", unit="sample", disable=not show_progress)
    try:
        for items in grouped.values():
            memory = EpisodicKeyframeMemory(memory_length=hl_config.memory_length, merge_distance=hl_config.merge_distance)
            frame_lookup = {
                frame_index: frame_path
                for item in items
                for frame_index, frame_path in zip(item.sample.recent_frame_indices, item.sample.recent_frame_paths)
            }
            language_memory = DEFAULT_LANGUAGE_MEMORY
            sequence_ok = True
            for item in items:
                runtime_sample = _with_ablation_context(
                    item.sample,
                    mode=mode,
                    memory=memory,
                    frame_lookup=frame_lookup,
                    language_memory=language_memory,
                )
                runtime_item = RuntimeSample(dataset_dir=item.dataset_dir, sample=runtime_sample)
                prediction = predict_fn(runtime_item)
                metrics = compute_prediction_metrics(
                    prediction,
                    item.sample,
                    target_protocol=hl_config.target_protocol,
                    keyframe_candidate_label_mode=hl_config.keyframe_candidate_label_mode,
                )
                for key, value in metrics.items():
                    totals[key] += value
                sequence_ok &= _normalize(prediction.current_objective) == _normalize(
                    item.sample.target_prediction(
                        target_protocol=hl_config.target_protocol,
                        keyframe_candidate_label_mode=hl_config.keyframe_candidate_label_mode,
                    ).current_objective
                )
                total_steps += 1
                if prediction_rows is not None:
                    prediction_rows.append(
                        _prediction_row(
                            item,
                            prediction,
                            metrics,
                            target_protocol=hl_config.target_protocol,
                            keyframe_candidate_label_mode=hl_config.keyframe_candidate_label_mode,
                        )
                    )
                if mode in ("language_memory_only", "full"):
                    language_memory = prediction.updated_language_memory
                if mode in ("keyframe_memory_only", "full"):
                    absolute_positions, _ = map_relative_positions_to_absolute(
                        prediction.keyframe_candidate_positions,
                        item.sample.recent_frame_indices,
                    )
                    memory.add_candidates(absolute_positions)
                progress.update(1)
            exact_sequence_episodes += int(sequence_ok)
    finally:
        progress.close()
    averaged = {key: value / max(total_steps, 1) for key, value in totals.items()}
    averaged["episode_sequence_accuracy"] = exact_sequence_episodes / max(len(grouped), 1)
    averaged["num_steps"] = float(total_steps)
    averaged["num_episodes"] = float(len(grouped))
    return dict(averaged)


def _run_sample_context_eval(
    items: Sequence[RuntimeSample],
    hl_config: HLMemoryConfig,
    predict_fn,
    *,
    prediction_rows: list[dict[str, object]] | None,
    show_progress: bool,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    grouped = _group_runtime_samples(items)
    progress = tqdm(total=len(items), desc="HL eval sample_context", unit="sample", disable=not show_progress)
    try:
        for item in items:
            prediction = predict_fn(item)
            metrics = compute_prediction_metrics(
                prediction,
                item.sample,
                target_protocol=hl_config.target_protocol,
                keyframe_candidate_label_mode=hl_config.keyframe_candidate_label_mode,
            )
            for key, value in metrics.items():
                totals[key] += value
            if prediction_rows is not None:
                prediction_rows.append(
                    _prediction_row(
                        item,
                        prediction,
                        metrics,
                        target_protocol=hl_config.target_protocol,
                        keyframe_candidate_label_mode=hl_config.keyframe_candidate_label_mode,
                    )
                )
            progress.update(1)
    finally:
        progress.close()
    total_steps = len(items)
    averaged = {key: value / max(total_steps, 1) for key, value in totals.items()}
    averaged["num_steps"] = float(total_steps)
    averaged["num_episodes"] = float(len(grouped))
    return dict(averaged)


def _with_ablation_context(sample, *, mode, memory, frame_lookup, language_memory):
    runtime_language_memory = language_memory if mode in ("language_memory_only", "full") else DEFAULT_LANGUAGE_MEMORY
    if mode in ("keyframe_memory_only", "full"):
        memory_indices = tuple(index for index in memory.visible_indices(sample.recent_frame_indices) if index in frame_lookup)
        memory_paths = tuple(frame_lookup[index] for index in memory_indices)
    else:
        memory_indices = ()
        memory_paths = ()
    return sample.with_runtime_context(
        language_memory=runtime_language_memory,
        memory_frame_paths=memory_paths,
        memory_frame_indices=memory_indices,
    )


def _prediction_row(
    item: RuntimeSample,
    prediction,
    metrics: dict[str, float],
    *,
    target_protocol: str = "hl_v1",
    keyframe_candidate_label_mode: str = "event_band",
) -> dict[str, object]:
    sample = item.sample
    return {
        "task_id": item.dataset_dir.parent.name,
        "dataset_dir": str(item.dataset_dir),
        "sample_id": sample.sample_id,
        "episode_index": sample.episode_index,
        "step_index": sample.step_index,
        "instruction": sample.instruction,
        "expected": sample.target_prediction(
            target_protocol=target_protocol,
            keyframe_candidate_label_mode=keyframe_candidate_label_mode,
        ).to_dict(),
        "prediction": prediction.to_dict(),
        "metrics": metrics,
    }


def _parse_eval_modes(value: str):
    allowed = {"sample_context", "no_memory", "language_memory_only", "keyframe_memory_only", "full"}
    modes = tuple(chunk.strip() for chunk in value.split(",") if chunk.strip())
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise ValueError(f"Unsupported --eval-modes: {unknown}")
    return modes  # type: ignore[return-value]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


if __name__ == "__main__":
    main(resolve_cli_args_with_yaml(EvalArgs, tyro))
