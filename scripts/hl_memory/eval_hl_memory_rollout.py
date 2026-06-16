from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time

import torch
from tqdm.auto import tqdm
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.config_io import resolve_cli_args_with_yaml
from openpi.hl_memory.data import FrameCache
from openpi.hl_memory.data import load_video_clips_for_sample
from openpi.hl_memory.data import load_exported_samples
from openpi.hl_memory.eval import AblationMode
from openpi.hl_memory.eval import EvalRolloutOptions
from openpi.hl_memory.eval import group_samples_by_episode
from openpi.hl_memory.eval import run_offline_rollout_batched
from openpi.hl_memory.hf_adapter import create_hf_adapter


@dataclasses.dataclass
class EvalArgs:
    dataset_dir: pathlib.Path
    model_path: pathlib.Path | None = None
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
    proprio_enabled: bool = False
    proprio_token_mode: str = "per_frame_plus_summary"
    proprio_state_dim: int = 14
    proprio_hidden_dim: int = 512
    proprio_dropout: float = 0.0
    proprio_noise_std: float = 0.0
    output_json: pathlib.Path | None = None
    prediction_jsonl: pathlib.Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    frame_cache_size: int = 512
    progress: bool = True
    eval_batch_size: int = 1
    eval_batch_log_interval: int = 10
    apply_rollout_memory_rule: bool = False
    known_prior_eval: bool = False
    known_prior_advance_threshold: float = 0.65
    known_prior_match_threshold: float = 0.62
    known_prior_max_advance_steps: int = 3
    eval_modes: str = "no_memory,language_memory_only,keyframe_memory_only,full"
    episode_indices: str | None = None
    exclude_episode_indices: str | None = None
    max_episodes: int | None = None
    max_samples: int | None = None


def main(args: EvalArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started_at = time.perf_counter()
    logging.info("[stage] loading exported samples from %s", args.dataset_dir)
    samples = load_exported_samples(args.dataset_dir)
    original_sample_count = len(samples)
    samples = _filter_samples(
        samples,
        episode_indices=args.episode_indices,
        exclude_episode_indices=args.exclude_episode_indices,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
    )
    logging.info(
        "[stage] loaded %d samples, selected %d samples in %.1fs",
        original_sample_count,
        len(samples),
        time.perf_counter() - started_at,
    )
    if not samples:
        raise ValueError("No samples selected for eval.")
    resolved_model_path = args.model_path or args.local_vlm_ckpt_path
    if resolved_model_path is None:
        raise ValueError("Set either `model_path` or `local_vlm_ckpt_path`.")
    logging.info("[stage] building HL config for backend=%s variant=%s", args.vlm_backend, args.vlm_variant)
    hl_config = HLMemoryConfig(
        vlm_backend=args.vlm_backend,
        vlm_variant=args.vlm_variant,
        vlm_hf_model_id=args.vlm_hf_model_id or str(resolved_model_path),
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
        proprio_enabled=args.proprio_enabled,
        proprio_token_mode=args.proprio_token_mode,
        proprio_state_dim=args.proprio_state_dim,
        proprio_hidden_dim=args.proprio_hidden_dim,
        proprio_dropout=args.proprio_dropout,
        proprio_noise_std=args.proprio_noise_std,
    )
    logging.info("[stage] creating adapter")
    adapter = create_hf_adapter(hl_config)
    logging.info("[stage] loading model from %s onto device=%s", resolved_model_path, args.device)
    loaded = adapter.load(model_path=str(resolved_model_path), device=args.device)
    logging.info("[stage] model ready; starting offline rollout evaluation")
    frame_cache = FrameCache(args.frame_cache_size)

    def predict_batch(batch_samples):
        batch_clips = [
            load_video_clips_for_sample(sample, args.dataset_dir, hl_config, frame_cache=frame_cache)
            for sample in batch_samples
        ]
        return adapter.predict_batch(loaded, list(batch_samples), batch_clips, device=args.device)

    grouped = group_samples_by_episode(samples)
    modes = _parse_eval_modes(args.eval_modes)
    rollout_options = EvalRolloutOptions(
        apply_rollout_memory_rule=args.apply_rollout_memory_rule,
        known_prior_eval=args.known_prior_eval,
        known_prior_advance_threshold=args.known_prior_advance_threshold,
        known_prior_match_threshold=args.known_prior_match_threshold,
        known_prior_max_advance_steps=args.known_prior_max_advance_steps,
    )
    prediction_writer = _PredictionJsonlWriter(args.prediction_jsonl, target_protocol=hl_config.target_protocol)
    try:
        metrics = _evaluate_with_progress(
            grouped,
            hl_config,
            predict_batch,
            modes=modes,
            batch_size=args.eval_batch_size,
            batch_log_interval=args.eval_batch_log_interval,
            rollout_options=rollout_options,
            prediction_writer=prediction_writer,
            show_progress=args.progress,
            total_samples=len(samples),
        )
    finally:
        prediction_writer.close()
    rendered = json.dumps(metrics, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
        logging.info("[stage] wrote metrics to %s", args.output_json)


def _evaluate_with_progress(
    grouped,
    hl_config: HLMemoryConfig,
    predict_batch,
    *,
    modes: tuple[AblationMode, ...],
    batch_size: int,
    batch_log_interval: int,
    rollout_options: EvalRolloutOptions,
    prediction_writer: "_PredictionJsonlWriter",
    show_progress: bool,
    total_samples: int,
):
    if batch_size <= 0:
        raise ValueError(f"--eval-batch-size must be positive, got {batch_size}")
    if batch_log_interval < 0:
        raise ValueError(f"--eval-batch-log-interval must be >= 0, got {batch_log_interval}")
    metrics = {}
    for mode in modes:
        logging.info(
            "[stage] evaluating mode=%s samples=%d episodes=%d batch_size=%d",
            mode,
            total_samples,
            len(grouped),
            batch_size,
        )
        progress_bar = tqdm(
            total=total_samples,
            desc=f"HL eval {mode}",
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        batch_stats = {
            "generate_batches": 0,
            "samples": 0,
            "batch_sum": 0,
            "batch_max": 0,
        }

        def on_batch_start(actual_batch_size: int) -> None:
            batch_stats["generate_batches"] += 1
            batch_stats["samples"] += actual_batch_size
            batch_stats["batch_sum"] += actual_batch_size
            batch_stats["batch_max"] = max(batch_stats["batch_max"], actual_batch_size)
            batch_mean = batch_stats["batch_sum"] / max(batch_stats["generate_batches"], 1)
            progress_bar.set_postfix(
                batch=actual_batch_size,
                batch_mean=f"{batch_mean:.2f}",
                batch_max=batch_stats["batch_max"],
                gen=batch_stats["generate_batches"],
                refresh=False,
            )
            if batch_log_interval and batch_stats["generate_batches"] % batch_log_interval == 0:
                logging.info(
                    "[stage] mode=%s generate_batches=%d samples=%d requested_batch=%d "
                    "last_batch=%d actual_batch_mean=%.2f actual_batch_max=%d",
                    mode,
                    batch_stats["generate_batches"],
                    batch_stats["samples"],
                    batch_size,
                    actual_batch_size,
                    batch_mean,
                    batch_stats["batch_max"],
                )

        def on_sample_done(sample):
            progress_bar.update(1)

        def on_prediction(mode, runtime_sample, source_sample, prediction, sample_metrics):
            prediction_writer.write(
                mode=mode,
                runtime_sample=runtime_sample,
                source_sample=source_sample,
                prediction=prediction,
                metrics=sample_metrics,
            )

        mode_started_at = time.perf_counter()
        try:
            metrics[mode] = run_offline_rollout_batched(
                grouped,
                hl_config,
                predict_batch,
                mode=mode,
                batch_size=batch_size,
                on_sample_done=on_sample_done,
                on_batch_start=on_batch_start,
                rollout_options=rollout_options,
                on_prediction=on_prediction,
            )
        finally:
            progress_bar.close()
        mode_metrics = metrics[mode]
        logging.info(
            "[stage] finished mode=%s in %.1fs requested_batch=%.0f actual_batch_mean=%.2f "
            "actual_batch_max=%.0f generate_batches=%.0f",
            mode,
            time.perf_counter() - mode_started_at,
            mode_metrics.get("eval_batch_size_requested", float(batch_size)),
            mode_metrics.get("actual_eval_batch_size_mean", 0.0),
            mode_metrics.get("actual_eval_batch_size_max", 0.0),
            mode_metrics.get("num_generate_batches", 0.0),
        )
    return metrics


def _filter_samples(
    samples,
    *,
    episode_indices: str | None,
    exclude_episode_indices: str | None,
    max_episodes: int | None,
    max_samples: int | None,
):
    include = _parse_episode_index_spec(episode_indices)
    exclude = _parse_episode_index_spec(exclude_episode_indices) or set()
    filtered = [sample for sample in samples if sample.episode_index not in exclude]
    if include is not None:
        filtered = [sample for sample in filtered if sample.episode_index in include]
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError(f"--max-episodes must be >= 1, got {max_episodes}")
        selected_episodes = []
        seen = set()
        for sample in filtered:
            if sample.episode_index in seen:
                continue
            seen.add(sample.episode_index)
            selected_episodes.append(sample.episode_index)
            if len(selected_episodes) >= max_episodes:
                break
        allowed = set(selected_episodes)
        filtered = [sample for sample in filtered if sample.episode_index in allowed]
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError(f"--max-samples must be >= 1, got {max_samples}")
        filtered = filtered[:max_samples]
        logging.warning(
            "--max-samples can cut episodes mid-sequence; use --max-episodes for cleaner rollout metrics."
        )
    return filtered


class _PredictionJsonlWriter:
    def __init__(self, path: pathlib.Path | None, *, target_protocol: str):
        self._handle = None
        self._target_protocol = target_protocol
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8")

    def write(self, *, mode, runtime_sample, source_sample, prediction, metrics) -> None:
        if self._handle is None:
            return
        expected = source_sample.target_prediction(target_protocol=self._target_protocol)
        payload = {
            "mode": mode,
            "sample_id": source_sample.sample_id,
            "episode_index": source_sample.episode_index,
            "step_index": source_sample.step_index,
            "frame_index": source_sample.frame_index,
            "instruction": source_sample.instruction,
            "step_prior": list(source_sample.step_prior),
            "runtime_language_memory": runtime_sample.language_memory,
            "runtime_memory_frame_indices": list(runtime_sample.memory_frame_indices),
            "recent_frame_indices": list(source_sample.recent_frame_indices),
            "expected": expected.to_dict(),
            "prediction": prediction.to_dict(),
            "metrics": metrics,
        }
        self._handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _parse_eval_modes(value: str) -> tuple[AblationMode, ...]:
    allowed: set[AblationMode] = {"no_memory", "language_memory_only", "keyframe_memory_only", "full"}
    modes = tuple(chunk.strip() for chunk in value.split(",") if chunk.strip())
    if not modes:
        raise ValueError("--eval-modes selected no modes")
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise ValueError(f"Unsupported --eval-modes value(s): {unknown}; allowed={sorted(allowed)}")
    return modes  # type: ignore[return-value]


def _parse_episode_index_spec(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    result: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid episode range {chunk!r}: end < start")
            result.update(range(start, end + 1))
        else:
            result.add(int(chunk))
    return result


if __name__ == "__main__":
    main(resolve_cli_args_with_yaml(EvalArgs, tyro))
