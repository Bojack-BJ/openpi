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
from openpi.hl_memory.eval import group_samples_by_episode
from openpi.hl_memory.eval import run_offline_rollout
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
    parallel_mode: str = "none"
    device_map: str = "auto"
    tensor_parallel_plan: str = "auto"
    output_json: pathlib.Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    frame_cache_size: int = 512
    progress: bool = True
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
        enable_thinking=args.enable_thinking,
        thinking_budget_tokens=args.thinking_budget_tokens,
        thinking_max_new_tokens=args.thinking_max_new_tokens,
        parallel_mode=args.parallel_mode,
        device_map=args.device_map,
        tensor_parallel_plan=args.tensor_parallel_plan,
    )
    logging.info("[stage] creating adapter")
    adapter = create_hf_adapter(hl_config)
    logging.info("[stage] loading model from %s onto device=%s", resolved_model_path, args.device)
    loaded = adapter.load(model_path=str(resolved_model_path), device=args.device)
    logging.info("[stage] model ready; starting offline rollout evaluation")
    frame_cache = FrameCache(args.frame_cache_size)

    def predict(sample):
        clips = load_video_clips_for_sample(sample, args.dataset_dir, hl_config, frame_cache=frame_cache)
        return adapter.predict(loaded, sample, clips, device=args.device)

    grouped = group_samples_by_episode(samples)
    modes = _parse_eval_modes(args.eval_modes)
    metrics = _evaluate_with_progress(
        grouped,
        hl_config,
        predict,
        modes=modes,
        show_progress=args.progress,
        total_samples=len(samples),
    )
    rendered = json.dumps(metrics, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
        logging.info("[stage] wrote metrics to %s", args.output_json)


def _evaluate_with_progress(
    grouped,
    hl_config: HLMemoryConfig,
    predict,
    *,
    modes: tuple[AblationMode, ...],
    show_progress: bool,
    total_samples: int,
):
    metrics = {}
    for mode in modes:
        logging.info("[stage] evaluating mode=%s samples=%d episodes=%d", mode, total_samples, len(grouped))
        progress_bar = tqdm(
            total=total_samples,
            desc=f"HL eval {mode}",
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        )

        def predict_with_progress(sample):
            progress_bar.set_postfix(
                episode=sample.episode_index,
                step=sample.step_index,
                refresh=False,
            )
            try:
                return predict(sample)
            finally:
                progress_bar.update(1)

        mode_started_at = time.perf_counter()
        try:
            metrics[mode] = run_offline_rollout(
                grouped,
                hl_config,
                predict_with_progress,
                mode=mode,
            )
        finally:
            progress_bar.close()
        logging.info("[stage] finished mode=%s in %.1fs", mode, time.perf_counter() - mode_started_at)
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
