from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import dataclasses
import io
import json
import logging
import pathlib
import types
import time
from typing import Any
from typing import Literal

import datasets
from datasets import load_dataset
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import load_annotations_jsonl
from openpi.hl_memory.labels import DEFAULT_LANGUAGE_MEMORY
from openpi.hl_memory.labels import TaskProgressState
from openpi.hl_memory.labels import derive_keyframe_positions
from openpi.hl_memory.labels import render_language_memory
from openpi.hl_memory.labels import render_language_memory_fields_from_state
from openpi.hl_memory.labels import update_progress_state
from openpi.hl_memory.memory import EpisodicKeyframeMemory
from openpi.hl_memory.memory import build_recent_context_indices
from openpi.hl_memory.memory import map_relative_positions_to_absolute
from openpi.hl_memory.frame_composer import compose_observation_frame
import openpi.training.data_loader as data_loader
import openpi.training.lerobot_dataset as openpi_lerobot_dataset
import openpi.transforms as _transforms


@dataclasses.dataclass
class ExportArgs:
    annotations_jsonl: pathlib.Path = tyro.MISSING
    source_config_name: str | None = None
    output_dir: pathlib.Path | None = None
    repo_id_override: str | None = None
    asset_id_override: str | None = None
    subtask_segments_path_override: str | None = None
    force_prompt_from_task: bool = True
    output_train_dir: pathlib.Path | None = None
    output_val_dir: pathlib.Path | None = None
    visual_mode: Literal["raw", "config"] = "raw"
    image_columns: str = "auto"
    missing_episode_policy: Literal["error", "skip"] = "error"
    episode_split: Literal["all", "train", "val"] = "all"
    val_ratio: float = 0.1
    split_seed: int = 42
    episode_indices: str | None = None
    exclude_episode_indices: str | None = None
    max_episodes: int | None = None
    recent_frames_length: int = 8
    training_fps: float = 20.0
    frame_subsample: int = 5
    recent_sample_hz: float = 2.0
    memory_length: int = 8
    merge_distance: int = 5
    frame_height: int = 224
    frame_width: int = 456
    subtask_progress_quantum: float = 0.05
    overwrite: bool = False


def main(args: ExportArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    export_targets = _resolve_export_targets(args)
    for _split_name, output_dir in export_targets:
        _validate_output_dir(output_dir, overwrite=args.overwrite)

    start_time = time.perf_counter()
    data_config = _resolve_export_data_config(args)
    logging.info(
        "Creating LeRobot dataset for HL export: repo_id=%s visual_mode=%s image_columns=%s",
        data_config.repo_id,
        args.visual_mode,
        args.image_columns,
    )
    raw_dataset = _create_hl_export_dataset(data_config, visual_mode=args.visual_mode, image_columns=args.image_columns)
    logging.info("HL export dataset ready in %.1fs; len=%d", time.perf_counter() - start_time, len(raw_dataset))
    export_dataset = data_loader.TransformedDataset(
        raw_dataset,
        [_HLVisualOnlyTransform()],
    )

    logging.info("Loading HL annotations: %s", args.annotations_jsonl)
    annotations = load_annotations_jsonl(args.annotations_jsonl)
    episode_to_annotations = _group_annotations_by_episode(annotations)
    logging.info(
        "Loaded %d annotations across %d episodes.",
        len(annotations),
        len(episode_to_annotations),
    )
    logging.info("Building episode index map from LeRobot dataset.")
    episode_to_indices = _build_episode_index_map(raw_dataset)
    logging.info("Episode index map ready for %d episodes.", len(episode_to_indices))
    hl_config = HLMemoryConfig(
        recent_frames_length=args.recent_frames_length,
        training_fps=args.training_fps,
        frame_subsample=args.frame_subsample,
        recent_sample_hz=args.recent_sample_hz,
        memory_length=args.memory_length,
        merge_distance=args.merge_distance,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
    )

    all_episode_items = sorted(episode_to_annotations.items())
    for split_name, output_dir in export_targets:
        split_start_time = time.perf_counter()
        sorted_episode_items = _filter_episode_items(
            all_episode_items,
            episode_split=split_name,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            episode_indices=args.episode_indices,
            exclude_episode_indices=args.exclude_episode_indices,
            max_episodes=args.max_episodes,
        )
        logging.info(
            "Selected %d/%d annotation episodes for split=%s val_ratio=%.3f split_seed=%d -> %s.",
            len(sorted_episode_items),
            len(episode_to_annotations),
            split_name,
            args.val_ratio,
            args.split_seed,
            output_dir,
        )
        _export_split(
            split_name=split_name,
            output_dir=output_dir,
            sorted_episode_items=sorted_episode_items,
            args=args,
            raw_dataset=raw_dataset,
            export_dataset=export_dataset,
            episode_to_indices=episode_to_indices,
            hl_config=hl_config,
            elapsed_start_time=split_start_time,
        )

    logging.info("HL export complete for %d output target(s) in %.1fs", len(export_targets), time.perf_counter() - start_time)


def _export_split(
    *,
    split_name: str,
    output_dir: pathlib.Path,
    sorted_episode_items: list[tuple[int, list[Any]]],
    args: ExportArgs,
    raw_dataset: Any,
    export_dataset: data_loader.Dataset,
    episode_to_indices: dict[int, list[int]],
    hl_config: HLMemoryConfig,
    elapsed_start_time: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    frame_cache: dict[int, str] = {}
    samples: list[ExportedHLMemorySample] = []
    skipped_episodes: list[int] = []
    episode_progress = tqdm(sorted_episode_items, desc=f"HL export {split_name}", dynamic_ncols=True, unit="episode")
    for episode_offset, (episode_index, episode_annotations) in enumerate(episode_progress, start=1):
        if episode_index not in episode_to_indices:
            message = _format_missing_episode_error(
                episode_index,
                source_config_name=args.source_config_name or f"repo_id={getattr(_unwrap_base_dataset(raw_dataset), 'repo_id', '<unknown>')}",
                available_episode_indices=episode_to_indices.keys(),
                dataset=_unwrap_base_dataset(raw_dataset),
            )
            if args.missing_episode_policy == "skip":
                skipped_episodes.append(episode_index)
                logging.warning("%s Skipping because missing_episode_policy=skip.", message)
                continue
            raise ValueError(message)
        episode_progress.set_postfix(
            episode=episode_index,
            annotations=len(episode_annotations),
            cached_frames=len(frame_cache),
        )
        logging.info(
            "Exporting episode %d/%d: episode_index=%d annotations=%d frames=%d cached_frames=%d",
            episode_offset,
            len(sorted_episode_items),
            episode_index,
            len(episode_annotations),
            len(episode_to_indices[episode_index]),
            len(frame_cache),
        )
        samples.extend(
            _export_episode(
                episode_index=episode_index,
                episode_annotations=episode_annotations,
                global_indices=episode_to_indices[episode_index],
                dataset=export_dataset,
                output_dir=output_dir,
                frame_cache=frame_cache,
                hl_config=hl_config,
                subtask_progress_quantum=args.subtask_progress_quantum,
            )
        )

    if skipped_episodes:
        logging.warning("Skipped %d annotation episodes missing from dataset: %s", len(skipped_episodes), skipped_episodes[:20])

    _write_jsonl(output_dir / "samples.jsonl", [sample.to_dict() for sample in samples])
    metadata = {
        "schema_version": "hl_memory_v1",
        "source_config_name": args.source_config_name,
        "annotations_jsonl": str(args.annotations_jsonl),
        "visual_mode": args.visual_mode,
        "image_columns": args.image_columns,
        "episode_split": split_name,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "episode_indices": args.episode_indices,
        "exclude_episode_indices": args.exclude_episode_indices,
        "max_episodes": args.max_episodes,
        "selected_episode_indices": [episode_index for episode_index, _ in sorted_episode_items],
        "num_samples": len(samples),
        "skipped_missing_episode_indices": skipped_episodes,
        "export_config": _export_config_metadata(hl_config),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")
    logging.info(
        "Exported %d HL memory samples and %d unique frames to %s in %.1fs",
        len(samples),
        len(frame_cache),
        output_dir,
        time.perf_counter() - elapsed_start_time,
    )


def _apply_export_overrides(config: Any, args: ExportArgs) -> Any:
    training_config = _load_training_config()
    if (
        args.repo_id_override is None
        and args.asset_id_override is None
        and args.subtask_segments_path_override is None
        and not args.force_prompt_from_task
    ):
        return config

    data_factory = config.data
    base_config = getattr(data_factory, "base_config", None) or training_config.DataConfig()
    subtask_segments_path = (
        args.subtask_segments_path_override
        if args.subtask_segments_path_override is not None
        else base_config.subtask_segments_path
    )
    if args.repo_id_override is not None and subtask_segments_path is None:
        subtask_segments_path = "subtask_segments.json"

    base_config = dataclasses.replace(
        base_config,
        prompt_from_task=True if args.force_prompt_from_task else base_config.prompt_from_task,
        subtask_segments_path=subtask_segments_path,
    )

    replacements: dict[str, Any] = {"base_config": base_config}
    if args.repo_id_override is not None:
        replacements["repo_id"] = args.repo_id_override
    if args.asset_id_override is not None or args.repo_id_override is not None:
        assets = getattr(data_factory, "assets", training_config.AssetsConfig())
        replacements["assets"] = dataclasses.replace(
            assets,
            asset_id=args.asset_id_override or args.repo_id_override,
        )
    return dataclasses.replace(config, data=dataclasses.replace(data_factory, **replacements))


def _export_config_metadata(hl_config: HLMemoryConfig) -> dict[str, Any]:
    return {
        "recent_frames_length": hl_config.recent_frames_length,
        "training_fps": hl_config.training_fps,
        "frame_subsample": hl_config.frame_subsample,
        "recent_sample_hz": hl_config.recent_sample_hz,
        "recent_window_sec": hl_config.recent_window_sec,
        "recent_step_sec": hl_config.recent_step_sec,
        "video_fps": hl_config.video_fps,
        "memory_length": hl_config.memory_length,
        "merge_distance": hl_config.merge_distance,
        "frame_height": hl_config.frame_height,
        "frame_width": hl_config.frame_width,
    }


def _resolve_hl_visual_config(config: Any, *, visual_mode: str) -> Any:
    if visual_mode == "config":
        logging.info("HL export visual_mode=config still loads only RGB image columns; masks and overlays are excluded.")
        return config
    if visual_mode != "raw":
        raise ValueError(f"Unsupported visual_mode: {visual_mode}")

    data_factory = config.data
    if hasattr(data_factory, "guidance_image_mode"):
        logging.info("HL export forcing raw visual mode for %s.", type(data_factory).__name__)
        return dataclasses.replace(config, data=dataclasses.replace(data_factory, guidance_image_mode="raw"))
    return config


def _resolve_export_targets(args: ExportArgs) -> list[tuple[str, pathlib.Path]]:
    paired_mode = args.output_train_dir is not None or args.output_val_dir is not None
    if paired_mode:
        if args.output_dir is not None:
            raise ValueError("Use either --output-dir or --output-train-dir/--output-val-dir, not both.")
        if args.output_train_dir is None or args.output_val_dir is None:
            raise ValueError("Paired export requires both --output-train-dir and --output-val-dir.")
        if args.episode_split != "all":
            raise ValueError("Paired export writes both train and val; leave --episode-split as all.")
        return [
            ("train", args.output_train_dir),
            ("val", args.output_val_dir),
        ]

    if args.output_dir is None:
        raise ValueError("Set --output-dir, or set both --output-train-dir and --output-val-dir.")
    return [(args.episode_split, args.output_dir)]


def _validate_output_dir(output_dir: pathlib.Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Use --overwrite to replace it.")


def _resolve_export_data_config(args: ExportArgs) -> Any:
    if args.source_config_name is None:
        if args.repo_id_override is None:
            raise ValueError("Set --repo-id-override when exporting without --source-config-name.")
        logging.info("Using explicit HL subtask schema without an OpenPI training config.")
        return types.SimpleNamespace(
            repo_id=args.repo_id_override,
            prompt_from_task=args.force_prompt_from_task,
            subtask_segments_path=args.subtask_segments_path_override or "subtask_segments.json",
            dataset_columns=None,
        )

    logging.info("Resolving training config: %s", args.source_config_name)
    training_config = _load_training_config()
    config = training_config.get_config(args.source_config_name)
    config = _apply_export_overrides(config, args)
    config = _resolve_hl_visual_config(config, visual_mode=args.visual_mode)
    return config.data.create(config.assets_dirs, config.model)


def _load_training_config() -> Any:
    import openpi.training.config as training_config  # pylint: disable=import-outside-toplevel

    return training_config


def _create_hl_export_dataset(data_config: Any, *, visual_mode: str, image_columns: str) -> data_loader.Dataset:
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create HL export dataset.")
    if repo_id == "fake":
        raise ValueError("HL export does not support fake datasets.")

    dataset_root = openpi_lerobot_dataset.HF_LEROBOT_HOME / repo_id
    dataset_meta = _load_lerobot_metadata(repo_id, dataset_root)
    selected_columns = _resolve_hl_export_columns(data_config, dataset_meta, image_columns=image_columns)
    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    logging.info(
        "Loading HL parquet dataset directly without LeRobot timestamp/delta checks; root=%s parquet_files=%d columns=%s",
        dataset_root,
        len(parquet_files),
        ", ".join(selected_columns),
    )
    hf_dataset = load_dataset("parquet", data_files=[str(path) for path in parquet_files], split="train", columns=list(selected_columns))
    dataset: data_loader.Dataset = _HLExportParquetDataset(repo_id=repo_id, root=dataset_root, hf_dataset=hf_dataset)

    if data_config.prompt_from_task:
        dataset = data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    if data_config.subtask_segments_path is not None:
        dataset = data_loader.TransformedDataset(
            dataset,
            [
                _transforms.SubtaskFromSegments(
                    data_loader._load_subtask_segments(data_config.subtask_segments_path, dataset_root)  # pylint: disable=protected-access
                )
            ],
        )
    return dataset


@dataclasses.dataclass(frozen=True)
class _LocalLeRobotMetadata:
    features: dict[str, Any]
    tasks: dict[int, str]


def _load_lerobot_metadata(repo_id: str, dataset_root: pathlib.Path) -> Any:
    try:
        return openpi_lerobot_dataset.LeRobotDatasetMetadata(repo_id, dataset_root)
    except Exception as exc:  # LeRobot v2.1 may query Hugging Face refs before reading local metadata.
        logging.warning("Falling back to local meta files for %s because metadata init failed: %s", repo_id, exc)
        info_path = dataset_root / "meta" / "info.json"
        tasks_path = dataset_root / "meta" / "tasks.jsonl"
        if not info_path.is_file():
            raise FileNotFoundError(info_path) from exc
        with info_path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
        tasks: dict[int, str] = {}
        if tasks_path.is_file():
            with tasks_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tasks[int(row["task_index"])] = str(row["task"])
        return _LocalLeRobotMetadata(features=dict(info.get("features", {})), tasks=tasks)


def _resolve_hl_export_columns(data_config: Any, dataset_meta: Any, *, image_columns: str) -> tuple[str, ...]:
    features = dict(getattr(dataset_meta, "features", {}) or {})
    existing_columns = set(features)
    selected: list[str] = []

    for column in ("timestamp", "frame_index", "episode_index", "index", "task_index", "subtask", "prompt"):
        _append_existing_column(selected, column, existing_columns)

    mode_or_columns = (image_columns or "auto").strip()
    if mode_or_columns == "config":
        configured_columns = tuple(data_config.dataset_columns or ())
        for column in configured_columns:
            if column.startswith("observation.images.") and not _is_guidance_visual_column(column):
                _append_existing_column(selected, column, existing_columns)
        if not any(column.startswith("observation.images.") for column in selected):
            for column in sorted(existing_columns):
                if column.startswith("observation.images.") and not _is_guidance_visual_column(column):
                    _append_existing_column(selected, column, existing_columns)
    elif mode_or_columns in ("auto", "all", "all_rgb"):
        for column in sorted(existing_columns):
            if column.startswith("observation.images."):
                if _is_guidance_visual_column(column):
                    continue
                _append_existing_column(selected, column, existing_columns)
    else:
        requested = _parse_image_columns(mode_or_columns)
        missing: list[str] = []
        for column in requested:
            before = len(selected)
            if column.startswith("observation.images.") and not _is_guidance_visual_column(column):
                _append_existing_column(selected, column, existing_columns)
            if len(selected) == before:
                missing.append(column)
        if missing:
            raise ValueError(
                "Requested image columns are missing from LeRobot metadata: "
                f"{missing}. Available image columns: {_available_image_columns(existing_columns)}"
            )

    if not any(column.startswith("observation.images.") for column in selected):
        raise ValueError(
            "Could not resolve any image columns for HL export. "
            f"image_columns={image_columns!r}, available={_available_image_columns(existing_columns)}"
        )
    return tuple(selected)


def _parse_image_columns(value: str) -> tuple[str, ...]:
    columns: list[str] = []
    for item in value.split(","):
        column = item.strip()
        if not column:
            continue
        if not column.startswith("observation.images."):
            column = f"observation.images.{column}"
        columns.append(column)
    if not columns:
        raise ValueError("--image-columns must be `auto`, `config`, or a comma-separated image column list.")
    return tuple(columns)


def _available_image_columns(existing_columns: set[str]) -> list[str]:
    return [
        column
        for column in sorted(existing_columns)
        if column.startswith("observation.images.") and not _is_guidance_visual_column(column)
    ]


def _append_existing_column(selected: list[str], column: str, existing_columns: set[str]) -> None:
    if column in existing_columns and column not in selected:
        selected.append(column)
        return
    fallback = data_loader._DATASET_COLUMN_FALLBACKS.get(column)  # pylint: disable=protected-access
    if fallback in existing_columns and fallback not in selected:
        selected.append(fallback)


def _is_guidance_visual_column(column: str) -> bool:
    lowered = column.lower()
    return "mask" in lowered or "overlay" in lowered


@dataclasses.dataclass
class _HLExportParquetDataset:
    repo_id: str
    root: pathlib.Path
    hf_dataset: datasets.Dataset

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.hf_dataset[int(index)]

    def __len__(self) -> int:
        return len(self.hf_dataset)


def _export_episode(
    *,
    episode_index: int,
    episode_annotations: list[Any],
    global_indices: list[int],
    dataset: data_loader.Dataset,
    output_dir: pathlib.Path,
    frame_cache: dict[int, str],
    hl_config: HLMemoryConfig,
    subtask_progress_quantum: float,
) -> list[ExportedHLMemorySample]:
    episode_start_time = time.perf_counter()
    progress_state = TaskProgressState()
    keyframe_memory = EpisodicKeyframeMemory(
        memory_length=hl_config.memory_length,
        merge_distance=hl_config.merge_distance,
    )
    samples: list[ExportedHLMemorySample] = []
    episode_step_prior = _episode_step_prior(episode_annotations)

    for step_index, annotation in enumerate(episode_annotations):
        if step_index > 0 and step_index % 50 == 0:
            logging.info(
                "Episode %d progress: %d/%d annotations, cached_frames=%d, elapsed=%.1fs",
                episode_index,
                step_index,
                len(episode_annotations),
                len(frame_cache),
                time.perf_counter() - episode_start_time,
            )
        if annotation.frame_index >= len(global_indices):
            raise ValueError(
                f"Annotation frame_index={annotation.frame_index} exceeds episode {episode_index} length {len(global_indices)}."
            )
        recent_local_indices = build_recent_context_indices(
            timestep=annotation.frame_index,
            frame_subsample=hl_config.frame_subsample,
            recent_frames_length=hl_config.recent_frames_length,
            recent_window_frames=int(round(hl_config.recent_window_sec * hl_config.training_fps)),
        )
        recent_global_indices = [global_indices[index] for index in recent_local_indices]
        recent_frame_paths = tuple(
            _ensure_frame_saved(global_index, dataset[global_index], output_dir=output_dir, cache=frame_cache, hl_config=hl_config)
            for global_index in recent_global_indices
        )
        memory_frame_indices = tuple(
            index for index in keyframe_memory.visible_indices(recent_local_indices) if global_indices[index] in frame_cache
        )
        memory_frame_paths = tuple(frame_cache[global_indices[index]] for index in memory_frame_indices)

        current_language_memory = render_language_memory(progress_state)
        next_progress_state = update_progress_state(progress_state, annotation)
        updated_language_memory = render_language_memory(next_progress_state)
        next_memory_fields = render_language_memory_fields_from_state(next_progress_state)
        keyframe_candidate_positions = derive_keyframe_positions(episode_annotations, step_index, recent_local_indices)
        instruction = annotation.instruction or _extract_instruction(dataset[global_indices[annotation.frame_index]])
        sample = ExportedHLMemorySample(
            sample_id=f"episode_{episode_index:06d}_step_{step_index:06d}",
            episode_index=episode_index,
            step_index=step_index,
            frame_index=annotation.frame_index,
            instruction=instruction,
            language_memory=current_language_memory or DEFAULT_LANGUAGE_MEMORY,
            updated_language_memory=updated_language_memory,
            current_subtask=annotation.current_subtask,
            phase=annotation.phase or annotation.current_subtask,
            target_query=annotation.target_query,
            goal_query=annotation.goal_query,
            keyframe_candidate_positions=keyframe_candidate_positions,
            memory_frame_paths=memory_frame_paths,
            memory_frame_indices=memory_frame_indices,
            memory_valid_length=len(memory_frame_paths),
            recent_frame_paths=recent_frame_paths,
            recent_frame_indices=tuple(recent_local_indices),
            recent_valid_length=len(recent_frame_paths),
            event_type=annotation.event_type,
            event_text=annotation.event_text,
            step_prior=episode_step_prior,
            task_progress=str(next_memory_fields["task_progress"]),
            current_objective=str(next_memory_fields["current_objective"]),
            relevant_objects=tuple(next_memory_fields["relevant_objects"]),  # type: ignore[arg-type]
            notes=str(next_memory_fields["notes"]),
            subtask_progress=_quantize_optional_progress(annotation.subtask_progress, subtask_progress_quantum),
            should_advance_objective=annotation.should_advance_objective,
            active_hand=annotation.active_hand,
            keyframe_label=annotation.keyframe_label,
            horizon_frame_index=annotation.horizon_frame_index,
            horizon_current_objective=annotation.horizon_current_objective,
            horizon_current_subtask=annotation.horizon_current_subtask,
            horizon_phase=annotation.horizon_phase,
        )
        samples.append(sample)

        absolute_keyframes, _ = map_relative_positions_to_absolute(keyframe_candidate_positions, recent_local_indices)
        keyframe_memory.add_candidates(absolute_keyframes)
        progress_state = next_progress_state

    return samples


def _episode_step_prior(episode_annotations: list[Any]) -> tuple[str, ...]:
    steps: list[str] = []
    previous = ""
    for annotation in sorted(episode_annotations, key=lambda item: int(item.frame_index)):
        current = str(annotation.current_subtask).strip()
        normalized = " ".join(current.lower().split())
        if not current or normalized == previous:
            continue
        steps.append(current)
        previous = normalized
    return tuple(steps)


def _build_episode_index_map(dataset: Any) -> dict[int, list[int]]:
    base_dataset = _unwrap_base_dataset(dataset)
    if not hasattr(base_dataset, "hf_dataset"):
        raise ValueError("Could not locate the base LeRobot dataset with `hf_dataset` attached.")
    episode_column = np.asarray(base_dataset.hf_dataset["episode_index"]).astype(int)
    grouped: dict[int, list[int]] = defaultdict(list)
    for global_index, episode_index in enumerate(episode_column.tolist()):
        grouped[int(episode_index)].append(global_index)
    return dict(grouped)


def _format_missing_episode_error(
    episode_index: int,
    *,
    source_config_name: str,
    available_episode_indices: Any,
    dataset: Any | None = None,
) -> str:
    available = sorted(int(index) for index in available_episode_indices)
    if available:
        available_summary = f"available_count={len(available)} range=[{available[0]}, {available[-1]}]"
    else:
        available_summary = "available_count=0"
    file_summary = _inspect_episode_file(dataset, episode_index)
    return (
        f"Episode {episode_index} was not found in dataset {source_config_name} ({available_summary}). "
        f"{file_summary} "
        "This usually means annotations.jsonl was generated from raw session ordering or from a different LeRobot repo. "
        "Regenerate annotations from the LeRobot sidecar with `scripts/hl_memory/export_hl_annotations_from_subtasks.py --repo-id ...`, "
        "or pass `--missing-episode-policy skip` only if a partial export is intended."
    )


def _inspect_episode_file(dataset: Any | None, episode_index: int) -> str:
    root = getattr(dataset, "root", None)
    if root is None:
        return "Could not inspect episode parquet files."
    matches = sorted(pathlib.Path(root).glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if not matches:
        return f"No parquet file named episode_{episode_index:06d}.parquet exists under {root / 'data'}."

    summaries: list[str] = []
    for path in matches[:3]:
        try:
            table = pq.read_table(path, columns=["episode_index"])
            values = np.asarray(table.column("episode_index")).astype(int)
            unique_values = sorted(set(values.tolist()))
            summaries.append(f"{path}: episode_index_values={unique_values[:5]} rows={len(values)}")
        except Exception as exc:  # pragma: no cover - diagnostic path.
            summaries.append(f"{path}: could not read episode_index ({exc})")
    return "Matching parquet file(s) exist, but their contents may not match: " + "; ".join(summaries)


def _unwrap_base_dataset(dataset: Any) -> Any:
    current = dataset
    while hasattr(current, "_dataset"):
        if hasattr(current, "hf_dataset"):
            return current
        current = current._dataset
    return current


def _group_annotations_by_episode(annotations: list[Any]) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.episode_index].append(annotation)
    return dict(grouped)


def _filter_episode_items(
    episode_items: list[tuple[int, list[Any]]],
    *,
    episode_split: str,
    val_ratio: float,
    split_seed: int,
    episode_indices: str | None,
    exclude_episode_indices: str | None,
    max_episodes: int | None,
) -> list[tuple[int, list[Any]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in [0, 1), got {val_ratio}")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError(f"--max-episodes must be >= 1, got {max_episodes}")

    include = _parse_episode_index_spec(episode_indices)
    exclude = _parse_episode_index_spec(exclude_episode_indices) or set()
    filtered = [(episode_index, annotations) for episode_index, annotations in episode_items if episode_index not in exclude]
    if include is not None:
        filtered = [(episode_index, annotations) for episode_index, annotations in filtered if episode_index in include]

    if episode_split != "all":
        val_indices = _deterministic_val_episode_indices(
            [episode_index for episode_index, _ in filtered],
            val_ratio=val_ratio,
            split_seed=split_seed,
        )
        if episode_split == "val":
            filtered = [(episode_index, annotations) for episode_index, annotations in filtered if episode_index in val_indices]
        elif episode_split == "train":
            filtered = [(episode_index, annotations) for episode_index, annotations in filtered if episode_index not in val_indices]
        else:
            raise ValueError(f"Unsupported --episode-split: {episode_split}")

    if max_episodes is not None:
        filtered = filtered[:max_episodes]
    return filtered


def _deterministic_val_episode_indices(episode_indices: list[int], *, val_ratio: float, split_seed: int) -> set[int]:
    if not episode_indices or val_ratio <= 0.0:
        return set()
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(sorted(episode_indices), dtype=np.int64)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled))
    return set(int(value) for value in shuffled[:val_count].tolist())


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


@dataclasses.dataclass(frozen=True)
class _HLVisualOnlyTransform:
    """Return only fields needed by HL export: RGB views, prompt text, and optional subtask."""

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        observation = sample.get("observation", {})
        images = observation.get("images", {}) if isinstance(observation, Mapping) else {}
        out: dict[str, Any] = {"image": {}}
        if isinstance(images, Mapping):
            out["image"] = _select_hl_image_views(images)
        for key, value in sample.items():
            key_text = str(key)
            if not key_text.startswith("observation.images.") or _is_guidance_visual_column(key_text):
                continue
            view_name = key_text.removeprefix("observation.images.")
            out["image"].setdefault(view_name, value)
        for key in ("prompt", "subtask", "current_subtask", "episode_index", "frame_index", "task_index"):
            if key in sample:
                out[key] = sample[key]
        return out


def _select_hl_image_views(images: Mapping[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, value in images.items():
        key_text = str(key)
        if _is_guidance_visual_column(key_text):
            continue
        selected[key_text] = value
    return selected


def _ensure_frame_saved(
    global_index: int,
    sample: Mapping[str, Any],
    *,
    output_dir: pathlib.Path,
    cache: dict[int, str],
    hl_config: HLMemoryConfig,
) -> str:
    if global_index in cache:
        return cache[global_index]
    image_views = _collect_image_views(sample)
    if not image_views:
        sample_keys = ", ".join(str(key) for key in sample.keys())
        raise ValueError(f"Sample {global_index} did not expose any image views after transforms. keys=[{sample_keys}]")
    composed = compose_observation_frame(
        image_views,
        frame_height=hl_config.frame_height,
        frame_width=hl_config.frame_width,
    )
    relative_path = pathlib.Path("frames") / f"frame_{global_index:08d}.png"
    composed.save(output_dir / relative_path)
    cache[global_index] = str(relative_path)
    return cache[global_index]


def _extract_instruction(sample: Mapping[str, Any]) -> str:
    for key in ("prompt", "instruction", "task"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, bytes) and value.strip():
            return value.decode("utf-8").strip()
    for value in sample.values():
        if isinstance(value, Mapping):
            nested = _extract_instruction(value)
            if nested:
                return nested
    return ""


def _collect_image_views(tree: Mapping[str, Any], *, prefix: str = "") -> dict[str, Image.Image | np.ndarray | torch.Tensor]:
    images: dict[str, Image.Image | np.ndarray | torch.Tensor] = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        image = _coerce_image_value(value)
        if image is not None:
            images[path] = image
            continue
        if isinstance(value, Mapping):
            images.update(_collect_image_views(value, prefix=path))
    return images


def _coerce_image_value(value: Any) -> Image.Image | np.ndarray | torch.Tensor | None:
    if isinstance(value, Image.Image) or isinstance(value, torch.Tensor) or _is_image_array(value):
        return value
    if isinstance(value, Mapping):
        image_bytes = value.get("bytes")
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return None


def _is_image_array(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        shape = tuple(value.shape)
    else:
        try:
            shape = tuple(np.asarray(value).shape)
        except Exception:
            return False
    if len(shape) != 3:
        return False
    channel_candidates = {shape[0], shape[-1]}
    return any(channel in channel_candidates for channel in (1, 3, 4))


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _quantize_optional_progress(value: float | None, quantum: float) -> float | None:
    if value is None:
        return None
    if quantum <= 0.0:
        return float(min(max(value, 0.0), 1.0))
    clipped = min(max(float(value), 0.0), 1.0)
    quantized = round(clipped / quantum) * quantum
    return float(round(min(max(quantized, 0.0), 1.0), 6))


if __name__ == "__main__":
    main(tyro.cli(ExportArgs))
