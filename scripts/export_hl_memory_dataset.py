from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import dataclasses
import json
import logging
import pathlib
import time
from typing import Any
from typing import Literal

import numpy as np
import torch
import tyro

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import load_annotations_jsonl
from openpi.hl_memory.labels import DEFAULT_LANGUAGE_MEMORY
from openpi.hl_memory.labels import TaskProgressState
from openpi.hl_memory.labels import derive_keyframe_positions
from openpi.hl_memory.labels import render_language_memory
from openpi.hl_memory.labels import update_progress_state
from openpi.hl_memory.memory import EpisodicKeyframeMemory
from openpi.hl_memory.memory import build_recent_context_indices
from openpi.hl_memory.memory import map_relative_positions_to_absolute
from openpi.hl_memory.frame_composer import compose_observation_frame
import openpi.training.config as training_config
import openpi.training.data_loader as data_loader


@dataclasses.dataclass
class ExportArgs:
    source_config_name: str
    annotations_jsonl: pathlib.Path
    output_dir: pathlib.Path
    visual_mode: Literal["raw", "config"] = "raw"
    recent_frames_length: int = 8
    frame_subsample: int = 5
    memory_length: int = 8
    merge_distance: int = 5
    frame_height: int = 224
    frame_width: int = 224
    overwrite: bool = False


def main(args: ExportArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists and is not empty. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frames").mkdir(exist_ok=True)

    start_time = time.perf_counter()
    logging.info("Resolving training config: %s", args.source_config_name)
    config = training_config.get_config(args.source_config_name)
    config = _resolve_hl_visual_config(config, visual_mode=args.visual_mode)
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(
        "Creating LeRobot dataset for HL export: repo_id=%s visual_mode=%s",
        data_config.repo_id,
        args.visual_mode,
    )
    raw_dataset = data_loader.create_torch_dataset(data_config, _hl_action_horizon(config, args.visual_mode), config.model)
    logging.info("LeRobot dataset ready in %.1fs; len=%d", time.perf_counter() - start_time, len(raw_dataset))
    export_dataset = data_loader.TransformedDataset(
        raw_dataset,
        [*data_config.repack_transforms.inputs, *_hl_export_input_transforms(data_config, args.visual_mode)],
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
    frame_cache: dict[int, str] = {}
    samples: list[ExportedHLMemorySample] = []
    hl_config = HLMemoryConfig(
        recent_frames_length=args.recent_frames_length,
        frame_subsample=args.frame_subsample,
        memory_length=args.memory_length,
        merge_distance=args.merge_distance,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
    )

    sorted_episode_items = sorted(episode_to_annotations.items())
    for episode_offset, (episode_index, episode_annotations) in enumerate(sorted_episode_items, start=1):
        if episode_index not in episode_to_indices:
            raise ValueError(f"Episode {episode_index} was not found in dataset {args.source_config_name}.")
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
                output_dir=args.output_dir,
                frame_cache=frame_cache,
                hl_config=hl_config,
            )
        )

    _write_jsonl(args.output_dir / "samples.jsonl", [sample.to_dict() for sample in samples])
    metadata = {
        "schema_version": "hl_memory_v1",
        "source_config_name": args.source_config_name,
        "annotations_jsonl": str(args.annotations_jsonl),
        "visual_mode": args.visual_mode,
        "num_samples": len(samples),
        "hl_memory_config": dataclasses.asdict(hl_config),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")
    logging.info(
        "Exported %d HL memory samples and %d unique frames to %s in %.1fs",
        len(samples),
        len(frame_cache),
        args.output_dir,
        time.perf_counter() - start_time,
    )


def _resolve_hl_visual_config(config: Any, *, visual_mode: str) -> Any:
    if visual_mode == "config":
        logging.warning("HL export visual_mode=config may expose mask overlays or mask images to the HL VLM.")
        return config
    if visual_mode != "raw":
        raise ValueError(f"Unsupported visual_mode: {visual_mode}")

    data_factory = config.data
    if hasattr(data_factory, "guidance_image_mode"):
        logging.info("HL export forcing raw visual mode for %s.", type(data_factory).__name__)
        return dataclasses.replace(config, data=dataclasses.replace(data_factory, guidance_image_mode="raw"))
    return config


def _hl_export_input_transforms(data_config: Any, visual_mode: str) -> list[Any]:
    if visual_mode == "config":
        return list(data_config.data_transforms.inputs)
    return [_StripGuidanceMasksForHLExport()]


def _hl_action_horizon(config: Any, visual_mode: str) -> int:
    if visual_mode == "config":
        return int(config.model.action_horizon)
    return 1


def _export_episode(
    *,
    episode_index: int,
    episode_annotations: list[Any],
    global_indices: list[int],
    dataset: data_loader.Dataset,
    output_dir: pathlib.Path,
    frame_cache: dict[int, str],
    hl_config: HLMemoryConfig,
) -> list[ExportedHLMemorySample]:
    episode_start_time = time.perf_counter()
    progress_state = TaskProgressState()
    keyframe_memory = EpisodicKeyframeMemory(
        memory_length=hl_config.memory_length,
        merge_distance=hl_config.merge_distance,
    )
    samples: list[ExportedHLMemorySample] = []

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
        )
        samples.append(sample)

        absolute_keyframes, _ = map_relative_positions_to_absolute(keyframe_candidate_positions, recent_local_indices)
        keyframe_memory.add_candidates(absolute_keyframes)
        progress_state = next_progress_state

    return samples


def _build_episode_index_map(dataset: Any) -> dict[int, list[int]]:
    base_dataset = _unwrap_base_dataset(dataset)
    if not hasattr(base_dataset, "hf_dataset"):
        raise ValueError("Could not locate the base LeRobot dataset with `hf_dataset` attached.")
    episode_column = np.asarray(base_dataset.hf_dataset["episode_index"]).astype(int)
    grouped: dict[int, list[int]] = defaultdict(list)
    for global_index, episode_index in enumerate(episode_column.tolist()):
        grouped[int(episode_index)].append(global_index)
    return dict(grouped)


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


@dataclasses.dataclass(frozen=True)
class _StripGuidanceMasksForHLExport:
    """Keep HL visual inputs clean: no masks, overlays, or LL model-specific padding images."""

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(sample)
        out.pop("mask", None)
        images = out.get("image")
        if isinstance(images, Mapping):
            out["image"] = {
                key: value
                for key, value in images.items()
                if "mask" not in str(key).lower() and "overlay" not in str(key).lower()
            }
        return out


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
        raise ValueError(f"Sample {global_index} did not expose any image views after transforms.")
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


def _collect_image_views(tree: Mapping[str, Any], *, prefix: str = "") -> dict[str, np.ndarray | torch.Tensor]:
    images: dict[str, np.ndarray | torch.Tensor] = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if _is_image_array(value):
            images[path] = value
            continue
        if isinstance(value, Mapping):
            images.update(_collect_image_views(value, prefix=path))
    return images


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


if __name__ == "__main__":
    main(tyro.cli(ExportArgs))
