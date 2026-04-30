#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect a converted LeRobot dataset.

Examples:
    python dataprocess_new/inspect_lerobot_dataset.py \
        --repo-id fastumi/20260312H081Ba_toy_block

    python dataprocess_new/inspect_lerobot_dataset.py \
        --dataset-root /path/to/lerobot_home/fastumi/20260312H081Ba_toy_block \
        --sample-indices 0,100,-1 \
        --preview-dir /tmp/lerobot_preview
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    from lerobot.common.constants import HF_LEROBOT_HOME
except Exception:
    from lerobot.constants import HF_LEROBOT_HOME

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _infer_repo_id(dataset_root: Path | None, repo_id: str | None) -> str:
    if repo_id:
        return repo_id
    if dataset_root is None:
        raise ValueError("Either --repo-id or --dataset-root must be provided.")
    try:
        return dataset_root.resolve().relative_to(Path(HF_LEROBOT_HOME).resolve()).as_posix()
    except ValueError:
        return dataset_root.name


def _load_metadata(repo_id: str, dataset_root: Path | None) -> LeRobotDatasetMetadata:
    kwargs: dict[str, Any] = {}
    if dataset_root is not None:
        kwargs["root"] = dataset_root
    try:
        return LeRobotDatasetMetadata(repo_id, **kwargs)
    except TypeError:
        if dataset_root is None:
            return LeRobotDatasetMetadata(repo_id)
        return LeRobotDatasetMetadata(repo_id, dataset_root)


def _load_dataset(repo_id: str, dataset_root: Path | None, episodes: list[int] | None) -> LeRobotDataset:
    kwargs: dict[str, Any] = {"repo_id": repo_id}
    if dataset_root is not None:
        kwargs["root"] = dataset_root
    if episodes is not None:
        kwargs["episodes"] = episodes
    try:
        return LeRobotDataset(**kwargs, download_videos=False)
    except TypeError:
        return LeRobotDataset(**kwargs)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _dtype_of(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def _to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except TypeError:
            pass
    return None


def _numeric_range(value: Any) -> tuple[float, float] | None:
    array = _to_numpy(value)
    if array is None or array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return None
    return float(np.nanmin(array)), float(np.nanmax(array))


def _short_value(value: Any, max_len: int = 160) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = repr(value)
    elif hasattr(value, "item") and _shape_of(value) in (None, ()):
        try:
            text = repr(value.item())
        except (TypeError, ValueError):
            text = repr(value)
    else:
        text = repr(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _flatten(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(child_prefix, child)
    else:
        yield prefix, value


def _summarize_value(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    shape = _shape_of(value)
    dtype = _dtype_of(value)
    value_range = _numeric_range(value)
    if shape is not None:
        summary["shape"] = shape
    if dtype is not None:
        summary["dtype"] = dtype
    if value_range is not None:
        summary["min"] = value_range[0]
        summary["max"] = value_range[1]
    if shape is None or shape == ():
        summary["value"] = _short_value(value)
    return summary


def _print_mapping(title: str, mapping: Any, *, max_items: int) -> None:
    print(f"\n[{title}]")
    if mapping is None:
        print("  <missing>")
        return
    if hasattr(mapping, "to_dict"):
        mapping = mapping.to_dict()
    if isinstance(mapping, Mapping):
        items = list(mapping.items())
        for key, value in items[:max_items]:
            print(f"  {key}: {json.dumps(value, ensure_ascii=False, default=_json_default)}")
        if len(items) > max_items:
            print(f"  ... ({len(items) - max_items} more)")
        return
    print(f"  {json.dumps(mapping, ensure_ascii=False, default=_json_default)}")


def _episode_rows(dataset: LeRobotDataset, max_episodes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_data_index = getattr(dataset, "episode_data_index", None)
    episodes_meta = getattr(getattr(dataset, "meta", None), "episodes", None)

    starts = episode_data_index.get("from") if isinstance(episode_data_index, Mapping) else None
    ends = episode_data_index.get("to") if isinstance(episode_data_index, Mapping) else None
    starts_np = _to_numpy(starts)
    ends_np = _to_numpy(ends)

    if starts_np is not None and ends_np is not None:
        for episode_index, (start, end) in enumerate(zip(starts_np.tolist(), ends_np.tolist(), strict=False)):
            row: dict[str, Any] = {
                "episode_index": episode_index,
                "from": int(start),
                "to": int(end),
                "num_frames": int(end) - int(start),
            }
            if isinstance(episodes_meta, list) and episode_index < len(episodes_meta):
                episode_meta = episodes_meta[episode_index]
                if isinstance(episode_meta, Mapping):
                    for key in ("tasks", "length", "num_frames"):
                        if key in episode_meta:
                            row[f"meta.{key}"] = episode_meta[key]
            rows.append(row)
            if len(rows) >= max_episodes:
                break
    return rows


def _parse_int_list(text: str | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_sample_indices(text: str, dataset_len: int) -> list[int]:
    if dataset_len <= 0:
        return []
    indices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        index = int(part)
        if index < 0:
            index = dataset_len + index
        if not 0 <= index < dataset_len:
            raise IndexError(f"sample index out of range: {part} for dataset length {dataset_len}")
        if index not in indices:
            indices.append(index)
    return indices


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _image_array(value: Any) -> np.ndarray | None:
    array = _to_numpy(value)
    if array is None or array.ndim not in (2, 3):
        return None

    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
        return None

    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.floating):
        min_value = float(np.nanmin(array)) if array.size else 0.0
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if min_value >= -1.0 and max_value <= 1.0:
            if min_value < 0.0:
                array = (array + 1.0) * 127.5
            else:
                array = array * 255.0
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8, copy=False)


def _save_previews(sample: Mapping[str, Any], sample_index: int, preview_dir: Path) -> list[Path]:
    try:
        from PIL import Image
    except Exception as exc:
        print(f"[Preview] Pillow is not available, skip image export: {exc}")
        return []

    preview_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key, value in _flatten("", sample):
        lowered = key.lower()
        if "image" not in lowered and "mask" not in lowered:
            continue
        array = _image_array(value)
        if array is None:
            continue
        path = preview_dir / f"sample_{sample_index:06d}_{_safe_name(key)}.png"
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths


def _load_subtask_sidecar(root: Path | None) -> Any:
    if root is None:
        return None
    path = root / "subtask_segments.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a converted LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, default=None, help="LeRobot repo id, e.g. fastumi/task_xxx.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Local dataset root. Overrides HF_LEROBOT_HOME lookup.")
    parser.add_argument("--episodes", type=str, default=None, help="Optional comma-separated episode indices to load.")
    parser.add_argument("--sample-indices", type=str, default="0,-1", help="Comma-separated global frame indices to inspect.")
    parser.add_argument("--max-features", type=int, default=80, help="Max feature/task entries to print.")
    parser.add_argument("--max-episodes", type=int, default=20, help="Max episode rows to print.")
    parser.add_argument("--preview-dir", type=Path, default=None, help="Optional directory to save image/mask previews.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write machine-readable summary JSON.")
    args = parser.parse_args()

    repo_id = _infer_repo_id(args.dataset_root, args.repo_id)
    metadata = _load_metadata(repo_id, args.dataset_root)
    episodes = _parse_int_list(args.episodes)
    dataset = _load_dataset(repo_id, args.dataset_root, episodes)
    root = Path(getattr(dataset, "root", args.dataset_root)) if getattr(dataset, "root", args.dataset_root) else None

    print("[Dataset]")
    print(f"  repo_id: {repo_id}")
    print(f"  root: {root}")
    print(f"  length: {len(dataset)} frames")
    print(f"  fps: {getattr(dataset, 'fps', getattr(metadata, 'fps', '<unknown>'))}")
    print(f"  codebase_version: {getattr(metadata, 'codebase_version', getattr(metadata, '_version', '<unknown>'))}")

    _print_mapping("Features", getattr(metadata, "features", None), max_items=args.max_features)
    _print_mapping("Tasks", getattr(metadata, "tasks", None), max_items=args.max_features)

    print("\n[Episodes]")
    episode_rows = _episode_rows(dataset, args.max_episodes)
    if episode_rows:
        for row in episode_rows:
            print(f"  {json.dumps(row, ensure_ascii=False, default=_json_default)}")
    else:
        print("  <episode index unavailable>")

    sidecar = _load_subtask_sidecar(root)
    print("\n[Subtask Sidecar]")
    if sidecar is None:
        print("  <missing>")
    else:
        episodes_payload = sidecar.get("episodes", {}) if isinstance(sidecar, Mapping) else {}
        print(f"  episodes: {len(episodes_payload)}")
        for episode_index, episode_payload in list(episodes_payload.items())[: min(args.max_episodes, 5)]:
            print(f"  episode {episode_index}: {json.dumps(episode_payload, ensure_ascii=False, default=_json_default)}")

    sample_indices = _parse_sample_indices(args.sample_indices, len(dataset))
    samples_summary: dict[int, dict[str, Any]] = {}
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        flat = dict(_flatten("", sample))
        print(f"\n[Sample {sample_index}]")
        sample_summary: dict[str, Any] = {}
        for key, value in flat.items():
            value_summary = _summarize_value(value)
            sample_summary[key] = value_summary
            print(f"  {key}: {json.dumps(value_summary, ensure_ascii=False, default=_json_default)}")
        samples_summary[sample_index] = sample_summary

        if args.preview_dir is not None:
            preview_paths = _save_previews(sample, sample_index, args.preview_dir)
            if preview_paths:
                print("  preview_files:")
                for path in preview_paths:
                    print(f"    {path}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "repo_id": repo_id,
            "root": root,
            "length": len(dataset),
            "fps": getattr(dataset, "fps", getattr(metadata, "fps", None)),
            "features": getattr(metadata, "features", None),
            "tasks": getattr(metadata, "tasks", None),
            "episodes": episode_rows,
            "subtask_sidecar": sidecar,
            "samples": samples_summary,
        }
        with args.json_out.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False, default=_json_default)
            stream.write("\n")
        print(f"\n[JSON] wrote {args.json_out}")


if __name__ == "__main__":
    main()
