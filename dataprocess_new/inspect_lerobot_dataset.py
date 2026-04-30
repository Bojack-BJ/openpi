#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect a converted LeRobot dataset.

Examples:
    python dataprocess_new/inspect_lerobot_dataset.py \
        --repo-id fastumi/20260312H081Ba_toy_block

    python dataprocess_new/inspect_lerobot_dataset.py \
        --dataset-root /root/Users/dataset/lerobot_home/fastumi/sponge_visual_guided \
        --sample-indices 0,100,-1 \
        --preview-dir /tmp/lerobot_preview
    
    python dataprocess_new/inspect_lerobot_dataset.py \
        --dataset-root /root/Users/dataset/lerobot_home/fastumi/sponge_visual_guided \
        --episodes 3 \
        --sample-indices 0,100,-1 \
        --preview-dir /tmp/lerobot_preview
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import re
from typing import Any


def _log(message: str) -> None:
    print(message, flush=True)


def _get_hf_lerobot_home() -> Path:
    try:
        from lerobot.common.constants import HF_LEROBOT_HOME
    except Exception:
        from lerobot.constants import HF_LEROBOT_HOME
    return Path(HF_LEROBOT_HOME)


def _get_lerobot_classes() -> tuple[Any, Any]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    return LeRobotDataset, LeRobotDatasetMetadata


def _infer_repo_id(dataset_root: Path | None, repo_id: str | None) -> str:
    if repo_id:
        return repo_id
    if dataset_root is None:
        raise ValueError("Either --repo-id or --dataset-root must be provided.")
    try:
        return dataset_root.resolve().relative_to(_get_hf_lerobot_home().resolve()).as_posix()
    except ValueError:
        return dataset_root.name


def _load_metadata(repo_id: str, dataset_root: Path | None) -> Any:
    _, metadata_cls = _get_lerobot_classes()
    kwargs: dict[str, Any] = {}
    if dataset_root is not None:
        kwargs["root"] = dataset_root
    try:
        return metadata_cls(repo_id, **kwargs)
    except TypeError:
        if dataset_root is None:
            return metadata_cls(repo_id)
        return metadata_cls(repo_id, dataset_root)


def _load_dataset(repo_id: str, dataset_root: Path | None, episodes: list[int] | None) -> Any:
    dataset_cls, _ = _get_lerobot_classes()
    kwargs: dict[str, Any] = {"repo_id": repo_id}
    if dataset_root is not None:
        kwargs["root"] = dataset_root
    if episodes is not None:
        kwargs["episodes"] = episodes
    try:
        return dataset_cls(**kwargs, download_videos=False)
    except TypeError:
        return dataset_cls(**kwargs)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
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
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except ValueError:
            return None
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
        value = f"<bytes:{len(value)}>"
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


def _read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl_file(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_local_meta(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {}
    meta_dir = root / "meta"
    if not meta_dir.exists():
        return {}

    meta: dict[str, Any] = {}
    for name in ("info", "tasks", "episodes", "episodes_stats", "stats"):
        json_path = meta_dir / f"{name}.json"
        jsonl_path = meta_dir / f"{name}.jsonl"
        if json_path.exists():
            meta[name] = _read_json_file(json_path)
        elif jsonl_path.exists():
            meta[name] = _read_jsonl_file(jsonl_path)
    return meta


def _features_from_meta(local_meta: Mapping[str, Any], metadata: Any | None) -> Any:
    info = local_meta.get("info", {})
    if isinstance(info, Mapping) and "features" in info:
        return info["features"]
    if metadata is not None:
        return getattr(metadata, "features", None)
    return None


def _tasks_from_meta(local_meta: Mapping[str, Any], metadata: Any | None) -> Any:
    tasks = local_meta.get("tasks")
    if tasks is not None:
        return tasks
    if metadata is not None:
        return getattr(metadata, "tasks", None)
    return None


def _fps_from_meta(local_meta: Mapping[str, Any], metadata: Any | None, dataset: Any | None = None) -> Any:
    info = local_meta.get("info", {})
    if isinstance(info, Mapping) and "fps" in info:
        return info["fps"]
    if dataset is not None and hasattr(dataset, "fps"):
        return getattr(dataset, "fps")
    if metadata is not None and hasattr(metadata, "fps"):
        return getattr(metadata, "fps")
    return "<unknown>"


def _version_from_meta(local_meta: Mapping[str, Any], metadata: Any | None) -> Any:
    info = local_meta.get("info", {})
    if isinstance(info, Mapping):
        for key in ("codebase_version", "version"):
            if key in info:
                return info[key]
    if metadata is not None:
        return getattr(metadata, "codebase_version", getattr(metadata, "_version", "<unknown>"))
    return "<unknown>"


def _episode_rows_from_meta(local_meta: Mapping[str, Any], max_episodes: int) -> list[dict[str, Any]]:
    episodes = local_meta.get("episodes")
    if not isinstance(episodes, list):
        return []

    rows: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes[:max_episodes]):
        if isinstance(episode, Mapping):
            row = dict(episode)
            row.setdefault("episode_index", episode.get("episode_index", index))
            rows.append(row)
        else:
            rows.append({"episode_index": index, "value": episode})
    return rows


def _episode_rows(dataset: Any, max_episodes: int) -> list[dict[str, Any]]:
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


def _find_parquet_files(root: Path, episodes: list[int] | None) -> list[Path]:
    data_dir = root / "data"
    files = sorted(data_dir.rglob("*.parquet"))
    if episodes is None:
        return files

    wanted = set(episodes)
    filtered: list[Path] = []
    for path in files:
        match = re.search(r"episode[_-](\d+)", path.stem)
        if match is not None and int(match.group(1)) in wanted:
            filtered.append(path)
    return filtered


def _parquet_file_infos(root: Path, episodes: list[int] | None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required for --backend parquet") from exc

    infos: list[dict[str, Any]] = []
    offset = 0
    for path in _find_parquet_files(root, episodes):
        parquet_file = pq.ParquetFile(path)
        num_rows = int(parquet_file.metadata.num_rows)
        infos.append({"path": path, "from": offset, "to": offset + num_rows, "num_rows": num_rows})
        offset += num_rows
    return infos


def _parquet_len(infos: list[dict[str, Any]]) -> int:
    return int(infos[-1]["to"]) if infos else 0


def _read_parquet_sample(infos: list[dict[str, Any]], sample_index: int) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required for --backend parquet") from exc

    for info in infos:
        if int(info["from"]) <= sample_index < int(info["to"]):
            local_index = sample_index - int(info["from"])
            table = pq.ParquetFile(info["path"]).read()
            rows = table.slice(local_index, 1).to_pylist()
            if not rows:
                raise IndexError(f"No row found for sample index {sample_index}")
            return rows[0]
    raise IndexError(f"sample index out of range: {sample_index}")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _candidate_paths(path_value: Any, dataset_root: Path | None) -> list[Path]:
    path = Path(str(path_value))
    candidates = [path]
    if dataset_root is not None and not path.is_absolute():
        candidates.append(dataset_root / path)
    return candidates


def _decode_video_frame(path: Path, *, frame_index: Any = None, timestamp: Any = None) -> np.ndarray | None:
    try:
        import cv2
    except Exception:
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        target_frame = None
        if frame_index is not None:
            target_frame = int(frame_index)
        elif timestamp is not None:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps > 0:
                target_frame = int(round(float(timestamp) * fps))
        if target_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(target_frame, 0))
        ok, frame_bgr = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _load_image_path(path: Path, *, frame_index: Any = None, timestamp: Any = None) -> np.ndarray | None:
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return _decode_video_frame(path, frame_index=frame_index, timestamp=timestamp)
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _image_array(value: Any, dataset_root: Path | None = None) -> np.ndarray | None:
    if isinstance(value, Mapping):
        if isinstance(value.get("bytes"), (bytes, bytearray, memoryview)):
            try:
                from PIL import Image

                return np.asarray(Image.open(io.BytesIO(bytes(value["bytes"]))).convert("RGB"))
            except Exception:
                return None
        if value.get("path") is not None:
            for candidate in _candidate_paths(value["path"], dataset_root):
                if not candidate.exists():
                    continue
                image = _load_image_path(
                    candidate,
                    frame_index=value.get("frame_index"),
                    timestamp=value.get("timestamp"),
                )
                if image is not None:
                    return image
        return None

    if isinstance(value, (str, Path)):
        for candidate in _candidate_paths(value, dataset_root):
            if not candidate.exists():
                continue
            image = _load_image_path(candidate)
            if image is not None:
                return image
        return None

    if hasattr(value, "__array__") and value.__class__.__module__.startswith("PIL"):
        return np.asarray(value)

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


def _iter_preview_candidates(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    lowered = prefix.lower()
    is_preview_key = prefix and ("image" in lowered or "mask" in lowered)
    if isinstance(value, Mapping):
        if is_preview_key and ("path" in value or "bytes" in value):
            yield prefix, value
            return
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_preview_candidates(child_prefix, child)
        return
    if is_preview_key:
        yield prefix, value


def _save_previews(
    sample: Mapping[str, Any],
    sample_index: int,
    preview_dir: Path,
    dataset_root: Path | None = None,
) -> list[Path]:
    try:
        from PIL import Image
    except Exception as exc:
        print(f"[Preview] Pillow is not available, skip image export: {exc}")
        return []

    preview_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    seen: set[str] = set()
    for key, value in _iter_preview_candidates("", sample):
        if key in seen:
            continue
        array = _image_array(value, dataset_root=dataset_root)
        if array is None:
            continue
        seen.add(key)
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
    parser.add_argument(
        "--backend",
        choices=["parquet", "lerobot"],
        default="parquet",
        help="Use fast direct parquet inspection or full LeRobotDataset loading.",
    )
    parser.add_argument("--metadata-only", action="store_true", help="Only print metadata and sidecar; skip sample loading.")
    parser.add_argument("--max-features", type=int, default=80, help="Max feature/task entries to print.")
    parser.add_argument("--max-episodes", type=int, default=20, help="Max episode rows to print.")
    parser.add_argument("--preview-dir", type=Path, default=None, help="Optional directory to save image/mask previews.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write machine-readable summary JSON.")
    args = parser.parse_args()

    _log("[Start] Inspecting LeRobot dataset")
    global np
    import numpy as np

    repo_id = _infer_repo_id(args.dataset_root, args.repo_id)
    root = args.dataset_root if args.dataset_root is not None else _get_hf_lerobot_home() / repo_id
    _log(f"[Start] repo_id={repo_id}")
    _log(f"[Start] root={root}")

    episodes = _parse_int_list(args.episodes)
    local_meta = _load_local_meta(root)
    metadata = None
    if not local_meta or args.backend == "lerobot":
        _log("[Load] Loading LeRobot metadata...")
        metadata = _load_metadata(repo_id, args.dataset_root)
        _log("[Load] LeRobot metadata loaded.")

    dataset = None
    parquet_infos: list[dict[str, Any]] = []
    if args.metadata_only:
        dataset_len = None
    elif args.backend == "parquet":
        _log("[Load] Scanning parquet files...")
        parquet_infos = _parquet_file_infos(root, episodes)
        dataset_len = _parquet_len(parquet_infos)
        _log(f"[Load] Found {len(parquet_infos)} parquet files, {dataset_len} frames.")
    else:
        _log("[Load] Loading full LeRobotDataset. This can be slow on large datasets...")
        dataset = _load_dataset(repo_id, args.dataset_root, episodes)
        root = Path(getattr(dataset, "root", root))
        dataset_len = len(dataset)
        _log("[Load] Full LeRobotDataset loaded.")

    print("[Dataset]")
    print(f"  repo_id: {repo_id}")
    print(f"  root: {root}")
    if dataset_len is not None:
        print(f"  length: {dataset_len} frames")
    print(f"  fps: {_fps_from_meta(local_meta, metadata, dataset)}")
    print(f"  codebase_version: {_version_from_meta(local_meta, metadata)}")
    print(f"  backend: {args.backend}")

    _print_mapping("Features", _features_from_meta(local_meta, metadata), max_items=args.max_features)
    _print_mapping("Tasks", _tasks_from_meta(local_meta, metadata), max_items=args.max_features)

    print("\n[Episodes]")
    episode_rows = _episode_rows_from_meta(local_meta, args.max_episodes)
    if not episode_rows and dataset is not None:
        episode_rows = _episode_rows(dataset, args.max_episodes)
    if not episode_rows and parquet_infos:
        episode_rows = [
            {
                "file": str(info["path"].relative_to(root)),
                "from": info["from"],
                "to": info["to"],
                "num_frames": info["num_rows"],
            }
            for info in parquet_infos[: args.max_episodes]
        ]
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

    if args.metadata_only:
        sample_indices = []
    else:
        sample_indices = _parse_sample_indices(args.sample_indices, int(dataset_len or 0))
    samples_summary: dict[int, dict[str, Any]] = {}
    for sample_index in sample_indices:
        _log(f"[Sample] Loading sample {sample_index}...")
        if args.backend == "parquet":
            sample = _read_parquet_sample(parquet_infos, sample_index)
        else:
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
            preview_paths = _save_previews(sample, sample_index, args.preview_dir, dataset_root=root)
            if preview_paths:
                print("  preview_files:")
                for path in preview_paths:
                    print(f"    {path}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "repo_id": repo_id,
            "root": root,
            "length": dataset_len,
            "fps": _fps_from_meta(local_meta, metadata, dataset),
            "features": _features_from_meta(local_meta, metadata),
            "tasks": _tasks_from_meta(local_meta, metadata),
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
