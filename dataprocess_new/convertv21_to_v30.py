#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import shutil
import glob
import os
from pathlib import Path
from typing import Any

import jsonlines
import pandas as pd
import pyarrow as pa
import tqdm
from datasets import Dataset, Features, Image

from lerobot.common.datasets.compute_stats import aggregate_stats
import numpy as np

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARQUET_PATH,   # <- 用这个，不是 DEFAULT_DATA_PATH
    DEFAULT_VIDEO_PATH,
    EPISODES_PATH,          # <- 用这个，不是 LEGACY_EPISODES_PATH
    EPISODES_STATS_PATH,    # <- 用这个，不是 LEGACY_EPISODES_STATS_PATH
    TASKS_PATH,             # <- 用这个，不是 LEGACY_TASKS_PATH
    cast_stats_to_numpy,
    flatten_dict,
    load_info,
    write_info,
    write_stats,
    write_episode,          # <- 单条写
    write_episode_stats,    # <- 单条写
    write_task,             # <- 单条写
)
from lerobot.common.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
from lerobot.common.utils.utils import init_logging

DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 500
from lerobot.common.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
from lerobot.common.utils.constants import HF_LEROBOT_HOME
from lerobot.common.utils.utils import init_logging

V21 = "v2.1"
V30 = "v3.0"

"""
-------------------------
OLD
data/chunk-000/episode_000000.parquet

NEW
data/chunk-000/file_000.parquet
-------------------------
OLD
videos/chunk-000/CAMERA/episode_000000.mp4

NEW
videos/CAMERA/chunk-000/file_000.mp4
-------------------------
OLD
episodes.jsonl
{"episode_index": 1, "tasks": ["Put the blue block in the green bowl"], "length": 266}

NEW
meta/episodes/chunk-000/episodes_000.parquet
episode_index | video_chunk_index | video_file_index | data_chunk_index | data_file_index | tasks | length
-------------------------
OLD
tasks.jsonl
{"task_index": 1, "task": "Put the blue block in the green bowl"}

NEW
meta/tasks/chunk-000/file_000.parquet
task_index | task
-------------------------
OLD
episodes_stats.jsonl

NEW
meta/episodes_stats/chunk-000/file_000.parquet
episode_index | mean | std | min | max
-------------------------
UPDATE
meta/info.json
-------------------------
"""


def load_jsonlines(fpath: Path) -> list[Any]:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def legacy_load_episodes(local_dir: Path) -> dict:
    episodes = load_jsonlines(local_dir / LEGACY_EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def legacy_load_episodes_stats(local_dir: Path) -> dict:
    episodes_stats = load_jsonlines(local_dir / LEGACY_EPISODES_STATS_PATH)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def legacy_load_tasks(local_dir: Path) -> tuple[dict, dict]:
    tasks = load_jsonlines(local_dir / LEGACY_TASKS_PATH)
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def validate_local_dataset_version(local_path: Path) -> None:
    """Validate that the local dataset has the expected v2.1 version."""
    info = load_info(local_path)
    dataset_version = info.get("codebase_version", "unknown")
    if dataset_version != V21:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V21}'. "
            f"This script is specifically for converting v2.1 datasets to v3.0."
        )


def convert_tasks(root: Path, new_root: Path):
    logging.info(f"Converting tasks from {root} to {new_root}")
    tasks, _ = legacy_load_tasks(root)
    task_indices = list(tasks.keys())
    task_strings = list(tasks.values())
    df_tasks = pd.DataFrame({"task_index": task_indices}, index=task_strings)
    write_tasks(df_tasks, new_root)


def get_video_keys(root: Path):
    info = load_info(root)
    features = info["features"]
    video_keys = [key for key, ft in features.items() if ft.get("dtype") == "video"]
    return video_keys


def get_image_keys(root: Path):
    info = load_info(root)
    features = info["features"]
    image_keys = [key for key, ft in features.items() if ft.get("dtype") == "image"]
    return image_keys


def concat_data_files(paths_to_cat, new_root: Path, chunk_idx: int, file_idx: int, image_keys):
    import pyarrow.parquet as pq

    # 1) Read all episode tables
    tables = [pq.read_table(f) for f in paths_to_cat]

    # 2) Concatenate (allow minor schema promotion)
    table = pa.concat_tables(tables, promote=True)

    # 3) Build HF Features from arrow schema
    features = Features.from_arrow_schema(table.schema)

    # 4) Override image columns to HF Image() so metadata/schema are correct
    for key in image_keys:
        if key in features:
            features[key] = Image()

    # 5) Convert to arrow schema with updated metadata
    arrow_schema = features.arrow_schema

    # 6) Write parquet
    out_path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.cast(arrow_schema), out_path)


def convert_data(root: Path, new_root: Path, data_file_size_in_mb: int):
    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    if len(ep_paths) == 0:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    image_keys = get_image_keys(root)

    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0.0
    num_frames = 0
    paths_to_cat = []
    episodes_metadata = []

    logging.info(f"Converting data files from {len(ep_paths)} episodes")

    for ep_path in tqdm.tqdm(ep_paths, desc="convert data files"):
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)
        ep_num_frames = get_parquet_num_frames(ep_path)

        ep_metadata = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": num_frames,
            "dataset_to_index": num_frames + ep_num_frames,
        }
        episodes_metadata.append(ep_metadata)

        # accumulate this episode into current target file
        size_in_mb += ep_size_in_mb
        num_frames += ep_num_frames
        paths_to_cat.append(ep_path)
        ep_idx += 1

        # flush when threshold reached
        if size_in_mb >= data_file_size_in_mb:
            concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

            # reset accumulator
            size_in_mb = 0.0
            paths_to_cat = []
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    # flush remaining
    if paths_to_cat:
        concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

    return episodes_metadata


def convert_videos(root: Path, new_root: Path, video_file_size_in_mb: int):
    logging.info(f"Converting videos from {root} to {new_root}")

    video_keys = get_video_keys(root)
    if len(video_keys) == 0:
        logging.info("No video keys found. Skip video conversion.")
        return None

    video_keys = sorted(video_keys)

    eps_metadata_per_cam = []
    for camera in video_keys:
        eps_metadata = convert_videos_of_camera(root, new_root, camera, video_file_size_in_mb)
        eps_metadata_per_cam.append(eps_metadata)

    num_eps_per_cam = [len(eps_cam_map) for eps_cam_map in eps_metadata_per_cam]
    if len(set(num_eps_per_cam)) != 1:
        raise ValueError(f"All cams dont have same number of episodes ({num_eps_per_cam}).")

    episodes_metadata = []
    num_cameras = len(video_keys)
    num_episodes = num_eps_per_cam[0]
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="convert videos"):
        # Sanity check
        ep_ids = [eps_metadata_per_cam[cam_idx][ep_idx]["episode_index"] for cam_idx in range(num_cameras)]
        ep_ids += [ep_idx]
        if len(set(ep_ids)) != 1:
            raise ValueError(f"All episode indices need to match ({ep_ids}).")

        ep_dict = {}
        for cam_idx in range(num_cameras):
            ep_dict.update(eps_metadata_per_cam[cam_idx][ep_idx])
        episodes_metadata.append(ep_dict)

    return episodes_metadata


def convert_videos_of_camera(root: Path, new_root: Path, video_key: str, video_file_size_in_mb: int):
    # Old layout: videos/chunk-000/CAMERA/episode_000000.mp4
    videos_dir = root / "videos"
    ep_paths = sorted(videos_dir.glob(f"*/{video_key}/*.mp4"))

    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0.0
    duration_in_s = 0.0
    paths_to_cat = []
    episodes_metadata = []

    for ep_path in tqdm.tqdm(ep_paths, desc=f"convert videos of {video_key}"):
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        # If adding this episode exceeds file size, flush existing accumulation first
        if size_in_mb + ep_size_in_mb >= video_file_size_in_mb and len(paths_to_cat) > 0:
            out_path = new_root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            concatenate_video_files(paths_to_cat, out_path)

            # Update metadata for the group just flushed
            for i, _ in enumerate(paths_to_cat):
                past_ep_idx = ep_idx - len(paths_to_cat) + i
                episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0.0
            duration_in_s = 0.0
            paths_to_cat = []

        ep_metadata = {
            "episode_index": ep_idx,
            f"videos/{video_key}/chunk_index": chunk_idx,  # may be overwritten on flush (same value usually)
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": duration_in_s,
            f"videos/{video_key}/to_timestamp": duration_in_s + ep_duration_in_s,
        }
        episodes_metadata.append(ep_metadata)

        paths_to_cat.append(ep_path)
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s
        ep_idx += 1

    # flush remaining
    if paths_to_cat:
        out_path = new_root / DEFAULT_VIDEO_PATH.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        concatenate_video_files(paths_to_cat, out_path)

        for i, _ in enumerate(paths_to_cat):
            past_ep_idx = ep_idx - len(paths_to_cat) + i
            episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

    return episodes_metadata


def generate_episode_metadata_dict(
    episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_videos=None
):
    num_episodes = len(episodes_metadata)
    episodes_legacy_metadata_vals = list(episodes_legacy_metadata.values())
    episodes_stats_vals = list(episodes_stats.values())
    episodes_stats_keys = list(episodes_stats.keys())

    for i in range(num_episodes):
        ep_legacy_metadata = episodes_legacy_metadata_vals[i]
        ep_metadata = episodes_metadata[i]
        ep_stats = episodes_stats_vals[i]

        ep_ids_set = {
            ep_legacy_metadata["episode_index"],
            ep_metadata["episode_index"],
            episodes_stats_keys[i],
        }

        if episodes_videos is None:
            ep_video = {}
        else:
            ep_video = episodes_videos[i]
            ep_ids_set.add(ep_video["episode_index"])

        if len(ep_ids_set) != 1:
            raise ValueError(f"Number of episodes is not the same ({ep_ids_set}).")

        ep_dict = {**ep_metadata, **ep_video, **ep_legacy_metadata, **flatten_dict({"stats": ep_stats})}
        ep_dict["meta/episodes/chunk_index"] = 0
        ep_dict["meta/episodes/file_index"] = 0
        yield ep_dict


def convert_episodes_metadata(root, new_root, episodes_metadata, episodes_video_metadata=None):
    logging.info(f"Converting episodes metadata from {root} to {new_root}")

    episodes_legacy_metadata = legacy_load_episodes(root)
    episodes_stats = legacy_load_episodes_stats(root)

    num_eps_set = {len(episodes_legacy_metadata), len(episodes_metadata)}
    if episodes_video_metadata is not None:
        num_eps_set.add(len(episodes_video_metadata))

    if len(num_eps_set) != 1:
        raise ValueError(f"Number of episodes is not the same ({num_eps_set}).")

    ds_episodes = Dataset.from_generator(
        lambda: generate_episode_metadata_dict(
            episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_video_metadata
        )
    )
    write_episodes(ds_episodes, new_root)

    stats = aggregate_stats(list(episodes_stats.values()))
    write_stats(stats, new_root)


def convert_info(root, new_root, data_file_size_in_mb, video_file_size_in_mb):
    info = load_info(root)
    info["codebase_version"] = V30

    info.pop("total_chunks", None)
    info.pop("total_videos", None)

    info["data_files_size_in_mb"] = data_file_size_in_mb
    info["video_files_size_in_mb"] = video_file_size_in_mb
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH if info.get("video_path") is not None else None
    if "fps" in info:
        info["fps"] = int(info["fps"])

    logging.info(f"Converting info from {root} to {new_root}")
    for key in info.get("features", {}):
        if info["features"][key].get("dtype") == "video":
            # already has fps in video_info
            continue
        if "fps" in info:
            info["features"][key]["fps"] = info["fps"]

    write_info(info, new_root)


def convert_dataset(
    load_path: str | Path | None = None,
    save_path: str | Path | None = None,
    branch: str | None = None,
    data_file_size_in_mb: int | None = None,
    video_file_size_in_mb: int | None = None,
    push_to_hub: bool = True,
    force_conversion: bool = False,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
):
    del branch, push_to_hub, start_ratio, end_ratio

    if data_file_size_in_mb is None:
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    root = Path(load_path)
    if not root.exists():
        raise FileNotFoundError(f"Input dataset path does not exist: {root}")

    validate_local_dataset_version(root)
    logging.info(f"Using local dataset at {root} (expected {V21})")

    new_root = Path(save_path)

    if new_root.is_dir():
        if force_conversion:
            logging.warning(f"Removing existing output dir: {new_root}")
            shutil.rmtree(new_root)
        else:
            logging.warning(f"Skip existing output dir: {new_root}")
            return

    try:
        convert_info(root, new_root, data_file_size_in_mb, video_file_size_in_mb)
        convert_tasks(root, new_root)
        episodes_metadata = convert_data(root, new_root, data_file_size_in_mb)
        episodes_videos_metadata = convert_videos(root, new_root, video_file_size_in_mb)
        convert_episodes_metadata(root, new_root, episodes_metadata, episodes_videos_metadata)
        logging.info(f"Conversion success: {root} -> {new_root}")
    except Exception as e:
        logging.exception(f"Conversion failed for {root}: {e}")
        if new_root.exists():
            shutil.rmtree(new_root)
        raise


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_ratio", type=float, default=0.0)
    parser.add_argument("--end_ratio", type=float, default=1.0)
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Unused in this local conversion script; kept for compatibility.",
    )
    parser.add_argument(
        "--data-file-size-in-mb",
        type=int,
        default=None,
        help=f"Data target file size (MB). Default from LeRobot utils ({DEFAULT_DATA_FILE_SIZE_IN_MB}).",
    )
    parser.add_argument(
        "--video-file-size-in-mb",
        type=int,
        default=None,
        help=f"Video target file size (MB). Default from LeRobot utils ({DEFAULT_VIDEO_FILE_SIZE_IN_MB}).",
    )
    parser.add_argument(
        "--push-to-hub",
        type=lambda input: input.lower() == "true",
        default=False,
        help="Unused here (local conversion only). Kept for compatibility.",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="If output path exists, delete and reconvert.",
    )

    args = parser.parse_args()

    load_root_path = "/mnt/shared-storage-gpfs2/internvla-gpfs2/zhaxizhuoma/lerobot_home/fastumi/double-Flower"
    save_root_path = "/mnt/shared-storage-gpfs2/internvla-gpfs2/zhaxizhuoma/lerobot_home/fastumi_30/double-Flower"

    load_paths = glob.glob(os.path.join(load_root_path, "long_horizon_tasks", "lift2", "*collaborate*"))

    load_paths.sort()
    num_datasets = len(load_paths)
    start_idx = int(num_datasets * args.start_ratio)
    end_idx = min(int(num_datasets * args.end_ratio) + 1, num_datasets)

    print(f"Matched datasets: {num_datasets}")
    print(f"Processing range: [{start_idx}, {end_idx})")

    for load_path in tqdm.tqdm(load_paths[start_idx:end_idx], desc="datasets"):
        save_path = load_path.replace(load_root_path, save_root_path)

        parts = load_path.split("/")
        repo_id = parts[-1] if len(parts) >= 1 else "unknown_repo"
        robot_id = parts[-2] if len(parts) >= 2 else "unknown_robot"
        task_type = parts[-3] if len(parts) >= 3 else "unknown_task_type"

        print(f"Converting {task_type} / {robot_id} / {repo_id} -> v3.0")

        args.load_path = load_path
        args.save_path = save_path
        convert_dataset(**vars(args))