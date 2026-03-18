#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fastumi_raw_to_lerobot_v30_parallel.py

FastUMI rawdata -> LeRobot dataset (v3.0 layout), SAFE parallel version.

Two-stage pipeline:
A) Worker processes (parallel per-session):
   - Read frames, align traj/clamp, apply Step1, resize
   - Dump episode payload to tmp .npz (NO dataset writing)
B) Main process (serial, deterministic order):
   - Load .npz and write into ONE LeRobotDataset via add_frame + save_episode
   - Optionally auto-convert legacy v2.x dataset to v3.0 layout in-place

Example

export HF_LEROBOT_HOME=/mnt/shared-storage-user/zhaxizhuoma/data/lerobot_home

python /mnt/shared-storage-user/internvla/Users/zhaxizhuoma/data/dataprocess_new/fastumi_raw_to_lerobot_v30.py \
  --raw-dir /mnt/shared-storage-user/zhaxizhuoma/fastumi_data/acone/2026-0208/Single-Flower \
  --repo-id fastumi/Single-Flower \
  --task "Grab the flower and insert it into the vase" \
  --fps 20 \
  --traj-source slam \
  --output-dir /mnt/shared-storage-user/internvla/Users/zhaxizhuoma/data/FastUMI/lerobot_data \
  --mode video \
  --next \
  --workers 16

"""

import argparse
import dataclasses
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.spatial.transform import Rotation as R


def _import_lerobot_dataset_and_home():
    """Return (LeRobotDataset, HF_LEROBOT_HOME). Supports both older and newer import paths."""
    # v3
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME
        return LeRobotDataset, HF_LEROBOT_HOME
    except Exception:
        pass

    # legacy-ish
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        try:
            from lerobot.common.constants import HF_LEROBOT_HOME
        except Exception:
            from lerobot.constants import HF_LEROBOT_HOME
        return LeRobotDataset, HF_LEROBOT_HOME
    except Exception as e:
        raise ImportError(
            "Cannot import LeRobot. Please ensure `lerobot` is installed in your environment. "
            f"Original error: {e}"
        )


LeRobotDataset, HF_LEROBOT_HOME = _import_lerobot_dataset_and_home()


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-4
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: Optional[str] = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

SOURCE_CAMERA_FPS = 60
STATE_NAMES_8 = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"]

STEP1_CFG_NON_NEXT = {
    "base_x": 0.05,
    "base_y": 0.01,
    "base_z": 0.09,
    "base_euler_deg": [0, 0, 0.0],
    "max_gripper": 84.0,
}

STEP1_CFG_NEXT = {
    "base_x": 0.07,
    "base_y": 0.02,
    "base_z": 0.11,
    "base_euler_deg": [0, 0, 0.0],
    "max_gripper": 84.0,
}


# -----------------------------
# Session discovery & layout
# -----------------------------

def find_all_sessions(raw_root: Path) -> List[Path]:
    found: List[Path] = []
    for r, dirs, _files in os.walk(raw_root):
        for d in dirs:
            if d.startswith("session"):
                found.append(Path(r) / d)
    return sorted(found)


def detect_layout(session_path: Path) -> Tuple[str, Dict[str, Path]]:
    """Returns: ("single"|"dual"|"invalid", paths_dict)."""
    if not session_path.is_dir():
        return "invalid", {}

    subdirs = [p for p in session_path.iterdir() if p.is_dir()]
    left = next((p for p in subdirs if p.name.startswith("left_hand")), None)
    right = next((p for p in subdirs if p.name.startswith("right_hand")), None)

    if left is not None and right is not None:
        return "dual", {"left": left, "right": right}
    if (session_path / "RGB_Images").exists():
        return "single", {"single": session_path}
    return "invalid", {}


# -----------------------------
# IO helpers
# -----------------------------

def ensure_clamp_csv(data_path: Path) -> None:
    clamp_dir = data_path / "Clamp_Data"
    if not clamp_dir.exists():
        return
    csv_path = clamp_dir / "clamp.csv"
    if csv_path.exists():
        return
    txt_path = clamp_dir / "clamp_data_tum.txt"
    if not txt_path.exists():
        return
    df = pd.read_csv(txt_path, sep=r"\s+", header=None)
    df.columns = ["timestamp", "clamp"]
    df.to_csv(csv_path, index=False)


def read_trj_txt(txt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(txt_path, sep=r"\s+", header=None)
    if df.shape[1] < 8:
        raise ValueError(f"trajectory txt expects >=8 cols, got {df.shape[1]}: {txt_path}")
    df = df.iloc[:, :8]
    df.columns = ["timestamp", "Pos X", "Pos Y", "Pos Z", "Q_X", "Q_Y", "Q_Z", "Q_W"]
    return df


def load_trajectory(data_path: Path, traj_source: str) -> pd.DataFrame:
    if traj_source == "merge":
        trj_dir = data_path / "Merged_Trajectory"
        trj_csv = trj_dir / "merged_trajectory.csv"
        trj_txt = trj_dir / "merged_trajectory.txt"
        if trj_csv.exists():
            df = pd.read_csv(trj_csv)
            cols = ["timestamp", "Pos X", "Pos Y", "Pos Z", "Q_X", "Q_Y", "Q_Z", "Q_W"]
            if all(c in df.columns for c in cols):
                return df[cols]
        if trj_txt.exists():
            return read_trj_txt(trj_txt)
        raise FileNotFoundError(f"No merged_trajectory.(csv|txt) under {trj_dir}")

    if traj_source == "slam":
        return read_trj_txt(data_path / "SLAM_Poses" / "slam_processed.txt")
    if traj_source == "vive":
        return read_trj_txt(data_path / "Vive_Poses" / "vive_data_tum.txt")

    raise ValueError(f"Unknown traj_source: {traj_source}")


def load_arm_sources(data_path: Path, traj_source: str):
    """Return dict with traj/clamp/timestamps/video_path or None."""
    if not data_path.is_dir():
        return None

    rgb_dir = data_path / "RGB_Images"
    video_path = rgb_dir / "video.mp4"
    ts_path = rgb_dir / "timestamps.csv"
    ensure_clamp_csv(data_path)
    clamp_path = data_path / "Clamp_Data" / "clamp.csv"

    if not (video_path.exists() and ts_path.exists() and clamp_path.exists()):
        return None

    traj = load_trajectory(data_path, traj_source)
    clamp = pd.read_csv(clamp_path)
    timestamps = pd.read_csv(ts_path)

    # normalize timestamps schema: 优先 header_stamp，其次 aligned_stamp
    if "header_stamp" in timestamps.columns:
        timestamps["timestamp"] = timestamps["header_stamp"]
    elif "aligned_stamp" in timestamps.columns:
        timestamps["timestamp"] = timestamps["aligned_stamp"]

    if "frame_index" not in timestamps.columns:
        timestamps["frame_index"] = np.arange(len(timestamps), dtype=int)

    cap = cv2.VideoCapture(str(video_path))
    ok = cap.isOpened()
    cap.release()
    if not ok:
        return None

    return {
        "traj": traj,
        "clamp": clamp,
        "timestamps": timestamps,
        "video_path": video_path,
    }


# -----------------------------
# Step1 + state/action
# -----------------------------

def build_T_base_to_local(step1_cfg: Dict[str, float]) -> np.ndarray:
    base_x = float(step1_cfg["base_x"])
    base_y = float(step1_cfg["base_y"])
    base_z = float(step1_cfg["base_z"])
    e = step1_cfg["base_euler_deg"]
    base_roll, base_pitch, base_yaw = np.deg2rad([float(e[0]), float(e[1]), float(e[2])])

    rotation_base_to_local = R.from_euler("xyz", [base_roll, base_pitch, base_yaw]).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation_base_to_local
    T[:3, 3] = [base_x, base_y, base_z]
    return T


def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local: np.ndarray):
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_local = np.eye(4, dtype=np.float64)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]

    T_base_r = np.dot(T_local[:3, :3], T_base_to_local[:3, :3])
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]

    rotation_base = R.from_matrix(T_base_r)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base


def normalize_state_from_raw(
    pose_row: pd.Series,
    clamp_value: float,
    T_base_to_local: np.ndarray,
    max_gripper: float,
) -> np.ndarray:
    x = float(pose_row["Pos X"])
    y = float(pose_row["Pos Y"])
    z = float(pose_row["Pos Z"])
    qx = float(pose_row["Q_X"])
    qy = float(pose_row["Q_Y"])
    qz = float(pose_row["Q_Z"])
    qw = float(pose_row["Q_W"])

    x_b, y_b, z_b, qx_b, qy_b, qz_b, qw_b = transform_to_base_quat(
        x, y, z, qx, qy, qz, qw, T_base_to_local
    )

    g = float(np.clip(float(clamp_value), 0.0, float(max_gripper)))
    g_norm = g / float(max_gripper)

    return np.array([x_b, y_b, z_b, qx_b, qy_b, qz_b, qw_b, g_norm], dtype=np.float32)


def build_actions_from_states(states: np.ndarray, use_next: bool) -> np.ndarray:
    if states.size == 0:
        return states.astype(np.float32)
    if not use_next:
        return states.astype(np.float32)

    T = states.shape[0]
    actions = np.zeros_like(states, dtype=np.float32)
    if T > 1:
        actions[:-1] = states[1:].astype(np.float32)
        actions[-1] = states[-1].astype(np.float32)
    else:
        actions[0] = states[0].astype(np.float32)
    return actions


def resize_rgb_no_crop(frame_bgr: np.ndarray, out_hw: int = 224) -> np.ndarray:
    rgb = frame_bgr[..., ::-1]  # BGR->RGB
    rgb = cv2.resize(rgb, (out_hw, out_hw), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


# -----------------------------
# Dataset creation (main only)
# -----------------------------

def create_empty_dataset(
    *,
    repo_id: str,
    fps: int,
    mode: Literal["video", "image"],
    bimanual: bool,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> "LeRobotDataset":
    features: Dict[str, Dict[str, object]] = {}

    if not bimanual:
        features["observation.state"] = {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8}
        features["action"] = {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8}
        features["robot_0_state"] = {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8}
        features["observation.images.front"] = {
            "dtype": mode,
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
    else:
        state_names_16 = [f"robot_0_{n}" for n in STATE_NAMES_8] + [f"robot_1_{n}" for n in STATE_NAMES_8]
        features["observation.state"] = {"dtype": "float32", "shape": (16,), "names": state_names_16}
        features["action"] = {"dtype": "float32", "shape": (16,), "names": state_names_16}
        features["robot_0_state"] = {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8}
        features["robot_1_state"] = {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8}
        features["observation.images.robot_0_image"] = {
            "dtype": mode,
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
        features["observation.images.robot_1_image"] = {
            "dtype": mode,
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }

    ds_root = HF_LEROBOT_HOME / repo_id
    if ds_root.exists():
        shutil.rmtree(ds_root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="fastumi",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


# -----------------------------
# Worker payload builders
# -----------------------------

def _build_single_payload(
    data_path: Path,
    *,
    fps: int,
    traj_source: str,
    use_next: bool,
    step1_cfg: Dict[str, float],
) -> Optional[Dict[str, np.ndarray]]:
    src = load_arm_sources(data_path, traj_source)
    if src is None:
        return None

    step = max(1, int(SOURCE_CAMERA_FPS / fps))
    ts_df = src["timestamps"].iloc[::step].reset_index(drop=True)
    if ts_df.empty:
        return None

    traj_df = src["traj"]
    clamp_df = src["clamp"]
    traj_ts = traj_df["timestamp"].to_numpy()
    clamp_ts = clamp_df["timestamp"].to_numpy()

    T_base = build_T_base_to_local(step1_cfg)
    max_g = float(step1_cfg["max_gripper"])

    cap = cv2.VideoCapture(str(src["video_path"]))
    states: List[np.ndarray] = []
    images: List[np.ndarray] = []

    for _, row in ts_df.iterrows():
        t = float(row["timestamp"])
        fidx = int(row["frame_index"])

        traj_i = int(np.argmin(np.abs(traj_ts - t)))
        clamp_i = int(np.argmin(np.abs(clamp_ts - t)))
        pose = traj_df.iloc[traj_i]
        clamp_v = float(clamp_df.iloc[clamp_i]["clamp"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        img = resize_rgb_no_crop(frame_bgr, out_hw=224)
        st = normalize_state_from_raw(pose, clamp_v, T_base, max_g)

        images.append(img)
        states.append(st)

    cap.release()
    if len(states) == 0:
        return None

    states_np = np.stack(states, axis=0).astype(np.float32)
    actions_np = build_actions_from_states(states_np, use_next).astype(np.float32)
    imgs_np = np.stack(images, axis=0).astype(np.uint8)

    return {
        "states": states_np,
        "actions": actions_np,
        "img_front": imgs_np,
    }


def _build_dual_payload(
    left_path: Path,
    right_path: Path,
    *,
    fps: int,
    traj_source: str,
    use_next: bool,
    step1_cfg: Dict[str, float],
) -> Optional[Dict[str, np.ndarray]]:
    left = load_arm_sources(left_path, traj_source)
    right = load_arm_sources(right_path, traj_source)
    if left is None or right is None:
        return None

    step = max(1, int(SOURCE_CAMERA_FPS / fps))
    master_ts = left["timestamps"].iloc[::step].reset_index(drop=True)
    if master_ts.empty:
        return None

    r_cam_ts = right["timestamps"]["timestamp"].to_numpy()
    r_cam_fidx = right["timestamps"]["frame_index"].to_numpy()

    l_traj_df = left["traj"]
    l_clamp_df = left["clamp"]
    l_traj_ts = l_traj_df["timestamp"].to_numpy()
    l_clamp_ts = l_clamp_df["timestamp"].to_numpy()

    r_traj_df = right["traj"]
    r_clamp_df = right["clamp"]
    r_traj_ts = r_traj_df["timestamp"].to_numpy()
    r_clamp_ts = r_clamp_df["timestamp"].to_numpy()

    T_base = build_T_base_to_local(step1_cfg)
    max_g = float(step1_cfg["max_gripper"])

    cap_l = cv2.VideoCapture(str(left["video_path"]))
    cap_r = cv2.VideoCapture(str(right["video_path"]))

    states0: List[np.ndarray] = []
    states1: List[np.ndarray] = []
    imgs_l: List[np.ndarray] = []
    imgs_r: List[np.ndarray] = []

    for _, row in master_ts.iterrows():
        t_master = float(row["timestamp"])

        # left aligned to master timestamp
        lti = int(np.argmin(np.abs(l_traj_ts - t_master)))
        lci = int(np.argmin(np.abs(l_clamp_ts - t_master)))
        l_pose = l_traj_df.iloc[lti]
        l_clamp = float(l_clamp_df.iloc[lci]["clamp"])

        cap_l.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
        ret_l, frame_l = cap_l.read()
        if not ret_l:
            continue

        # right: nearest camera timestamp to master, then align traj/clamp to that
        r_cam_i = int(np.argmin(np.abs(r_cam_ts - t_master)))
        t_right = float(r_cam_ts[r_cam_i])

        rti = int(np.argmin(np.abs(r_traj_ts - t_right)))
        rci = int(np.argmin(np.abs(r_clamp_ts - t_right)))
        r_pose = r_traj_df.iloc[rti]
        r_clamp = float(r_clamp_df.iloc[rci]["clamp"])

        cap_r.set(cv2.CAP_PROP_POS_FRAMES, int(r_cam_fidx[r_cam_i]))
        ret_r, frame_r = cap_r.read()
        if not ret_r:
            continue

        img_left = resize_rgb_no_crop(frame_l, out_hw=224)
        img_right = resize_rgb_no_crop(frame_r, out_hw=224)
        s0 = normalize_state_from_raw(l_pose, l_clamp, T_base, max_g)
        s1 = normalize_state_from_raw(r_pose, r_clamp, T_base, max_g)

        imgs_l.append(img_left)
        imgs_r.append(img_right)
        states0.append(s0)
        states1.append(s1)

    cap_l.release()
    cap_r.release()
    if len(states0) == 0:
        return None

    s0_np = np.stack(states0, axis=0).astype(np.float32)
    s1_np = np.stack(states1, axis=0).astype(np.float32)
    states_np = np.concatenate([s0_np, s1_np], axis=1).astype(np.float32)
    actions_np = build_actions_from_states(states_np, use_next).astype(np.float32)
    imgs_l_np = np.stack(imgs_l, axis=0).astype(np.uint8)
    imgs_r_np = np.stack(imgs_r, axis=0).astype(np.uint8)

    return {
        "states16": states_np,
        "actions16": actions_np,
        "robot0_states": s0_np,
        "robot1_states": s1_np,
        "img0": imgs_l_np,
        "img1": imgs_r_np,
    }


def _worker_build_episode_npz(
    index: int,
    session_path: Path,
    mode: str,
    paths: Dict[str, str],
    *,
    fps: int,
    traj_source: str,
    use_next: bool,
    step1_cfg: Dict[str, float],
    tmp_dir: Path,
) -> Tuple[int, Optional[str], int, Optional[str]]:
    """
    Worker entry:
    Returns (index, npz_path_or_None, num_frames, err_or_None)
    """
    try:
        if mode == "single":
            payload = _build_single_payload(Path(paths["single"]), fps=fps, traj_source=traj_source, use_next=use_next, step1_cfg=step1_cfg)
            if payload is None:
                return index, None, 0, f"empty/failed single: {session_path}"
            npz_path = tmp_dir / f"ep_{index:06d}.npz"
            np.savez_compressed(npz_path, **payload)
            return index, str(npz_path), int(payload["states"].shape[0]), None

        if mode == "dual":
            payload = _build_dual_payload(Path(paths["left"]), Path(paths["right"]), fps=fps, traj_source=traj_source, use_next=use_next, step1_cfg=step1_cfg)
            if payload is None:
                return index, None, 0, f"empty/failed dual: {session_path}"
            npz_path = tmp_dir / f"ep_{index:06d}.npz"
            np.savez_compressed(npz_path, **payload)
            return index, str(npz_path), int(payload["states16"].shape[0]), None

        return index, None, 0, f"unknown mode={mode}"
    except Exception as e:
        return index, None, 0, f"worker exception: {session_path} | {repr(e)}"


# -----------------------------
# v2.x -> v3.0 converter (same as your original)
# -----------------------------

def _read_codebase_version(dataset_root: Path) -> str:
    p3 = dataset_root / "meta" / "info.json"
    p2 = dataset_root / "info.json"
    p = p3 if p3.exists() else p2
    if not p.exists():
        return "unknown"
    try:
        with open(p, "r", encoding="utf-8") as f:
            info = json.load(f)
        return str(info.get("codebase_version", "unknown"))
    except Exception:
        return "unknown"


def _convert_local_dataset_to_v30_inplace(dataset_root: Path) -> None:
    # The official conversion code depends on LeRobot v3 modules.
    try:
        import jsonlines
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import Dataset, Features, Image

        from lerobot.datasets.compute_stats import aggregate_stats
        from lerobot.datasets.utils import (
            DEFAULT_CHUNK_SIZE,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_DATA_PATH,
            DEFAULT_VIDEO_FILE_SIZE_IN_MB,
            DEFAULT_VIDEO_PATH,
            LEGACY_EPISODES_PATH,
            LEGACY_EPISODES_STATS_PATH,
            LEGACY_TASKS_PATH,
            cast_stats_to_numpy,
            flatten_dict,
            get_file_size_in_mb,
            get_parquet_file_size_in_mb,
            get_parquet_num_frames,
            load_info,
            update_chunk_file_indices,
            write_episodes,
            write_info,
            write_stats,
            write_tasks,
        )
        from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
    except Exception as e:
        raise RuntimeError(
            "Your dataset was written in a legacy LeRobot v2.x format, but I couldn't import the LeRobot v3 "
            "conversion utilities. Please install/upgrade to LeRobot v3 (codebase_version v3.0) in this "
            "environment, then re-run.\n\n"
            f"Import error: {e}"
        )

    V30 = "v3.0"

    def load_jsonlines(fpath: Path):
        with jsonlines.open(fpath, "r") as reader:
            return list(reader)

    def legacy_load_episodes(local_dir: Path) -> dict:
        episodes = load_jsonlines(local_dir / LEGACY_EPISODES_PATH)
        return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}

    def legacy_load_episodes_stats(local_dir: Path) -> dict:
        fpath = local_dir / LEGACY_EPISODES_STATS_PATH
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing legacy episodes stats file: {fpath}. "
                "(LeRobot v2.x usually writes episodes_stats.jsonl; if yours doesn't, you need to generate it first.)"
            )
        episodes_stats = load_jsonlines(fpath)
        return {
            item["episode_index"]: cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
        }

    def legacy_load_tasks(local_dir: Path) -> tuple[dict, dict]:
        tasks = load_jsonlines(local_dir / LEGACY_TASKS_PATH)
        tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        task_to_task_index = {task: task_index for task_index, task in tasks.items()}
        return tasks, task_to_task_index

    def get_video_keys(root: Path):
        info = load_info(root)
        features = info["features"]
        return sorted([key for key, ft in features.items() if ft["dtype"] == "video"])

    def get_image_keys(root: Path):
        info = load_info(root)
        features = info["features"]
        return sorted([key for key, ft in features.items() if ft["dtype"] == "image"])

    def convert_tasks(root: Path, new_root: Path):
        tasks, _ = legacy_load_tasks(root)
        df_tasks = pd.DataFrame({"task_index": list(tasks.keys())}, index=list(tasks.values()))
        write_tasks(df_tasks, new_root)

    def concat_data_files(paths_to_cat, new_root: Path, chunk_idx: int, file_idx: int, image_keys: list[str]):
        import pyarrow.parquet as pq
        import pyarrow as pa
        from datasets import Features, Image

        tables = [pq.read_table(f) for f in paths_to_cat]
        table = pa.concat_tables(tables, promote=True)
        features = Features.from_arrow_schema(table.schema)
        for key in image_keys:
            features[key] = Image()
        arrow_schema = features.arrow_schema
        path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.cast(arrow_schema), path)

    def convert_data(root: Path, new_root: Path, data_file_size_in_mb: int):
        data_dir = root / "data"
        ep_paths = sorted(data_dir.glob("*/*.parquet"))
        image_keys = get_image_keys(root)

        ep_idx = 0
        chunk_idx = 0
        file_idx = 0
        size_in_mb = 0.0
        num_frames = 0
        paths_to_cat = []
        episodes_metadata = []

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
            size_in_mb += float(ep_size_in_mb)
            num_frames += int(ep_num_frames)
            episodes_metadata.append(ep_metadata)
            ep_idx += 1

            if size_in_mb < data_file_size_in_mb:
                paths_to_cat.append(ep_path)
                continue

            if paths_to_cat:
                concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

            size_in_mb = float(ep_size_in_mb)
            paths_to_cat = [ep_path]
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

        if paths_to_cat:
            concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

        return episodes_metadata

    def convert_videos_of_camera(root: Path, new_root: Path, video_key: str, video_file_size_in_mb: int):
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
            ep_size_in_mb = float(get_file_size_in_mb(ep_path))
            ep_duration_in_s = float(get_video_duration_in_s(ep_path))

            if size_in_mb + ep_size_in_mb >= video_file_size_in_mb and len(paths_to_cat) > 0:
                concatenate_video_files(
                    paths_to_cat,
                    new_root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
                )
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
                f"videos/{video_key}/chunk_index": chunk_idx,
                f"videos/{video_key}/file_index": file_idx,
                f"videos/{video_key}/from_timestamp": duration_in_s,
                f"videos/{video_key}/to_timestamp": duration_in_s + ep_duration_in_s,
            }
            episodes_metadata.append(ep_metadata)
            paths_to_cat.append(ep_path)
            size_in_mb += ep_size_in_mb
            duration_in_s += ep_duration_in_s
            ep_idx += 1

        if paths_to_cat:
            concatenate_video_files(
                paths_to_cat,
                new_root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
            )
            for i, _ in enumerate(paths_to_cat):
                past_ep_idx = ep_idx - len(paths_to_cat) + i
                episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

        return episodes_metadata

    def convert_videos(root: Path, new_root: Path, video_file_size_in_mb: int):
        video_keys = get_video_keys(root)
        if len(video_keys) == 0:
            return None

        eps_metadata_per_cam = [convert_videos_of_camera(root, new_root, k, video_file_size_in_mb) for k in video_keys]
        num_eps_per_cam = [len(m) for m in eps_metadata_per_cam]
        if len(set(num_eps_per_cam)) != 1:
            raise ValueError(f"All cams dont have same number of episodes ({num_eps_per_cam}).")

        num_cameras = len(video_keys)
        num_episodes = num_eps_per_cam[0]
        episodes_metadata = []
        for ep_idx in range(num_episodes):
            ep_ids = [eps_metadata_per_cam[cam_idx][ep_idx]["episode_index"] for cam_idx in range(num_cameras)] + [ep_idx]
            if len(set(ep_ids)) != 1:
                raise ValueError(f"All episode indices need to match ({ep_ids}).")
            ep_dict = {"episode_index": ep_idx}
            for cam_idx in range(num_cameras):
                ep_dict.update(eps_metadata_per_cam[cam_idx][ep_idx])
            episodes_metadata.append(ep_dict)

        return episodes_metadata

    def generate_episode_metadata_dict(episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_videos=None):
        legacy_vals = list(episodes_legacy_metadata.values())
        stats_vals = list(episodes_stats.values())
        stats_keys = list(episodes_stats.keys())

        for i in range(len(episodes_metadata)):
            ep_legacy = legacy_vals[i]
            ep_meta = episodes_metadata[i]
            ep_stats = stats_vals[i]

            ep_ids_set = {ep_legacy["episode_index"], ep_meta["episode_index"], stats_keys[i]}
            if episodes_videos is None:
                ep_video = {}
            else:
                ep_video = episodes_videos[i]
                ep_ids_set.add(ep_video["episode_index"])
            if len(ep_ids_set) != 1:
                raise ValueError(f"Number of episodes is not the same ({ep_ids_set}).")

            ep_dict = {**ep_meta, **ep_video, **ep_legacy, **flatten_dict({"stats": ep_stats})}
            ep_dict["meta/episodes/chunk_index"] = 0
            ep_dict["meta/episodes/file_index"] = 0
            yield ep_dict

    def convert_episodes_metadata(root: Path, new_root: Path, episodes_metadata, episodes_video_metadata=None):
        episodes_legacy_metadata = legacy_load_episodes(root)
        episodes_stats = legacy_load_episodes_stats(root)

        num_eps_set = {len(episodes_legacy_metadata), len(episodes_metadata)}
        if episodes_video_metadata is not None:
            num_eps_set.add(len(episodes_video_metadata))
        if len(num_eps_set) != 1:
            raise ValueError(f"Number of episodes is not the same ({num_eps_set}).")

        from datasets import Dataset
        ds_episodes = Dataset.from_generator(
            lambda: generate_episode_metadata_dict(
                episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_video_metadata
            )
        )
        write_episodes(ds_episodes, new_root)
        stats = aggregate_stats(list(episodes_stats.values()))
        write_stats(stats, new_root)

    def convert_info(root: Path, new_root: Path, data_file_size_in_mb: int, video_file_size_in_mb: int):
        info = load_info(root)
        info["codebase_version"] = V30
        info.pop("total_chunks", None)
        info.pop("total_videos", None)
        info["data_files_size_in_mb"] = data_file_size_in_mb
        info["video_files_size_in_mb"] = video_file_size_in_mb
        info["data_path"] = DEFAULT_DATA_PATH
        info["video_path"] = DEFAULT_VIDEO_PATH if info.get("video_path", None) is not None else None
        info["fps"] = int(info["fps"])
        for key in info["features"]:
            if info["features"][key]["dtype"] == "video":
                continue
            info["features"][key]["fps"] = info["fps"]
        write_info(info, new_root)

    DEFAULT_DATA_FILE_SIZE_IN_MB = 128
    DEFAULT_VIDEO_FILE_SIZE_IN_MB = 512

    data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    new_root = dataset_root.parent / (dataset_root.name + "__v30_tmp")
    if new_root.exists():
        shutil.rmtree(new_root)
    new_root.mkdir(parents=True, exist_ok=True)

    try:
        convert_info(dataset_root, new_root, data_file_size_in_mb, video_file_size_in_mb)
        convert_tasks(dataset_root, new_root)
        episodes_metadata = convert_data(dataset_root, new_root, data_file_size_in_mb)
        episodes_videos_metadata = convert_videos(dataset_root, new_root, video_file_size_in_mb)
        convert_episodes_metadata(dataset_root, new_root, episodes_metadata, episodes_videos_metadata)
    except Exception:
        if new_root.exists():
            shutil.rmtree(new_root)
        raise

    backup_root = dataset_root.parent / (dataset_root.name + "__v2_backup")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    shutil.move(str(dataset_root), str(backup_root))
    shutil.move(str(new_root), str(dataset_root))
    shutil.rmtree(backup_root)


def maybe_convert_repo_to_v30(repo_id: str) -> None:
    dataset_root = HF_LEROBOT_HOME / repo_id
    ver = _read_codebase_version(dataset_root)
    if ver.startswith("v3"):
        return
    print(f"[Info] Detected legacy LeRobot dataset version={ver}. Converting to v3.0 ...")
    _convert_local_dataset_to_v30_inplace(dataset_root)
    ver2 = _read_codebase_version(dataset_root)
    print(f"[Info] Conversion done. codebase_version={ver2}")


# -----------------------------
# Main: parallel stage A + serial stage B
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="FastUMI rawdata -> LeRobot v3.0 (SAFE parallel; auto-convert from v2.x if needed)")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Raw data root (search recursively for session*)")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot repo id, e.g. fastumi_delta/unplug_charger")
    parser.add_argument("--task", type=str, required=True, help="Task string stored in each frame")
    parser.add_argument("--fps", type=int, default=20, choices=[20, 30, 60], help="Target fps (downsample from 60Hz)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional copy-out directory")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video"], help="Store images as image or video")
    parser.add_argument("--traj-source", type=str, default="slam", choices=["merge", "slam", "vive"], help="Which trajectory source")
    parser.add_argument("--next", action="store_true", help="Use next-step action: action[t]=state[t+1], last repeats")
    parser.add_argument("--workers", type=int, default=8, help="Num worker processes for per-session preprocessing")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Where to store tmp npz payloads (default: system tmp)")
    parser.add_argument(
        "--no-v30-convert",
        action="store_true",
        help="If set, do NOT auto-convert legacy v2.x dataset to v3.0 layout",
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    repo_id: str = args.repo_id
    task: str = args.task
    fps: int = args.fps
    traj_source: str = args.traj_source
    use_next: bool = args.next
    mode: Literal["image", "video"] = args.mode
    output_dir: Optional[Path] = args.output_dir
    workers: int = int(args.workers)

    if not raw_dir.exists():
        raise FileNotFoundError(f"--raw-dir does not exist: {raw_dir}")

    sessions = find_all_sessions(raw_dir)
    if not sessions:
        raise RuntimeError(f"No session* directories found under: {raw_dir}")

    layout = None
    valid = []
    for s in sessions:
        m, paths = detect_layout(s)
        if m == "invalid":
            continue
        valid.append((s, m, paths))
        if layout is None:
            layout = m
        elif layout != m:
            raise RuntimeError(
                f"Mixed layouts found under {raw_dir}: already have '{layout}', but session {s} is '{m}'. "
                "Please convert single/dual tasks separately."
            )

    if not valid or layout is None:
        raise RuntimeError(f"No valid sessions found under: {raw_dir}")

    bimanual = layout == "dual"
    step1_cfg = STEP1_CFG_NEXT if use_next else STEP1_CFG_NON_NEXT

    # tmp dir for npz payloads
    if args.tmp_dir is None:
        tmp_root = Path(tempfile.mkdtemp(prefix="fastumi_lerobot_npz_"))
        auto_cleanup_tmp = True
    else:
        tmp_root = args.tmp_dir
        tmp_root.mkdir(parents=True, exist_ok=True)
        auto_cleanup_tmp = False

    print(f"[Info] Using tmp_dir={tmp_root} (auto_cleanup={auto_cleanup_tmp})")

    # -------------------------
    # Stage A: parallel build npz
    # -------------------------
    futures = []
    results: Dict[int, Tuple[Optional[str], int, Optional[str]]] = {}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, (sess_path, m, paths) in enumerate(valid):
            # convert paths to str for pickling safety
            spaths = {k: str(v) for k, v in paths.items()}
            fut = ex.submit(
                _worker_build_episode_npz,
                idx,
                sess_path,
                m,
                spaths,
                fps=fps,
                traj_source=traj_source,
                use_next=use_next,
                step1_cfg=step1_cfg,
                tmp_dir=tmp_root,
            )
            futures.append(fut)

        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="StageA preprocess (npz)"):
            idx, npz_path, nframes, err = fut.result()
            results[idx] = (npz_path, nframes, err)

    # -------------------------
    # Stage B: serial write dataset
    # -------------------------
    dataset = create_empty_dataset(
        repo_id=repo_id,
        fps=fps,
        mode=mode,
        bimanual=bimanual,
        dataset_config=DEFAULT_DATASET_CONFIG,
    )

    ok_count = 0
    skipped = 0
    total_frames_written = 0

    for idx, (sess_path, m, _paths) in enumerate(valid):
        npz_path, nframes, err = results.get(idx, (None, 0, "missing result"))
        if npz_path is None:
            skipped += 1
            if err:
                print(f"[Warn] Skip episode idx={idx} session={sess_path} reason={err}")
            continue

        data = np.load(npz_path, allow_pickle=False)

        if m == "single":
            states = data["states"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            img_front = data["img_front"].astype(np.uint8)

            T = states.shape[0]
            for i in range(T):
                frame = {
                    "task": task,
                    "observation.state": torch.from_numpy(states[i]).to(torch.float32),
                    "action": torch.from_numpy(actions[i]).to(torch.float32),
                    "robot_0_state": torch.from_numpy(states[i]).to(torch.float32),
                    "observation.images.front": img_front[i],
                }
                dataset.add_frame(frame)
            dataset.save_episode()
            ok_count += 1
            total_frames_written += T

        else:
            states16 = data["states16"].astype(np.float32)
            actions16 = data["actions16"].astype(np.float32)
            s0 = data["robot0_states"].astype(np.float32)
            s1 = data["robot1_states"].astype(np.float32)
            img0 = data["img0"].astype(np.uint8)
            img1 = data["img1"].astype(np.uint8)

            T = states16.shape[0]
            for i in range(T):
                frame = {
                    "task": task,
                    "observation.state": torch.from_numpy(states16[i]).to(torch.float32),
                    "action": torch.from_numpy(actions16[i]).to(torch.float32),
                    "robot_0_state": torch.from_numpy(s0[i]).to(torch.float32),
                    "robot_1_state": torch.from_numpy(s1[i]).to(torch.float32),
                    "observation.images.robot_0_image": img0[i],
                    "observation.images.robot_1_image": img1[i],
                }
                dataset.add_frame(frame)
            dataset.save_episode()
            ok_count += 1
            total_frames_written += T

    # finalize dataset
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    elif hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "close"):
        dataset.close()

    # convert to v3.0 layout
    if not args.no_v30_convert:
        maybe_convert_repo_to_v30(repo_id)

    # copy-out
    if output_dir is not None:
        target = output_dir / repo_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(HF_LEROBOT_HOME / repo_id, target)
        print(f"[Info] Dataset copied to {target}")

    ver = _read_codebase_version(HF_LEROBOT_HOME / repo_id)
    print(
        f"[Done] repo_id={repo_id} bimanual={bimanual} episodes_saved={ok_count} skipped={skipped} "
        f"frames_written={total_frames_written} codebase_version={ver} HF_LEROBOT_HOME={HF_LEROBOT_HOME}"
    )

    # cleanup tmp
    if auto_cleanup_tmp:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            print(f"[Warn] Failed to cleanup tmp_dir={tmp_root}")


if __name__ == "__main__":
    main()
