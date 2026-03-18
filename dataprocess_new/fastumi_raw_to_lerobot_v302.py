#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

FastUMI rawdata -> LeRobot dataset (v3.0 layout), SAFE parallel version (NO RGB dump).

核心思路（推荐）：
- Stage A（并行 Worker）：只做对齐与Step1，输出 states + 需要读取的 frame_index + video_path（不存任何RGB数组）
- Stage B（主进程串行写入）：打开原始 mp4，按 frame_index 解码采样帧、resize，写入 LeRobotDataset
  * 为了稳妥处理“读帧失败”，Stage B 会只保留成功读到帧的那些 timestep，并基于保留下来的 states 重新构建 actions



用法示例：

export HF_LEROBOT_HOME=/mnt/shared-storage-gpfs2/internvla-gpfs2/zhaxizhuoma/lerobot_home

python /mnt/shared-storage-gpfs2/internvla-gpfs2/zhaxizhuoma/data/dataprocess_new/fastumi_raw_to_lerobot_v302.py \
  --raw-dir /mnt/shared-storage-gpfs2/internvla-gpfs2/zhaxizhuoma/fastumi_data/acone_new/task_20260121H011 \
  --repo-id fastumi_32/Fold_square_towels_new \
  --task "Fold square towel" \
  --fps 20 \
  --traj-source slam \
  --mode video \
  --next \
  --workers 32 \
  --max-inflight 16 \
  --tmp-dir /mnt/shared-storage-gpfs2/internvla-gpfs2/zhaxizhuoma/tmp/fastumi_npz \
  --output-dir /mnt/shared-storage-gpfs2/internvla-gpfs2/zhaxizhuoma/backup
"""

import argparse
import contextlib
import dataclasses
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.spatial.transform import Rotation as R


# -----------------------------
# LeRobot imports (compat)
# -----------------------------
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


# -----------------------------
# Configs
# -----------------------------
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-4
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: Optional[str] = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

SOURCE_CAMERA_FPS = 60  # raw video fps (FastUMI)
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
        # 与 v21 一样保留兼容 fallback
        slam_dir = data_path / "SLAM_Poses"
        cand = [
            slam_dir / "slam_processed.txt",
            slam_dir / "slam_raw.txt",
            data_path / "slam_processed.txt",
            data_path / "slam_raw.txt",
        ]
        for p in cand:
            if p.exists():
                return read_trj_txt(p)
        raise FileNotFoundError(f"No slam trajectory found. Tried: {', '.join(str(p) for p in cand)}")

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
    pose_row: np.ndarray,
    clamp_value: float,
    T_base_to_local: np.ndarray,
    max_gripper: float,
) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = [float(v) for v in pose_row.tolist()]

    x_b, y_b, z_b, qx_b, qy_b, qz_b, qw_b = transform_to_base_quat(
        x, y, z, qx, qy, qz, qw, T_base_to_local
    )

    g = float(np.clip(float(clamp_value), 0.0, float(max_gripper)))
    g_norm = g / float(max_gripper)

    return np.array([x_b, y_b, z_b, qx_b, qy_b, qz_b, qw_b, g_norm], dtype=np.float32)


def build_actions_from_states(states: np.ndarray, use_next: bool) -> np.ndarray:
    """actions shape == states shape. If use_next: action[t]=state[t+1], last repeats."""
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


def nearest_indices(sorted_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """
    sorted_ts must be ascending.
    For each query, return index of nearest value in sorted_ts.
    """
    idx = np.searchsorted(sorted_ts, query_ts, side="left")
    idx0 = np.clip(idx - 1, 0, len(sorted_ts) - 1)
    idx1 = np.clip(idx, 0, len(sorted_ts) - 1)

    d0 = np.abs(sorted_ts[idx0] - query_ts)
    d1 = np.abs(sorted_ts[idx1] - query_ts)
    choose1 = d1 < d0
    out = np.where(choose1, idx1, idx0)
    return out.astype(np.int64)


@contextlib.contextmanager
def suppress_stderr(enabled: bool):
    """Temporarily redirect process stderr to /dev/null."""
    if not enabled:
        yield
        return

    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


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
        use_videos=(mode == "video"),
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


# -----------------------------
# Worker payload builders (NO frames decode)
# -----------------------------
def _build_single_payload(
    data_path: Path,
    *,
    fps: int,
    traj_source: str,
    step1_cfg: Dict[str, float],
) -> Optional[Dict[str, np.ndarray]]:
    src = load_arm_sources(data_path, traj_source)
    if src is None:
        return None

    step = max(1, int(SOURCE_CAMERA_FPS / fps))
    ts_df = src["timestamps"].iloc[::step].reset_index(drop=True)
    if ts_df.empty:
        return None

    t_arr = ts_df["timestamp"].to_numpy(dtype=np.float64)
    frame_np = ts_df["frame_index"].to_numpy(dtype=np.int32)

    traj_df = src["traj"].sort_values("timestamp").reset_index(drop=True)
    clamp_df = src["clamp"].sort_values("timestamp").reset_index(drop=True)
    traj_ts = traj_df["timestamp"].to_numpy(dtype=np.float64)
    clamp_ts = clamp_df["timestamp"].to_numpy(dtype=np.float64)

    traj_i = nearest_indices(traj_ts, t_arr)
    clamp_i = nearest_indices(clamp_ts, t_arr)
    pose_np = traj_df.loc[traj_i, ["Pos X", "Pos Y", "Pos Z", "Q_X", "Q_Y", "Q_Z", "Q_W"]].to_numpy(dtype=np.float64)
    clamp_np = clamp_df.loc[clamp_i, "clamp"].to_numpy(dtype=np.float64)

    T_base = build_T_base_to_local(step1_cfg)
    max_g = float(step1_cfg["max_gripper"])

    T = pose_np.shape[0]
    if T == 0:
        return None

    states_np = np.empty((T, 8), dtype=np.float32)
    for k in range(T):
        states_np[k] = normalize_state_from_raw(pose_np[k], float(clamp_np[k]), T_base, max_g)

    # 注意：存成 numpy unicode，避免 object dtype 导致 np.load(allow_pickle=False) 失败
    vpath = np.asarray(str(src["video_path"]))

    return {
        "video_path": vpath,
        "frame_index": frame_np,
        "states": states_np,
    }


def _build_dual_payload(
    left_path: Path,
    right_path: Path,
    *,
    fps: int,
    traj_source: str,
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

    t_master = master_ts["timestamp"].to_numpy(dtype=np.float64)
    f0_np = master_ts["frame_index"].to_numpy(dtype=np.int32)

    right_ts_df = right["timestamps"]
    r_cam_ts = right_ts_df["timestamp"].to_numpy(dtype=np.float64)
    r_cam_fidx = right_ts_df["frame_index"].to_numpy(dtype=np.int32)
    r_cam_pick = nearest_indices(r_cam_ts, t_master)
    f1_np = r_cam_fidx[r_cam_pick]
    t_right = r_cam_ts[r_cam_pick]

    l_traj_df = left["traj"].sort_values("timestamp").reset_index(drop=True)
    l_clamp_df = left["clamp"].sort_values("timestamp").reset_index(drop=True)
    r_traj_df = right["traj"].sort_values("timestamp").reset_index(drop=True)
    r_clamp_df = right["clamp"].sort_values("timestamp").reset_index(drop=True)

    l_traj_ts = l_traj_df["timestamp"].to_numpy(dtype=np.float64)
    l_clamp_ts = l_clamp_df["timestamp"].to_numpy(dtype=np.float64)
    r_traj_ts = r_traj_df["timestamp"].to_numpy(dtype=np.float64)
    r_clamp_ts = r_clamp_df["timestamp"].to_numpy(dtype=np.float64)

    lti = nearest_indices(l_traj_ts, t_master)
    lci = nearest_indices(l_clamp_ts, t_master)
    rti = nearest_indices(r_traj_ts, t_right)
    rci = nearest_indices(r_clamp_ts, t_right)

    l_pose_np = l_traj_df.loc[lti, ["Pos X", "Pos Y", "Pos Z", "Q_X", "Q_Y", "Q_Z", "Q_W"]].to_numpy(dtype=np.float64)
    r_pose_np = r_traj_df.loc[rti, ["Pos X", "Pos Y", "Pos Z", "Q_X", "Q_Y", "Q_Z", "Q_W"]].to_numpy(dtype=np.float64)
    l_clamp_np = l_clamp_df.loc[lci, "clamp"].to_numpy(dtype=np.float64)
    r_clamp_np = r_clamp_df.loc[rci, "clamp"].to_numpy(dtype=np.float64)

    T_base = build_T_base_to_local(step1_cfg)
    max_g = float(step1_cfg["max_gripper"])

    T = min(len(t_master), l_pose_np.shape[0], r_pose_np.shape[0])
    if T <= 0:
        return None

    s0_np = np.empty((T, 8), dtype=np.float32)
    s1_np = np.empty((T, 8), dtype=np.float32)
    for k in range(T):
        s0_np[k] = normalize_state_from_raw(l_pose_np[k], float(l_clamp_np[k]), T_base, max_g)
        s1_np[k] = normalize_state_from_raw(r_pose_np[k], float(r_clamp_np[k]), T_base, max_g)

    f0_np = f0_np[:T]
    f1_np = f1_np[:T]

    v0 = np.asarray(str(left["video_path"]))
    v1 = np.asarray(str(right["video_path"]))

    return {
        "video_path_0": v0,
        "video_path_1": v1,
        "frame_index_0": f0_np,
        "frame_index_1": f1_np,
        "robot0_states": s0_np,
        "robot1_states": s1_np,
    }


def _worker_build_episode_npz(
    index: int,
    session_path: Path,
    mode: str,
    paths: Dict[str, str],
    *,
    fps: int,
    traj_source: str,
    step1_cfg: Dict[str, float],
    tmp_dir: Path,
) -> Tuple[int, Optional[str], int, Optional[str], float]:
    """
    Worker entry:
    Returns (index, npz_path_or_None, num_steps, err_or_None)
    """
    t0 = time.perf_counter()
    try:
        if mode == "single":
            payload = _build_single_payload(
                Path(paths["single"]),
                fps=fps,
                traj_source=traj_source,
                step1_cfg=step1_cfg,
            )
            if payload is None:
                return index, None, 0, f"empty/failed single: {session_path}", time.perf_counter() - t0
            npz_path = tmp_dir / f"ep_{index:06d}.npz"
            np.savez_compressed(npz_path, **payload)
            return index, str(npz_path), int(payload["states"].shape[0]), None, time.perf_counter() - t0

        if mode == "dual":
            payload = _build_dual_payload(
                Path(paths["left"]),
                Path(paths["right"]),
                fps=fps,
                traj_source=traj_source,
                step1_cfg=step1_cfg,
            )
            if payload is None:
                return index, None, 0, f"empty/failed dual: {session_path}", time.perf_counter() - t0
            npz_path = tmp_dir / f"ep_{index:06d}.npz"
            np.savez_compressed(npz_path, **payload)
            return index, str(npz_path), int(payload["robot0_states"].shape[0]), None, time.perf_counter() - t0

        return index, None, 0, f"unknown mode={mode}", time.perf_counter() - t0
    except Exception as e:
        return index, None, 0, f"worker exception: {session_path} | {repr(e)}", time.perf_counter() - t0


# -----------------------------
# v2.x -> v3.0 converter (same logic as你原始)
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
# Stage B writers (decode frames + strict fail-on-any-frame-read-error)
# -----------------------------
def _read_frames_by_indices(video_path: str, frame_indices: np.ndarray) -> Optional[List[np.ndarray]]:
    """
    Strict reader: fail the whole episode if any target frame cannot be read.
    """
    if frame_indices.size == 0:
        return []

    # Fast path: non-decreasing indices can be read sequentially to reduce random seeks.
    if np.any(frame_indices[1:] < frame_indices[:-1]):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None

        imgs: List[np.ndarray] = []
        for fidx in frame_indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame_bgr = cap.read()
            if not ok:
                cap.release()
                return None
            imgs.append(resize_rgb_no_crop(frame_bgr, out_hw=224))

        cap.release()
        return imgs

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    imgs: List[np.ndarray] = []
    first = int(frame_indices[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    cur = first

    for target in frame_indices.tolist():
        target = int(target)
        while cur < target:
            ok, _ = cap.read()
            if not ok:
                cap.release()
                return None
            cur += 1

        ok, frame_bgr = cap.read()
        if not ok:
            cap.release()
            return None
        imgs.append(resize_rgb_no_crop(frame_bgr, out_hw=224))
        cur += 1

    cap.release()
    return imgs


def _write_single_episode_from_npz(
    dataset: "LeRobotDataset",
    npz_path: str,
    task: str,
    use_next: bool,
) -> int:
    data = np.load(npz_path, allow_pickle=False)
    video_path = str(data["video_path"])
    frame_index = data["frame_index"].astype(np.int32)
    states_all = data["states"].astype(np.float32)

    imgs = _read_frames_by_indices(video_path, frame_index)
    if imgs is None:
        return 0

    states = states_all
    actions = build_actions_from_states(states, use_next=use_next)

    for i in range(states.shape[0]):
        frame = {
            "task": task,
            "observation.state": torch.from_numpy(states[i]).to(torch.float32),
            "action": torch.from_numpy(actions[i]).to(torch.float32),
            "robot_0_state": torch.from_numpy(states[i]).to(torch.float32),
            "observation.images.front": imgs[i],
        }
        dataset.add_frame(frame)

    dataset.save_episode()
    return int(states.shape[0])


def _write_dual_episode_from_npz(
    dataset: "LeRobotDataset",
    npz_path: str,
    task: str,
    use_next: bool,
) -> int:
    data = np.load(npz_path, allow_pickle=False)
    v0 = str(data["video_path_0"])
    v1 = str(data["video_path_1"])
    f0 = data["frame_index_0"].astype(np.int32)
    f1 = data["frame_index_1"].astype(np.int32)
    s0_all = data["robot0_states"].astype(np.float32)
    s1_all = data["robot1_states"].astype(np.float32)

    imgs0 = _read_frames_by_indices(v0, f0)
    imgs1 = _read_frames_by_indices(v1, f1)
    if imgs0 is None or imgs1 is None:
        return 0

    s0 = s0_all
    s1 = s1_all
    states16 = np.concatenate([s0, s1], axis=1).astype(np.float32)
    actions16 = build_actions_from_states(states16, use_next=use_next)

    for i in range(states16.shape[0]):
        frame = {
            "task": task,
            "observation.state": torch.from_numpy(states16[i]).to(torch.float32),
            "action": torch.from_numpy(actions16[i]).to(torch.float32),
            "robot_0_state": torch.from_numpy(s0[i]).to(torch.float32),
            "robot_1_state": torch.from_numpy(s1[i]).to(torch.float32),
            "observation.images.robot_0_image": imgs0[i],
            "observation.images.robot_1_image": imgs1[i],
        }
        dataset.add_frame(frame)

    dataset.save_episode()
    return int(states16.shape[0])


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="FastUMI rawdata -> LeRobot v3.0 (SAFE parallel; NO RGB dump)")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Raw data root (search recursively for session*)")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot repo id, e.g. fastumi_delta/unplug_charger")
    parser.add_argument("--task", type=str, required=True, help="Task string stored in each frame")
    parser.add_argument("--fps", type=int, default=30, choices=[20, 30, 60], help="Target fps (downsample from 60Hz)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional copy-out directory")
    parser.add_argument("--mode", type=str, default="video", choices=["image", "video"], help="Store images as image or video")
    parser.add_argument("--traj-source", type=str, default="slam", choices=["merge", "slam", "vive"], help="Which trajectory source")
    parser.add_argument("--next", action="store_true", help="Use next-step action: action[t]=state[t+1], last repeats")
    parser.add_argument("--workers", type=int, default=8, help="Num worker processes for per-session preprocessing")
    parser.add_argument("--max-inflight", type=int, default=0,
                        help="Max inflight sessions submitted to workers (0 => 2*workers)")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Where to store tmp npz payloads (default: system tmp)")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep tmp npz payloads for debugging")
    parser.add_argument("--log-every", type=int, default=1, help="Print progress logs every N StageA completions / StageB writes")
    parser.add_argument("--quiet-ffmpeg", action="store_true", help="Suppress ffmpeg/SVT stderr logs")
    parser.add_argument("--image-writer-processes", type=int, default=DEFAULT_DATASET_CONFIG.image_writer_processes)
    parser.add_argument("--image-writer-threads", type=int, default=DEFAULT_DATASET_CONFIG.image_writer_threads)
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
    mode: Literal["image", "video"] = args.mode  # type: ignore
    output_dir: Optional[Path] = args.output_dir
    workers: int = int(args.workers)
    max_inflight: int = int(args.max_inflight) if int(args.max_inflight) > 0 else max(1, 2 * workers)
    log_every: int = max(1, int(args.log_every))
    quiet_ffmpeg: bool = bool(args.quiet_ffmpeg)

    if not raw_dir.exists():
        raise FileNotFoundError(f"--raw-dir does not exist: {raw_dir}")

    sessions = find_all_sessions(raw_dir)
    if not sessions:
        raise RuntimeError(f"No session* directories found under: {raw_dir}")

    layout = None
    valid: List[Tuple[Path, str, Dict[str, Path]]] = []
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
    print(f"[Info] workers={workers}, max_inflight={max_inflight}, sessions={len(valid)}")

    # dataset writer config (避免 LeRobot 内部过度并发放大内存压力)
    dataset_cfg = DatasetConfig(
        use_videos=(mode == "video"),
        tolerance_s=DEFAULT_DATASET_CONFIG.tolerance_s,
        image_writer_processes=int(args.image_writer_processes),
        image_writer_threads=int(args.image_writer_threads),
        video_backend=DEFAULT_DATASET_CONFIG.video_backend,
    )

    # 创建数据集（主进程）
    dataset = create_empty_dataset(
        repo_id=repo_id,
        fps=fps,
        mode=mode,
        bimanual=bimanual,
        dataset_config=dataset_cfg,
    )

    # -------------------------
    # Streaming pipeline: Stage A (parallel npz) + Stage B (serial write) with backpressure
    # -------------------------
    total = len(valid)
    ok_eps = 0
    skipped = 0
    total_frames_written = 0
    stagea_done = 0
    stagea_worker_time_sum = 0.0
    stageb_write_count = 0
    stageb_write_time_sum = 0.0
    ready: Dict[int, Tuple[Optional[str], int, Optional[str]]] = {}
    next_to_write = 0

    def submit_job(ex, idx: int):
        sess_path, m, paths = valid[idx]
        spaths = {k: str(v) for k, v in paths.items()}
        fut = ex.submit(
            _worker_build_episode_npz,
            idx,
            sess_path,
            m,
            spaths,
            fps=fps,
            traj_source=traj_source,
            step1_cfg=step1_cfg,
            tmp_dir=tmp_root,
        )
        return fut

    pending = {}  # future -> idx

    with ProcessPoolExecutor(max_workers=workers) as ex:
        submitted = 0
        pbar = tqdm.tqdm(total=total, desc=f"StageA preprocess (workers={workers})", dynamic_ncols=True)

        # 先填满队列（受 max_inflight 限制）
        while submitted < total and len(pending) < max_inflight:
            fut = submit_job(ex, submitted)
            pending[fut] = submitted
            submitted += 1

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)

            for fut in done:
                idx = pending.pop(fut)
                try:
                    ridx, npz_path, nsteps, err, worker_s = fut.result()
                except Exception as e:
                    ridx, npz_path, nsteps, err, worker_s = idx, None, 0, f"future_exception: {repr(e)}", 0.0

                ready[ridx] = (npz_path, nsteps, err)
                stagea_done += 1
                stagea_worker_time_sum += float(worker_s)
                pbar.update(1)
                if stagea_done % log_every == 0 or stagea_done == total:
                    avg_worker = stagea_worker_time_sum / max(1, stagea_done)
                    print(
                        f"[StageA] done={stagea_done}/{total} avg_worker_s={avg_worker:.2f} "
                        f"pending={len(pending)} ready={len(ready)} next_to_write={next_to_write}"
                    )

                # 保持流水线满载（但不超过 max_inflight）
                while submitted < total and len(pending) < max_inflight:
                    nfut = submit_job(ex, submitted)
                    pending[nfut] = submitted
                    submitted += 1

                # 按 session 原顺序写入，确保 deterministic，不会乱序
                while next_to_write in ready:
                    npz_path2, _nsteps2, err2 = ready.pop(next_to_write)
                    sess_path2, m2, _ = valid[next_to_write]

                    if npz_path2 is None:
                        skipped += 1
                        print(f"[Warn] Skip episode idx={next_to_write} session={sess_path2} reason={err2}")
                        next_to_write += 1
                        continue

                    try:
                        tw0 = time.perf_counter()
                        with suppress_stderr(quiet_ffmpeg):
                            if m2 == "single":
                                written = _write_single_episode_from_npz(dataset, npz_path2, task=task, use_next=use_next)
                            else:
                                written = _write_dual_episode_from_npz(dataset, npz_path2, task=task, use_next=use_next)
                        stageb_write_time_sum += (time.perf_counter() - tw0)
                        stageb_write_count += 1

                        if written <= 0:
                            skipped += 1
                            print(f"[Warn] Episode idx={next_to_write} session={sess_path2} wrote 0 frames. skipped.")
                        else:
                            ok_eps += 1
                            total_frames_written += int(written)
                        if stageb_write_count % log_every == 0:
                            avg_w = stageb_write_time_sum / max(1, stageb_write_count)
                            print(
                                f"[StageB] writes={stageb_write_count} ok={ok_eps} skipped={skipped} "
                                f"avg_write_s={avg_w:.2f} frames_written={total_frames_written}"
                            )

                    except Exception as e:
                        skipped += 1
                        print(f"[WriteFail] idx={next_to_write} session={sess_path2} err={repr(e)}")

                    finally:
                        # 写完立刻删 npz，避免 /dev/shm 或 tmp 堆积
                        if (not args.keep_tmp) and npz_path2:
                            try:
                                Path(npz_path2).unlink(missing_ok=True)
                            except Exception:
                                pass

                    next_to_write += 1

        pbar.close()

    # finalize dataset
    with suppress_stderr(quiet_ffmpeg):
        if hasattr(dataset, "consolidate"):
            dataset.consolidate()
        elif hasattr(dataset, "finalize"):
            dataset.finalize()
        elif hasattr(dataset, "close"):
            dataset.close()

    # convert to v3.0 layout if needed
    if not args.no_v30_convert:
        with suppress_stderr(quiet_ffmpeg):
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
        f"[Done] repo_id={repo_id} bimanual={bimanual} episodes_saved={ok_eps} skipped={skipped} "
        f"frames_written={total_frames_written} codebase_version={ver} HF_LEROBOT_HOME={HF_LEROBOT_HOME}"
    )

    # cleanup tmp
    if auto_cleanup_tmp and (not args.keep_tmp):
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            print(f"[Warn] Failed to cleanup tmp_dir={tmp_root}")
    elif not auto_cleanup_tmp and (not args.keep_tmp):
        # best-effort: if custom tmp dir was provided and now empty, don't删目录（通常用户想复用）
        pass


if __name__ == "__main__":
    main()
