#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件用途：
        将 FastUMI 原始采集数据并行转换为 LeRobot Dataset v2.0 数据格式。
        本脚本同时支持单臂(session/RGB_Images)和双臂(session/left_hand + right_hand)目录布局，
        并在转换过程中完成位姿归一化、夹爪归一化、图像缩放、动作构建与数据落盘。

主要功能：
        1. 递归发现 session* 数据目录，并自动识别 single/dual 两种数据布局。
        2. 读取并对齐视频时间戳、轨迹(traj)与夹爪(clamp)信号，按目标 FPS 下采样。
        3. 执行 Step1 坐标变换与夹爪归一化，生成 observation.state 和 action。
        4. 采用"并行提取 + 串行写入"两阶段流程，保证转换过程高吞吐且结果顺序确定。
        5. 可选将最终生成的 LeRobot 数据集从 HF_LEROBOT_HOME 复制到指定输出目录。

适用场景：
        - 将 FastUMI 采集的原始任务数据批量整理为训练/评测可直接使用的 LeRobot 数据集。
        - 在多核服务器上进行大规模 session 数据转换，需要兼顾速度与可复现顺序。
        - 需要在 merge/slam/vive 多种轨迹来源之间切换，并统一输出数据字段。

核心逻辑：
        1. 主进程扫描 session，校验布局一致性，初始化目标 LeRobotDataset。
        2. Worker 进程按 session 并行读取视频帧和传感数据，完成对齐、归一化、图像 resize，
             并将中间结果写入临时 .npz（不直接写数据集，避免并发写冲突）。
        3. 主进程按 session 索引顺序依次加载 .npz，通过 add_frame + save_episode 串行写入同一数据集。
        4. 最后执行 consolidate/finalize/close，并按需复制输出、清理临时文件。

运行方式：
        python dataprocess_new/fastumi_raw_to_lerobot_v21.py --raw-dir <raw_dir> --repo-id <repo_id> --task <task_name>

运行示例：
        export HF_LEROBOT_HOME=/root/Users/dataset/lerobot_home
        
        # 示例一：单臂数据，使用 merge 轨迹，生成 image 模式数据集
        python dataprocess_new/fastumi_raw_to_lerobot_v21.py \
            --raw-dir /data/fastumi/task_20260121H011 \
            --repo-id fastumi_new/Fold_square_towels \
            --task "Fold square towels" \
            --fps 20 \
            --traj-source merge \
            --mode image \
            --workers 8 \
            --max-inflight 16

        # 示例二：双臂数据，使用 next-step action，并将结果复制到指定目录
        python dataprocess_new/fastumi_raw_to_lerobot_v21.py \
            --raw-dir /root/Users/dataset/task_20260310H072Aa/good \
            --repo-id fastumi/Waste_sorting_a \
            --task "Sort the recyclable waste and hazardous waste into the corresponding bins" \
            --fps 30 \
            --traj-source merge \
            --mode image \
            --workers 48 \
            --next \
            # --output-dir /root/Users/dataset/lerobot_home/fastumi/fruit_classification \
            # --tmp-dir /lumos-vePFS/tmp/fastumi_tmp \

参数说明：
        --raw-dir
            含义：FastUMI 原始数据根目录，脚本会递归查找所有 session* 子目录。
            类型：Path
            是否必填：是
            示例：--raw-dir /data/fastumi/task_20260121H011

        --repo-id
            含义：LeRobot 数据集仓库 ID（相对 HF_LEROBOT_HOME 的目标路径）。
            类型：字符串
            是否必填：是
            示例：--repo-id fastumi_new/Fold_square_towels

        --task
            含义：写入每一帧的任务文本标签。
            类型：字符串
            是否必填：是
            示例：--task "Fold square towels"

        --fps
            含义：目标输出帧率（从 60Hz 原始相机帧下采样）。
            类型：整数
            是否必填：否
            可选值：20 / 30 / 60
            默认值：20
            示例：--fps 30

        --output-dir
            含义：可选复制输出目录；若提供，将把 HF_LEROBOT_HOME/<repo-id> 复制到该目录。
            类型：Path
            是否必填：否
            默认值：None（不复制）
            示例：--output-dir /data/converted

        --mode
            含义：图像存储模式，image 为逐帧图像，video 为视频后端存储。
            类型：字符串
            是否必填：否
            可选值：image / video
            默认值：image
            示例：--mode image

        --traj-source
            含义：轨迹来源类型。
            类型：字符串
            是否必填：否
            可选值：merge / slam / vive
            默认值：merge
            示例：--traj-source slam

        --next
            含义：启用 next-step 动作构建，即 action[t] = state[t+1]（最后一帧重复末状态）。
            类型：布尔标志（flag）
            是否必填：否
            默认值：False（不传即关闭）
            示例：--next

        --workers
            含义：并行提取阶段使用的进程数。
            类型：整数
            是否必填：否
            默认值：max(1, cpu_count // 4)
            示例：--workers 8

        --max-inflight
            含义：同时在队列中的最大 session 任务数；0 表示自动设为 2 * workers。
            类型：整数
            是否必填：否
            默认值：0
            示例：--max-inflight 16

        --tmp-dir
            含义：中间 .npz 临时文件目录，建议使用高速盘或 /dev/shm。
            类型：Path
            是否必填：否
            默认值：$TMPDIR/fastumi_tmp_<repo_id>
            示例：--tmp-dir /dev/shm/fastumi_tmp

        --keep-tmp
            含义：保留中间 .npz 文件，便于调试排查。
            类型：布尔标志（flag）
            是否必填：否
            默认值：False（不传即清理）
            示例：--keep-tmp

        --image-writer-processes
            含义：LeRobot 图像写入进程数。
            类型：整数
            是否必填：否
            默认值：8
            示例：--image-writer-processes 8

        --image-writer-threads
            含义：LeRobot 图像写入线程数。
            类型：整数
            是否必填：否
            默认值：8
            示例：--image-writer-threads 8

注意事项：
        - 运行前需安装依赖（如：opencv-python、numpy、pandas、torch、scipy、tqdm、lerobot）。
        - 运行前需正确配置环境变量 HF_LEROBOT_HOME，并确保目标路径可写。
        - 输入目录下 session 数据布局必须一致，不能 single/dual 混用。
        - traj/clamp/video 任一关键数据缺失时对应 session 会被跳过并计入失败数。
        - 建议先小规模试跑（少量 session）验证轨迹来源与参数设置，再全量转换。

"""

import argparse
import dataclasses
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from scipy.spatial.transform import Rotation as R

try:
    from lerobot.common.constants import HF_LEROBOT_HOME
except Exception:
    from lerobot.constants import HF_LEROBOT_HOME  

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 1e-4
    image_writer_processes: int = 8
    image_writer_threads: int = 8
    video_backend: Optional[str] = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

SOURCE_CAMERA_FPS = 60
STATE_NAMES_8 = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"]

STEP1_CFG = {
    "base_x": 0.3,
    "base_y": 0.0,
    "base_z": 0.145,
    "base_euler_deg": [180, -90, 0.0],
    "max_gripper": 88.0, # 84 for acone/fasttouch, 88 for robotiq
}

# # Xarm6
# STEP1_CFG_DUAL = {
#     "base_x_0": 0.250,
#     "base_y_0": 0.2,
#     "base_z_0": 0.145,
#     "base_euler_deg_0": [180, -90, 0.0],
#     "max_gripper_0": 88.0, # 84 for acone/fasttouch, 88 for robotiq
#     "base_x_1": 0.250,
#     "base_y_1": -0.2,
#     "base_z_1": 0.145,
#     "base_euler_deg_1": [180, -90, 0.0],
#     "max_gripper_1": 88.0, # 84 for acone/fasttouch, 88 for robotiq
# }

# fasttouch
STEP1_CFG_DUAL = {
    "base_x_0": 0.3,
    "base_y_0": 0.04,
    "base_z_0": 0.16,
    "base_euler_deg_0": [0.0, 0.0, 0.0],
    "max_gripper_0": 84.0, # 84 for acone/fasttouch, 88 for robotiq
    "base_x_1": 0.3,
    "base_y_1": -0.04,
    "base_z_1": 0.16,
    "base_euler_deg_1": [0.0, 0.0, 0.0],
    "max_gripper_1": 84.0, # 84 for acone/fasttouch, 88 for robotiq
}


def make_dual_step1_cfgs(step1_cfg_dual: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Create per-arm step1 configs from explicit dual-arm keys.

        Expected keys:
            base_x_0, base_y_0, base_z_0, base_euler_deg_0, max_gripper_0
            base_x_1, base_y_1, base_z_1, base_euler_deg_1, max_gripper_1
        """
        required = [
            "base_x_0", "base_y_0", "base_z_0", "base_euler_deg_0", "max_gripper_0",
            "base_x_1", "base_y_1", "base_z_1", "base_euler_deg_1", "max_gripper_1",
        ]
        missing = [k for k in required if k not in step1_cfg_dual]
        if missing:
            raise ValueError(f"Dual Step1 config missing keys: {missing}")

        cfg0 = {
            "base_x": float(step1_cfg_dual["base_x_0"]),
            "base_y": float(step1_cfg_dual["base_y_0"]),
            "base_z": float(step1_cfg_dual["base_z_0"]),
            "base_euler_deg": step1_cfg_dual["base_euler_deg_0"],
            "max_gripper": float(step1_cfg_dual["max_gripper_0"]),
        }
        cfg1 = {
            "base_x": float(step1_cfg_dual["base_x_1"]),
            "base_y": float(step1_cfg_dual["base_y_1"]),
            "base_z": float(step1_cfg_dual["base_z_1"]),
            "base_euler_deg": step1_cfg_dual["base_euler_deg_1"],
            "max_gripper": float(step1_cfg_dual["max_gripper_1"]),
        }
        return cfg0, cfg1


def find_all_sessions(raw_root: Path) -> List[Path]:
    found: List[Path] = []
    for r, dirs, _files in os.walk(raw_root):
        for d in dirs:
            if d.startswith("session"):
                found.append(Path(r) / d)
    return sorted(found)


def detect_layout(session_path: Path) -> Tuple[str, Dict[str, Path]]:
    """
    Returns:
      ("single", {"single": path}) or ("dual", {"left": path, "right": path}) or ("invalid", {})
    """
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
        # Newer layout: SLAM_Poses/slam_processed.txt
        # Older layout: SLAM_Poses/slam_raw.txt
        slam_dir = data_path / "SLAM_Poses"
        cand = [
            slam_dir / "slam_processed.txt",
            slam_dir / "slam_raw.txt",
            # optional extra fallback if some old dumps put it at root:
            data_path / "slam_processed.txt",
            data_path / "slam_raw.txt",
        ]
        for p in cand:
            if p.exists():
                return read_trj_txt(p)
        raise FileNotFoundError(
            f"No slam trajectory found. Tried: {', '.join(str(p) for p in cand)}"
        )

    if traj_source == "vive":
        return read_trj_txt(data_path / "Vive_Poses" / "vive_data_tum.txt")

    raise ValueError(f"Unknown traj_source: {traj_source}")



def load_arm_sources(data_path: Path, traj_source: str):
    """
    Returns dict with:
      traj (df), clamp (df), timestamps (df), video_path (Path)
    or None if missing.
    """
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
    pose_row: np.ndarray,  # [PosX, PosY, PosZ, Qx, Qy, Qz, Qw]
    clamp_value: float,
    T_base_to_local: np.ndarray,
    max_gripper: float,
) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = [float(v) for v in pose_row.tolist()]
    x_b, y_b, z_b, qx_b, qy_b, qz_b, qw_b = transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local)

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


def resize_mask_no_crop(mask: np.ndarray, out_hw: int = 224) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = cv2.resize(mask, (out_hw, out_hw), interpolation=cv2.INTER_NEAREST)
    return ((mask > 0).astype(np.uint8) * 255)[..., None]


def _parse_frame_range(frame_range: str) -> Tuple[int, int]:
    start, end = frame_range.strip().strip("[]").split(",", maxsplit=1)
    return int(start), int(end) + 1


def _segments_from_subtask_json(path: Path, *, fallback_end_frame: int) -> List[Tuple[int, int, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    segments: List[Tuple[int, int, str]] = []
    for item in payload.get("subtask_instruction", []):
        if not isinstance(item, dict):
            continue
        for frame_range, subtask in item.items():
            start, end = _parse_frame_range(str(frame_range))
            text = str(subtask).strip()
            if text and start < end:
                segments.append((start, end, text))
    if segments:
        return segments

    subtasks = [str(item).strip() for item in payload.get("subtask", [])]
    boundaries = [int(item) for item in payload.get("boundaries_frame_indices", [])]
    if not subtasks:
        return []
    starts = [0, *boundaries]
    ends = [*boundaries, int(fallback_end_frame)]
    return [
        (start, end, subtask)
        for start, end, subtask in zip(starts, ends, subtasks)
        if subtask and start < end
    ]


def _subtasks_for_frame_indices(session_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    if frame_indices.size == 0:
        return np.asarray([], dtype="<U1")
    candidates = [
        session_path / "subtask.json",
        session_path.parent / "subtask.json",
    ]
    fallback_end_frame = int(np.max(frame_indices)) + 1
    segments: List[Tuple[int, int, str]] = []
    for candidate in candidates:
        segments = _segments_from_subtask_json(candidate, fallback_end_frame=fallback_end_frame)
        if segments:
            break

    values: List[str] = []
    for frame_index in frame_indices.tolist():
        value = ""
        for start, end, subtask in segments:
            if start <= int(frame_index) < end:
                value = subtask
                break
        values.append(value)
    max_len = max([1, *(len(value) for value in values)])
    return np.asarray(values, dtype=f"<U{max_len}")


def _masks_for_frame_indices(session_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    mask_dir = session_path / "annotation" / "masks"
    masks = np.zeros((len(frame_indices), 224, 224, 1), dtype=np.uint8)
    if not mask_dir.exists():
        return masks
    for out_index, frame_index in enumerate(frame_indices.tolist()):
        mask_path = mask_dir / f"{int(frame_index):06d}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        masks[out_index] = resize_mask_no_crop(mask, out_hw=224)
    return masks


def _segments_from_subtask_array(subtasks: np.ndarray) -> List[Dict[str, object]]:
    values = [str(value).strip() for value in subtasks.tolist()]
    segments: List[Dict[str, object]] = []
    start = 0
    current = values[0] if values else ""
    for index, value in enumerate(values[1:], start=1):
        if value == current:
            continue
        if current:
            segments.append({"start_frame": start, "end_frame": index, "subtask": current})
        start = index
        current = value
    if current:
        segments.append({"start_frame": start, "end_frame": len(values), "subtask": current})
    return segments


def _subtask_segments_from_npz(npz_path: Path) -> List[Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=False)
    if "subtask" not in z.files:
        return []
    return _segments_from_subtask_array(z["subtask"])


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


def read_frames_by_indices(video_path: Path, frame_indices: np.ndarray) -> Optional[List[np.ndarray]]:
    if frame_indices.size == 0:
        return []

    # Require non-decreasing indices
    if np.any(frame_indices[1:] < frame_indices[:-1]):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frames: List[np.ndarray] = []
        for fidx in frame_indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, fr = cap.read()
            if not ok:
                cap.release()
                return None
            frames.append(fr)
        cap.release()
        return frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames: List[np.ndarray] = []
    first = int(frame_indices[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    cur = first

    for target in frame_indices.tolist():
        target = int(target)
        # skip until reach target
        while cur < target:
            ok, _ = cap.read()
            if not ok:
                cap.release()
                return None
            cur += 1
        ok, fr = cap.read()
        if not ok:
            cap.release()
            return None
        frames.append(fr)
        cur += 1

    cap.release()
    return frames


def create_empty_dataset(
    *,
    repo_id: str,
    fps: int,
    mode: Literal["video", "image"],
    bimanual: bool,
    include_guidance: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
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
        if include_guidance:
            features["observation.masks.front_mask"] = {
                "dtype": "uint8",
                "shape": (224, 224, 1),
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
        if include_guidance:
            features["observation.masks.robot_0_mask"] = {
                "dtype": "uint8",
                "shape": (224, 224, 1),
                "names": ["height", "width", "channels"],
            }
            features["observation.masks.robot_1_mask"] = {
                "dtype": "uint8",
                "shape": (224, 224, 1),
                "names": ["height", "width", "channels"],
            }

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

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



def _extract_single_to_npz(
    data_path: str,
    out_npz: str,
    *,
    fps: int,
    traj_source: str,
    use_next: bool,
    step1_cfg: Dict[str, float],
    include_guidance: bool,
) -> Tuple[bool, str]:
    data_path = str(data_path)
    out_npz = str(out_npz)
    try:
        src = load_arm_sources(Path(data_path), traj_source)
        if src is None:
            return False, "missing_sources"

        step = max(1, int(SOURCE_CAMERA_FPS / fps))
        ts_df = src["timestamps"].iloc[::step].reset_index(drop=True)
        if ts_df.empty:
            return False, "empty_timestamps"

        # arrays
        t_arr = ts_df["timestamp"].to_numpy(dtype=np.float64)
        fidx_arr = ts_df["frame_index"].to_numpy(dtype=np.int64)

        traj_df = src["traj"].sort_values("timestamp").reset_index(drop=True)
        clamp_df = src["clamp"].sort_values("timestamp").reset_index(drop=True)

        traj_ts = traj_df["timestamp"].to_numpy(dtype=np.float64)
        clamp_ts = clamp_df["timestamp"].to_numpy(dtype=np.float64)

        traj_i = nearest_indices(traj_ts, t_arr)
        clamp_i = nearest_indices(clamp_ts, t_arr)

        pose_np = traj_df.loc[traj_i, ["Pos X", "Pos Y", "Pos Z", "Q_X", "Q_Y", "Q_Z", "Q_W"]].to_numpy(dtype=np.float64)
        clamp_np = clamp_df.loc[clamp_i, "clamp"].to_numpy(dtype=np.float64)

        frames_bgr = read_frames_by_indices(src["video_path"], fidx_arr)
        if frames_bgr is None:
            return False, "video_read_failed"

        T_base = build_T_base_to_local(step1_cfg)
        max_g = float(step1_cfg["max_gripper"])

        T = len(frames_bgr)
        imgs = np.empty((T, 224, 224, 3), dtype=np.uint8)
        states = np.empty((T, 8), dtype=np.float32)

        for k in range(T):
            imgs[k] = resize_rgb_no_crop(frames_bgr[k], out_hw=224)
            states[k] = normalize_state_from_raw(pose_np[k], float(clamp_np[k]), T_base, max_g)

        actions = build_actions_from_states(states, use_next)

        payload = {
            "kind": np.array("single"),
            "states": states,
            "actions": actions,
            "images": imgs,
        }
        if include_guidance:
            path_obj = Path(data_path)
            payload["front_mask"] = _masks_for_frame_indices(path_obj, fidx_arr)
            payload["subtask"] = _subtasks_for_frame_indices(path_obj, fidx_arr)

        np.savez_compressed(out_npz, **payload)
        return True, "ok"
    except Exception as e:
        return False, f"exception: {e}\n{traceback.format_exc()}"


def _extract_dual_to_npz(
    left_path: str,
    right_path: str,
    out_npz: str,
    *,
    fps: int,
    traj_source: str,
    use_next: bool,
    step1_cfg_robot0: Dict[str, float],
    step1_cfg_robot1: Dict[str, float],
    include_guidance: bool,
) -> Tuple[bool, str]:
    left_path = str(left_path)
    right_path = str(right_path)
    out_npz = str(out_npz)
    try:
        left = load_arm_sources(Path(left_path), traj_source)
        right = load_arm_sources(Path(right_path), traj_source)
        if left is None or right is None:
            return False, "missing_sources"

        step = max(1, int(SOURCE_CAMERA_FPS / fps))
        master_ts = left["timestamps"].iloc[::step].reset_index(drop=True)
        if master_ts.empty:
            return False, "empty_timestamps"

        t_master = master_ts["timestamp"].to_numpy(dtype=np.float64)
        fidx_l = master_ts["frame_index"].to_numpy(dtype=np.int64)

        # right camera mapping by nearest timestamp
        right_ts_df = right["timestamps"]
        r_cam_ts = right_ts_df["timestamp"].to_numpy(dtype=np.float64)
        r_cam_fidx = right_ts_df["frame_index"].to_numpy(dtype=np.int64)
        r_cam_pick = nearest_indices(r_cam_ts, t_master)
        fidx_r = r_cam_fidx[r_cam_pick]
        t_right = r_cam_ts[r_cam_pick]

        # sort traj/clamp
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

        frames_l = read_frames_by_indices(left["video_path"], fidx_l)
        frames_r = read_frames_by_indices(right["video_path"], fidx_r)
        if frames_l is None or frames_r is None:
            return False, "video_read_failed"

        T_base_0 = build_T_base_to_local(step1_cfg_robot0)
        T_base_1 = build_T_base_to_local(step1_cfg_robot1)
        max_g_0 = float(step1_cfg_robot0["max_gripper"])
        max_g_1 = float(step1_cfg_robot1["max_gripper"])

        T = min(len(frames_l), len(frames_r), len(t_master))
        if T <= 0:
            return False, "no_frames"

        imgs_l = np.empty((T, 224, 224, 3), dtype=np.uint8)
        imgs_r = np.empty((T, 224, 224, 3), dtype=np.uint8)
        s0 = np.empty((T, 8), dtype=np.float32)
        s1 = np.empty((T, 8), dtype=np.float32)

        for k in range(T):
            imgs_l[k] = resize_rgb_no_crop(frames_l[k], out_hw=224)
            imgs_r[k] = resize_rgb_no_crop(frames_r[k], out_hw=224)
            s0[k] = normalize_state_from_raw(l_pose_np[k], float(l_clamp_np[k]), T_base_0, max_g_0)
            s1[k] = normalize_state_from_raw(r_pose_np[k], float(r_clamp_np[k]), T_base_1, max_g_1)

        states = np.concatenate([s0, s1], axis=1).astype(np.float32)  # (T,16)
        actions = build_actions_from_states(states, use_next)

        payload = {
            "kind": np.array("dual"),
            "s0": s0,
            "s1": s1,
            "states": states,
            "actions": actions,
            "img_l": imgs_l,
            "img_r": imgs_r,
        }
        if include_guidance:
            left_path_obj = Path(left_path)
            right_path_obj = Path(right_path)
            payload["robot_0_mask"] = _masks_for_frame_indices(left_path_obj, fidx_l[:T])
            payload["robot_1_mask"] = _masks_for_frame_indices(right_path_obj, fidx_r[:T])
            payload["subtask"] = _subtasks_for_frame_indices(left_path_obj, fidx_l[:T])

        np.savez_compressed(out_npz, **payload)
        return True, "ok"
    except Exception as e:
        return False, f"exception: {e}\n{traceback.format_exc()}"


def write_single_from_npz(dataset: LeRobotDataset, npz_path: Path, *, task: str, include_guidance: bool) -> bool:
    z = np.load(npz_path, allow_pickle=False)
    states = z["states"].astype(np.float32)
    actions = z["actions"].astype(np.float32)
    images = z["images"].astype(np.uint8)
    masks = z["front_mask"].astype(np.uint8) if include_guidance and "front_mask" in z.files else None

    T = states.shape[0]
    for i in range(T):
        frame = {
            "task": task,
            "observation.state": torch.from_numpy(states[i]).to(torch.float32),
            "action": torch.from_numpy(actions[i]).to(torch.float32),
            "robot_0_state": torch.from_numpy(states[i]).to(torch.float32),
            "observation.images.front": images[i],
        }
        if include_guidance:
            frame["observation.masks.front_mask"] = masks[i] if masks is not None else np.zeros((224, 224, 1), dtype=np.uint8)
        dataset.add_frame(frame)
    dataset.save_episode()
    return True


def write_dual_from_npz(dataset: LeRobotDataset, npz_path: Path, *, task: str, include_guidance: bool) -> bool:
    z = np.load(npz_path, allow_pickle=False)
    s0 = z["s0"].astype(np.float32)
    s1 = z["s1"].astype(np.float32)
    states = z["states"].astype(np.float32)
    actions = z["actions"].astype(np.float32)
    img_l = z["img_l"].astype(np.uint8)
    img_r = z["img_r"].astype(np.uint8)
    mask_l = z["robot_0_mask"].astype(np.uint8) if include_guidance and "robot_0_mask" in z.files else None
    mask_r = z["robot_1_mask"].astype(np.uint8) if include_guidance and "robot_1_mask" in z.files else None

    T = states.shape[0]
    for i in range(T):
        frame = {
            "task": task,
            "observation.state": torch.from_numpy(states[i]).to(torch.float32),
            "action": torch.from_numpy(actions[i]).to(torch.float32),
            "robot_0_state": torch.from_numpy(s0[i]).to(torch.float32),
            "robot_1_state": torch.from_numpy(s1[i]).to(torch.float32),
            "observation.images.robot_0_image": img_l[i],
            "observation.images.robot_1_image": img_r[i],
        }
        if include_guidance:
            frame["observation.masks.robot_0_mask"] = mask_l[i] if mask_l is not None else np.zeros((224, 224, 1), dtype=np.uint8)
            frame["observation.masks.robot_1_mask"] = mask_r[i] if mask_r is not None else np.zeros((224, 224, 1), dtype=np.uint8)
        dataset.add_frame(frame)
    dataset.save_episode()
    return True


def main():
    parser = argparse.ArgumentParser(description="FastUMI rawdata -> LeRobot v2.1 (safe parallel) with Step1 processing.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Raw data root (search recursively for session*)")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot repo id, e.g. fastumi/task_xxx")
    parser.add_argument("--task", type=str, required=True, help="Task string stored in each frame")
    parser.add_argument("--fps", type=int, default=20, choices=[20, 30, 60], help="Target fps (downsample from 60Hz)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional copy-out directory")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video"], help="Store images as image or video")
    parser.add_argument("--traj-source", type=str, default="merge", choices=["merge", "slam", "vive"], help="Trajectory source")
    parser.add_argument("--next", action="store_true", help="Use next-step action: action[t]=state[t+1], last repeats")

    # parallel knobs
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 4),
                        help="Number of worker processes for per-session extraction")
    parser.add_argument("--max-inflight", type=int, default=0,
                        help="Max inflight sessions submitted to workers (0 => 2*workers)")
    parser.add_argument("--tmp-dir", type=Path, default=None,
                        help="Temp dir for episode .npz (recommend /dev/shm if enough RAM)")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep tmp npz files (debug)")
    parser.add_argument(
        "--include-guidance",
        action="store_true",
        help="Store optional annotation/masks/*.png and subtask.json fields in the LeRobot 2.1 dataset.",
    )

    # writer config (avoid oversubscribe)
    parser.add_argument("--image-writer-processes", type=int, default=DEFAULT_DATASET_CONFIG.image_writer_processes)
    parser.add_argument("--image-writer-threads", type=int, default=DEFAULT_DATASET_CONFIG.image_writer_threads)

    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    repo_id: str = args.repo_id
    task: str = args.task
    fps: int = args.fps
    traj_source: str = args.traj_source
    use_next: bool = args.next
    mode: Literal["image", "video"] = args.mode  # type: ignore
    output_dir: Optional[Path] = args.output_dir
    include_guidance: bool = bool(args.include_guidance)

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
                f"Mixed layouts under {raw_dir}: already '{layout}', but session {s} is '{m}'. "
                f"Please convert single/dual tasks separately."
            )

    if not valid or layout is None:
        raise RuntimeError(f"No valid sessions found under: {raw_dir}")

    bimanual = (layout == "dual")
    step1_cfg = STEP1_CFG
    step1_cfg_dual = STEP1_CFG_DUAL
    step1_cfg_robot0, step1_cfg_robot1 = make_dual_step1_cfgs(step1_cfg_dual)

    # tmp dir
    if args.tmp_dir is None:
        tmp_dir = Path(os.getenv("TMPDIR", "/tmp")) / f"fastumi_tmp_{repo_id.replace('/', '_')}"
    else:
        tmp_dir = args.tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)

    workers = int(args.workers)
    max_inflight = int(args.max_inflight) if int(args.max_inflight) > 0 else max(1, 2 * workers)

    # dataset config
    dataset_cfg = DatasetConfig(
        use_videos=(mode == "video"),
        tolerance_s=DEFAULT_DATASET_CONFIG.tolerance_s,
        image_writer_processes=int(args.image_writer_processes),
        image_writer_threads=int(args.image_writer_threads),
        video_backend=DEFAULT_DATASET_CONFIG.video_backend,
    )

    dataset = create_empty_dataset(
        repo_id=repo_id,
        fps=fps,
        mode=mode,
        bimanual=bimanual,
        include_guidance=include_guidance,
        dataset_config=dataset_cfg,
    )

    total = len(valid)
    ok_count = 0
    fail_count = 0
    subtask_sidecar: Dict[str, Dict[str, List[Dict[str, object]]]] = {"episodes": {}}

    ready: Dict[int, Tuple[bool, str, Path]] = {} 
    next_to_write = 0

    def submit_job(ex, idx: int):
        s_path, m, paths = valid[idx]
        npz_path = tmp_dir / f"{idx:06d}_{s_path.name}.npz"
        if m == "single":
            fut = ex.submit(
                _extract_single_to_npz,
                str(paths["single"]),
                str(npz_path),
                fps=fps,
                traj_source=traj_source,
                use_next=use_next,
                step1_cfg=step1_cfg,
                include_guidance=include_guidance,
            )
        else:
            fut = ex.submit(
                _extract_dual_to_npz,
                str(paths["left"]),
                str(paths["right"]),
                str(npz_path),
                fps=fps,
                traj_source=traj_source,
                use_next=use_next,
                step1_cfg_robot0=step1_cfg_robot0,
                step1_cfg_robot1=step1_cfg_robot1,
                include_guidance=include_guidance,
            )
        return fut, npz_path

    pending = {}  # future -> (idx, npz_path)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        submitted = 0

        pbar = tqdm.tqdm(total=total, desc=f"Extract (workers={workers})", dynamic_ncols=True)
        while submitted < total and len(pending) < max_inflight:
            fut, npz = submit_job(ex, submitted)
            pending[fut] = (submitted, npz)
            submitted += 1

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                idx, npz_path = pending.pop(fut)
                try:
                    ok, msg = fut.result()
                except Exception as e:
                    ok, msg = False, f"future_exception: {e}\n{traceback.format_exc()}"

                ready[idx] = (ok, msg, npz_path)
                pbar.update(1)

                # keep pipeline full
                while submitted < total and len(pending) < max_inflight:
                    nfut, nnpz = submit_job(ex, submitted)
                    pending[nfut] = (submitted, nnpz)
                    submitted += 1

                # write in order as much as possible
                while next_to_write in ready:
                    ok2, msg2, npz2 = ready.pop(next_to_write)
                    if not ok2:
                        fail_count += 1
                        # print a compact error
                        print(f"[Skip] idx={next_to_write} reason={msg2.splitlines()[0]}")
                        # remove partial npz if any
                        if npz2.exists() and not args.keep_tmp:
                            try:
                                npz2.unlink()
                            except Exception:
                                pass
                        next_to_write += 1
                        continue

                    # write episode
                    try:
                        episode_index = ok_count
                        segments = _subtask_segments_from_npz(npz2) if include_guidance else []
                        if not bimanual:
                            write_single_from_npz(dataset, npz2, task=task, include_guidance=include_guidance)
                        else:
                            write_dual_from_npz(dataset, npz2, task=task, include_guidance=include_guidance)
                        if segments:
                            subtask_sidecar["episodes"][str(episode_index)] = {"segments": segments}
                        ok_count += 1
                    except Exception as e:
                        fail_count += 1
                        print(f"[WriteFail] idx={next_to_write} err={e}")
                    finally:
                        if npz2.exists() and not args.keep_tmp:
                            try:
                                npz2.unlink()
                            except Exception:
                                pass
                    next_to_write += 1

        pbar.close()

    # finalize dataset
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    elif hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "close"):
        dataset.close()

    if include_guidance and subtask_sidecar["episodes"]:
        sidecar_path = HF_LEROBOT_HOME / repo_id / "subtask_segments.json"
        with sidecar_path.open("w", encoding="utf-8") as stream:
            json.dump(subtask_sidecar, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(f"[Info] Wrote subtask sidecar: {sidecar_path}")

    # copy out
    if output_dir is not None:
        target = output_dir / repo_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(HF_LEROBOT_HOME / repo_id, target)
        print(f"[Info] Dataset copied to {target}")

    if not args.keep_tmp:
        # best-effort cleanup tmp dir if empty
        try:
            if tmp_dir.exists() and len(list(tmp_dir.glob("*.npz"))) == 0:
                tmp_dir.rmdir()
        except Exception:
            pass

    print(
        f"[Done] repo_id={repo_id} bimanual={bimanual} "
        f"episodes_ok={ok_count} episodes_fail={fail_count} "
        f"HF_LEROBOT_HOME={HF_LEROBOT_HOME}"
    )


if __name__ == "__main__":
    main()
