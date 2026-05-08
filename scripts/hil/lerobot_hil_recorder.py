from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


STATE_NAMES_8 = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"]


@dataclasses.dataclass
class HilFrame:
    state_8d: np.ndarray
    action_8d: np.ndarray
    image_front: np.ndarray
    task: str
    subtask: str
    timestamp: float
    control_mode: str
    base_action_7d: np.ndarray | None
    human_action_7d: np.ndarray | None
    takeover_id: int | None


def action7_rpy_deg_to_action8_quat(action_7d: np.ndarray) -> np.ndarray:
    action = np.asarray(action_7d, dtype=np.float32)
    quat = R.from_euler("xyz", action[3:6].astype(np.float64), degrees=True).as_quat().astype(np.float32)
    return np.concatenate([action[:3], quat, action[6:7]]).astype(np.float32)


class SingleArmHilRecorder:
    """Buffers HIL rollout frames and writes successful episodes to LeRobot format."""

    def __init__(
        self,
        *,
        repo_id: str,
        fps: int,
        robot_type: str = "fasttouch",
        overwrite: bool = False,
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
    ) -> None:
        self.repo_id = repo_id
        self.fps = int(fps)
        self.robot_type = robot_type
        self.overwrite = bool(overwrite)
        self.image_writer_processes = int(image_writer_processes)
        self.image_writer_threads = int(image_writer_threads)
        self._frames: list[HilFrame] = []
        self._dataset = None
        self._dataset_root: pathlib.Path | None = None
        self._episode_index = 0

    def record(
        self,
        *,
        state_8d: np.ndarray,
        action_8d: np.ndarray,
        image_front: np.ndarray,
        task: str,
        subtask: str = "",
        timestamp: float,
        control_mode: str,
        base_action_7d: np.ndarray | None = None,
        human_action_7d: np.ndarray | None = None,
        takeover_id: int | None = None,
    ) -> None:
        self._frames.append(
            HilFrame(
                state_8d=np.asarray(state_8d, dtype=np.float32).copy(),
                action_8d=np.asarray(action_8d, dtype=np.float32).copy(),
                image_front=np.asarray(image_front, dtype=np.uint8).copy(),
                task=str(task),
                subtask=str(subtask),
                timestamp=float(timestamp),
                control_mode=str(control_mode),
                base_action_7d=None if base_action_7d is None else np.asarray(base_action_7d, dtype=np.float32).copy(),
                human_action_7d=(
                    None if human_action_7d is None else np.asarray(human_action_7d, dtype=np.float32).copy()
                ),
                takeover_id=takeover_id,
            )
        )

    def drop_recent_policy_frames(self, count: int) -> int:
        if count <= 0:
            return 0
        drop_indices: list[int] = []
        for index in range(len(self._frames) - 1, -1, -1):
            if self._frames[index].control_mode != "policy":
                continue
            drop_indices.append(index)
            if len(drop_indices) >= count:
                break
        for index in drop_indices:
            del self._frames[index]
        return len(drop_indices)

    def discard_episode(self) -> int:
        count = len(self._frames)
        self._frames.clear()
        return count

    def has_frames(self) -> bool:
        return bool(self._frames)

    def save_episode(self) -> int:
        if not self._frames:
            return 0
        dataset = self._get_dataset()
        episode_index = self._episode_index
        for frame in self._frames:
            payload = {
                "task": frame.task,
                "observation.state": frame.state_8d,
                "action": frame.action_8d,
                "robot_0_state": frame.state_8d,
                "observation.images.front": frame.image_front,
                "subtask": frame.subtask,
            }
            dataset.add_frame(payload)
        dataset.save_episode()
        self._write_metadata_sidecar(episode_index, self._frames)
        count = len(self._frames)
        self._frames.clear()
        self._episode_index += 1
        return count

    def close(self) -> None:
        dataset = self._dataset
        if dataset is None:
            return
        if hasattr(dataset, "consolidate"):
            dataset.consolidate()
        elif hasattr(dataset, "finalize"):
            dataset.finalize()
        elif hasattr(dataset, "close"):
            dataset.close()

    def _get_dataset(self):
        if self._dataset is not None:
            return self._dataset

        try:
            from lerobot.common.constants import HF_LEROBOT_HOME
        except Exception:
            from lerobot.constants import HF_LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self._dataset_root = pathlib.Path(HF_LEROBOT_HOME) / self.repo_id
        if self._dataset_root.exists():
            if self.overwrite:
                shutil.rmtree(self._dataset_root)
            else:
                raise FileExistsError(
                    f"HIL output dataset already exists: {self._dataset_root}. "
                    "Use --hil_overwrite_dataset or choose a new --hil_output_repo_id."
                )

        features: dict[str, dict[str, Any]] = {
            "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8},
            "action": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8},
            "robot_0_state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8},
            "subtask": {"dtype": "string", "shape": (1,), "names": ["text"]},
            "observation.images.front": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            },
        }
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            robot_type=self.robot_type,
            features=features,
            use_videos=False,
            tolerance_s=1e-4,
            image_writer_processes=self.image_writer_processes,
            image_writer_threads=self.image_writer_threads,
        )
        return self._dataset

    def _write_metadata_sidecar(self, episode_index: int, frames: list[HilFrame]) -> None:
        if self._dataset_root is None:
            return
        metadata_path = self._dataset_root / "hil_metadata.jsonl"
        with metadata_path.open("a", encoding="utf-8") as stream:
            for frame_index, frame in enumerate(frames):
                row = {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": frame.timestamp,
                    "control_mode": frame.control_mode,
                    "takeover_id": frame.takeover_id,
                    "base_action_7d": None
                    if frame.base_action_7d is None
                    else frame.base_action_7d.astype(float).tolist(),
                    "human_action_7d": None
                    if frame.human_action_7d is None
                    else frame.human_action_7d.astype(float).tolist(),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
