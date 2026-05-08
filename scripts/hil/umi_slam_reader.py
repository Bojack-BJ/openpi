from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class UmiSlamSample:
    timestamp: float
    received_at: float
    position_xyz_m: np.ndarray
    quat_xyzw: np.ndarray
    gripper_open: float
    gripper_raw: float
    confidence: float | None = None


class UmiSlamReader:
    """ROS reader for UMI XV SLAM pose and clamp data."""

    def __init__(
        self,
        *,
        xv_serial: str,
        max_gripper: float,
        pose_max_age_s: float = 0.25,
        gripper_max_age_s: float = 0.5,
        node_name: str = "openpi_hil_umi_slam_reader",
    ) -> None:
        self._xv_serial = xv_serial
        self._max_gripper = float(max_gripper)
        self._pose_max_age_s = float(pose_max_age_s)
        self._gripper_max_age_s = float(gripper_max_age_s)
        self._lock = threading.Lock()
        self._pose: tuple[float, float, np.ndarray, np.ndarray, float | None] | None = None
        self._gripper: tuple[float, float, float] | None = None
        self._clamp_msg_class: type[Any] | None = None

        try:
            import rospy
            from roslib.message import get_message_class
            from xv_sdk.msg import PoseStampedConfidence
        except Exception as exc:
            raise RuntimeError(
                "HIL UMI SLAM reader requires ROS1 packages: rospy, roslib, and xv_sdk.msg.PoseStampedConfidence."
            ) from exc

        self._rospy = rospy
        self._get_message_class = get_message_class
        self._pose_msg_cls = PoseStampedConfidence
        is_initialized = getattr(getattr(rospy, "core", None), "is_initialized", lambda: False)
        if not is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)

        self.slam_topic = f"/xv_sdk/{xv_serial}/slam/pose"
        self.clamp_topic = f"/xv_sdk/{xv_serial}/clamp/Data"
        self._slam_subscriber = rospy.Subscriber(
            self.slam_topic,
            self._pose_msg_cls,
            self._slam_callback,
            queue_size=100,
            buff_size=2**20,
            tcp_nodelay=True,
        )
        self._clamp_subscriber = rospy.Subscriber(
            self.clamp_topic,
            rospy.AnyMsg,
            self._clamp_callback,
            queue_size=100,
            buff_size=2**20,
            tcp_nodelay=True,
        )

    def close(self) -> None:
        for subscriber in (self._slam_subscriber, self._clamp_subscriber):
            try:
                subscriber.unregister()
            except Exception:
                pass

    def wait_until_ready(self, *, timeout_s: float) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.latest(allow_stale=True) is not None:
                return True
            time.sleep(0.02)
        return False

    def latest(self, *, allow_stale: bool = False) -> UmiSlamSample | None:
        with self._lock:
            pose = self._pose
            gripper = self._gripper
        if pose is None or gripper is None:
            return None

        pose_stamp, pose_received_at, pos, quat, confidence = pose
        gripper_stamp, gripper_received_at, gripper_raw = gripper
        now = time.monotonic()
        if not allow_stale:
            if now - pose_received_at > self._pose_max_age_s:
                return None
            if now - gripper_received_at > self._gripper_max_age_s:
                return None

        gripper_open = float(np.clip(gripper_raw / max(self._max_gripper, 1e-6), 0.0, 1.0))
        return UmiSlamSample(
            timestamp=max(float(pose_stamp), float(gripper_stamp)),
            received_at=max(float(pose_received_at), float(gripper_received_at)),
            position_xyz_m=pos.copy(),
            quat_xyzw=quat.copy(),
            gripper_open=gripper_open,
            gripper_raw=float(gripper_raw),
            confidence=confidence,
        )

    def _slam_callback(self, msg: Any) -> None:
        try:
            pose_msg = msg.poseMsg
            stamp = float(pose_msg.header.stamp.to_sec())
            position = pose_msg.pose.position
            orientation = pose_msg.pose.orientation
            pos = np.asarray([position.x, position.y, position.z], dtype=np.float64)
            quat = np.asarray([orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float64)
            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
                return
            norm = float(np.linalg.norm(quat))
            if norm < 1e-8:
                return
            quat = quat / norm
            confidence = float(msg.confidence) if hasattr(msg, "confidence") else None
            with self._lock:
                self._pose = (stamp, time.monotonic(), pos, quat, confidence)
        except Exception:
            return

    def _clamp_callback(self, msg: Any) -> None:
        try:
            if isinstance(msg, self._rospy.AnyMsg):
                if self._clamp_msg_class is None:
                    type_str = msg._connection_header.get("type", "") if hasattr(msg, "_connection_header") else ""
                    if type_str:
                        self._clamp_msg_class = self._get_message_class(type_str)
                if self._clamp_msg_class is None:
                    return
                real_msg = self._clamp_msg_class()
                real_msg.deserialize(msg._buff)
            else:
                real_msg = msg

            value = getattr(real_msg, "data", None)
            if value is None:
                return
            stamp = None
            if hasattr(real_msg, "header") and hasattr(real_msg.header, "stamp"):
                if real_msg.header.stamp.secs != 0 or real_msg.header.stamp.nsecs != 0:
                    stamp = float(real_msg.header.stamp.to_sec())
            if stamp is None:
                stamp = float(self._rospy.get_time())
            with self._lock:
                self._gripper = (stamp, time.monotonic(), float(value))
        except Exception:
            return
