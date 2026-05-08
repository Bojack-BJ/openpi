from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclasses.dataclass(frozen=True)
class RelativePoseTarget:
    position_xyz_m: np.ndarray
    euler_xyz_rad: np.ndarray
    translation_clamped: bool
    rotation_clamped: bool


def parse_signed_axes(spec: str) -> tuple[tuple[int, float], tuple[int, float], tuple[int, float]]:
    """Parse mappings like 'x,y,z' or '-y,x,z'."""
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    parts = [part.strip().lower() for part in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated axes, got: {spec!r}")

    parsed = []
    used = set()
    for part in parts:
        sign = -1.0 if part.startswith("-") else 1.0
        axis = part[1:] if part.startswith("-") else part
        if axis not in axis_to_index:
            raise ValueError(f"Unsupported axis {part!r}; expected x, y, z with optional '-'")
        index = axis_to_index[axis]
        if index in used:
            raise ValueError(f"Axis {axis!r} is repeated in mapping {spec!r}")
        used.add(index)
        parsed.append((index, sign))
    return tuple(parsed)  # type: ignore[return-value]


def apply_signed_axes(
    value: np.ndarray,
    axes: tuple[tuple[int, float], tuple[int, float], tuple[int, float]],
) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.asarray([sign * value[index] for index, sign in axes], dtype=np.float64)


class RelativePoseMapper:
    """Maps UMI relative SLAM motion onto a robot TCP pose locked at takeover start."""

    def __init__(
        self,
        *,
        axes: tuple[tuple[int, float], tuple[int, float], tuple[int, float]],
        translation_scale: float,
        delta_frame: str,
        max_delta_xyz: float,
        max_delta_rpy_deg: float,
    ) -> None:
        if delta_frame not in ("local", "world"):
            raise ValueError(f"delta_frame must be 'local' or 'world', got {delta_frame!r}")
        self._axes = axes
        self._translation_scale = float(translation_scale)
        self._delta_frame = delta_frame
        self._max_delta_xyz = float(max_delta_xyz)
        self._max_delta_rot_rad = np.deg2rad(float(max_delta_rpy_deg))
        self._robot_start_pos: np.ndarray | None = None
        self._robot_start_rot: R | None = None
        self._umi_start_pos: np.ndarray | None = None
        self._umi_start_rot: R | None = None

    def begin(
        self,
        *,
        robot_tcp_position_xyz_m: np.ndarray,
        robot_tcp_euler_xyz_rad: np.ndarray,
        umi_position_xyz_m: np.ndarray,
        umi_quat_xyzw: np.ndarray,
    ) -> None:
        self._robot_start_pos = np.asarray(robot_tcp_position_xyz_m, dtype=np.float64)
        self._robot_start_rot = R.from_euler("xyz", np.asarray(robot_tcp_euler_xyz_rad, dtype=np.float64))
        self._umi_start_pos = np.asarray(umi_position_xyz_m, dtype=np.float64)
        self._umi_start_rot = R.from_quat(np.asarray(umi_quat_xyzw, dtype=np.float64))

    def reset(self) -> None:
        self._robot_start_pos = None
        self._robot_start_rot = None
        self._umi_start_pos = None
        self._umi_start_rot = None

    @property
    def active(self) -> bool:
        return self._robot_start_pos is not None

    def target(self, *, umi_position_xyz_m: np.ndarray, umi_quat_xyzw: np.ndarray) -> RelativePoseTarget:
        if (
            self._robot_start_pos is None
            or self._robot_start_rot is None
            or self._umi_start_pos is None
            or self._umi_start_rot is None
        ):
            raise RuntimeError("RelativePoseMapper.begin() must be called before target().")

        umi_pos = np.asarray(umi_position_xyz_m, dtype=np.float64)
        umi_rot = R.from_quat(np.asarray(umi_quat_xyzw, dtype=np.float64))

        delta_xyz = umi_pos - self._umi_start_pos
        if self._delta_frame == "local":
            delta_xyz = self._umi_start_rot.inv().apply(delta_xyz)
        delta_xyz = apply_signed_axes(delta_xyz, self._axes) * self._translation_scale

        translation_clamped = False
        delta_norm = float(np.linalg.norm(delta_xyz))
        if self._max_delta_xyz > 0.0 and delta_norm > self._max_delta_xyz:
            delta_xyz = delta_xyz * (self._max_delta_xyz / max(delta_norm, 1e-9))
            translation_clamped = True

        if self._delta_frame == "local":
            target_pos = self._robot_start_pos + self._robot_start_rot.apply(delta_xyz)
        else:
            target_pos = self._robot_start_pos + delta_xyz

        delta_rot = self._umi_start_rot.inv() * umi_rot
        delta_rotvec = delta_rot.as_rotvec()
        rotation_clamped = False
        rot_norm = float(np.linalg.norm(delta_rotvec))
        if self._max_delta_rot_rad > 0.0 and rot_norm > self._max_delta_rot_rad:
            delta_rotvec = delta_rotvec * (self._max_delta_rot_rad / max(rot_norm, 1e-9))
            delta_rot = R.from_rotvec(delta_rotvec)
            rotation_clamped = True

        target_rot = self._robot_start_rot * delta_rot
        return RelativePoseTarget(
            position_xyz_m=np.asarray(target_pos, dtype=np.float64),
            euler_xyz_rad=np.asarray(target_rot.as_euler("xyz"), dtype=np.float64),
            translation_clamped=translation_clamped,
            rotation_clamped=rotation_clamped,
        )
