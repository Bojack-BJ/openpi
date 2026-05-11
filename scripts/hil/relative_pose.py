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
    raw_delta_xyz_m: np.ndarray
    mapped_delta_xyz_m: np.ndarray
    command_delta_xyz_m: np.ndarray
    raw_delta_rotvec_rad: np.ndarray
    mapped_delta_rotvec_rad: np.ndarray


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


def signed_axes_to_matrix(axes: tuple[tuple[int, float], tuple[int, float], tuple[int, float]]) -> np.ndarray:
    """Return a matrix that maps raw SLAM xyz vectors into the configured frame."""
    matrix = np.zeros((3, 3), dtype=np.float64)
    for output_index, (input_index, sign) in enumerate(axes):
        matrix[output_index, input_index] = sign
    return matrix


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
        self._frame_matrix = signed_axes_to_matrix(axes)
        determinant = float(np.linalg.det(self._frame_matrix))
        if determinant < 0.0:
            raise ValueError(
                "hil_slam_axes must describe a right-handed frame when mapping orientation; "
                f"got determinant {determinant:.1f} for axes={axes!r}"
            )
        self._frame_rot = R.from_matrix(self._frame_matrix)
        self._translation_scale = float(translation_scale)
        self._delta_frame = delta_frame
        self._max_delta_xyz = float(max_delta_xyz)
        self._max_delta_rot_rad = np.deg2rad(float(max_delta_rpy_deg))
        self._robot_start_pos: np.ndarray | None = None
        self._robot_start_rot: R | None = None
        self._umi_start_raw_pos: np.ndarray | None = None
        self._umi_start_raw_rot: R | None = None
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
        self._umi_start_raw_pos = np.asarray(umi_position_xyz_m, dtype=np.float64)
        self._umi_start_raw_rot = R.from_quat(np.asarray(umi_quat_xyzw, dtype=np.float64))
        self._umi_start_pos, self._umi_start_rot = self.map_umi_pose(
            position_xyz_m=umi_position_xyz_m,
            quat_xyzw=umi_quat_xyzw,
        )

    def reset(self) -> None:
        self._robot_start_pos = None
        self._robot_start_rot = None
        self._umi_start_raw_pos = None
        self._umi_start_raw_rot = None
        self._umi_start_pos = None
        self._umi_start_rot = None

    @property
    def active(self) -> bool:
        return self._robot_start_pos is not None

    @property
    def frame_matrix(self) -> np.ndarray:
        return self._frame_matrix.copy()

    def map_umi_pose(self, *, position_xyz_m: np.ndarray, quat_xyzw: np.ndarray) -> tuple[np.ndarray, R]:
        """Map raw UMI SLAM pose into the configured robot/base-aligned frame."""
        raw_pos = np.asarray(position_xyz_m, dtype=np.float64)
        raw_rot = R.from_quat(np.asarray(quat_xyzw, dtype=np.float64))
        return self._frame_rot.apply(raw_pos), self._frame_rot * raw_rot * self._frame_rot.inv()

    def orientation_error_deg(
        self,
        *,
        robot_tcp_euler_xyz_rad: np.ndarray,
        umi_quat_xyzw: np.ndarray,
    ) -> float:
        """Return angular distance between current robot TCP and mapped UMI orientation."""
        robot_rot = R.from_euler("xyz", np.asarray(robot_tcp_euler_xyz_rad, dtype=np.float64))
        umi_rot = self.map_umi_pose(position_xyz_m=np.zeros(3, dtype=np.float64), quat_xyzw=umi_quat_xyzw)[1]
        return float(np.rad2deg((robot_rot.inv() * umi_rot).magnitude()))

    def target(self, *, umi_position_xyz_m: np.ndarray, umi_quat_xyzw: np.ndarray) -> RelativePoseTarget:
        if (
            self._robot_start_pos is None
            or self._robot_start_rot is None
            or self._umi_start_raw_pos is None
            or self._umi_start_raw_rot is None
            or self._umi_start_pos is None
            or self._umi_start_rot is None
        ):
            raise RuntimeError("RelativePoseMapper.begin() must be called before target().")

        raw_umi_pos = np.asarray(umi_position_xyz_m, dtype=np.float64)
        raw_umi_rot = R.from_quat(np.asarray(umi_quat_xyzw, dtype=np.float64))
        raw_delta_xyz = raw_umi_pos - self._umi_start_raw_pos
        if self._delta_frame == "local":
            raw_delta_local_xyz = self._umi_start_raw_rot.inv().apply(raw_delta_xyz)
            mapped_delta_xyz = self._frame_rot.apply(raw_delta_local_xyz)
        else:
            umi_pos, _umi_rot = self.map_umi_pose(position_xyz_m=umi_position_xyz_m, quat_xyzw=umi_quat_xyzw)
            mapped_delta_xyz = umi_pos - self._umi_start_pos
        delta_xyz = mapped_delta_xyz.copy()
        delta_xyz = delta_xyz * self._translation_scale

        translation_clamped = False
        delta_norm = float(np.linalg.norm(delta_xyz))
        if self._max_delta_xyz > 0.0 and delta_norm > self._max_delta_xyz:
            delta_xyz = delta_xyz * (self._max_delta_xyz / max(delta_norm, 1e-9))
            translation_clamped = True

        if self._delta_frame == "local":
            target_pos = self._robot_start_pos + self._robot_start_rot.apply(delta_xyz)
        else:
            target_pos = self._robot_start_pos + delta_xyz

        raw_delta_rot = self._umi_start_raw_rot.inv() * raw_umi_rot
        delta_rot = self._frame_rot * raw_delta_rot * self._frame_rot.inv()
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
            raw_delta_xyz_m=np.asarray(raw_delta_xyz, dtype=np.float64),
            mapped_delta_xyz_m=np.asarray(mapped_delta_xyz, dtype=np.float64),
            command_delta_xyz_m=np.asarray(target_pos - self._robot_start_pos, dtype=np.float64),
            raw_delta_rotvec_rad=np.asarray(raw_delta_rot.as_rotvec(), dtype=np.float64),
            mapped_delta_rotvec_rad=np.asarray(delta_rot.as_rotvec(), dtype=np.float64),
        )
