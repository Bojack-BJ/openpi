#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from hil.relative_pose import RelativePoseMapper
from hil.relative_pose import parse_signed_axes
from hil.umi_slam_reader import UmiSlamReader


def _parse_pose_xyz_rpy_deg(value: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected x,y,z,roll,pitch,yaw, got: {value!r}")
    return np.asarray(parts[:3], dtype=np.float64), np.deg2rad(np.asarray(parts[3:], dtype=np.float64))


def _fmt_vec(value: np.ndarray, *, decimals: int = 4) -> str:
    return str(np.round(np.asarray(value, dtype=np.float64), decimals).tolist())


def _rpy_deg_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler("xyz", degrees=True)


def _read_xarm_pose(*, robot_ip: str, bestman_path: str) -> tuple[np.ndarray, np.ndarray]:
    if bestman_path and bestman_path not in sys.path:
        sys.path.append(bestman_path)
    from Bestman_real_xarm6 import Bestman_Real_Xarm6  # noqa: PLC0415

    arm = Bestman_Real_Xarm6(robot_ip, None, None)
    arm.robot.set_state(0)
    code, qpos_raw = arm.robot.get_position()
    if code != 0:
        print(f"[WARN] xArm get_position return code={code}")
    x_mm, y_mm, z_mm, roll, pitch, yaw = qpos_raw
    return (
        np.asarray([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0,
        np.deg2rad(np.asarray([roll, pitch, yaw], dtype=np.float64)),
    )


def _print_pose_block(
    *,
    sample,
    mapper: RelativePoseMapper,
    target,
    robot_start_pos: np.ndarray,
    robot_start_rpy_rad: np.ndarray,
) -> None:
    mapped_pos, mapped_rot = mapper.map_umi_pose(
        position_xyz_m=sample.position_xyz_m,
        quat_xyzw=sample.quat_xyzw,
    )
    pose_age_s = time.monotonic() - sample.received_at
    print(
        "[UMI RAW]    "
        f"xyz_m={_fmt_vec(sample.position_xyz_m)} "
        f"rpy_deg={_fmt_vec(_rpy_deg_from_quat_xyzw(sample.quat_xyzw), decimals=2)} "
        f"gripper_open={sample.gripper_open:.3f} raw={sample.gripper_raw:.3f} "
        f"age_s={pose_age_s:.3f} confidence={sample.confidence}"
    )
    print(
        "[UMI MAPPED] "
        f"xyz_m={_fmt_vec(mapped_pos)} "
        f"rpy_deg={_fmt_vec(mapped_rot.as_euler('xyz', degrees=True), decimals=2)}"
    )
    print(
        "[BASELINE]   "
        f"xarm_xyz_m={_fmt_vec(robot_start_pos)} "
        f"xarm_rpy_deg={_fmt_vec(np.rad2deg(robot_start_rpy_rad), decimals=2)}"
    )
    print(
        "[DELTA]      "
        f"raw_xyz_m={_fmt_vec(target.raw_delta_xyz_m)} "
        f"mapped_xyz_m={_fmt_vec(target.mapped_delta_xyz_m)} "
        f"cmd_xyz_m={_fmt_vec(target.command_delta_xyz_m)}"
    )
    print(
        "[ROT DELTA]  "
        f"raw_rotvec_deg={_fmt_vec(np.rad2deg(target.raw_delta_rotvec_rad), decimals=2)} "
        f"mapped_rotvec_deg={_fmt_vec(np.rad2deg(target.mapped_delta_rotvec_rad), decimals=2)}"
    )
    print(
        "[XARM TARGET] "
        f"xyz_m={_fmt_vec(target.position_xyz_m)} "
        f"rpy_deg={_fmt_vec(np.rad2deg(target.euler_xyz_rad), decimals=2)} "
        f"translation_clamped={target.translation_clamped} rotation_clamped={target.rotation_clamped}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug UMI SLAM pose mapping used by HIL takeover.")
    parser.add_argument("--umi_xv_serial", required=True, help="UMI XV 设备序列号，用于订阅 /xv_sdk/<serial>/slam/pose")
    parser.add_argument("--umi_max_gripper", type=float, default=84.0, help="UMI clamp 原始值对应 open=1 的量程")
    parser.add_argument("--umi_pose_max_age_s", type=float, default=0.25, help="UMI SLAM pose 最大允许延迟")
    parser.add_argument("--umi_gripper_max_age_s", type=float, default=0.5, help="UMI clamp 最大允许延迟")
    parser.add_argument("--umi_ros_queue_size", type=int, default=1, help="UMI ROS subscriber queue size；默认只保留最新消息以降低延迟")
    parser.add_argument("--hil_ready_timeout_s", type=float, default=10.0, help="启动时等待 UMI SLAM+clamp 首帧的超时时间")
    parser.add_argument("--hil_max_delta_xyz", type=float, default=0.0, help="从接管起点算的累计 TCP 平移范数上限（米）；0 表示不限制")
    parser.add_argument("--hil_max_delta_rpy_deg", type=float, default=0.0, help="从接管起点算的累计 TCP 旋转角上限（度）；0 表示不限制")
    parser.add_argument("--hil_slam_axes", default="x,y,z", help="UMI raw SLAM xyz/rpy -> robot base xyz/rpy 的右手系轴映射，例如 z,-x,-y")
    parser.add_argument("--hil_slam_translation_scale", type=float, default=0.5, help="UMI 平移到 TCP 平移的比例；UMI 与 TCP 均按米处理")
    parser.add_argument("--hil_require_umi_tcp_alignment", action="store_true", help="打印 UMI/TCP orientation 对齐误差")
    parser.add_argument("--hil_umi_tcp_alignment_threshold_deg", type=float, default=25.0, help="--hil_require_umi_tcp_alignment 的角度阈值")
    parser.add_argument("--hil_log_interval_s", type=float, default=0.5, help="debug 打印间隔；0 表示尽快打印")
    parser.add_argument("--use_xarm", action="store_true", help="读取真实 xArm 当前 TCP pose 作为 baseline；默认使用 --robot_tcp_pose")
    parser.add_argument("--robot_ip", default="192.168.1.217", help="--use_xarm 时读取的 xArm 控制盒 IP")
    parser.add_argument(
        "--bestman_path",
        default="/home/lumos/lxt/BestMan_Xarm/RoboticsToolBox/",
        help="BestMan_Xarm RoboticsToolBox 路径；--use_xarm 时使用",
    )
    parser.add_argument(
        "--robot_tcp_pose",
        default="0.4,0.0,0.146,180,-90,0.0",
        help="不连接 xArm 时使用的 baseline TCP pose: x,y,z,roll,pitch,yaw（米、度）",
    )
    parser.add_argument("--no_wait_for_enter", action="store_true", help="收到 UMI 首帧后立即锁定 baseline；默认按 Enter 后锁定")
    parser.add_argument("--once", action="store_true", help="只打印一次 target 后退出")
    args = parser.parse_args()

    axes = parse_signed_axes(args.hil_slam_axes)
    mapper = RelativePoseMapper(
        axes=axes,
        translation_scale=args.hil_slam_translation_scale,
        max_delta_xyz=args.hil_max_delta_xyz,
        max_delta_rpy_deg=args.hil_max_delta_rpy_deg,
    )
    print("[CONFIG]", f"axes={args.hil_slam_axes}", f"scale={args.hil_slam_translation_scale}")
    print(f"[CONFIG] frame_matrix raw_umi_xyz -> robot_xyz:\n{mapper.frame_matrix}")

    umi_reader = UmiSlamReader(
        xv_serial=args.umi_xv_serial,
        max_gripper=args.umi_max_gripper,
        pose_max_age_s=args.umi_pose_max_age_s,
        gripper_max_age_s=args.umi_gripper_max_age_s,
        queue_size=args.umi_ros_queue_size,
    )
    print(f"[UMI] slam_topic={umi_reader.slam_topic} clamp_topic={umi_reader.clamp_topic}")
    if not umi_reader.wait_until_ready(timeout_s=args.hil_ready_timeout_s):
        raise RuntimeError("UMI SLAM reader did not receive pose+clamp before timeout.")

    if args.use_xarm:
        robot_start_pos, robot_start_rpy_rad = _read_xarm_pose(robot_ip=args.robot_ip, bestman_path=args.bestman_path)
    else:
        robot_start_pos, robot_start_rpy_rad = _parse_pose_xyz_rpy_deg(args.robot_tcp_pose)

    if not args.no_wait_for_enter:
        input("Move UMI/TCP to the takeover baseline, then press Enter to lock baseline...")

    baseline_sample = umi_reader.latest()
    if baseline_sample is None:
        raise RuntimeError("UMI SLAM/clamp data is stale when locking baseline.")
    mapper.begin(
        robot_tcp_position_xyz_m=robot_start_pos,
        robot_tcp_euler_xyz_rad=robot_start_rpy_rad,
        umi_position_xyz_m=baseline_sample.position_xyz_m,
        umi_quat_xyzw=baseline_sample.quat_xyzw,
    )
    if args.hil_require_umi_tcp_alignment:
        alignment_error = mapper.orientation_error_deg(
            robot_tcp_euler_xyz_rad=robot_start_rpy_rad,
            umi_quat_xyzw=baseline_sample.quat_xyzw,
        )
        print(
            "[ALIGN]",
            f"error_deg={alignment_error:.2f}",
            f"threshold_deg={args.hil_umi_tcp_alignment_threshold_deg:.2f}",
            f"ok={alignment_error <= args.hil_umi_tcp_alignment_threshold_deg}",
        )
    print("[BASELINE LOCKED]")

    interval_s = max(float(args.hil_log_interval_s), 0.0)
    try:
        while True:
            sample = umi_reader.latest()
            if sample is None:
                print("[WARN] UMI SLAM/clamp data is stale.")
                time.sleep(max(interval_s, 0.02))
                continue
            target = mapper.target(
                umi_position_xyz_m=sample.position_xyz_m,
                umi_quat_xyzw=sample.quat_xyzw,
            )
            _print_pose_block(
                sample=sample,
                mapper=mapper,
                target=target,
                robot_start_pos=robot_start_pos,
                robot_start_rpy_rad=robot_start_rpy_rad,
            )
            print("-" * 120)
            if args.once:
                break
            time.sleep(interval_s)
    finally:
        umi_reader.close()


if __name__ == "__main__":
    main()
