#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件用途：
    该文件用于双臂 startouch 的在线策略回放控制：采集双路 YU12 摄像头图像与双臂状态，
    调用远程 policy server 推理动作，再将动作下发到两台机械臂与夹爪执行。

主要功能：
    1. 以 YU12(I420) 模式初始化摄像头并转为 RGB 图像。
    2. 读取双臂末端位姿（SDK 为法兰；若训练标签为夹爪尖 TCP，则用 --tcp_offset 与训练对齐）、
       四元数与夹爪开合（0~1），组装策略输入观测 obs。
    3. 通过 websocket 调用策略服务获取动作序列。
    4. 解析双臂 14 维动作（xyz + RPY + gripper），转为四元数后发送给 startouch 执行。

适用场景：
    1. 真机部署时的双臂策略联调与在线回放。
    2. 已有远程策略服务（serve_policy）并需要实时闭环执行的场景。
    3. 需要验证视觉输入 + 机器人状态联合推理效果的场景。

核心逻辑：
    启动后先连接机器人（via CAN）、策略服务器与双路摄像头，循环中持续读取当前机器人状态与图像，
    将其打包后发送到策略服务器获取动作序列，再逐步下发给双臂执行并控制夹爪。

运行方式：
    sudo ip link set can0 up type can bitrate 1000000
    sudo ip link set can1 up type can bitrate 1000000
    PYTHONPATH=/home/lumos/openpi/startouch-v1-fast_touch/interface_py

    python pi0_rollout_serve_pro_v1_remote_dyh_0318_startouch_rpy.py \
        --description "Pick up the pink mug from the three mugs on the table with your right hand and place it on the round, off-white coaster in front of your right hand."

    uv run pi0_rollout_serve_pro_v1_remote_lxh_0318_startouch_rpy.py \
        --description "Pick up the pink mug from the three mugs on the table with your right hand and place it on the round, off-white coaster in front of your right hand."

运行示例：
    # 示例一：只传必填任务描述，CAN 接口使用默认值
    python pi0_rollout_serve_pro_v1_remote_dyh_0318_startouch.py \
        --description "Bimanual task"

    # 示例二：指定双臂 CAN 接口
    python pi0_rollout_serve_pro_v1_remote_dyh_0318_startouch.py \
        --description "pick two bottles and place in tray" \
        --left_can can0 --right_can can1

运行前提：
    1. Python >= 3.8。
    2. 已安装依赖：opencv-python、numpy、scipy、openpi_client、startouchclass。
    3. startouchclass 中 SingleArm 可正常导入。
    4. 双臂 CAN 总线可达，且策略服务在 SERVER_IP:PORT 正常运行。
    5. 双路摄像头设备节点存在（默认 /dev/video0 与 /dev/video2）。

参数说明：
    --description
        含义：任务自然语言描述，会直接传给策略服务作为 prompt。
        类型：字符串
        是否必填：是
        示例：--description "pick drinks and toothpaste then put them in box"

    --left_can
        含义：左臂（arm0）对应的 CAN 接口名。
        类型：字符串
        是否必填：否
        默认值：can0
        示例：--left_can can0

    --right_can
        含义：右臂（arm1）对应的 CAN 接口名。
        类型：字符串
        是否必填：否
        默认值：can1
        示例：--right_can can1

注意事项：
    1. 当前 SERVER_IP/PORT、摄像头分辨率与设备编号为硬编码参数，部署前请按现场环境确认。
    2. 该脚本会直接驱动真机运动，建议在安全区域、低速和有人监护条件下调试。
    3. 策略返回的动作中 RPY 为度；脚本会转为弧度再下发，与 SDK 读写的欧拉角（弧度）一致。
"""

import argparse
import pathlib
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np

from openpi_client import websocket_client_policy, image_tools

from startouchclass import SingleArm
from tcp_compensation import (
    flange_position_to_tcp,
    parse_tool_offset_xyz,
    tcp_position_to_flange,
)

# ====== 本机 policy server 参数 ======
SERVER_IP = "180.184.74.93"   # 中转服务器ip地址
# SERVER_IP = "127.0.0.1"   # 本地ip地址
PORT = 8009             # DevA=8002、DevB=8003、DevC=8004、DevD=8005、本地=8001，DevI=8010

# ====== 摄像头（YU12 / I420）参数 ======
DEV = 0                   # /dev/video0
W, H, FPS = 1280, 1280, 100
MASK_OVERLAY_KEY = "__mask_overlay"


def init_yu12_camera(DEV):
    """按 YU12(I420) 模式初始化摄像头。"""
    cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 /dev/video{DEV}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YU12'))  # YU12 == I420
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # 打印一下驱动返回的实际 FOURCC，确认真的是 YU12
    fcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = "".join([chr((fcc_int >> (8 * i)) & 0xFF) for i in range(4)])
    print("FOURCC from driver:", fcc)
    return cap


def grab_rgb(cap):
    """
    从 YU12(I420) 摄像头抓一帧，返回 RGB 图像 (H, W, 3, uint8)。
    逻辑：raw(YU12) -> BGR -> RGB
    """
    ok, raw = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败")

    # raw 一般是 (H*3//2, W) 或 (H*3//2, W, 1)，统一拉平成 2D 再转
    yuv = np.ascontiguousarray(raw).reshape(H * 3 // 2, W)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)   # I420 -> BGR
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)        # BGR -> RGB
    return rgb

def grab_rgb_latest(cap, flush_n=4):
    """丢弃旧帧后抓取最新一帧，并输出 RGB 图像。"""
    for _ in range(flush_n):
        cap.grab()
        print("fuck")
    ok, raw = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败")
    yuv = np.ascontiguousarray(raw).reshape(H * 3 // 2, W)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _stdin_read_char_nonblocking():
    """终端模式下非阻塞读一个字符；非 tty 或无可读数据时返回 None。"""
    if not sys.stdin.isatty():
        return None
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if not r:
        return None
    try:
        return sys.stdin.read(1)
    except Exception:
        return None


def _stdin_wants_reset_pause() -> bool:
    ch = _stdin_read_char_nonblocking()
    return ch in ("s", "S")


def _parse_xy(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    x_str, y_str = value.split(",", maxsplit=1)
    return int(float(x_str)), int(float(y_str))


def _select_point(image_rgb: np.ndarray, *, title: str) -> tuple[int, int]:
    clicked: list[tuple[int, int]] = []
    display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[:] = [(int(x), int(y))]

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    while not clicked:
        frame = display.copy()
        cv2.putText(frame, "Click target point, q/esc to cancel", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(title, frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(title)
            raise RuntimeError("Mask prompt selection cancelled.")
    cv2.destroyWindow(title)
    return clicked[0]


def _array_stats(name: str, value) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return f"{name}: shape={arr.shape} dtype={arr.dtype} empty"
    return (
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min()} max={arr.max()} mean={float(arr.mean()):.2f} nonzero={int(np.count_nonzero(arr))}"
    )


def _save_debug_image(path: pathlib.Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_np = np.asarray(image_rgb)
    if image_np.ndim == 2:
        cv2.imwrite(str(path), image_np)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))


def _dump_mask_preview_debug(
    *,
    debug_dir: pathlib.Path,
    view: str,
    point_xy: tuple[int, int],
    client_image: np.ndarray,
    payload: dict,
) -> None:
    overlay = np.asarray(payload["overlay"])
    mask = np.asarray(payload.get("mask", np.zeros(client_image.shape[:2], dtype=np.uint8)))
    server_image = payload.get("image")
    print(
        "[MASK DEBUG]",
        f"view={view}",
        f"point={point_xy}",
        f"initialized={payload.get('initialized')}",
        f"score={payload.get('score')}",
        f"bbox={payload.get('bbox_xyxy')}",
    )
    print("[MASK DEBUG]", _array_stats("client_image", client_image))
    if server_image is not None:
        print("[MASK DEBUG]", _array_stats("server_image", server_image))
    print("[MASK DEBUG]", _array_stats("overlay", overlay))
    print("[MASK DEBUG]", _array_stats("mask", mask))

    prefix = debug_dir / f"{view}_x{point_xy[0]}_y{point_xy[1]}"
    _save_debug_image(prefix.with_name(prefix.name + "_client.png"), client_image)
    if server_image is not None:
        _save_debug_image(prefix.with_name(prefix.name + "_server.png"), np.asarray(server_image))
    _save_debug_image(prefix.with_name(prefix.name + "_overlay.png"), overlay)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_mask.png")), mask)
    print(f"[MASK DEBUG] saved preview images under {debug_dir}")


def _dump_mask_rollout_debug(
    *,
    debug_dir: pathlib.Path,
    view: str,
    infer_index: int,
    payload: dict,
) -> None:
    overlay = payload.get("overlay")
    if overlay is None:
        print(f"[MASK DEBUG] infer={infer_index:06d} has no overlay payload; keys={sorted(payload)}")
        return

    overlay = np.asarray(overlay)
    mask = np.asarray(payload.get("mask", np.zeros(overlay.shape[:2], dtype=np.uint8)))
    server_image = payload.get("image")
    print(
        "[MASK DEBUG]",
        f"infer={infer_index:06d}",
        f"view={view}",
        f"initialized={payload.get('initialized')}",
        f"score={payload.get('score')}",
        f"bbox={payload.get('bbox_xyxy')}",
        _array_stats("overlay", overlay),
        _array_stats("mask", mask),
    )

    prefix = debug_dir / "rollout" / f"{infer_index:06d}_{view}"
    if server_image is not None:
        _save_debug_image(prefix.with_name(prefix.name + "_server.png"), np.asarray(server_image))
    _save_debug_image(prefix.with_name(prefix.name + "_overlay.png"), overlay)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_mask.png")), mask)


def _preview_mask_overlay(
    policy_client,
    *,
    image_rgb: np.ndarray,
    view: str,
    point_xy: tuple[int, int] | None,
    alpha: float,
    debug_dir: pathlib.Path | None,
) -> None:
    while True:
        if point_xy is None:
            point_xy = _select_point(image_rgb, title=f"Select SAM3 point: {view}")
        mask_request = {
            "enabled": True,
            "view": view,
            "reset": True,
            "points": np.asarray([point_xy], dtype=np.float32),
            "point_labels": np.asarray([1], dtype=np.int32),
            "preview_only": True,
            "alpha": float(alpha),
        }
        if debug_dir is not None:
            mask_request["return_image"] = True

        obs = {
            "image": {view: image_tools.convert_to_uint8(image_rgb)},
            MASK_OVERLAY_KEY: mask_request,
        }
        resp = policy_client.infer(obs)
        payload = resp.get(MASK_OVERLAY_KEY, {})
        overlay = payload.get("overlay")
        if overlay is None:
            raise RuntimeError(f"Policy server did not return mask overlay preview: keys={sorted(resp)}")
        if debug_dir is not None:
            _dump_mask_preview_debug(
                debug_dir=debug_dir,
                view=view,
                point_xy=point_xy,
                client_image=image_rgb,
                payload=payload,
            )

        window = f"SAM3 overlay preview: {view}"
        cv2.imshow(window, cv2.cvtColor(np.asarray(overlay), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        answer = input("SAM3 overlay 是否正确？[y=确认 / r=重新点击 / q=退出]: ").strip().lower()
        cv2.destroyWindow(window)
        if answer in ("y", "yes", ""):
            return
        if answer in ("q", "quit"):
            raise RuntimeError("Mask overlay preview rejected.")
        point_xy = None


def _arm_index(arm_name: str) -> int:
    if arm_name == "robot_0":
        return 0
    if arm_name == "robot_1":
        return 1
    raise ValueError(f"Unsupported arm name: {arm_name}")


def _select_by_arm(arm_name: str, robot_0_value, robot_1_value):
    return robot_0_value if _arm_index(arm_name) == 0 else robot_1_value


def _capture_resized(cap) -> np.ndarray:
    return cv2.resize(grab_rgb_latest(cap), (224, 224), interpolation=cv2.INTER_AREA)


def _slice_actions(actions_all: np.ndarray, start: int, end: int) -> tuple[int, np.ndarray] | None:
    n_act = len(actions_all)
    if start < 0 or end < 0:
        print(f"[WARN] 忽略非法 action 区间: action_start={start} action_end={end}")
        return None
    if start > end:
        print(f"[WARN] action_start 不能大于 action_end: {start} > {end}")
        return None
    if n_act == 0:
        print("[WARN] 策略返回的 action 序列为空")
        return None
    lo = start
    hi = min(end, n_act - 1)
    if lo >= n_act:
        print(f"[WARN] action_start 越界: start={start} 序列长度={n_act}")
        return None
    if lo > hi:
        print(f"[WARN] 无 action 可执行: 区间 [{start},{end}] 与长度 {n_act} 无交集")
        return None
    return lo, actions_all[lo : hi + 1]


def _adjust_gripper_actions(actions_all: np.ndarray, *, arm_mode: str, single_arm_index: int) -> np.ndarray:
    if arm_mode == "single":
        if actions_all.shape[-1] < 7:
            raise ValueError(f"single mode expects action dim >= 7, got shape={actions_all.shape}")
        threshold = 0.8 if single_arm_index == 0 else 0.95
        actions_all[..., 6] = np.where(actions_all[..., 6] < threshold, actions_all[..., 6] - 0.2, actions_all[..., 6])
        actions_all[..., 6] = np.clip(actions_all[..., 6], 0.0, 1.0)
        return actions_all

    if actions_all.shape[-1] < 14:
        raise ValueError(f"dual mode expects action dim >= 14, got shape={actions_all.shape}")
    actions_all[..., 6] = np.where(actions_all[..., 6] < 0.8, actions_all[..., 6] - 0.2, actions_all[..., 6])
    actions_all[..., 6] = np.clip(actions_all[..., 6], 0.0, 1.0)
    actions_all[..., 13] = np.where(actions_all[..., 13] < 0.95, actions_all[..., 13] - 0.2, actions_all[..., 13])
    actions_all[..., 13] = np.clip(actions_all[..., 13], 0.0, 1.0)
    return actions_all


def reset_arms_to_init(arm0: SingleArm, arm1: SingleArm,
                       init_pos0, init_euler0, init_pos1, init_euler1) -> None:
    arm0.set_end_effector_pose_euler(pos=init_pos0, euler=init_euler0, tf=2)
    arm1.set_end_effector_pose_euler(pos=init_pos1, euler=init_euler1, tf=2)
    arm0.setGripperPosition(1.0)
    arm1.setGripperPosition(1.0)


def reset_arm_to_init(arm: SingleArm, init_pos, init_euler) -> None:
    arm.set_end_effector_pose_euler(pos=init_pos, euler=init_euler, tf=2)
    arm.setGripperPosition(1.0)


def interp_and_move_one(arm: SingleArm,
                        start_pos, start_euler_rad, target_pos, target_euler_rad,
                        g_open: float,
                        step_size: float, dt: float) -> None:
    """线性位置插值 + 线性欧拉角插值，下发单臂（插值过程中不响应 s）。"""
    ps = np.asarray(start_pos, dtype=float)
    pe = np.asarray(target_pos, dtype=float)
    es = np.asarray(start_euler_rad, dtype=float)
    ee = np.asarray(target_euler_rad, dtype=float)
    if step_size <= 0:
        steps = 1
    else:
        steps = max(int(np.ceil(np.linalg.norm(pe - ps) / max(step_size, 1e-6))), 1)

    ts = np.linspace(0.0, 1.0, steps + 1)
    pos_arr = np.linspace(ps, pe, steps + 1)
    euler_arr_rad = np.outer(1.0 - ts, es) + np.outer(ts, ee)

    for i in range(steps + 1):
        arm.set_end_effector_pose_euler_raw(pos=pos_arr[i].tolist(), euler=euler_arr_rad[i].tolist())
        time.sleep(dt)

    arm.setGripperPosition(g_open)


def interp_and_move_both(arm0: SingleArm, arm1: SingleArm,
                         start_pos0, start_euler_rad0, target_pos0, target_euler_rad0,
                         start_pos1, start_euler_rad1, target_pos1, target_euler_rad1,
                         g0: float, g1: float,
                         step_size: float, dt: float) -> None:
    """线性位置插值 + 线性欧拉角插值，同步下发双臂（插值过程中不响应 s）。"""
    p0s = np.asarray(start_pos0, dtype=float)
    p0e = np.asarray(target_pos0, dtype=float)
    e0s = np.asarray(start_euler_rad0, dtype=float)
    e0e = np.asarray(target_euler_rad0, dtype=float)
    p1s = np.asarray(start_pos1, dtype=float)
    p1e = np.asarray(target_pos1, dtype=float)
    e1s = np.asarray(start_euler_rad1, dtype=float)
    e1e = np.asarray(target_euler_rad1, dtype=float)
    dist0 = np.linalg.norm(p0e - p0s)
    dist1 = np.linalg.norm(p1e - p1s)
    steps = max(int(np.ceil(max(dist0, dist1) / max(step_size, 1e-6))), 1)

    ts = np.linspace(0.0, 1.0, steps + 1)
    pos0_arr = np.linspace(p0s, p0e, steps + 1)
    pos1_arr = np.linspace(p1s, p1e, steps + 1)
    euler0_arr_rad = np.outer(1.0 - ts, e0s) + np.outer(ts, e0e)
    euler1_arr_rad = np.outer(1.0 - ts, e1s) + np.outer(ts, e1e)

    for i in range(steps + 1):
        arm0.set_end_effector_pose_euler_raw(pos=pos0_arr[i].tolist(), euler=euler0_arr_rad[i].tolist())
        arm1.set_end_effector_pose_euler_raw(pos=pos1_arr[i].tolist(), euler=euler1_arr_rad[i].tolist())
        time.sleep(dt)

    arm0.setGripperPosition(g0)
    arm1.setGripperPosition(g1)


def main():
    """主流程：连接设备，循环推理并下发动作。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="任务自然语言指令")
    parser.add_argument("--arm_mode", choices=("dual", "single"), default="dual", help="dual=双臂 14D；single=单臂 7D")
    parser.add_argument("--single_arm", choices=("robot_0", "robot_1"), default=None, help="single 模式必须显式指定使用哪只 arm")
    parser.add_argument("--single_image_key", default="front", help="single 模式发给 policy 的 image key，FastUMI 单臂默认 front")
    parser.add_argument("--left_can", default="can0", help="左臂（robot_0）CAN 接口")
    parser.add_argument("--right_can", default="can1", help="右臂（robot_1）CAN 接口")
    parser.add_argument("--camera_dev0", type=int, default=DEV, help="robot_0 图像对应的视频设备编号")
    parser.add_argument("--camera_dev1", type=int, default=DEV + 2, help="robot_1 图像对应的视频设备编号")
    parser.add_argument("--dt", type=float, default=0.025, help="插值每步的睡眠时间（秒）")
    parser.add_argument("--interp_step_size", type=float, default=0.0025, help="插值最大平移步长（米），超过此值时自动细分")
    parser.add_argument("--init_pose_left", type=str, default="0.3,0.0,0.16,0.0,0.0,0.0", help="robot_0 初始位姿: x,y,z,roll,pitch,yaw（弧度）")
    parser.add_argument("--init_pose_right", type=str, default="0.3,0.0,0.16,0.0,0.0,0.0", help="robot_1 初始位姿: x,y,z,roll,pitch,yaw（弧度）")
    parser.add_argument(
        "--tcp_offset",
        type=str,
        default="0.0,0.0,0.0",
        help=(
            "法兰系下「法兰原点→夹爪尖 TCP」的平移 x,y,z（米），与训练 TCP 定义一致。"
            "训练 state/action 为 TCP、真机读法兰时填标定；训练为法兰则 0,0,0。"
        ),
    )
    parser.add_argument("--tcp_debug", action="store_true", help="打印法兰/TCP 互转调试信息")
    parser.add_argument("--action_start", type=int, default=0, help="本次推理返回的 action 序列起始下标（含）")
    parser.add_argument("--action_end", type=int, default=20, help="本次推理返回的 action 序列结束下标（含）")
    parser.add_argument("--mask_overlay", action="store_true", help="请求 policy server 使用 SAM3 mask overlay 后再推理")
    parser.add_argument("--mask_view", default=None, help="需要做 SAM3 overlay 的 image key；single 默认 front，dual 下建议显式指定 robot_0/robot_1")
    parser.add_argument("--mask_prompt_point", default=None, help="初始正点 prompt，格式 x,y；不填则弹窗点击")
    parser.add_argument("--mask_alpha", type=float, default=0.35, help="SAM3 overlay alpha")
    parser.add_argument("--mask_debug_dir", default=None, help="显式提供目录时，保存 SAM3 preview 输入/输出调试图并请求 server 回传原图")
    args = parser.parse_args()

    is_dual = args.arm_mode == "dual"
    if not is_dual and args.single_arm is None:
        parser.error("--arm_mode single 需要显式指定 --single_arm robot_0 或 robot_1，避免误用默认 robot_0")
    if args.mask_overlay and is_dual and args.mask_view is None:
        parser.error("--arm_mode dual 开启 --mask_overlay 时需要显式指定 --mask_view robot_0 或 robot_1")

    single_arm = args.single_arm
    single_arm_index = _arm_index(single_arm) if single_arm is not None else 0
    effective_mask_view = args.mask_view or args.single_image_key

    tcp_off = parse_tool_offset_xyz(args.tcp_offset)
    if np.any(tcp_off != 0.0):
        print(f"[INFO] TCP 工具偏移（法兰系，米）={tcp_off.tolist()}")

    def parse_init_pose(value: str):
        parts = [float(v) for v in value.split(",")]
        if len(parts) != 6:
            raise ValueError(f"init_pose 需要 6 个值 x,y,z,r,p,y，实际得到: {value}")
        x, y, z, roll, pitch, yaw = parts
        return [x, y, z], [roll, pitch, yaw]

    init_pos0, init_euler0 = parse_init_pose(args.init_pose_left)
    init_pos1, init_euler1 = parse_init_pose(args.init_pose_right)
    init_by_arm = {
        "robot_0": (init_pos0, init_euler0),
        "robot_1": (init_pos1, init_euler1),
    }

    arms: dict[str, SingleArm] = {}
    if is_dual or single_arm == "robot_0":
        arms["robot_0"] = SingleArm(can_interface_=args.left_can)
    if is_dual or single_arm == "robot_1":
        arms["robot_1"] = SingleArm(can_interface_=args.right_can)
    time.sleep(2)

    def reset_active_arms() -> None:
        if is_dual:
            reset_arms_to_init(arms["robot_0"], arms["robot_1"], init_pos0, init_euler0, init_pos1, init_euler1)
            return
        assert single_arm is not None
        init_pos, init_euler = init_by_arm[single_arm]
        reset_arm_to_init(arms[single_arm], init_pos, init_euler)

    print(
        f"[INFO] 移动到初始位姿: "
        f"{'robot_0=' + args.init_pose_left + ' robot_1=' + args.init_pose_right if is_dual else single_arm + '=' + _select_by_arm(single_arm, args.init_pose_left, args.init_pose_right)}"
    )
    reset_active_arms()
    print("[INFO] 已到达初始位姿")

    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=SERVER_IP,
        port=PORT,
    )
    print(f"[INFO] 已连接策略服务器：ws://{SERVER_IP}:{PORT}")
    if args.mask_overlay:
        metadata = policy_client.get_server_metadata()
        if not metadata.get("mask_overlay", {}).get("enabled", False):
            raise RuntimeError("client 开启了 --mask_overlay，但 server 未启用 --mask-overlay")

    camera_devs = {
        "robot_0": args.camera_dev0,
        "robot_1": args.camera_dev1,
    }
    caps = {}
    if is_dual:
        caps["robot_0"] = init_yu12_camera(camera_devs["robot_0"])
        caps["robot_1"] = init_yu12_camera(camera_devs["robot_1"])
    else:
        assert single_arm is not None
        caps[single_arm] = init_yu12_camera(camera_devs[single_arm])

    for _ in range(50):
        for cap in caps.values():
            _ = cap.read()
    print("[INFO] 摄像头预热完成，开始循环")

    input("按 Enter 开始")
    mask_debug_dir = pathlib.Path(args.mask_debug_dir) if args.mask_debug_dir else None
    if args.mask_overlay:
        if is_dual:
            preview_images = {
                "robot_0": _capture_resized(caps["robot_0"]),
                "robot_1": _capture_resized(caps["robot_1"]),
            }
        else:
            assert single_arm is not None
            preview_images = {args.single_image_key: _capture_resized(caps[single_arm])}
        if effective_mask_view not in preview_images:
            raise ValueError(f"--mask_view must be one of {sorted(preview_images)}, got {effective_mask_view}")
        _preview_mask_overlay(
            policy_client,
            image_rgb=preview_images[effective_mask_view],
            view=effective_mask_view,
            point_xy=_parse_xy(args.mask_prompt_point),
            alpha=args.mask_alpha,
            debug_dir=mask_debug_dir,
        )
        print("[INFO] SAM3 mask overlay 已确认，后续推理会持续请求 overlay")

    stdin_fd = sys.stdin.fileno()
    old_term = None
    try:
        if sys.stdin.isatty():
            old_term = termios.tcgetattr(stdin_fd)
            tty.setcbreak(stdin_fd)
        print("[INFO] 按 s：复位到初始位姿并暂停推理；按 c：继续推理")

        paused = False
        infer_index = 0
        while True:
            ch = _stdin_read_char_nonblocking()
            if ch in ("s", "S"):
                reset_active_arms()
                paused = True
                print("[INFO] 已复位到初始位姿，推理已暂停；按 c 继续")
                continue
            if ch in ("c", "C"):
                paused = False
                print("[INFO] 继续推理")

            if paused:
                time.sleep(0.02)
                continue

            if is_dual:
                pos0, quat_wxyz0 = arms["robot_0"].get_ee_pose_quat()
                qw0, qx0, qy0, qz0 = quat_wxyz0
                quat0 = np.array([qx0, qy0, qz0, qw0])
                p_tcp0 = flange_position_to_tcp(pos0, quat_wxyz0, tcp_off)

                pos1, quat_wxyz1 = arms["robot_1"].get_ee_pose_quat()
                qw1, qx1, qy1, qz1 = quat_wxyz1
                quat1 = np.array([qx1, qy1, qz1, qw1])
                p_tcp1 = flange_position_to_tcp(pos1, quat_wxyz1, tcp_off)

                if args.tcp_debug:
                    print(
                        "[TCP DEBUG][obs] flange -> state(TCP) | "
                        f"L {np.asarray(pos0).round(4)} -> {np.asarray(p_tcp0).round(4)} | "
                        f"R {np.asarray(pos1).round(4)} -> {np.asarray(p_tcp1).round(4)} | "
                        f"off {tcp_off.tolist()}"
                    )

                gripper_open0 = float(arms["robot_0"].get_gripper_position())
                gripper_open1 = float(arms["robot_1"].get_gripper_position())
                gripper_open0 = float(np.clip(np.where(gripper_open0 < 0.3, gripper_open0 + 0.2, gripper_open0), 0.0, 1.0))
                gripper_open1 = float(np.clip(np.where(gripper_open1 < 0.3, gripper_open1 + 0.2, gripper_open1), 0.0, 1.0))

                pose_euler0 = arms["robot_0"].get_ee_pose_euler()
                pose_euler1 = arms["robot_1"].get_ee_pose_euler()
                cur_pos0 = pose_euler0[0].tolist()
                cur_euler0 = pose_euler0[1].tolist()
                cur_pos1 = pose_euler1[0].tolist()
                cur_euler1 = pose_euler1[1].tolist()

                state_vec = np.array([*p_tcp0, *quat0, gripper_open0, *p_tcp1, *quat1, gripper_open1], dtype=np.float32)
                image_obs = {
                    "robot_0": image_tools.convert_to_uint8(_capture_resized(caps["robot_0"])),
                    "robot_1": image_tools.convert_to_uint8(_capture_resized(caps["robot_1"])),
                }
            else:
                assert single_arm is not None
                pos, quat_wxyz = arms[single_arm].get_ee_pose_quat()
                qw, qx, qy, qz = quat_wxyz
                quat = np.array([qx, qy, qz, qw])
                p_tcp = flange_position_to_tcp(pos, quat_wxyz, tcp_off)
                gripper_open = float(arms[single_arm].get_gripper_position())
                gripper_open = float(np.clip(np.where(gripper_open < 0.3, gripper_open + 0.2, gripper_open), 0.0, 1.0))
                pose_euler = arms[single_arm].get_ee_pose_euler()
                cur_pos_single = pose_euler[0].tolist()
                cur_euler_single = pose_euler[1].tolist()

                if args.tcp_debug:
                    print(
                        "[TCP DEBUG][obs] flange -> state(TCP) | "
                        f"{single_arm} {np.asarray(pos).round(4)} -> {np.asarray(p_tcp).round(4)} | "
                        f"off {tcp_off.tolist()}"
                    )

                state_vec = np.array([*p_tcp, *quat, gripper_open], dtype=np.float32)
                image_obs = {
                    args.single_image_key: image_tools.convert_to_uint8(_capture_resized(caps[single_arm])),
                }

            print("[INFO] 机器人状态：", state_vec)
            obs = {
                "state": state_vec,
                "image": image_obs,
                "prompt": args.description,
            }
            if args.mask_overlay:
                obs[MASK_OVERLAY_KEY] = {
                    "enabled": True,
                    "view": effective_mask_view,
                    "alpha": args.mask_alpha,
                }
                if mask_debug_dir is not None:
                    obs[MASK_OVERLAY_KEY]["return_image"] = True

            resp = policy_client.infer(obs)
            if args.mask_overlay and mask_debug_dir is not None:
                _dump_mask_rollout_debug(
                    debug_dir=mask_debug_dir,
                    view=effective_mask_view,
                    infer_index=infer_index,
                    payload=resp.get(MASK_OVERLAY_KEY, {}),
                )
            infer_index += 1
            actions_all = resp["actions"] if "actions" in resp else resp["action"]
            actions_all = np.array(actions_all, dtype=np.float64, copy=True)
            if actions_all.ndim == 1:
                actions_all = actions_all.reshape(1, -1)
            actions_all = _adjust_gripper_actions(actions_all, arm_mode=args.arm_mode, single_arm_index=single_arm_index)

            action_slice_result = _slice_actions(actions_all, args.action_start, args.action_end)
            if action_slice_result is None:
                continue
            lo, action_slice = action_slice_result

            for step_idx, action in enumerate(action_slice, start=lo):
                if _stdin_wants_reset_pause():
                    reset_active_arms()
                    paused = True
                    print("[INFO] 已复位到初始位姿，推理已暂停；按 c 继续")
                    break

                print(f"第 {step_idx} 步, action:", action)
                if is_dual:
                    x_a0, y_a0, z_a0, roll0, pitch0, yaw0, g_open0, x_a1, y_a1, z_a1, roll1, pitch1, yaw1, g_open1 = action
                    g0 = float(np.clip(g_open0, 0.0, 1.0))
                    g1 = float(np.clip(g_open1, 0.0, 1.0))
                    tgt_euler_rad0 = np.deg2rad([roll0, pitch0, yaw0]).tolist()
                    tgt_euler_rad1 = np.deg2rad([roll1, pitch1, yaw1]).tolist()
                    tgt_pos0 = tcp_position_to_flange([x_a0, y_a0, z_a0], tgt_euler_rad0, tcp_off).tolist()
                    tgt_pos1 = tcp_position_to_flange([x_a1, y_a1, z_a1], tgt_euler_rad1, tcp_off).tolist()

                    if args.tcp_debug:
                        print(
                            "[TCP DEBUG][cmd] step=%d TCP->flange | "
                            "L [%.4f,%.4f,%.4f] -> %s | "
                            "R [%.4f,%.4f,%.4f] -> %s"
                            % (step_idx, x_a0, y_a0, z_a0, tgt_pos0, x_a1, y_a1, z_a1, tgt_pos1)
                        )

                    interp_and_move_both(
                        arms["robot_0"], arms["robot_1"],
                        cur_pos0, cur_euler0, tgt_pos0, tgt_euler_rad0,
                        cur_pos1, cur_euler1, tgt_pos1, tgt_euler_rad1,
                        g0, g1,
                        step_size=args.interp_step_size,
                        dt=args.dt,
                    )
                    cur_pos0, cur_euler0 = tgt_pos0, tgt_euler_rad0
                    cur_pos1, cur_euler1 = tgt_pos1, tgt_euler_rad1
                else:
                    assert single_arm is not None
                    x_a, y_a, z_a, roll, pitch, yaw, g_open = action[:7]
                    gripper_open = float(np.clip(g_open, 0.0, 1.0))
                    tgt_euler_rad = np.deg2rad([roll, pitch, yaw]).tolist()
                    tgt_pos = tcp_position_to_flange([x_a, y_a, z_a], tgt_euler_rad, tcp_off).tolist()

                    if args.tcp_debug:
                        print(
                            "[TCP DEBUG][cmd] step=%d TCP->flange | %s [%.4f,%.4f,%.4f] -> %s"
                            % (step_idx, single_arm, x_a, y_a, z_a, tgt_pos)
                        )

                    interp_and_move_one(
                        arms[single_arm],
                        cur_pos_single,
                        cur_euler_single,
                        tgt_pos,
                        tgt_euler_rad,
                        gripper_open,
                        step_size=args.interp_step_size,
                        dt=args.dt,
                    )
                    cur_pos_single, cur_euler_single = tgt_pos, tgt_euler_rad

    finally:
        if old_term is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)
        for cap in caps.values():
            cap.release()
        for arm in arms.values():
            arm.cleanup()
        print("[INFO] 结束，摄像头与机械臂已释放。")


if __name__ == "__main__":
    main()
