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


def reset_arms_to_init(arm0: SingleArm, arm1: SingleArm,
                       init_pos0, init_euler0, init_pos1, init_euler1) -> None:
    arm0.set_end_effector_pose_euler(pos=init_pos0, euler=init_euler0, tf=2)
    arm1.set_end_effector_pose_euler(pos=init_pos1, euler=init_euler1, tf=2)
    arm0.setGripperPosition(1.0)
    arm1.setGripperPosition(1.0)


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
    """主流程：连接设备，循环推理并下发双臂动作。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="任务自然语言指令")
    parser.add_argument("--left_can", default="can0", help="左臂（arm0）CAN 接口")
    parser.add_argument("--right_can", default="can1", help="右臂（arm1）CAN 接口")
    parser.add_argument("--dt", type=float, default=0.025,
                        help="插值每步的睡眠时间（秒），默认 0.04")
    parser.add_argument("--interp_step_size", type=float, default=0.0025,
                        help="插值最大平移步长（米），超过此值时自动细分，默认 0.01")
    parser.add_argument('--init_pose_left', type=str, default="0.3,0.0,0.16,0.0,0.0,0.0", help='模拟左臂当前位姿: x,y,z,roll,pitch,yaw（弧度）')
    parser.add_argument('--init_pose_right', type=str, default="0.3,0.0,0.16,0.0,0.0,0.0", help='模拟右臂当前位姿: x,y,z,roll,pitch,yaw（弧度）')
    parser.add_argument(
        "--tcp_offset",
        type=str,
        # default="0.0,0.0,0.0",
        # default="0.0,0.0,0.0",
        default="0.0,0.0,0.0",
        help=(
            "双臂共用：法兰系下「法兰原点→夹爪尖 TCP」的平移 x,y,z（米），与训练 TCP 定义一致。"
            "训练 state/action 为 TCP、真机读法兰时填标定；训练为法兰则 0,0,0。"
        ),
    )
    parser.add_argument(
        "--tcp_debug",
        action="store_true",
        help="每一步打印：观测侧 法兰→TCP；chunk 内每一步 策略TCP→下发法兰",
    )
    parser.add_argument(
        "--action_start",
        type=int,
        default=0,
        help="本次推理返回的 action 序列：起始下标（0 起算，含）",
    )

    parser.add_argument(
        "--action_end",
        type=int,
        default=20,
        help="结束下标（0 起算，含）；与 action_start 闭区间，若超出序列长度则截断到末尾",
    )
    args = parser.parse_args()

    tcp_off = parse_tool_offset_xyz(args.tcp_offset)
    if np.any(tcp_off != 0.0):
        print(f"[INFO] TCP 工具偏移（法兰系，米，双臂共用）={tcp_off.tolist()}")

    # ---- 解析初始位姿参数 ----
    def parse_init_pose(s):
        parts = [float(v) for v in s.split(',')]
        if len(parts) != 6:
            raise ValueError(f"init_pose 需要 6 个值 x,y,z,r,p,y，实际得到: {s}")
        x, y, z, roll, pitch, yaw = parts
        return [x, y, z], [roll, pitch, yaw]

    init_pos0, init_euler0 = parse_init_pose(args.init_pose_left)
    init_pos1, init_euler1 = parse_init_pose(args.init_pose_right)

    # ---- 初始化机器人 ----
    arm0 = SingleArm(can_interface_=args.left_can)
    arm1 = SingleArm(can_interface_=args.right_can)
    time.sleep(2)

    # ---- 移动到初始位姿 ----
    print(f"[INFO] 移动到初始位姿: left={args.init_pose_left}  right={args.init_pose_right}")
    arm0.set_end_effector_pose_euler(pos=init_pos0, euler=init_euler0, tf=1)
    arm1.set_end_effector_pose_euler(pos=init_pos1, euler=init_euler1, tf=1)
    arm0.setGripperPosition(1.0)
    arm1.setGripperPosition(1.0)
    print("[INFO] 已到达初始位姿")
    print(SERVER_IP, PORT)
    # ---- 连接 policy server（本机）----
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=SERVER_IP,
        port=PORT,
    )
    print(f"[INFO] 已连接策略服务器：ws://{SERVER_IP}:{PORT}")

    # ---- 摄像头 ----
    cam0 = init_yu12_camera(DEV)
    cam1 = init_yu12_camera(DEV + 2)

    # 预热
    for _ in range(50):
        _ = cam0.read()
        _ = cam1.read()
    print("[INFO] 摄像头预热完成，开始循环")

    input("按 Enter 开始")

    stdin_fd = sys.stdin.fileno()
    old_term = None
    try:
        if sys.stdin.isatty():
            old_term = termios.tcgetattr(stdin_fd)
            tty.setcbreak(stdin_fd)
        print("[INFO] 按 s：双臂复位到初始位姿并暂停推理；按 c：继续推理")

        paused = False
        while True:
            ch = _stdin_read_char_nonblocking()
            if ch in ("s", "S"):
                reset_arms_to_init(arm0, arm1, init_pos0, init_euler0, init_pos1, init_euler1)
                paused = True
                print("[INFO] 已复位到初始位姿，推理已暂停；按 c 继续")
                continue
            if ch in ("c", "C"):
                paused = False
                print("[INFO] 继续推理")

            if paused:
                time.sleep(0.02)
                continue

            # 1. 读取机器人实时状态 --------------------------------------
            # get_ee_pose_quat 返回 (pos_m, quat_wxyz)

            # time.sleep(0.3)

            pos0, quat_wxyz0 = arm0.get_ee_pose_quat()
            qw0, qx0, qy0, qz0 = quat_wxyz0
            quat = np.array([qx0, qy0, qz0, qw0])  # scipy xyzw 格式
            p_tcp0 = flange_position_to_tcp(pos0, quat_wxyz0, tcp_off)
            x0, y0, z0 = p_tcp0

            pos1, quat_wxyz1 = arm1.get_ee_pose_quat()
            qw1, qx1, qy1, qz1 = quat_wxyz1
            quat1 = np.array([qx1, qy1, qz1, qw1])  # scipy xyzw 格式
            p_tcp1 = flange_position_to_tcp(pos1, quat_wxyz1, tcp_off)
            x1, y1, z1 = p_tcp1

            if args.tcp_debug:
                print(
                    "[TCP DEBUG][obs] flange -> state(TCP) | "
                    f"L {np.asarray(pos0).round(4)} -> {np.asarray(p_tcp0).round(4)} | "
                    f"R {np.asarray(pos1).round(4)} -> {np.asarray(p_tcp1).round(4)} | "
                    f"off {tcp_off.tolist()}"
                )

            gripper_open = float(arm0.get_gripper_position())
            gripper_open1 = float(arm1.get_gripper_position())
            # 与 chunk 内夹爪维一致：下发 <0.3 则 −0.2 → 观测送入策略时对 <0.3 读数 +0.2
            gripper_open = float(np.clip(np.where(gripper_open < 0.3, gripper_open + 0.2, gripper_open), 0.0, 1.0))
            gripper_open1 = float(np.clip(np.where(gripper_open1 < 0.3, gripper_open1 + 0.2, gripper_open1), 0.0, 1.0))

            # 记录当前位姿（欧拉角单位为弧度），用于后续 action 插值的起点
            cur_pos0 = arm0.get_ee_pose_euler()[0].tolist()
            cur_euler0 = arm0.get_ee_pose_euler()[1].tolist()
            cur_pos1 = arm1.get_ee_pose_euler()[0].tolist()
            cur_euler1 = arm1.get_ee_pose_euler()[1].tolist()
        

            state_vec = np.array(
                [x0, y0, z0, *quat, gripper_open, x1, y1, z1, *quat1, gripper_open1], dtype=np.float32
            )

            print("[INFO] 机器人状态：", state_vec)

            # 2. 读取摄像头图像（YU12 -> RGB）----------------------------
            img_rgb0 = grab_rgb_latest(cam0)  # (H, W, 3), RGB, uint8
            img_rgb1 = grab_rgb_latest(cam1)  # (H, W, 3), RGB, uint8

            # resize 到 224x224
            img_rgb0 = cv2.resize(img_rgb0, (224, 224), interpolation=cv2.INTER_AREA)
            img_rgb1 = cv2.resize(img_rgb1, (224, 224), interpolation=cv2.INTER_AREA)

            # debug 保存一张
            # cv2.imwrite("cam_debug0.png", cv2.cvtColor(img_rgb0, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("cam_debug1.png", cv2.cvtColor(img_rgb1, cv2.COLOR_RGB2BGR))

            # 3. 组装 observation & 远程推理 ----------------------------
            # 注意这里的 key 要和你 pi0 的 FastUMIInputs 配置对应：
            # - state -> "state"
            # - front 图 -> data["image"]["front"]
            print("state_vec:", state_vec)
            # input("按 Enter 继续组装 obs 并推理...")
            obs = {
                "state": state_vec,
                "image": { 
                    "robot_0": image_tools.convert_to_uint8(img_rgb0),
                    "robot_1": image_tools.convert_to_uint8(img_rgb1)
                },
                "prompt": args.description,
            }
            # cv2.putText(img_rgb0, f"robot0  /dev/video{cam0}", (10, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # cv2.putText(img_rgb1, f"robot1  /dev/video{cam1}", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # combined = np.concatenate([img_rgb0, img_rgb1], axis=1)
            # cv2.imshow("Camera Test  (press q to quit)", combined)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            resp = policy_client.infer(obs)

            # 兼容 "actions" / "action" 两种命名
            if "actions" in resp:
                actions_all = resp["actions"]
            else:
                actions_all = resp["action"]

            # 策略侧可能是只读 numpy / torch，需 copy 后才能原地改夹爪维
            actions_all = np.array(actions_all, dtype=np.float64, copy=True)
            if actions_all.ndim == 1:
                actions_all = actions_all.reshape(1, -1)
            # actions_all[..., 6] = np.where(((0.2 < actions_all[..., 6]) & (actions_all[..., 6] < 0.3)), actions_all[..., 6] - 0.2, actions_all[..., 6])
            # actions_all[..., 6] = np.clip(actions_all[..., 6], 0.0, 1.0)
            # actions_all[..., 13] = np.where(((0.2 < actions_all[..., 13]) & (actions_all[..., 13] < 0.3)), actions_all[..., 13] - 0.2, actions_all[..., 13])
            # actions_all[..., 13] = np.clip(actions_all[..., 13], 0.0, 1.0)
            actions_all[..., 6] = np.where((actions_all[..., 6] < 0.8), actions_all[..., 6] - 0.2, actions_all[..., 6])
            actions_all[..., 6] = np.clip(actions_all[..., 6], 0.0, 1.0)
            actions_all[..., 13] = np.where((actions_all[..., 13] < 0.95), actions_all[..., 13] - 0.2, actions_all[..., 13])
            actions_all[..., 13] = np.clip(actions_all[..., 13], 0.0, 1.0)
            # if len(actions_all) == 0:
            #     print("[WARN] 策略返回空 action。")
            #     time.sleep(0.1)
            #     continue
            # time.sleep(0.1)
            a0, a1 = args.action_start, args.action_end
            n_act = len(actions_all)
            if a0 < 0 or a1 < 0:
                print(f"[WARN] 忽略非法 action 区间: action_start={a0} action_end={a1}")
                continue
            if a0 > a1:
                print(f"[WARN] action_start 不能大于 action_end: {a0} > {a1}")
                continue
            if n_act == 0:
                print("[WARN] 策略返回的 action 序列为空")
                continue
            lo = a0
            hi = min(a1, n_act - 1)
            if lo >= n_act:
                print(f"[WARN] action_start 越界: start={a0} 序列长度={n_act}")
                continue
            if lo > hi:
                print(f"[WARN] 无 action 可执行: 区间 [{a0},{a1}] 与长度 {n_act} 无交集")
                continue
            action_slice = actions_all[lo : hi + 1]
            # 日志「第 N 步」的 N 与 actions_all 的下标一致
            for step_idx, action in enumerate(action_slice, start=lo):
                # 每步都检测 s 键，及时响应复位
                if _stdin_wants_reset_pause():
                    reset_arms_to_init(arm0, arm1, init_pos0, init_euler0, init_pos1, init_euler1)
                    paused = True
                    print("[INFO] 已复位到初始位姿，推理已暂停；按 c 继续")
                    break

                print(f"第 {step_idx} 步, action:", action)

                # 4. 解析 action 并下发到机械臂 -----------------------------
                # 双臂 action：14 维
                # [x0,y0,z0,r0,p0,y0,g0, x1,y1,z1,r1,p1,y1,g1]
                # rpy 单位：度（策略输出）；下发前转为弧度与 SDK 一致
                x_a0, y_a0, z_a0, roll0, pitch0, yaw0, g_open0, x_a1, y_a1, z_a1, roll1, pitch1, yaw1, g_open1 = action

                # 夹爪：模型输出 g_open 0~1，直接传给 startouch
                g0 = float(np.clip(g_open0, 0.0, 1.0))
                g1 = float(np.clip(g_open1, 0.0, 1.0))

                tgt_euler_rad0 = np.deg2rad([roll0, pitch0, yaw0]).tolist()
                tgt_euler_rad1 = np.deg2rad([roll1, pitch1, yaw1]).tolist()
                # 策略输出 TCP 目标；SDK 需要法兰位置（随目标姿态旋转的偏移）
                tgt_pos0 = tcp_position_to_flange(
                    [x_a0, y_a0, z_a0], tgt_euler_rad0, tcp_off
                ).tolist()
                tgt_pos1 = tcp_position_to_flange(
                    [x_a1, y_a1, z_a1], tgt_euler_rad1, tcp_off
                ).tolist()

                if args.tcp_debug:
                    print(
                        "[TCP DEBUG][cmd] step=%d TCP->flange | "
                        "L [%.4f,%.4f,%.4f] -> %s | "
                        "R [%.4f,%.4f,%.4f] -> %s"
                        % (
                            step_idx,
                            x_a0,
                            y_a0,
                            z_a0,
                            tgt_pos0,
                            x_a1,
                            y_a1,
                            z_a1,
                            tgt_pos1,
                        )
                    )

                # 模型输出 rpy 为度，已转弧度后下发
                # arm0.set_end_effector_pose_euler(pos=tgt_pos0, euler=tgt_euler_rad0, tf=args.dt)
                # arm1.set_end_effector_pose_euler(pos=tgt_pos1, euler=tgt_euler_rad1, tf=args.dt)
                # arm0.setGripperPosition(g0)
                # arm1.setGripperPosition(g1)

                # 如需启用插值平滑控制，可改用下面这段（当前默认注释）：
                interp_and_move_both(
                    arm0, arm1,
                    cur_pos0, cur_euler0, tgt_pos0, tgt_euler_rad0,
                    cur_pos1, cur_euler1, tgt_pos1, tgt_euler_rad1,
                    g0, g1,
                    step_size=args.interp_step_size,
                    dt=args.dt,
                )
                cur_pos0, cur_euler0 = tgt_pos0, tgt_euler_rad0
                cur_pos1, cur_euler1 = tgt_pos1, tgt_euler_rad1
                # input("按 Enter 执行下一步 action...")
                # time.sleep(args.dt)

            # 可选：按 ESC 退出（不需要显示窗口也没关系）
            # if cv2.waitKey(1) == 27:
            #     break

    finally:
        if old_term is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)
        cam0.release()
        cam1.release()
        arm0.cleanup()
        arm1.cleanup()
        # cv2.destroyAllWindows()
        print("[INFO] 结束，摄像头与机械臂已释放。")


if __name__ == "__main__":
    main()
