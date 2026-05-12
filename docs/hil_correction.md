# HIL Correction Rollout

本文档描述 `codex-hil-correction-fasttouch-touch` 分支里的 single-arm HIL correction 流程：策略先正常闭环控制，人工用 UMI XV SLAM + clamp 接管修正，成功后把 policy + human correction 片段保存为 LeRobot episode。

## 适用范围

- 当前只支持 `scripts/pi0_rollout_client_xarm_rpy.py` 的 `--arm_mode single`。
- 必须指定 `--single_arm robot_0` 或 `--single_arm robot_1`。
- HIL 输入来自 UMI XV ROS topic：`/xv_sdk/<serial>/slam/pose` 和 `/xv_sdk/<serial>/clamp/Data`。
- 输出是一个新的 LeRobot dataset，路径为 `$HF_LEROBOT_HOME/<hil_output_repo_id>`。
- 可以和 `serve_policy.py --mask-overlay ...` 联合使用；HIL 本身只记录 RGB front image，不记录 mask。

## 启动 Policy Server

普通 checkpoint：

```bash
python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config sponge_visual_guided_pi05 \
  --policy.dir /path/to/checkpoint
```

如果需要 SAM3 mask overlay，在 `policy:checkpoint` 前加 mask 参数：

```bash
python scripts/serve_policy.py \
  --mask-overlay \
  --mask-overlay-view front \
  --sam3-checkpoint-path /root/Users/lixiaotong/openpi/third_party/sam3/ckpt/sam3.pt \
  --mask-overlay-tracking-mode video_window \
  --mask-overlay-video-window-size 8 \
  --sam3-video-version sam3 \
  policy:checkpoint \
  --policy.config sponge_visual_guided_pi05 \
  --policy.dir /root/Users/lixiaotong/openpi/checkpoints/sponge_visual_mask_keys_pi05_xarm/sponge_visual_mask_keys_pi05_xarm/60000
```

## 启动 Single-Arm HIL Rollout

`robot_0` 示例:

```bash
export HF_LEROBOT_HOME=/root/Users/dataset/lerobot_home
export PYTHONPATH=/root/Users/lixiaotong/openpi/src

python scripts/pi0_rollout_client_xarm_rpy.py \
  --arm_mode single \
  --single_arm robot_0 \
  --server_ip 127.0.0.1 \
  --port 8000 \
  --robot_ip <robot_0_xarm_ip> \
  --camera_dev0 0 \
  --description "Put the target object into the target slot" \
  --hil_correction \
  --umi_xv_serial <umi_serial> \
  --hil_slam_axes z,-x,-y \
  --hil_require_umi_tcp_alignment \
  --hil_umi_tcp_alignment_threshold_deg 25 \
  --hil_output_repo_id fastumi/sponge_visual_guided_xarm_hil \
  --hil_fps 20
```

`robot_1` 时改成：

```bash
--single_arm robot_1 \
--robot_ip_2 <robot_1_xarm_ip> \
--camera_dev1 <camera_id>
```

同一个任务可以复用同一个 `--hil_output_repo_id`。如果 dataset 已存在，`e` 每次保存都会 append 一个新 episode；只有确认要清空重建时才加 `--hil_overwrite_dataset`。

## 键盘控制

- `Enter`：开始主循环。
- `s`：reset arm 到 init pose，丢弃当前 HIL buffer，暂停 inference。
- `c`：继续 policy inference。
- `t`：切换人工接管。第一次按会在下一帧锁定当前 UMI pose 和 xArm TCP 作为相对运动原点；再次按会结束接管并恢复 policy。开启 `--hil_require_umi_tcp_alignment` 后，如果 UMI orientation 和当前 TCP orientation 差距超过阈值，会持续打印误差并等待操作员调整，达标后自动开始接管。
- `e`：把当前 buffer append 为一个成功 LeRobot episode，然后清空 buffer 并暂停。
- `x`：丢弃当前 buffer，不保存。

## 训练数据语义

HIL recorder 会把当前成功片段写成 LeRobot episode：

- `observation.state`：8D TCP state，`x,y,z,qx,qy,qz,qw,gripper_width`。
- `action`：8D action，同样是 position + quaternion + gripper。
- `robot_0_state`：兼容字段，内容等于 `observation.state`。
- `observation.images.front`：224x224 RGB front image。
- `subtask`：policy frame 为 `policy_success_candidate`，人工接管 frame 为 `hil_correction`。
- `task`：来自 `--description`。

如果 `$HF_LEROBOT_HOME/<hil_output_repo_id>` 已存在，recorder 会打开现有 dataset 并从下一个 episode index 继续写；不会要求每次换 repo。

额外 sidecar 文件：

```text
$HF_LEROBOT_HOME/<hil_output_repo_id>/hil_metadata.jsonl
```

里面逐帧记录：

- `control_mode`: `policy` 或 `human`。
- `takeover_id`: 同一次人工接管的编号。
- `base_action_7d`: policy 原始 7D action。
- `human_action_7d`: 人工接管后的 7D action，policy frame 为 `null`。

`e` 保存的是当前 buffer 里的 policy + human frames。`t` 开始接管时会按 `--hil_pre_takeover_drop` 删除最近若干 policy frames，避免把接管前明显失败的动作写进成功片段。

## SLAM 坐标系与 UMI-TCP 对齐

UMI XV SLAM pose 是 camera frame：`z` 向前、`x` 向右、`y` 向下；SLAM world 原点和轴方向来自 UMI 开机/初始化时的姿态。因此推荐流程是：

- UMI 开机/初始化时先按固定姿态握持，让 raw SLAM world frame 和 robot base frame 只有确定的轴映射关系。
- 用 `--hil_slam_axes` 把 raw SLAM xyz 映射到 robot/base xyz。常见 xArm base 约定是 `x` 向前、`y` 向左、`z` 向上，对应 UMI camera frame 可先试 `--hil_slam_axes z,-x,-y`。
- 当前 HIL target 固定按 robot base frame 应用 mapped UMI delta：mapped translation 直接加到 TCP position，mapped rotation delta 左乘到 TCP orientation。
- `--hil_slam_axes` 现在会同时作用于 UMI position 和 orientation；它必须是右手系映射，否则 orientation 变换没有物理意义，程序会拒绝启动。
- 如果要让操作更直观，开启 `--hil_require_umi_tcp_alignment`。接管前先把 UMI 转到接近当前 TCP 的 orientation，再按 `t`；未达标时会持续打印 TCP pose、映射后的 UMI pose 和 orientation error，操作员继续调整，直到误差进入阈值后自动开始接管。
- 轴方向一定要用小位移验证。若 TCP 运动方向反了，优先改 `--hil_slam_axes` 的符号或排列。

## UMI 数据处理

- `UmiSlamReader` 订阅 UMI pose 和 clamp topic，callback 只保留最新一帧 pose/clamp；控制循环调用 `latest()` 时读取当前最新值。
- UMI SLAM position 按米处理；robot state/action 也按米处理。xArm SDK 需要 mm 的转换只发生在最终下发 `_set_xarm_pose()` 时。
- ROS subscriber 默认 `queue_size=1`，旧 SLAM 消息会被丢弃，避免高频 SLAM 在 ROS 层堆积造成控制延迟。
- `latest()` 会检查 `--umi_pose_max_age_s` 和 `--umi_gripper_max_age_s`，超过阈值则不发 HIL command。
- HIL 接管默认使用 `--hil_xarm_control_mode servo`：开始接管时切到 xArm `mode=1`，每步用 `set_servo_cartesian` 非阻塞下发 TCP pose，退出接管/复位/保存/丢弃/程序结束时切回 `mode=0`。如果 SDK 或机器人固件不支持，可用 `--hil_xarm_control_mode position` 回退到旧的 `set_position(wait=True)`。
- `servo` 模式下 `--dt` 是 HIL command 最小下发间隔，默认 `0.025s`；夹爪命令按时间和开合变化量节流，避免每个 pose 都等待夹爪动作。
- `servo` 模式会在下发前做平滑：位置用 EMA，旋转用 slerp。`--hil_servo_pos_alpha` / `--hil_servo_rot_alpha` 越小越稳但滞后越大；`1.0` 表示关闭对应平滑。
- `servo` 模式会先下发 xArm command，再按 `--hil_servo_record_interval_s` 记录 LeRobot frame。这样相机读图和 recorder 不再阻塞每一条 servo command；如果仍有一顿一顿的体感，先把记录间隔临时调到 `0.2` 验证控制链路。
- HIL action 日志默认按 `--hil_log_interval_s 0.5` 秒节流；如果设成 `0` 会每步打印，可能显著拖慢控制环。
- 坐标系排查使用 `scripts/debug_umi_hil_pose.py`，主 rollout 不再暴露 pose debug 参数。
- 当前 HIL servo command 已和 LeRobot frame 记录降频解耦；如果体感仍慢，下一步应把 xArm command loop 放到独立线程，彻底避免记录过程影响控制节奏。

## 关键参数

- `--umi_xv_serial`：UMI XV serial，HIL 模式必填。
- `--umi_max_gripper`：clamp raw value 对应 fully open 的值，用于归一化 gripper。
- `--umi_pose_max_age_s` / `--umi_gripper_max_age_s`：UMI pose / gripper 允许的最大数据延迟。
- `--umi_ros_queue_size`：UMI ROS subscriber queue size，默认 `1`，用于只保留最新消息。
- `--hil_ready_timeout_s`：启动时等待第一帧 UMI pose + gripper 的超时。
- `--hil_output_repo_id`：输出 LeRobot repo id。
- `--hil_fps`：写入 LeRobot 的 fps。
- `--hil_pre_takeover_drop`：开始接管时丢弃最近多少个 policy frames。
- `--hil_max_delta_xyz`：从接管起点算的累计 TCP 平移范数上限，单位米；`0` 表示不限制。
- `--hil_max_delta_rpy_deg`：从接管起点算的累计 TCP 旋转角上限，单位度；`0` 表示不限制。
- `--hil_slam_axes`：UMI raw SLAM xyz/rpy 到 robot base xyz/rpy 的右手系轴映射，例如 `z,-x,-y`。
- `--hil_slam_translation_scale`：UMI 平移到 TCP 平移的比例；UMI 与 TCP 均按米处理。
- `--hil_require_umi_tcp_alignment`：开始接管前检查 UMI orientation 是否接近当前 TCP orientation。
- `--hil_umi_tcp_alignment_threshold_deg`：UMI-TCP orientation 对齐角度阈值，默认 `25` 度。
- `--hil_xarm_control_mode`：HIL 接管期间的 xArm 下发模式，默认 `servo`；`position` 是阻塞式回退。
- `--hil_servo_pos_alpha`：HIL servo 位置平滑系数，默认 `0.35`；越小越稳但越滞后。
- `--hil_servo_rot_alpha`：HIL servo 旋转平滑系数，默认 `0.25`；越小越稳但越滞后。
- `--hil_servo_record_interval_s`：HIL servo 模式下记录 LeRobot frame 的最小间隔，默认 `0.1s`；`0` 表示每条 command 都记录。
- `--hil_log_interval_s`：HIL 状态日志打印间隔，默认 `0.5` 秒；设为 `0` 表示每步打印。

## 安全检查

- 第一次测试建议把 `--hil_max_delta_xyz` 降到 `0.02`，确认方向正确后再提高。
- 正式采集前先用很小的 UMI 位移检查 `--hil_slam_axes`；建议先开 `--hil_require_umi_tcp_alignment`。
- 出现 clipping warning 时，优先减小 UMI 手部动作；只有确认安全后再提高限幅。
- 失败 episode 用 `x` 丢弃，不要按 `e`。
- 不要在不确认的情况下使用 `--hil_overwrite_dataset`。

## 常见问题

- `HIL correction requires --arm_mode single`：当前实现只支持 single-arm。
- `HIL correction requires --umi_xv_serial`：需要提供 UMI XV serial。
- 等待 UMI 数据超时：检查 ROS 是否启动、topic 名是否包含正确 serial、pose 和 clamp topic 是否都有数据。
- 运行中提示 stale data：UMI pose 或 clamp 数据延迟超过阈值；检查 ROS 负载/网络，必要时谨慎增大 max age。
- 运动方向反了：调整 `--hil_slam_axes`，例如给某个轴加 `-` 或交换轴顺序。
- 想重新采集同名 dataset：确认后加 `--hil_overwrite_dataset`；默认行为是 append 新 episode。
