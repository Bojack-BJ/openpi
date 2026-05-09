# HL Memory V1

HL Memory 是高层 VLM 子系统，负责从历史 keyframes、recent observation clip、任务文本和上一轮 language memory 中预测：

- `current_subtask`：给低层 policy 的当前可执行子任务
- `updated_language_memory`：给下一轮 HL 和低层 policy 读的紧凑状态
- `keyframe_candidate_positions`：recent clip 内值得进入 historical memory 的候选帧，1-indexed
- `phase` / `target_query` / `goal_query`

代码入口：

- `src/openpi/hl_memory/`
- `scripts/export_hl_annotations_from_subtasks.py`
- `scripts/export_hl_memory_dataset.py`
- `scripts/train_hl_memory.py`
- `scripts/eval_hl_memory_rollout.py`
- `scripts/run_hl_memory_zero_shot.py`

当前支持 Qwen2.5-VL / Qwen3.5-VL。当前不做在线 robot wrapper、target mask 训练或低层 action 闭环。

## FastUMI Pipeline

生产路径必须以 LeRobot episode index 为准：先转 LeRobot，再从 `subtask_segments.json` 生成 HL annotations。不要直接用 raw session 排序做训练 annotations，因为 raw session 可能被过滤、跳过或合并，顺序不一定等于最终 `episode_index`。

### 1. Raw To LeRobot

raw session 里放 `subtask.json`：

```text
session_001/
  RGB_Images/video.mp4
  Merged_Trajectory/merged_trajectory.txt
  annotation/
  subtask.json
```

推荐 `subtask.json`：

```json
{
  "video_fps": 60.0,
  "interval_convention": "half_open",
  "segments": [
    {"start_frame": 0, "end_frame": 276, "subtask": "pick up the motor"},
    {"start_frame": 276, "end_frame": 501, "subtask": "place the motor into the box"}
  ]
}
```

转换：

```bash
export HF_LEROBOT_HOME=/root/Users/dataset/lerobot_home

python dataprocess_new/fastumi_raw_to_lerobot_v21.py \
  --raw-dir /path/to/raw_task_root \
  --repo-id fastumi/sponge_visual_guided \
  --task "Put the target object into the target slot" \
  --robot-type fasttouch \
  --fps 20 \
  --traj-source merge \
  --mode image \
  --include-guidance
```

关键输出：

```text
$HF_LEROBOT_HOME/fastumi/sponge_visual_guided/subtask_segments.json
```

这个文件里的 episode key 就是后续 HL/LL 训练使用的真实 `episode_index`。

### 2. Subtasks To HL Annotation JSONL

推荐从 LeRobot root 或 repo id 生成：

```bash
python scripts/export_hl_annotations_from_subtasks.py \
  --repo-id fastumi/sponge_visual_guided_xarm \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --instruction "Put the target object into the target slot" \
  --overwrite
```

等价路径输入：

```bash
python scripts/export_hl_annotations_from_subtasks.py \
  --lerobot-dir /root/Users/dataset/lerobot_home/fastumi/sponge_visual_guided \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --overwrite
```

raw fallback 只用于检查或没有 LeRobot sidecar 的临时场景：

```bash
python scripts/export_hl_annotations_from_subtasks.py \
  --raw-dir /path/to/raw_task_root \
  --output-jsonl /tmp/annotations.jsonl \
  --overwrite
```

默认每个 segment 导出两条 annotation：`start_frame` 的 `subtask_boundary` 和 segment 中点的 `progress`。这样 HL 同时学习 subtask 切换点和 subtask 进行中的稳定状态。`--emit-success-events` 会额外在 `end_frame - 1` 导出 `success`，当前默认不启用。

长 segment 可以按长度补充 progress 样本，但建议加上 cap，避免长时间段主导训练：

```bash
python scripts/export_hl_annotations_from_subtasks.py \
  --repo-id fastumi/sponge_visual_guided_xarm \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --instruction "Put the target object into the target slot" \
  --progress-sample-stride 50 \
  --max-progress-samples-per-segment 4 \
  --progress-min-gap 10 \
  --overwrite
```

### 3. HL Annotation JSONL To HL Dataset

```bash
python scripts/export_hl_memory_dataset.py \
  --source-config-name sponge_visual_guided_qwen3_5_2b_400m_touch \
  --annotations-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --output-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported \
  --visual-mode raw \
  --recent-frames-length 8 \
  --frame-subsample 5 \
  --memory-length 8 \
  --merge-distance 5 \
  --overwrite
```

输出：

```text
exported/
  samples.jsonl
  metadata.json
  frames/frame_XXXXXXXX.png
```

`samples.jsonl` 是 HL VLM 训练和评估的数据集。

`--visual-mode raw` 是默认值，也是推荐值。HL export 只读取必要的 parquet columns：episode/frame/task/subtask/prompt 和 RGB image columns；不会读取 state/action，也不会读取或传给 HL 任何 mask / overlay / 高亮目标。`--visual-mode config` 只影响优先选择哪些已配置的 RGB camera column，仍然会排除 mask/overlay。

推荐按 episode 做 held-out split，而不是直接用同一个 `exported/` 同时训练和评估。下面这条命令会在一次运行里计算一次 split，并同时生成互斥的 train / val episode，避免两次运行时误填不同 ratio/seed：

```bash
python scripts/export_hl_memory_dataset.py \
  --source-config-name sponge_visual_guided_qwen3_5_2b_400m_touch \
  --annotations-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --output-train-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_train \
  --output-val-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_val \
  --val-ratio 0.1 \
  --split-seed 42 \
  --visual-mode raw \
  --overwrite
```

也可以显式指定 episode：`--episode-indices 0,3,7-12` 或排除 episode：`--exclude-episode-indices 100-120`。`metadata.json` 会记录 selected episode 列表，方便确认 train/val 没有重叠。旧的单目录模式仍然可用：`--output-dir ... --episode-split train|val|all`。

如果报 `Episode ... was not found in dataset`，说明 `annotations.jsonl` 的 episode index 和当前 LeRobot dataset 不匹配；优先用上一步的 `--repo-id` / `--lerobot-dir` 从 `subtask_segments.json` 重新生成 annotations。只有确认要导出交集时才加 `--missing-episode-policy skip`。

### 4. Train

```bash
python scripts/train_hl_memory.py \
  --dataset-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_train \
  --output-dir /root/Users/checkpoints/hl_memory/sponge_visual_guided_qwen35 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-2B \
  --precision float16 \
  --device cuda \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-train-steps 1000 \
  --save-interval 200
```

多卡数据并行训练用 `torchrun`。每个 rank 独立采样 HL sample，同步梯度；只有 rank0 打印 tqdm/loss 和保存 checkpoint：

```bash
export PYTHONPATH=/lumos-vePFS/suzhou/Users/lixiaotong/openpi_second_branch/src
export WANDB_API_KEY="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"
export WANDB_MODE=online

torchrun --standalone --nproc_per_node 8 scripts/train_hl_memory.py \
  --dataset-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_train \
  --output-dir /root/Users/checkpoints/hl_memory/sponge_visual_guided_qwen35 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-2B \
  --precision bfloat16 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --frame-cache-size 4096 \
  --num-train-steps 2000 \
  --save-interval 200 \
  --wandb-enabled \
  --wandb-project openpi-hl-memory \
  --wandb-run-name sponge-qwen35-2b-hl
```

这里 `--batch-size` 是每张卡的 micro batch；有效全局 batch 约等于 `batch_size * grad_accum_steps * nproc_per_node`。训练进度条会显示 ETA、`s/it` / `it/s`、`data_s/it` 和 `step_s/it`。开启 wandb 后 rank0 会记录 `train/loss`、`time/data_s_per_it`、`time/step_s_per_it`、`time/data_fraction`、`train/lr` 和 `train/global_batch_size`。`--frame-cache-size` 是每个 rank 缓存的 resized frame 数；如果 `data_s/it` 占比高，可以适当增大，前提是 CPU 内存足够。

可用 backend / variant：

- `--vlm-backend qwen2_5_vl`
- `--vlm-backend qwen3_5_vl --vlm-variant qwen3_5_2b|qwen3_5_4b|qwen3_5_27b`
- variant 也支持 `2b` / `4b` / `27b`

Qwen3.5 建议：

- 默认可用 `bfloat16`，GPU 不支持时 runtime 会降到 fp16。
- vision `conv3d` / cuDNN 报错时先显式 `--precision float16`。
- 如果 `loss=nan`，优先在 A100/H100 等支持 bf16 的 GPU 上改用 `--precision bfloat16`；full fine-tune 时模型参数本身是 fp16/bf16，GradScaler 会自动禁用，不要依赖它解决纯 fp16 溢出。
- 训练多卡 DDP 不要同时设置 `--parallel-mode device_map|tensor_parallel`。
- 27B 单卡推理 OOM 时可用 `--parallel-mode device_map --device-map auto`；这是模型切分，不是数据并行训练。

也可用 YAML：

```bash
python scripts/train_hl_memory.py --config-yaml src/openpi/hl_memory/train_hl_memory.yaml
```

### 5. Eval

```bash
python scripts/eval_hl_memory_rollout.py \
  --dataset-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_val \
  --model-path /root/Users/checkpoints/hl_memory/sponge_visual_guided_qwen35/checkpoint-step-001000 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --device cuda \
  --frame-cache-size 512 \
  --output-json /root/Users/dataset/hl_memory/sponge_visual_guided/eval_metrics.json
```

评估默认跑四种 ablation：`no_memory`、`language_memory_only`、`keyframe_memory_only`、`full`。核心指标是 subtask match、phase/target/goal accuracy、keyframe precision/recall、memory similarity/drift、event accuracy。

快速 smoke eval 可以先只跑少量未见 episode 或单个 ablation：

```bash
python scripts/eval_hl_memory_rollout.py \
  --dataset-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_val \
  --model-path /root/Users/checkpoints/hl_memory/sponge_visual_guided_qwen35/checkpoint-step-001000 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --device cuda \
  --max-episodes 10 \
  --eval-modes full \
  --output-json /root/Users/dataset/hl_memory/sponge_visual_guided/eval_metrics_smoke.json
```

`--max-samples` 也可用于快速调试，但它可能截断 episode 中间序列；正式 rollout metrics 更推荐用 `--max-episodes`。

Eval 会打印阶段日志并显示每种 ablation 的 tqdm 进度条：

- `[stage] loading processor` / `[stage] loading model weights`：正在读取 processor 和 checkpoint。
- `[stage] moving model to cuda`：正在把模型搬到 GPU；如果长时间停在这里，优先检查 GPU 显存、CUDA 初始化和 `--parallel-mode device_map`。
- `HL eval <mode>`：已经进入逐 sample rollout；如果停在某个 sample，通常是该 sample 的视频 processor encode 或 `generate()` 慢。
- `--frame-cache-size` 控制 resized frame LRU cache，默认 `512`；如果 eval 反复读取同一批帧且 CPU 内存足够，可以适当增大。

### 6. Video Inference

单视角：

```bash
python scripts/run_hl_memory_zero_shot.py \
  --video-path /path/to/front.mp4 \
  --instruction "Put the target object into the target slot" \
  --language-memory "Task started." \
  --recent-end-sec 42 \
  --recent-step-sec 1 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --local-vlm-ckpt-path /path/to/Qwen3.5-2B-or-hl-checkpoint \
  --precision float16 \
  --device cuda \
  --debug-dir /tmp/hl_debug
```

双视角 rollout：

```bash
python scripts/run_hl_memory_zero_shot.py \
  --left-video-path /path/to/left.mp4 \
  --right-video-path /path/to/right.mp4 \
  --instruction "Put the target object into the target slot" \
  --language-memory "Task started." \
  --rollout-interval-sec 2 \
  --rollout-start-sec 0 \
  --rollout-end-sec 40 \
  --recent-step-sec 1 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --local-vlm-ckpt-path /path/to/Qwen3.5-2B-or-hl-checkpoint \
  --precision float16 \
  --device cuda \
  --debug-dir /tmp/hl_rollout_debug \
  --output-json /tmp/hl_rollout_debug/summary.json
```

Input rules：

- 单视角 `--video-path` 映射为 `front`。
- 双视角 `--left-video-path` / `--right-video-path` 映射为 `robot_0` / `robot_1`，同一时间抽帧后横向拼接。
- memory clip 和 recent clip 都按时间从旧到新排列。
- recent clip 最后一张有效帧定义当前状态，`current_subtask` 必须描述最后有效帧。
- rollout 会把上一轮 `updated_language_memory` 和 keyframe candidates 带到下一轮，并在 `debug_dir/rollout_step_XXX/` 保存实际输入帧。

常用参数说明：

| 参数 | 作用 |
| --- | --- |
| `--instruction` | 任务指令，会进入 HL VLM prompt；如果配合 `--task-config-path`，还会拼接 nominal primitive plan。 |
| `--video-path` | 单视角输入视频，内部视角名为 `front`。 |
| `--left-video-path` / `--right-video-path` | 双视角输入视频，内部视角名为 `robot_0` / `robot_1`，同一时间抽帧后横向拼接成一张 HL frame。 |
| `--task-config-path` | JSON task plan，可提供 `task_description` 和 `steps`，作为分段先验；视觉证据优先级仍高于 plan。 |
| `--model-path` | 已训练 HL checkpoint 或 Hugging Face/local model 路径。 |
| `--local-vlm-ckpt-path` | 本地 VLM/checkpoint 路径；设置后优先覆盖 `--model-path`。 |
| `--vlm-backend` | VLM 后端，常用 `qwen2_5_vl` 或 `qwen3_5_vl`。 |
| `--vlm-variant` | Qwen3.5 变体，例如 `qwen3_5_2b`，也可用短名 `2b` / `4b` / `27b`。 |
| `--precision` | 推理精度，常用 `float16` 或 `bfloat16`；遇到 vision/cuDNN 问题先试 `float16`。 |
| `--device` | 推理设备，通常是 `cuda`；CPU 只适合小模型/调试。 |

clip 和 memory 参数：

| 参数 | 作用 |
| --- | --- |
| `--recent-end-sec` | 单次预测时 recent clip 的结束时间；不填时默认取视频末尾。 |
| `--recent-step-sec` | recent clip 抽帧间隔，默认每 1 秒取一帧。 |
| `--recent-seconds` | 手动指定 recent clip 秒数列表，例如 `10,12,14`；不能和 rollout interval 模式一起用。 |
| `--memory-seconds` | 手动指定历史 memory keyframe 秒数列表，例如 `2,8,15`。 |
| `--auto-memory` | 单次预测时自动从 recent 前面的时间段取 memory frames；rollout 模式会关闭它并使用上一步 keyframes。 |
| `--recent-frames-length` | recent clip 最大帧数，默认 `8`。 |
| `--memory-length` | memory clip 最大帧数，默认 `8`。 |
| `--frame-height` / `--frame-width` | 输入 VLM 前的 frame resize 尺寸，默认 `224x224`。 |
| `--allow-single-frame-fallback` | 视频太短或时间点不足时允许复用单帧补齐，方便调试短视频。 |

rollout 和调试参数：

| 参数 | 作用 |
| --- | --- |
| `--rollout-interval-sec` | 开启 recurrent rollout；每隔 N 秒推理一次，并把上一轮 language memory/keyframes 传到下一轮。 |
| `--rollout-start-sec` / `--rollout-end-sec` | rollout 起止时间；不填 end 时默认到视频末尾。 |
| `--keyframe-merge-distance-sec` | 合并距离过近的 keyframe candidates，默认 `2.0` 秒。 |
| `--language-memory` | 初始语言记忆；空字符串表示从默认初始状态开始。 |
| `--output-json` | 保存最终 summary JSON。 |
| `--rollout-jsonl` | rollout 模式下逐步保存 compact JSONL。 |
| `--rollout-pretty-json` | rollout 模式下逐步保存可读 JSON。 |
| `--debug-dir` | 保存输入帧、keyframe candidates、debug panel；排查模型预测时建议开启。 |
| `--debug-video-fps` | 用 debug panels 合成 rollout debug 视频的 FPS。 |
| `--max-new-tokens` | 非 thinking 模式的最大生成 token 数。 |
| `--enable-thinking` | 打开 Qwen thinking；默认关闭，建议先关闭以保证 JSON 输出稳定。 |
| `--parallel-mode` | 大模型加载方式；单卡默认 `none`，27B OOM 时可试 `device_map`。 |
| `--device-map` / `--tensor-parallel-plan` | 配合 `--parallel-mode device_map|tensor_parallel` 使用。 |

## Data Formats

### HL Annotation JSONL

一行一个 JSON，是 `scripts/export_hl_memory_dataset.py` 的输入。

最小行：

```json
{"episode_index": 0, "frame_index": 276, "current_subtask": "place the motor into the box"}
```

推荐行：

```json
{
  "episode_index": 0,
  "frame_index": 276,
  "current_subtask": "place the motor into the box",
  "instruction": "put the motor into the box",
  "phase": "place the motor into the box",
  "target_query": "motor",
  "goal_query": "box",
  "event_type": "progress",
  "event_text": "Continuing placing the motor into the box."
}
```

`event_type` 支持 `none`、`subtask_boundary`、`success`、`failure`、`progress`、`discovery`。它会影响导出的 target `updated_language_memory`。

HL prediction 还支持可选 SAM grounding 字段：

```json
{
  "sam_text_prompt": "motor",
  "sam_point_xy": [120, 85]
}
```

`sam_point_xy` 是 last valid recent frame 上的像素点，用于给 SAM point prompt；`sam_text_prompt` 是给 SAM/Grounding-SAM 的短文本 prompt。没有人工 point label 时，训练 target 通常只监督 `sam_text_prompt`，point prompt 主要靠 VLM zero-shot/推理能力生成。

### Language Memory

`updated_language_memory` 是给下游低层 VLM/action policy 读的 compact context，不是 debug log。Rollout fallback 会规整成四行：

```text
Task progress: <one short sentence>
Current objective: <one short executable objective>
Relevant objects: <object/location phrases, or none>
Notes: <one short caution/spatial fact, or none>
```

不要写逐帧时间戳、frame id、长日志或 raw model output。调试看 `raw_model_output`、`model_prediction`、`rollout_pretty.json`。

### LL Mask/Subtask Guidance

LL policy 不新增模型字段，仍走 `prompt + observation.images`。

可选 LeRobot 字段：

```text
observation.images.front
observation.masks.front_mask
observation.images.robot_0_image
observation.images.robot_1_image
observation.masks.robot_0_mask
observation.masks.robot_1_mask
subtask
```

当前行为：

- `FastUMIData7DRPYGuidedConfig` / `FastUMIdualData14DRPYGuidedConfig` 在 dataloader transform 阶段做 mask overlay。
- 有 `subtask` 或 `current_subtask` 时拼进 prompt。
- mask 缺失时自动补零，等价于原图。
- LeRobot 2.1 训练时通过 `DataConfig(subtask_segments_path="subtask_segments.json")` 动态注入当前帧 subtask。

## CrossTask Smoke

CrossTask 只用于 action-free HL smoke。建议只用 18 个有 temporal boundary 的 primary tasks。

```bash
python scripts/export_hl_memory_crosstask.py \
  --crosstask-release-dir /path/to/crosstask_release \
  --videos-root /path/to/missing_videos \
  --split train \
  --output-dir /path/to/hl_memory_train \
  --recent-frames-length 8 \
  --frame-subsample 1 \
  --memory-length 8 \
  --merge-distance 1 \
  --overwrite
```

一键 smoke：

```bash
DATA_ROOT=/path/to/crosstask \
DEVICE=cuda \
MODEL_BACKEND=qwen2_5_vl \
MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
TRAIN_STEPS=20 \
scripts/hl_memory_crosstask_smoke.sh
```

可先检查本地视频覆盖率：

```bash
python scripts/check_crosstask_video_coverage.py \
  --crosstask-release-dir /path/to/crosstask_release \
  --videos-root /path/to/missing_videos \
  --split train \
  --verify-decodable
```

## Troubleshooting

- Episode 错位：用 `--repo-id` / `--lerobot-dir` 从 `subtask_segments.json` 生成 annotations；确认 `source_config_name` 指向同一个 LeRobot `repo_id`。
- 缺依赖：HL export/train/eval 需要 `Pillow`、`torch`、`transformers`；`--config-yaml` 额外需要 `pyyaml`。
- Qwen3.5 JSON 不稳定：默认关闭 thinking；只有需要 reasoning trace 时加 `--enable-thinking --thinking-budget-tokens 128 --thinking-max-new-tokens 1024`。
- 多机 `.arrow` / `.incomplete`：`HF_LEROBOT_HOME` 是数据根目录，`HF_DATASETS_CACHE` 是 HuggingFace parquet/Arrow 派生缓存。多机不要共享同一个可写 cache，建议 `export HF_DATASETS_CACHE=/tmp/openpi_hf_datasets_cache_${HOSTNAME}`。
