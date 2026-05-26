# HL Memory V1

HL Memory 是高层 VLM 子系统，负责从历史 keyframes、recent observation clip、任务文本和上一轮 language memory 中预测结构化高层状态：

- `task_progress`：历史状态，累积已经完成或稳定成立的任务进展
- `current_objective`：给低层 policy 的当前可执行目标
- `subtask_progress` / `should_advance_objective`：当前目标完成度和是否建议切换目标
- `active_hand`：当前主要执行手，取 `left` / `right` / `both` / 空字符串
- `relevant_objects` / `notes`：当前目标相关的物体、位置和动作约束
- `keyframe_candidate_positions`：recent clip 内值得进入 historical memory 的候选帧，1-indexed
- `phase` / `target_query` / `goal_query`
- `target_bbox_xyxy`：推理 debug 可选字段，训练 GT 不要求

代码入口：

- `src/openpi/hl_memory/`
- `scripts/hl_memory/export_hl_annotations_from_subtasks.py`
- `scripts/hl_memory/batch_export_hl_annotations_from_subtasks.py`
- `scripts/hl_memory/batch_normalize_hl_annotations_with_llm.py`
- `scripts/hl_memory/export_hl_memory_dataset.py`
- `scripts/hl_memory/batch_export_hl_memory_dataset_from_subtasks.py`
- `scripts/hl_memory/train_hl_memory.py`
- `scripts/hl_memory/train_hl_memory_multitask.py`
- `scripts/hl_memory/eval_hl_memory_rollout.py`
- `scripts/hl_memory/run_hl_memory_zero_shot.py`

当前支持 Qwen2.5-VL / Qwen3.5-VL。当前不做在线 robot wrapper、target mask 训练或低层 action 闭环。

兼容性说明：`current_subtask` 和 `updated_language_memory` 仍然会读写，用于兼容旧数据和旧 checkpoint；新协议以 `current_objective` 和四字段 language memory 为准。

## FastUMI Pipeline

生产路径必须以 LeRobot episode index 为准：先转 LeRobot，再从 `subtask_segments.json` 生成 raw HL annotations，然后用 LLM normalize，最后从 normalized annotations 导出 train/val samples。不要直接用 raw session 排序做训练 annotations，因为 raw session 可能被过滤、跳过或合并，顺序不一定等于最终 `episode_index`。

推荐顺序：

```text
raw sessions
  -> LeRobot repo + subtask_segments.json
  -> hl_annotations.jsonl
  -> hl_annotations_llm_normalized.jsonl
  -> HL train/val samples.jsonl
```

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

### 2. Subtasks To Raw HL Annotation JSONL

主力路径是 batch wrapper：先在每个 LeRobot task repo 旁边生成 raw `hl_annotations.jsonl`，后续再统一做 LLM normalize 和 HL dataset export。推荐采样策略是“按 segment 长度动态分配 progress 样本 + late fractions 强化切换前状态 + short segment 单独补内部 progress”，避免只学到 start/mid/end。

```bash
PYTHONPATH=src python scripts/hl_memory/batch_export_hl_annotations_from_subtasks.py \
  --subtask-root /root/Users/dataset/lerobot_home/subtask \
  --workers 8 \
  --progress-sample-target-frames 30 \
  --progress-extra-fractions 0.85,0.9,0.95 \
  --min-progress-samples-per-segment 2 \
  --max-progress-samples-per-segment 10 \
  --progress-sample-jitter 0.05 \
  --progress-sample-seed 42 \
  --progress-min-gap 10 \
  --short-segment-max-frames 40 \
  --short-segment-progress-fractions 0.2,0.4,0.6,0.75,0.9 \
  --short-segment-progress-min-gap -1 \
  --emit-success-events \
  --overwrite \
  --continue-on-error
```

输出：

```text
/root/Users/dataset/lerobot_home/subtask/<task_id>/hl_annotations.jsonl
```

单任务命令主要用于 debug、只重跑某个 task，或本地检查一个 repo 的采样分布：

```bash
python scripts/hl_memory/export_hl_annotations_from_subtasks.py \
  --repo-id fastumi/sponge_visual_guided_xarm \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --instruction "Put the target object into the target slot" \
  --overwrite
```

等价路径输入：

```bash
python scripts/hl_memory/export_hl_annotations_from_subtasks.py \
  --lerobot-dir /root/Users/dataset/lerobot_home/fastumi/sponge_visual_guided \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --overwrite
```

raw fallback 只用于检查或没有 LeRobot sidecar 的临时场景：

```bash
python scripts/hl_memory/export_hl_annotations_from_subtasks.py \
  --raw-dir /path/to/raw_task_root \
  --output-jsonl /tmp/annotations.jsonl \
  --overwrite
```

如果不传 progress 参数，默认每个 segment 导出两条 annotation：`start_frame` 的 `subtask_boundary` 和 segment 中点的 `progress`。这只适合 smoke test，不适合作为最终训练集。`--emit-success-events` 会额外在 `end_frame - 1` 导出 `success`，推荐训练时开启，让模型看到 `should_advance_objective=true` 附近的画面。

旧的 stride 采样仍然可用，但不推荐作为主路径，因为它不保证覆盖固定 progress 位置：

```bash
python scripts/hl_memory/export_hl_annotations_from_subtasks.py \
  --repo-id fastumi/sponge_visual_guided_xarm \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --instruction "Put the target object into the target slot" \
  --progress-sample-stride 50 \
  --max-progress-samples-per-segment 4 \
  --progress-min-gap 10 \
  --overwrite
```

注意采样分布：如果只用默认两点，`subtask_progress` 基本只覆盖 `0.0` 和 `0.5`，模型很难学到 late-stage 状态和 `should_advance_objective=true` 的切换时机。推荐主路径是“动态数量 + 比例位置 + 可选 jitter”：

```text
num_samples = clamp(segment_length / target_frames, min_samples, max_samples)
fractions = 1/(n+1), 2/(n+1), ..., n/(n+1)
optional jitter: fraction += uniform(-jitter, +jitter)
extra_fractions: add late-stage candidates such as 0.85/0.9/0.95
```

也就是说，`--progress-sample-target-frames` 不是直接每 N 帧固定 stride 取点，而是用 N 决定这个 segment 应该采几个内部 progress 点；位置仍按 segment 内部比例均匀铺开。`--progress-extra-fractions` 会在动态/固定/stride/midpoint 采样基础上额外补 late-stage 候选点，仍然受 `--progress-min-gap` 和 `--max-progress-samples-per-segment` 约束。`subtask_boundary` 覆盖 start，`--emit-success-events` 覆盖 end，所以 progress samples 不包含两端。

固定比例采样适合 segment 长度差异不大、希望分布稳定的任务：

```bash
python scripts/hl_memory/export_hl_annotations_from_subtasks.py \
  --repo-id fastumi/sponge_visual_guided_xarm \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --instruction "Put the target object into the target slot" \
  --progress-sample-fractions 0.2,0.4,0.6,0.8 \
  --progress-sample-jitter 0.03 \
  --progress-sample-seed 42 \
  --progress-min-gap 10 \
  --emit-success-events \
  --overwrite
```

动态采样适合 segment 长度差异较大的任务：短 segment 采 2–4 个，长 segment 最多采 5–10 个。batch wrapper 和单任务 exporter 支持同一组参数。

```bash
python scripts/hl_memory/export_hl_annotations_from_subtasks.py \
  --repo-id fastumi/sponge_visual_guided_xarm \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --instruction "Put the target object into the target slot" \
  --progress-sample-target-frames 30 \
  --progress-extra-fractions 0.85,0.9,0.95 \
  --min-progress-samples-per-segment 2 \
  --max-progress-samples-per-segment 10 \
  --progress-sample-jitter 0.05 \
  --progress-sample-seed 42 \
  --progress-min-gap 10 \
  --emit-success-events \
  --overwrite
```

如果存在很短的 segment，例如 20–40 帧，普通 `--progress-min-gap` 和 late fractions 可能只留下 start/success，导致模型看不到连续 progress。可以对短 segment 单独启用自适应固定比例采样：

```bash
python scripts/hl_memory/export_hl_annotations_from_subtasks.py \
  --repo-id fastumi/sponge_visual_guided_xarm \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --instruction "Put the target object into the target slot" \
  --progress-sample-target-frames 30 \
  --progress-extra-fractions 0.85,0.9,0.95 \
  --min-progress-samples-per-segment 2 \
  --max-progress-samples-per-segment 10 \
  --progress-sample-jitter 0.05 \
  --progress-sample-seed 42 \
  --progress-min-gap 10 \
  --short-segment-max-frames 40 \
  --short-segment-progress-fractions 0.2,0.4,0.6,0.75,0.9 \
  --short-segment-progress-min-gap -1 \
  --emit-success-events \
  --overwrite
```

`--short-segment-progress-min-gap -1` 会自动把全局 `--progress-min-gap` cap 到短 segment 内部采样点和 start/end 的最小间距，避免 20 帧 segment 里的 `0.75` 因为离 success 太近被删掉。如果想完全手动控制，可以设成 `2` 或 `3`。

`--progress-sample-fractions` 优先级高于 `--progress-sample-target-frames`，两者都不设置时才回退到 `--progress-sample-stride` / midpoint 旧逻辑。`--progress-extra-fractions` 不参与优先级判断，它总是在主采样模式后额外添加候选点，适合强化切换前的 late-stage 状态。`--progress-sample-jitter` 是 deterministic 的：同一 seed、episode、segment 会生成相同采样点，重跑可复现；不同 episode/segment 的 progress 不会永远落在完全相同的比例位置。正式训练前建议抽查 `hl_annotations.jsonl` 的 progress 分布，确认每个常见 subtask 至少包含 early/mid/late/end 样本。

### 3. LLM GT Normalization Before Dataset Export

这一步在导出 train/val samples 之前执行。它是第 3 步，因为它依赖第 2 步生成的 raw `hl_annotations.jsonl`；不要先 export train/val 再 normalize，否则旧 `samples.jsonl` 不会自动带上新字段。

如果 rule-based annotations 的 `current_subtask` 太像状态文本，或者你希望 `task_progress` 是更自然的累积历史，用离线 LLM 归一化 annotations。推荐用大模型只生成 GT sidecar，不要在训练或 rollout 时在线调用。

多任务批量归一化推荐用 batch 脚本。默认 `--granularity task`，每个 task 先生成一个 task-level segment sidecar，再把 sidecar 展开到所有 episode 的 annotation rows。这样同一个任务的重复 episode 不会反复调用 LLM。

```bash
PYTHONPATH=src python scripts/hl_memory/batch_normalize_hl_annotations_with_llm.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --model-path /root/Users/lixiaotong/Qwen3.5-27B \
  --device-map auto \
  --granularity task \
  --memory-summary-mode llm \
  --max-new-tokens 128 \
  --skip-existing \
  --continue-on-error
```

如果模型能放进单张 GPU，可以用多进程按 task 并行。每个 worker 会绑定一个 `CUDA_VISIBLE_DEVICES` 分组、独立加载一次模型、处理一部分 task：

```bash
PYTHONPATH=src python scripts/hl_memory/batch_normalize_hl_annotations_with_llm.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --model-path /root/Users/lixiaotong/Qwen3.5-27B \
  --device-map auto \
  --parallel-workers 4 \
  --granularity task \
  --memory-summary-mode code \
  --max-new-tokens 128 \
  --skip-existing \
  --continue-on-error
```

`--parallel-workers 4` 会自动把 4 个进程绑定到前 4 张可见 GPU。如果需要手动分组，可以用 `--worker-gpu-groups "0;1;2;3"` 表示 4 个进程各用一张卡，或 `"0,1;2,3"` 表示 2 个进程、每个进程内部用 `device_map auto` 切 2 张卡。27B 如果单卡放不下，就不要用一卡一进程，改用多卡分组。`--memory-summary-mode code --max-new-tokens 128` 是最快组合；如果希望 history summary 也由 LLM 精修，用 `--memory-summary-mode llm`，但它仍然只在 task sidecar 阶段执行一次。

如果要多台机器同时跑，先用 divide 脚本按 task 的 input row 数做均衡切分。它不会加载模型，只会生成 `summary.json`、每个 shard 的 task list 和可直接复制执行的 command：

```bash
PYTHONPATH=src python scripts/hl_memory/divide_hl_annotation_normalize_tasks.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --shards 4 \
  --output-dir /root/Users/dataset/hl_memory/normalize_shards \
  --skip-existing \
  -- \
  --model-path /root/Users/lixiaotong/Qwen3.5-27B \
  --device-map auto \
  --granularity task \
  --memory-summary-mode llm \
  --max-new-tokens 128 \
  --skip-existing \
  --continue-on-error
```

输出示例：

```text
/root/Users/dataset/hl_memory/normalize_shards/
  summary.json
  shard_000_tasks.txt
  shard_000_command.sh
  shard_001_tasks.txt
  shard_001_command.sh
```

每台机器只运行一个 shard 的 command，例如：

```bash
bash /root/Users/dataset/hl_memory/normalize_shards/shard_000_command.sh
```

`--skip-existing` 在 divide 阶段会把已经完成的 task 排除；生成的 command 里也建议继续保留 `--skip-existing`，这样某台机器中断后可以重跑同一个 shard。切分逻辑按 `hl_annotations.jsonl` 行数做贪心均衡，不是简单按 task 数平均，所以大任务不会集中到同一台机器。

输出默认写在每个 task 目录：

```text
/root/Users/dataset/lerobot_home/subtask/<task_id>/hl_annotations_llm_normalized.jsonl
/root/Users/dataset/lerobot_home/subtask/<task_id>/hl_segments_llm_sidecar.json
```

默认 `--granularity task` 的流程是：

```text
all episode annotations in one task
  -> infer one canonical ordered segment list for the task
  -> LLM normalize each segment once
  -> LLM/code summarize completed segment prefixes once
  -> save hl_segments_llm_sidecar.json
  -> expand every episode row from the sidecar
```

这样每个 row 会得到：

```json
{
  "task_progress": "The pen holder has been placed on the table.",
  "current_objective": "grasp the glue stick with the right hand and place it into the pen holder",
  "subtask_progress": 0.42,
  "should_advance_objective": false,
  "active_hand": "right",
  "relevant_objects": ["glue stick", "pen holder"],
  "target_query": "glue stick",
  "goal_query": "pen holder",
  "notes": "left hand keeps the pen holder stable"
}
```

`subtask_progress` 和 `should_advance_objective` 由代码按每个 episode 的实际 segment 时间范围展开生成；LLM 不处理 episode。默认 `--memory-summary-mode llm` 只在 task sidecar 阶段为 completed segment prefixes 生成一次 `task_progress` 摘要；如果想让 `task_progress` 也完全由规则生成，可以用 `--memory-summary-mode code`。

如果只处理部分任务，`--only-task-id` 支持一次给多个 id，也支持重复：

```bash
PYTHONPATH=src python scripts/hl_memory/batch_normalize_hl_annotations_with_llm.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --only-task-id 20260323O058A 20260324H125A 20260324H127A \
  --model-path /root/Users/lixiaotong/Qwen3.5-27B \
  --skip-existing
```

单文件归一化仍然可以直接调用底层脚本：

```bash
PYTHONPATH=src python scripts/hl_memory/normalize_hl_annotations_with_llm.py \
  --input-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations.jsonl \
  --output-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations_llm_normalized.jsonl \
  --model-path /root/Users/lixiaotong/Qwen3.5-27B \
  --device-map auto \
  --granularity task \
  --memory-summary-mode llm \
  --resume
```

归一化输出会在原 row 上补充这些字段：

```json
{
  "task_progress": "The motor has been picked up.",
  "current_objective": "place the motor into the box",
  "subtask_progress": 0.42,
  "should_advance_objective": false,
  "active_hand": "right",
  "relevant_objects": ["motor", "box"],
  "notes": "keep the motor aligned with the box opening",
  "target_query": "motor",
  "goal_query": "box"
}
```

`task_progress` 只能描述历史和已完成状态，不能把未来完整流程提前写进去。`current_objective` 必须是当前帧对应的低层可执行目标，不能写成 `motor is picked up` 这种被动状态。

### 4. HL Sidecar To LL Objective Segments

LL VLA 训练不会直接读取 `hl_segments_llm_sidecar.json`。现有 LL dataloader 只认识 episode-level `subtask_segments_path`，再把当前帧的 `subtask` 拼进 prompt。因此需要把 HL normalized `current_objective` 映射回每个 episode 的 frame ranges，生成同目录下的 `ll_current_objective_segments.json`：

```bash
PYTHONPATH=src python scripts/hl_memory/batch_export_ll_objective_segments_from_hl_sidecars.py \
  --subtask-root /root/Users/dataset/lerobot_home/subtask \
  --workers 8 \
  --overwrite \
  --continue-on-error
```

单任务也可以直接跑：

```bash
PYTHONPATH=src python scripts/hl_memory/export_ll_objective_segments_from_hl_sidecar.py \
  --task-dir /root/Users/dataset/lerobot_home/subtask/20260323O058A \
  --overwrite
```

输出：

```text
/root/Users/dataset/lerobot_home/subtask/<task_id>/ll_current_objective_segments.json
```

文件格式仍然是 dataloader 已支持的 `episodes -> segments`，字段名仍叫 `subtask`，但语义是 HL normalized `current_objective`：

```json
{
  "field_semantics": {"subtask": "current_objective"},
  "episodes": {
    "0": {
      "segments": [
        {
          "start_frame": 0,
          "end_frame": 276,
          "subtask": "grasp the sponge with the right hand",
          "source_subtask": "pick up the sponge"
        }
      ]
    }
  }
}
```

LL config 里用这个文件替代 raw `subtask_segments.json`：

```python
DataConfig(
    prompt_from_task=True,
    subtask_segments_path="ll_current_objective_segments.json",
)
```

这样训练时 LL 看到的 prompt 会变成：

```text
Overall instruction: <task prompt>
Current subtask: <HL current_objective>
```

这和 rollout 时 HL server 低频输出 `current_objective`、LL server 高频消费当前 objective 的接口保持一致。

### 5. HL Annotation JSONL To HL Dataset

批量导出 train/val 时，推荐显式读取 LLM-normalized annotations：

```bash
PYTHONPATH=src python scripts/hl_memory/batch_export_hl_memory_dataset_from_subtasks.py \
  --source-config-name sponge_visual_guided_qwen3_5_2b_400m_touch \
  --subtask-root /root/Users/dataset/lerobot_home/subtask \
  --output-root /root/Users/dataset/hl_memory/subtask \
  --repo-prefix subtask/ \
  --annotations-name hl_annotations_llm_normalized.jsonl \
  --visual-mode raw \
  --workers 4 \
  --overwrite \
  --continue-on-error \
  -- --recent-frames-length 8 --training-fps 20 --frame-subsample 5 --frame-height 224 --frame-width 456 --memory-length 8 --merge-distance 5
```

不要在这一步同时依赖 `--auto-export-annotations` 生成 raw annotations 再导出 train/val；正确顺序是先生成 `hl_annotations.jsonl`，再 normalize 成 `hl_annotations_llm_normalized.jsonl`，最后从 normalized 文件导出 samples。

如果之前已经用旧 annotations 导出过 `samples.jsonl`，新增的 `task_progress` 修正、`subtask_progress`、`should_advance_objective`、`active_hand` 和 `step_prior` 不会自动写进旧 samples；需要从 normalized annotations 重新跑这一节的 HL dataset export。Raw -> LeRobot 不需要重跑。

```bash
python scripts/hl_memory/export_hl_memory_dataset.py \
  --source-config-name sponge_visual_guided_qwen3_5_2b_400m_touch \
  --annotations-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations_llm_normalized.jsonl \
  --output-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported \
  --visual-mode raw \
  --recent-frames-length 8 \
  --training-fps 20 \
  --frame-subsample 5 \
  --frame-height 224 \
  --frame-width 456 \
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

每条 sample 的核心字段：

```json
{
  "sample_id": "episode_000000_step_000003",
  "episode_index": 0,
  "step_index": 3,
  "frame_index": 276,
  "instruction": "put the motor into the box",
  "language_memory": "Task progress: ...\nCurrent objective: ...\nRelevant objects: ...\nNotes: ...",
  "updated_language_memory": "Task progress: ...\nCurrent objective: ...\nRelevant objects: ...\nNotes: ...",
  "step_prior": ["approach motor", "grasp motor", "place motor into box"],
  "task_progress": "The motor has been picked up.",
  "current_objective": "place the motor into the box",
  "subtask_progress": 0.42,
  "should_advance_objective": false,
  "active_hand": "right",
  "relevant_objects": ["motor", "box"],
  "notes": "none",
  "current_subtask": "place the motor into the box",
  "phase": "place the motor into the box",
  "target_query": "motor",
  "goal_query": "box",
  "keyframe_candidate_positions": [1, 8],
  "recent_frame_paths": ["frames/frame_....png"],
  "memory_frame_paths": ["frames/frame_....png"]
}
```

训练 target JSON 来自 `task_progress/current_objective/subtask_progress/should_advance_objective/active_hand/relevant_objects/notes/...`。`step_prior` 只进入 prompt 作为 nominal ordered plan，不作为 target JSON 监督。`updated_language_memory/current_subtask` 是兼容旧代码的派生字段，不应该作为新协议的唯一真值。

推荐按 episode 做 held-out split，而不是直接用同一个 `exported/` 同时训练和评估。下面这条命令会在一次运行里计算一次 split，并同时生成互斥的 train / val episode，避免两次运行时误填不同 ratio/seed：

```bash
python scripts/hl_memory/export_hl_memory_dataset.py \
  --source-config-name sponge_visual_guided_qwen3_5_2b_400m_touch \
  --annotations-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations_llm_normalized.jsonl \
  --output-train-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_train \
  --output-val-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_val \
  --val-ratio 0.1 \
  --split-seed 42 \
  --visual-mode raw \
  --overwrite
```

也可以显式指定 episode：`--episode-indices 0,3,7-12` 或排除 episode：`--exclude-episode-indices 100-120`。`metadata.json` 会记录 selected episode 列表，方便确认 train/val 没有重叠。旧的单目录模式仍然可用：`--output-dir ... --episode-split train|val|all`。

如果报 `Episode ... was not found in dataset`，说明 `annotations.jsonl` 的 episode index 和当前 LeRobot dataset 不匹配；优先用上一步的 `--repo-id` / `--lerobot-dir` 从 `subtask_segments.json` 重新生成 annotations。只有确认要导出交集时才加 `--missing-episode-policy skip`。

### 6. Train

多任务 / batch pipeline 推荐直接用 `train_hl_memory_multitask.py`。它可以从 batch export 的 root 下自动发现多个 task 的 `train/` 目录，把所有 task 的 samples 合并成一个训练池：

```bash
export PYTHONPATH=/lumos-vePFS/suzhou/Users/lixiaotong/openpi/src
export WANDB_API_KEY="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"
export WANDB_MODE=online

torchrun --standalone --nproc_per_node 8 scripts/hl_memory/train_hl_memory_multitask.py \
  --dataset-root /root/Users/dataset/hl_memory/subtask \
  --dataset-glob '*/train' \
  --val-dataset-root /root/Users/dataset/hl_memory/subtask \
  --val-dataset-glob '*/val' \
  --output-dir /root/Users/checkpoints/hl_memory/subtask_multitask_qwen35_lora \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-4B \
  --precision bfloat16 \
  --training-fps 20 \
  --frame-subsample 5 \
  --frame-height 224 \
  --frame-width 456 \
  --learning-rate 5e-6 \
  --lora-enabled \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --language-memory-dropout 0.3 \
  --step-prior-dropout 0.3 \
  --distributed-strategy fsdp \
  --fsdp-min-num-params 20000000 \
  --batch-size 1 \
  --grad-accum-steps 32 \
  --frame-cache-size 4096 \
  --num-train-steps 2000 \
  --save-interval 200 \
  --log-interval 10 \
  --val-interval 100 \
  --val-batches 10 \
  --wandb-enabled \
  --wandb-project openpi-hl-memory \
  --wandb-run-name subtask-multitask-qwen35-4b-lora
```

数据选择方式：

- `--dataset-root /path/to/root --dataset-glob '*/train'`：推荐 batch 训练入口，自动加载每个 task 的 train split。
- `--dataset-dir /path/to/task/train`：只训练单个导出目录，主要用于 smoke/debug。
- `--dataset-dirs ...` 或 `--dataset-dirs-json /path/to/list.json`：显式指定多个导出目录，适合排除坏任务或只跑子集。

`train_hl_memory_multitask.py` 支持 `torchrun` 多卡 DDP/FSDP。每张卡启动一个 rank，每个 rank 独立采样不同 HL samples、各自 forward/backward，然后在 optimizer step 前同步梯度。`--batch-size` 是每张卡的 micro batch；有效全局 batch 约等于 `batch_size * grad_accum_steps * nproc_per_node`。

当前脚本会把每个 rank 内的 `--batch-size` 个 samples 合成一个 VLM batch，一次 processor encode 和一次 model forward/backward。因此它同时支持“多卡 batch 并行”和“单卡 batch 内并行”。把 `--batch-size` 调大通常会提高 GPU 吞吐，但也会增加 peak activation memory；如果 OOM，优先降低 `--batch-size`，再用 `--grad-accum-steps` 保持全局 batch。

`--distributed-strategy ddp` 是默认值，每张卡各自保留一份完整模型，速度直接但显存压力最大。4B/27B 或 batch 内并行 OOM 时用 `--distributed-strategy fsdp`，FSDP 会 shard 参数、梯度和 optimizer states，显存更低，但通信和 checkpoint 保存更慢。FSDP 的参数 shard 数由 `torchrun --nproc_per_node` / world size 决定，不由 `--fsdp-min-num-params` 决定；`--fsdp-min-num-params` 只是 auto-wrap 阈值，控制多大的子模块会被单独包成一个 FSDP unit。阈值越小，wrap 越细，峰值显存通常越低但通信开销越高；阈值越大，wrap 越粗，速度可能更好但峰值显存更高。LoRA + FSDP 需要 `use_orig_params=True`，脚本内部已启用。FSDP flatten 同一个 unit 时要求浮点参数 dtype 一致；脚本会在 FSDP wrapping 前把 LoRA 等浮点参数对齐到 `--precision` 指定的 dtype，避免 bf16 base model + fp32 LoRA 混合报错。FSDP 路径不会在 grad accumulation 期间使用 `no_sync()`，因为 FSDP `no_sync()` 会保留未 shard 梯度、增加峰值显存；如果仍然 OOM，优先用 `--batch-size 1` 并增大 `--grad-accum-steps`，再把 `--fsdp-min-num-params` 降到 `50000000` 或 `20000000`，最后再考虑 `--fsdp-cpu-offload`，CPU offload 会明显变慢。

开启 `--val-interval` 后，脚本会每隔 N 个 optimizer steps 从 val split 随机抽 `--val-batches` 个 batch 做 forward-only loss，并在 rank0 记录 `val/loss`、`val/time_s`、`val/batches_per_rank` 和 `val/effective_samples` 到 wandb。默认不启用 validation；多任务 batch pipeline 推荐显式设置 `--val-dataset-root ... --val-dataset-glob '*/val'`。

Loss 计算逻辑：输入序列是 `prompt + target JSON`，labels 会把 prompt tokens 和 padding tokens 置为 `-100`，只监督 target JSON token。Hugging Face causal LM loss 是所有未 mask target tokens 的平均 cross entropy。`grad_accum_steps` 内每个 micro batch 的 loss 会除以 accum steps 再 backward；日志里的 `train/loss` 也按 accum steps 做平均，因此不同 accum 配置下数值可比。DDP 下每个 rank 算本地 loss，日志再跨 rank 求平均。

多任务训练建议优先用 LoRA + `--language-memory-dropout` + `--step-prior-dropout`。原因是不同 task 的语言目标差异大，full finetune 更容易记住任务模板或破坏原 VLM 的通用视觉能力；step prior dropout 则避免模型只背 plan，不看 recent clip。`--num-train-steps` 是 optimizer steps，不是 epoch；实际见过的样本数约等于 `num_train_steps * global_batch_size`，需要按总 sample 数和任务数量估算。

单任务训练仍然保留，适合先确认某一个 dataset export 没有格式问题：

```bash
python scripts/hl_memory/train_hl_memory.py \
  --dataset-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_train \
  --output-dir /root/Users/checkpoints/hl_memory/sponge_visual_guided_qwen35 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-2B \
  --precision float16 \
  --training-fps 20 \
  --frame-subsample 5 \
  --frame-height 224 \
  --frame-width 456 \
  --device cuda \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-train-steps 1000 \
  --save-interval 200
```

多卡数据并行训练用 `torchrun`。每个 rank 独立采样 HL sample，同步梯度；只有 rank0 打印 tqdm/loss 和保存 checkpoint：

```bash
export PYTHONPATH=/lumos-vePFS/suzhou/Users/lixiaotong/openpi/src
export WANDB_API_KEY="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"
export WANDB_MODE=online

torchrun --standalone --nproc_per_node 8 scripts/hl_memory/train_hl_memory.py \
  --dataset-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_train \
  --output-dir /root/Users/checkpoints/hl_memory/sponge_visual_guided_qwen35 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-2B \
  --precision bfloat16 \
  --training-fps 20 \
  --frame-subsample 5 \
  --frame-height 224 \
  --frame-width 456 \
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

如果 full finetune 出现过度依赖 language memory 或通用视觉能力退化，优先尝试 LoRA + language memory dropout：

```bash
torchrun --standalone --nproc_per_node 8 scripts/hl_memory/train_hl_memory.py \
  --dataset-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_train \
  --output-dir /root/Users/checkpoints/hl_memory/sponge_visual_guided_qwen35_lora \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-2B \
  --precision bfloat16 \
  --learning-rate 5e-6 \
  --lora-enabled \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --language-memory-dropout 0.3 \
  --language-memory-dropout-value "No progress has been recorded yet." \
  --step-prior-dropout 0.3 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-train-steps 2000 \
  --save-interval 200
```

LoRA 需要安装 `peft`。默认 LoRA target modules 是 Qwen 常见 attention/MLP 线性层：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。`--language-memory-dropout` 会在训练时随机把输入 `language_memory` 替换成指定文本，`--step-prior-dropout` 会随机移除 `step_prior`，逼模型更多依赖视觉和 recent clip。

当前脚本默认值已经偏保守：`--learning-rate 5e-6`、`--lora-r 8`、`--lora-alpha 16`、`--language-memory-dropout 0.3`、`--step-prior-dropout 0.3`。如果模型第一步就背出完整流程，优先检查 GT 是否泄漏未来，再降低 learning rate / 训练步数，而不是增大 LoRA rank。

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
python scripts/hl_memory/train_hl_memory.py --config-yaml src/openpi/hl_memory/train_hl_memory.yaml
```

### 7. Eval

```bash
python scripts/hl_memory/eval_hl_memory_rollout.py \
  --dataset-dir /root/Users/dataset/hl_memory/subtask/20260116W001/val \
  --model-path /root/Users/checkpoints/hl_memory/subtask_multitask_qwen35_lora/checkpoint-step-002000 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --device cuda \
  --frame-cache-size 512 \
  --known-prior-eval \
  --output-json /root/Users/eval/hl_memory/subtask_multitask_qwen35_lora_step2000/20260116W001/eval_metrics.json \
  --prediction-jsonl /root/Users/eval/hl_memory/subtask_multitask_qwen35_lora_step2000/20260116W001/full_predictions.jsonl
```

评估默认跑四种 ablation：`no_memory`、`language_memory_only`、`keyframe_memory_only`、`full`。核心指标包括 `objective_exact_match` / `objective_normalized_match`、progress/advance/active-hand accuracy、target/goal accuracy、keyframe precision/recall、memory similarity/drift。

当前 HL V1 主协议以 `current_objective`、`subtask_progress`、`should_advance_objective` 为主。Eval 的主指标是 `objective_*`、`subtask_progress_mae`、`subtask_progress_accuracy_0_1`、`should_advance_accuracy`、`active_hand_accuracy`、`target_query_accuracy`、`goal_query_accuracy`。`legacy_subtask_*`、`legacy_phase_accuracy`、`legacy_event_accuracy` 只用于排查旧字段兼容问题，不建议作为是否训好的主判断。

快速 smoke eval 可以先只跑少量未见 episode 或单个 ablation：

```bash
python scripts/hl_memory/eval_hl_memory_rollout.py \
  --dataset-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_val \
  --model-path /root/Users/checkpoints/hl_memory/sponge_visual_guided_qwen35/checkpoint-step-001000 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --device cuda \
  --max-episodes 10 \
  --eval-modes full \
  --output-json /root/Users/dataset/hl_memory/sponge_visual_guided/eval_metrics_smoke.json
```

多任务 batch eval 推荐对每个 task 的 `val/` split 单独跑一次，再汇总 per-task 和 overall metrics：

```bash
PYTHONPATH=src python scripts/hl_memory/batch_eval_hl_memory_rollout.py \
  --dataset-root /root/Users/dataset/hl_memory/subtask \
  --dataset-glob '*/val' \
  --output-root /root/Users/eval/hl_memory/subtask_multitask_qwen35_lora_step200 \
  --workers 8 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --continue-on-error \
  -- \
  --local-vlm-ckpt-path /root/Users/checkpoints/hl_memory/subtask_multitask_qwen35_lora/checkpoint-step-000200 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --precision bfloat16 \
  --device cuda \
  --eval-batch-size 4 \
  --apply-rollout-memory-rule \
  --known-prior-eval \
  --frame-cache-size 512
```

输出包括每个 task 的 `eval_metrics.json` / `eval.log`，以及 `batch_hl_memory_eval_summary.json`。默认 `--workers 1` 是因为每个 eval 子进程都会加载一份 VLM；同一张 GPU 上并行通常只会更慢或 OOM。

`batch_eval_hl_memory_rollout.py` 是 task 级并行，不是 DDP。`--workers` 控制同时跑多少个 eval 子进程；`--gpu-ids 0,1,...` 会把 task 按 round-robin 分配到不同物理 GPU，并给每个子进程设置 `CUDA_VISIBLE_DEVICES=<gpu_id>`。如果不传 `--gpu-ids`，所有子进程继承当前 shell 的 CUDA 环境，通常只会用默认可见 GPU。

单个 eval 子进程内部可用 `--eval-batch-size N` 做 sample batch 并行。由于 `full` / `language_memory_only` rollout 有 episode 内状态依赖，batching 只会跨 episode frontier 合批：每个 episode 当前待评估 sample 进入同一个 VLM batch，预测返回后再推进各自 memory 状态。因此 `--eval-batch-size` 不能把单个 episode 的未来 samples 乱序并行；如果只评估 1 个 episode，它基本不会提速。显存不够时先降 `--eval-batch-size`，再降 `--workers`。

如果要让 offline eval 更接近 `run_hl_memory_zero_shot.py --known-prior-mode` 的真实 rollout，打开 `--apply-rollout-memory-rule --known-prior-eval`。前者复用 rollout 端的 language memory 清理/压缩规则；后者用 sample 里的 `step_prior` 做 known-prior 状态机，按 `subtask_progress/should_advance_objective` 推进当前 step。若只想评估纯模型原始输出，不要打开这两个开关。

`--max-samples` 也可用于快速调试，但它可能截断 episode 中间序列；正式 rollout metrics 更推荐用 `--max-episodes`。

Eval 会打印阶段日志并显示每种 ablation 的 tqdm 进度条：

- `[stage] loading processor` / `[stage] loading model weights`：正在读取 processor 和 checkpoint。
- `[stage] moving model to cuda`：正在把模型搬到 GPU；如果长时间停在这里，优先检查 GPU 显存、CUDA 初始化和 `--parallel-mode device_map`。
- `HL eval <mode>`：已经进入逐 sample rollout；如果停在某个 sample，通常是该 sample 的视频 processor encode 或 `generate()` 慢。
- `--frame-cache-size` 控制 resized frame LRU cache，默认 `512`；如果 eval 反复读取同一批帧且 CPU 内存足够，可以适当增大。

### 8. Video Inference

单视角：

```bash
python scripts/hl_memory/run_hl_memory_zero_shot.py \
  --video-path /path/to/front.mp4 \
  --instruction "Put the target object into the target slot" \
  --language-memory "Task started." \
  --recent-end-sec 42 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --local-vlm-ckpt-path /path/to/Qwen3.5-2B-or-hl-checkpoint \
  --precision float16 \
  --device cuda \
  --debug-dir /tmp/hl_debug
```

双视角 rollout：

```bash
python scripts/hl_memory/run_hl_memory_zero_shot.py \                             
  --left-video-path /root/Users/segmentation_data_dtw/20260116W001/20260414/task_20260116W001_Light_bulb_packing/background/multi_session_20260414/session_095153/left_hand_250801DR48FB25002358/RGB_Images/video.mp4 \
  --right-video-path /root/Users/segmentation_data_dtw/20260116W001/20260414/task_20260116W001_Light_bulb_packing/background/multi_session_20260414/session_095153/right_hand_250801DR48FP25002672/RGB_Images/video.mp4 \
  --instruction "Pack the light bulb into the box" \
  --language-memory "" \
  --rollout-interval-sec 1 \
  --rollout-start-sec 0 \
  --rollout-end-sec 40 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --local-vlm-ckpt-path /root/Users/checkpoints/hl_memory/subtask_multitask_qwen35_lora/checkpoint-step-000200 \
  --precision float16 \
  --device cuda \
  --debug-dir /tmp/hl_rollout_debug_new \
  --embedding-debug-dir /tmp/hl_rollout_embedding_debug \
  --output-json /tmp/hl_rollout_debug_new/summary.json
```

known-prior rollout：

```bash
python scripts/hl_memory/run_hl_memory_zero_shot.py \
  --left-video-path /path/to/left.mp4 \
  --right-video-path /path/to/right.mp4 \
  --instruction "Pack the light bulb into the box" \
  --task-config-path /path/to/subtask.json \
  --known-prior-mode \
  --known-prior-advance-threshold 0.65 \
  --known-prior-match-threshold 0.62 \
  --known-prior-max-advance-steps 3 \
  --rollout-interval-sec 1 \
  --rollout-start-sec 0 \
  --rollout-end-sec 40 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --local-vlm-ckpt-path /root/Users/checkpoints/hl_memory/subtask_multitask_qwen35_lora/checkpoint-step-001000 \
  --precision bfloat16 \
  --device cuda \
  --debug-dir /tmp/hl_rollout_known_prior \
  --output-json /tmp/hl_rollout_known_prior/summary.json
```

Input rules：

- 单视角 `--video-path` 映射为 `front`。
- 双视角 `--left-video-path` / `--right-video-path` 映射为 `robot_0` / `robot_1`，同一时间抽帧后横向拼接。
- HL frame 固定使用双槽画布，默认 `456x224`：每个视角槽位 `224x224`，中间 gap `8`。单视角放左槽，右槽为黑色 padding；双视角左右各一槽。
- memory clip 和 recent clip 都按时间从旧到新排列。
- recent clip 最后一张有效帧定义当前状态，`current_objective` 必须描述最后有效帧对应的当前低层目标。
- 不传 `--recent-step-sec` 时，rollout 会按训练导出节奏自动取 `--frame-subsample / --training-fps` 秒；默认 `5 / 20 = 0.25s`，对应 LeRobot 20Hz 上每 5 帧取一帧。原始输入视频即使是 60Hz，也应按秒抽 `0.25s` 间隔来匹配训练时域。
- Qwen video metadata 的 `fps` 同样按 `--training-fps / --frame-subsample` 推导，默认 `4.0Hz`；不要再把 HL clip 当作 1Hz 视频。
- rollout 会把上一轮四字段 language memory 和 keyframe candidates 带到下一轮，并在 `debug_dir/rollout_step_XXX/` 保存实际输入帧。
- `known-prior` 会把 `task-config-path` 里的 `steps` 或 `segments[*].subtask` 当成固定 prior plan。模型判断当前 prior step 的进度/是否完成，也可以报告当前画面最接近的后续 prior step；rollout 侧维护单调 step pointer，并把最终 `current_objective` 重写成 pointer 对应的 prior step。

常用参数说明：

| 参数 | 作用 |
| --- | --- |
| `--instruction` | 任务指令，会进入 HL VLM prompt；如果配合 `--task-config-path`，还会拼接 nominal primitive plan。 |
| `--video-path` | 单视角输入视频，内部视角名为 `front`。 |
| `--left-video-path` / `--right-video-path` | 双视角输入视频，内部视角名为 `robot_0` / `robot_1`，同一时间抽帧后横向拼接成一张 HL frame。 |
| `--task-config-path` | JSON task plan，可提供 `task_description` 和 `steps`，作为分段先验；视觉证据优先级仍高于 plan。 |
| `--known-prior-mode` | 将 `task-config-path` 的 step/subtask 序列作为显式状态机，不让模型自由决定下一步 objective；适合已知流程或先由 thinking 大模型离线拆解出 steps 的任务。 |
| `--known-prior-start-index` | known-prior 初始 step index，0-based，默认 `0`。 |
| `--known-prior-advance-threshold` | 如果模型没有显式输出 `should_advance_objective=true`，则用 `subtask_progress >= threshold` 推进到下一 prior step，默认 `0.65`；当前模型常输出粗粒度 `0.33/0.66`，用 `0.95` 很容易永远不切。 |
| `--known-prior-match-threshold` | 如果模型输出的 `current_objective/current_subtask/phase` 与后续 prior step 的文本相似度超过该阈值，则直接推进到该 step，默认 `0.62`。 |
| `--known-prior-max-advance-steps` | 每轮 rollout 最多向后匹配/跳过多少个 prior steps，默认 `3`；如果 `--rollout-interval-sec` 大、subtask 很短，需要调大，否则 pointer 会追不上。 |
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
| `--recent-step-sec` | recent clip 抽帧间隔；不传时由 `--frame-subsample / --training-fps` 推导，默认 `0.25s`。 |
| `--training-fps` | HL 训练导出所基于的 LeRobot fps，默认 `20.0`。 |
| `--frame-subsample` | HL 训练导出 recent clip 的帧间隔，默认 `5`。 |
| `--recent-seconds` | 手动指定 recent clip 秒数列表，例如 `10,12,14`；不能和 rollout interval 模式一起用。 |
| `--memory-seconds` | 手动指定历史 memory keyframe 秒数列表，例如 `2,8,15`。 |
| `--auto-memory` | 单次预测时自动从 recent 前面的时间段取 memory frames；rollout 模式会关闭它并使用上一步 keyframes。 |
| `--recent-frames-length` | recent clip 最大帧数，默认 `8`。 |
| `--memory-length` | memory clip 最大帧数，默认 `8`。 |
| `--frame-height` / `--frame-width` | 输入 VLM 前的固定双槽画布尺寸，默认 `456x224`（width x height），即两个 `224x224` 视角槽加 `8px` gap。 |
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
| `--embedding-debug-dir` | 保存 prompt token、last hidden state、可用 attention heatmap、text/image top attention、image-token latent PCA；用于分析模型是否看图/看 memory。 |
| `--embedding-debug-max-tokens` | attention heatmap 最多可视化多少个 token，默认 `160`，避免长 prompt 图片过大。 |
| `--debug-video-fps` | 用 debug panels 合成 rollout debug 视频的 FPS。 |
| `--max-new-tokens` | 非 thinking 模式的最大生成 token 数。 |
| `--enable-thinking` | 打开 Qwen thinking；默认关闭，建议先关闭以保证 JSON 输出稳定。 |
| `--parallel-mode` | 大模型加载方式；单卡默认 `none`，27B OOM 时可试 `device_map`。 |
| `--device-map` / `--tensor-parallel-plan` | 配合 `--parallel-mode device_map|tensor_parallel` 使用。 |

推理输出 JSON 使用新 schema。`target_bbox_xyxy` 是可选 debug 字段，模型看到目标时可以输出当前帧像素坐标 `[x1, y1, x2, y2]`。开启 `--debug-dir` 时 debug panel 会把 bbox 画到 current frame 上；训练数据不需要提供 bbox GT。

## Data Formats

### HL Annotation JSONL

一行一个 JSON，是 `scripts/hl_memory/export_hl_memory_dataset.py` 的输入。

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
  "current_objective": "place the motor into the box",
  "task_progress": "The motor has been picked up.",
  "subtask_progress": 0.42,
  "should_advance_objective": false,
  "active_hand": "right",
  "relevant_objects": ["motor", "box"],
  "notes": "none",
  "instruction": "put the motor into the box",
  "phase": "place the motor into the box",
  "target_query": "motor",
  "goal_query": "box",
  "event_type": "progress",
  "event_text": "Continuing placing the motor into the box."
}
```

`event_type` 支持 `none`、`subtask_boundary`、`success`、`failure`、`progress`、`discovery`。它会影响 rule-based fallback 的 `task_progress` / rendered language memory；如果 annotation row 已经提供 LLM-normalized `task_progress`，则优先使用该字段。

如果同时存在 `current_objective` 和 `current_subtask`，新训练 target 使用 `current_objective`。`current_subtask` 只作为 legacy alias 保留。

HL prediction 还支持可选 SAM grounding 字段：

```json
{
  "sam_text_prompt": "motor",
  "sam_point_xy": [120, 85]
}
```

`sam_point_xy` 是 last valid recent frame 上的像素点，用于给 SAM point prompt；`sam_text_prompt` 是给 SAM/Grounding-SAM 的短文本 prompt。没有人工 point label 时，训练 target 通常只监督 `sam_text_prompt`，point prompt 主要靠 VLM zero-shot/推理能力生成。

### Language Memory

language memory 是给下游低层 VLM/action policy 读的 compact context，不是 debug log。新协议固定规整成四行：

```text
Task progress: <one short sentence>
Current objective: <one short executable objective>
Relevant objects: <object/location phrases, or none>
Notes: <one short caution/spatial fact, or none>
```

不要写逐帧时间戳、frame id、长日志或 raw model output。调试看 `raw_model_output`、`model_prediction`、`rollout_pretty.json`。

四行文本由结构化字段确定性渲染。不要让模型直接自由发挥一整段 `updated_language_memory`；训练目标和 rollout parser 都优先读结构化 JSON 字段。

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
python scripts/hl_memory/export_hl_memory_crosstask.py \
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
scripts/hl_memory/hl_memory_crosstask_smoke.sh
```

可先检查本地视频覆盖率：

```bash
python scripts/hl_memory/check_crosstask_video_coverage.py \
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
