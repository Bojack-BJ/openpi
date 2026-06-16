# HL Memory Pipeline

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

## 1. Raw To LeRobot

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

## 2. Subtasks To Raw HL Annotation JSONL

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

MEMER-style 对照实验可以复用同一套 `samples.jsonl` schema，只在 raw annotation 里额外写入 horizon label 和显式 keyframe label，再由训练时 `--target-protocol` 决定监督目标。dense 采样命令示例：

```bash
PYTHONPATH=src python scripts/hl_memory/batch_export_hl_annotations_from_subtasks.py \
  --subtask-root /root/Users/dataset/lerobot_home/subtask \
  --workers 8 \
  --sampling-mode dense-stride \
  --dense-sample-stride-frames 2 \
  --prediction-horizon-steps 2 \
  --keyframe-label-mode segment_end \
  --overwrite \
  --continue-on-error
```

新增字段是可选的，旧数据仍可读：

```json
{
  "horizon_frame_index": 1234,
  "horizon_current_objective": "place the cup into the cabinet",
  "horizon_current_subtask": "place the cup into the cabinet",
  "horizon_phase": "place the cup into the cabinet",
  "keyframe_label": true
}
```

`--sampling-mode fraction-rules` 是当前默认路径：先生成 segment boundary/progress/success rows，再叠加 `--progress-sample-*`、short-segment 和 late-fraction 等规则。`--sampling-mode annotations` 只是兼容旧命令的 alias，行为等同于 `fraction-rules`。`--sampling-mode dense-stride` 会在每个 segment 内每 `--dense-sample-stride-frames` 帧生成一个 sample，并额外强制包含 `--keyframe-label-mode` 选中的显式 keyframe 帧。`--prediction-horizon-steps 2` 表示 horizon label 来自 `sample_frame + 2 * dense_sample_stride_frames` 所在 segment，超过 episode 末尾会 clip 到最后一帧；设为 `0` 就是 current-frame objective baseline。

`--keyframe-label-mode event_boundary` 保持旧 keyframe 规则。`--keyframe-label-mode segment_end` 会把每个 segment 的 `end_frame - 1` 标为显式 keyframe，是当前推荐 baseline：它比 `memer_rules` 稠密得多，避免模型在 `keyframe_candidate_positions` 上学成永远输出 `[]`。`--keyframe-label-mode memer_rules` 会用文本规则把 state-changing segments 的代表帧写成显式 keyframe label：默认 `place/release/put/insert/stack/open/close/press/handover/pick up stack` 选 segment 最后一帧，approach/return/retreat/move back/observation 等默认不选，适合作为更稀疏 keyframe 消融。需要改规则时传 `--keyframe-rule-path rules.json`，格式为：

```json
{
  "default_select": "none",
  "rules": [
    {"match": "place|release|put|insert|stack", "select": "last"},
    {"match": "pick", "select": "last"}
  ]
}
```

支持的 `select` 是 `none`、`first`、`last`、`both`。导出到 HL dataset 后仍然只写现有 `keyframe_candidate_positions`，不引入新的 keyframe schema。

推荐 ablation matrix：

| Sample row selection | Keyframe labels | Target protocol | 用途 |
| --- | --- | --- | --- |
| `fraction-rules` | `event_boundary` | `hl_v1` | 当前完整 HL baseline，使用 boundary/progress/success + fraction/dynamic rules |
| `dense-stride` | `event_boundary` | `hl_v1` | 单独验证 dense 采样是否解决节奏问题 |
| `dense-stride` | `segment_end` | `hl_v1` | 当前推荐 keyframe baseline，验证稠密 segment-end keyframe 是否帮助完整 HL |
| `dense-stride` | `segment_end` | `subtask_keyframe` | 当前帧 objective + keyframe baseline，最接近给 LL VLA 的目标 |
| `dense-stride`, horizon=0 | `segment_end` | `memer_objective` | current-frame objective + keyframe baseline |
| `dense-stride`, horizon=2 | `segment_end` | `memer_objective` | current + short-horizon objective baseline |
| `dense-stride` | `memer_rules` | `hl_v1` | 稀疏 keyframe 消融；如果 keyframe 非空比例过低，模型通常会输出空数组 |

## 3. LLM GT Normalization Before Dataset Export

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
  --llm-batch-size 4 \
  --subtask-progress-quantum 0.05 \
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
  --llm-batch-size 4 \
  --subtask-progress-quantum 0.05 \
  --max-new-tokens 128 \
  --skip-existing \
  --continue-on-error
```

`--parallel-workers 4` 会自动把 4 个进程绑定到前 4 张可见 GPU。如果需要手动分组，可以用 `--worker-gpu-groups "0;1;2;3"` 表示 4 个进程各用一张卡，或 `"0,1;2,3"` 表示 2 个进程、每个进程内部用 `device_map auto` 切 2 张卡。27B 如果单卡放不下，就不要用一卡一进程，改用多卡分组。`--memory-summary-mode code --max-new-tokens 128` 是最快组合；如果希望 history summary 也由 LLM 精修，用 `--memory-summary-mode llm`，但它仍然只在 task sidecar 阶段执行一次。

`--llm-batch-size` 控制同一个 worker 的 `model.generate` 一次吃多少个 text prompts。旧逻辑等价于 `--llm-batch-size 1`，GPU 利用率通常较低；可以先试 `4`，显存足够再试 `8`。这个 batch 是 LLM prompt batch，不改变 task sharding，也不改变最终 row 顺序。

`subtask_progress` 的 GT 原始值来自 `(frame_index - segment_start) / segment_length`。默认 `--subtask-progress-quantum 0.05` 会量化到 `0.00, 0.05, ..., 1.00`，避免训练目标里出现长小数。训练/推理 schema 保持数值 `[0, 1]`，不要写成 `40%` 字符串；百分号会增加格式解析和 token 对齐难度。

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
  --llm-batch-size 4 \
  --subtask-progress-quantum 0.05 \
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

如果已经生成了 normalized annotations，只想批量 round 现有 `subtask_progress`，可以直接原地改写：

```bash
PYTHONPATH=src python scripts/hl_memory/batch_round_hl_annotation_progress.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --input-name hl_annotations_llm_normalized.jsonl \
  --quantum 0.05 \
  --advance-threshold 0.85
```

这会同时更新 top-level `subtask_progress` 和 `llm_gt.subtask_progress`；传 `--advance-threshold` 时也会按 round 后的 progress 重新计算 `should_advance_objective`。

如果每个 task 的 `hl_segments_llm_sidecar.json` 已经生成过，但你改了 sample row selection，例如从 `fraction-rules` 换成 `dense-stride`，不需要重新调用 27B。先重新生成新的 raw `hl_annotations.jsonl`，再复用旧 sidecar 重新展开每个 sample row：

```bash
PYTHONPATH=src python scripts/hl_memory/batch_normalize_hl_annotations_with_llm.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --input-name hl_annotations.jsonl \
  --output-name hl_annotations_llm_normalized.jsonl \
  --reuse-sidecar \
  --overwrite \
  --continue-on-error
```

`--reuse-sidecar` 会跳过模型加载和 LLM generate，只读取每个 task 目录下已有的 `hl_segments_llm_sidecar.json`，把当前 input rows 重新展开成 normalized rows。这里推荐配合 `--overwrite`，因为 row 采样变了以后旧 output 的 done/resume 记录不再代表完整新数据。对于 MEMER-style horizon 数据，reuse-sidecar 展开时也会把 `horizon_current_objective/horizon_current_subtask/horizon_phase` 映射成 sidecar 里的 normalized objective，而不是保留 raw subtask 文本。

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
  "subtask_progress": 0.4,
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
  "subtask_progress": 0.4,
  "should_advance_objective": false,
  "active_hand": "right",
  "relevant_objects": ["motor", "box"],
  "notes": "keep the motor aligned with the box opening",
  "target_query": "motor",
  "goal_query": "box"
}
```

`task_progress` 只能描述历史和已完成状态，不能把未来完整流程提前写进去。`current_objective` 必须是当前帧对应的低层可执行目标，不能写成 `motor is picked up` 这种被动状态。

## 4. HL Sidecar To LL Objective Segments

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
