# HL Memory Data Formats

## HL Annotation JSONL

一行一个 JSON，是 `scripts/hl_memory/export_hl_memory_dataset.py` 的输入。

最小行：

```json
{"episode_index": 0, "frame_index": 276, "current_subtask": "place the motor into the box"}
```

MEMER-style 对照实验可选字段：

```json
{
  "episode_index": 0,
  "frame_index": 276,
  "current_subtask": "place the motor into the box",
  "horizon_frame_index": 286,
  "horizon_current_objective": "return the hand to the observation region",
  "horizon_current_subtask": "return the hand to the observation region",
  "horizon_phase": "return the hand to the observation region",
  "keyframe_label": true
}
```

这些字段向后兼容：旧 `hl_v1` 训练仍读当前帧字段；`subtask_keyframe` 读取当前帧 `current_objective`，`memer_objective` 同时读取当前帧 `current_objective` 和 `horizon_current_objective`。`keyframe_label` 只影响导出 samples 时的 `keyframe_candidate_positions` 计算，不改变最终 keyframe schema。

推荐行：

```json
{
  "episode_index": 0,
  "frame_index": 276,
  "current_subtask": "place the motor into the box",
  "current_objective": "place the motor into the box",
  "task_progress": "The motor has been picked up.",
  "subtask_progress": 0.4,
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

## Language Memory

language memory 是给下游低层 VLM/action policy 读的 compact context，不是 debug log。新协议固定规整成四行：

```text
Task progress: <one short sentence>
Current objective: <one short executable objective>
Relevant objects: <object/location phrases, or none>
Notes: <one short caution/spatial fact, or none>
```

不要写逐帧时间戳、frame id、长日志或 raw model output。调试看 `raw_model_output`、`model_prediction`、`rollout_pretty.json`。

四行文本由结构化字段确定性渲染。不要让模型直接自由发挥一整段 `updated_language_memory`；训练目标和 rollout parser 都优先读结构化 JSON 字段。

## LL Mask/Subtask Guidance

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

- Episode 错位：用 `--repo-id` / `--lerobot-dir` 从 `subtask_segments.json` 生成 annotations；确认 export 时的 `--repo-id-override` 或 `--source-config-name` 指向同一个 LeRobot `repo_id`。
- 缺依赖：HL export/train/eval 需要 `Pillow`、`torch`、`transformers`；`--config-yaml` 额外需要 `pyyaml`。
- Qwen3.5 JSON 不稳定：默认关闭 thinking；只有需要 reasoning trace 时加 `--enable-thinking --thinking-budget-tokens 128 --thinking-max-new-tokens 1024`。
- 多机 `.arrow` / `.incomplete`：`HF_LEROBOT_HOME` 是数据根目录，`HF_DATASETS_CACHE` 是 HuggingFace parquet/Arrow 派生缓存。多机不要共享同一个可写 cache，建议 `export HF_DATASETS_CACHE=/tmp/openpi_hf_datasets_cache_${HOSTNAME}`。
