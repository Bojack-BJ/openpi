# HL Dataset Export

## HL Annotation JSONL To HL Dataset

批量导出 train/val 时，推荐显式读取 LLM-normalized annotations：

```bash
PYTHONPATH=src python scripts/hl_memory/batch_export_hl_memory_dataset_from_subtasks.py \
  --subtask-root /root/Users/dataset/lerobot_home/subtask \
  --output-root /root/Users/dataset/hl_memory/subtask \
  --repo-prefix subtask/ \
  --annotations-name hl_annotations_llm_normalized.jsonl \
  --visual-mode raw \
  --image-columns auto \
  --workers 4 \
  --overwrite \
  --continue-on-error \
  -- --recent-frames-length 8 --training-fps 20 --frame-subsample 5 --recent-sample-hz 2.0 --frame-height 224 --frame-width 456 --subtask-progress-quantum 0.05 --memory-length 8 --merge-distance 5
```

Subtask 数据 schema 基本统一时，batch export 推荐不传 `--source-config-name`，而是用 `--repo-prefix subtask/` 和每个 task 目录名组成 `repo_id=subtask/<task_id>`。脚本会直接读 `<HF_LEROBOT_HOME>/<repo_id>`，默认只加载 episode/frame/task/prompt/subtask 和 RGB image columns；只有显式传 `--proprio-enabled` 时才额外读取 robot state。

`--source-config-name` 仍可作为兼容路径使用。它主要提供 base LeRobot repo、prompt_from_task、subtask sidecar、assets 和原始 `dataset_columns`。但如果这个 config 只覆盖单臂或双臂其中一种，可能会把 HL 图像列限制错；因此 batch 路径默认 `--image-columns auto`，会忽略 config 的 image column 限制，按每个 repo 实际存在的非 mask/overlay `observation.images.*` 自动选择单视角或双视角。

`--image-columns` 可选值：

- `auto`：默认，读取所有非 mask/overlay 的 RGB image columns，适合混合单/双视角 subtask repos。
- `config`：严格使用 `--source-config-name` 里配置的 RGB camera columns，适合复现某个固定 config。
- `front,robot_0,robot_1` 或完整列名：显式指定视角列；短名会自动补成 `observation.images.<name>`。

`--frame-width 456 --frame-height 224` 不是按原图自适应，而是固定 VLM 输入画布：两个 `224x224` view slot，中间 `8px` gap。单视角放左槽、右槽黑 padding；双视角左右各一槽。固定尺寸是为了让 train/eval/rollout 的视觉分布一致，不让单/双视角或不同原图尺寸改变 token layout。

如果要训练 proprio/state soft tokens，导出时加：

```bash
--proprio-enabled \
--proprio-state-columns auto \
--proprio-state-dim 14
```

`auto` 会优先读取 `observation.state`，否则读取按列名排序的 `observation.state.*`。导出后的 `samples.jsonl` 会包含 `recent_robot_states`、`recent_robot_state_masks` 和 `robot_state_dim_names`；state 统一为 14 维 fastumi/DRPY 协议，双臂 14 维全有效，单臂默认前 7 维有效、后 7 维为 0 且 mask 为 0。脚本会在 train split 上计算 `proprio_norm_stats.json`，同一次 train/val 导出时 val 复用 train stats；samples 内保存的是 normalized state。

不要在这一步同时依赖 `--auto-export-annotations` 生成 raw annotations 再导出 train/val；正确顺序是先生成 `hl_annotations.jsonl`，再 normalize 成 `hl_annotations_llm_normalized.jsonl`，最后从 normalized 文件导出 samples。

如果之前已经用旧 annotations 导出过 `samples.jsonl`，新增的 `task_progress` 修正、`subtask_progress`、`should_advance_objective`、`active_hand` 和 `step_prior` 不会自动写进旧 samples；需要从 normalized annotations 重新跑这一节的 HL dataset export。Raw -> LeRobot 不需要重跑。

```bash
python scripts/hl_memory/export_hl_memory_dataset.py \
  --repo-id-override fastumi/sponge_visual_guided \
  --annotations-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations_llm_normalized.jsonl \
  --output-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported \
  --visual-mode raw \
  --image-columns auto \
  --recent-frames-length 8 \
  --training-fps 20 \
  --frame-subsample 5 \
  --recent-sample-hz 2.0 \
  --frame-height 224 \
  --frame-width 456 \
  --subtask-progress-quantum 0.05 \
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

`metadata.json` 只记录数据导出信息，例如 selected episodes、visual/image column 选择、frame layout 和 clip fps。它不记录 VLM backend/variant；训练和推理实际使用的 Qwen 版本由 train/eval/zero-shot 命令参数或 checkpoint metadata 决定。

`--visual-mode raw` 是默认值，也是推荐值。HL export 只读取必要的 parquet columns：episode/frame/task/subtask/prompt 和 RGB image columns；不会读取 state/action，也不会读取或传给 HL 任何 mask / overlay / 高亮目标。图像列由 `--image-columns` 控制；`--visual-mode config` 只影响是否沿用 config 的视觉模式变换，不会让 mask/overlay 进入 HL。

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
  "subtask_progress": 0.4,
  "should_advance_objective": false,
  "active_hand": "right",
  "horizon_frame_index": 286,
  "horizon_current_objective": "return the right hand to the observation region",
  "horizon_current_subtask": "return the right hand to the observation region",
  "horizon_phase": "return the right hand to the observation region",
  "keyframe_label": true,
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

训练 target JSON 默认来自 `task_progress/current_objective/subtask_progress/should_advance_objective/active_hand/relevant_objects/notes/...`。`step_prior` 只进入 prompt 作为 nominal ordered plan，不作为 target JSON 监督。`updated_language_memory/current_subtask` 是兼容旧代码的派生字段，不应该作为新协议的唯一真值。`horizon_*` 和 `keyframe_label` 是 MEMER-style 对照实验的可选字段；`hl_v1` 默认忽略 horizon objective，`subtask_keyframe` 只监督当前帧 objective + keyframes，`memer_objective` 同时监督当前帧 objective 和 short-horizon objective。

推荐按 episode 做 held-out split，而不是直接用同一个 `exported/` 同时训练和评估。下面这条命令会在一次运行里计算一次 split，并同时生成互斥的 train / val episode，避免两次运行时误填不同 ratio/seed：

```bash
python scripts/hl_memory/export_hl_memory_dataset.py \
  --repo-id-override fastumi/sponge_visual_guided \
  --annotations-jsonl /root/Users/dataset/hl_memory/sponge_visual_guided/annotations_llm_normalized.jsonl \
  --output-train-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_train \
  --output-val-dir /root/Users/dataset/hl_memory/sponge_visual_guided/exported_val \
  --val-ratio 0.1 \
  --split-seed 42 \
  --visual-mode raw \
  --image-columns auto \
  --training-fps 20 \
  --frame-subsample 5 \
  --recent-sample-hz 2.0 \
  --frame-height 224 \
  --frame-width 456 \
  --subtask-progress-quantum 0.05 \
  --overwrite
```

也可以显式指定 episode：`--episode-indices 0,3,7-12` 或排除 episode：`--exclude-episode-indices 100-120`。`metadata.json` 的 `selected_episode_indices` 会记录 selected episode 列表，方便确认 train/val 没有重叠。旧的单目录模式仍然可用：`--output-dir ... --episode-split train|val|all`。

如果报 `Episode ... was not found in dataset`，说明 `annotations.jsonl` 的 episode index 和当前 LeRobot dataset 不匹配；优先用上一步的 `--repo-id` / `--lerobot-dir` 从 `subtask_segments.json` 重新生成 annotations。只有确认要导出交集时才加 `--missing-episode-policy skip`。
