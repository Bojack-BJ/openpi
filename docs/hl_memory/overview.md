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

实验 setting 对照表见 [`experiment_settings.md`](experiment_settings.md)。
