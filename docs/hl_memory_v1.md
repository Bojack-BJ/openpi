# HL Memory V1

V1 在仓库中新增一个独立的 `src/openpi/hl_memory/` 子系统，用于做高层 memory/subtask 训练和离线 rollout 评估。

当前范围：

- `language memory` 更新
- `current_subtask` 预测
- MemER 风格 `episodic keyframe memory`

当前主路径是 **Qwen-only 双 clip 输入**：

- 一个固定长度的 `recent observation clip`
- 一个固定长度的 `historical memory keyframe clip`
- 一段文本 `language memory`

当前不包含：

- segmentation
- target selection
- LL/action 接入
- online runtime wrapper

## 核心接口

- `HLMemoryConfig`
- `HLMemoryPrediction`
- `EpisodicKeyframeMemory`
- `ExportedHLMemorySample`

当前 HL 设计借鉴 MemER 的两个关键点：

- 高层策略输入 `selected keyframes + recent frames + task instruction`，输出当前 subtask 和 recent-window candidate keyframes。
- candidate keyframes 不应是普通显著帧，而是未来决策可能需要回忆的 task-relevant state，例如对象位置、计数事件、完成/失败动作或关键状态切换。
- rollout 侧对 candidate keyframes 做时间聚类，避免相邻重复帧不断堆积成 memory。

`HLMemoryPrediction` 当前输出固定为：

- `updated_language_memory`
- `current_subtask`
- `keyframe_candidate_positions`
- `phase`
- `target_query`
- `goal_query`

## 标注 JSONL

V1 导出脚本读取的标注格式为一行一个 JSON 对象，最小字段：

```json
{
  "episode_index": 0,
  "frame_index": 42,
  "current_subtask": "pick apple"
}
```

可选字段：

```json
{
  "instruction": "sort all fruits into the basket",
  "phase": "pick",
  "target_query": "apple",
  "goal_query": "basket",
  "event_type": "success",
  "event_text": "Completed pick apple."
}
```

`event_type` 支持：

- `none`
- `subtask_boundary`
- `success`
- `failure`
- `progress`
- `discovery`

## 导出

先基于现有训练 config 复用数据集和基础 transforms，再导出 HL 样本：

```bash
python scripts/export_hl_memory_dataset.py \
  --source-config-name <train_config_name> \
  --annotations-jsonl <annotations.jsonl> \
  --output-dir <out_dir>
```

导出物：

- `samples.jsonl`
- `metadata.json`
- `frames/frame_XXXXXXXX.png`

每个样本包含：

- instruction
- previous language memory
- updated language memory
- recent frame paths
- recent frame indices
- recent valid length
- historical keyframe paths
- historical keyframe indices
- historical memory valid length
- target `HLMemoryPrediction`

注意：

- 训练时不会再把这些帧拼成一张 panel 图。
- runtime 会把它们恢复成两个有序 clip：
  - `memory clip`
  - `recent clip`
- 对于通用机器人数据导出，如果单个 timestep 有多个 camera views，当前导出器会先把这些 views 合成一张 RGB observation frame，再把这些 frame 组织成 clip。
- `keyframe_candidate_positions` 永远只指向 `recent clip` 内的 1-indexed 位置。
- 如果某一步只有 1 张当前图，也允许通过 padding/fallback 扩成固定长度 clip。

## 训练

```bash
python scripts/train_hl_memory.py \
  --dataset-dir <out_dir> \
  --output-dir <ckpt_dir> \
  --vlm-backend qwen2_5_vl
```

也支持从 YAML 读取参数：

```bash
python scripts/train_hl_memory.py --config-yaml <train_hl_memory.yaml>
```

最小示例：

```yaml
dataset_dir: /path/to/hl_memory_train_local
output_dir: /path/to/hl_memory_ckpts
vlm_backend: qwen2_5_vl
local_vlm_ckpt_path: /path/to/local/qwen-vl-checkpoint
device: cuda
batch_size: 1
grad_accum_steps: 4
num_train_steps: 1000
```

如果你已经把 VLM 权重提前下载到本地路径，也可以显式传：

```bash
python scripts/train_hl_memory.py \
  --dataset-dir <out_dir> \
  --output-dir <ckpt_dir> \
  --local-vlm-ckpt-path </path/to/local/qwen-vl-checkpoint>
```

当前 V1 runtime backend：

- `qwen2_5_vl`
- `qwen3_5_vl`

当前约束：

- `qwen2_5_vl`：已实现 Torch/HF runtime adapter
- `qwen3_5_vl`：已实现 Torch/HF runtime adapter，支持 `vlm_variant=qwen3_5_2b` / `qwen3_5_4b`
- `paligemma`：明确不支持这条双 clip / 视频式 HL 路径，会直接报错

Qwen3.5 variant 选择：

```bash
python scripts/run_hl_memory_zero_shot.py \
  --video-path /path/to/video.mp4 \
  --instruction "Fold the shoebox" \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --device cuda

python scripts/run_hl_memory_zero_shot.py \
  --video-path /root/Users/lixiaotong/openpi/cross_task_datasets/video_block.mp4 \
  --instruction "Put the crescent and square blocks into their corresponding slots" \
  --language-memory "Task started." \
  --rollout-interval-sec 2 \
  --rollout-start-sec 0 \
  --rollout-end-sec 20 \
  --recent-step-sec 1 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-4B \
  --device cuda \
  --debug-dir cross_task_datasets/debug_rollout_block \
  --output-json cross_task_datasets/debug_rollout_block/summary.json
```

也可以用简写：

- `--vlm-variant 2b` -> `Qwen/Qwen3.5-2B`
- `--vlm-variant 4b` -> `Qwen/Qwen3.5-4B`

如果传了 `--local-vlm-ckpt-path`，会优先从本地路径加载；否则用 variant 对应的默认 HF model id。

Qwen3.5 依赖较新的 Transformers image-text-to-text 支持。如果你的环境报缺少
`AutoModelForImageTextToText`，按 Qwen3.5 官方模型卡建议安装更新的 `transformers`。

## 离线 rollout 评估

```bash
python scripts/eval_hl_memory_rollout.py \
  --dataset-dir <out_dir> \
  --model-path <ckpt_dir/checkpoint-step-XXXXXX> \
  --vlm-backend qwen2_5_vl
```

如果评估时你更想显式写本地 checkpoint 路径，也可以用：

```bash
python scripts/eval_hl_memory_rollout.py \
  --dataset-dir <out_dir> \
  --local-vlm-ckpt-path </path/to/local/hl-memory-checkpoint>

python scripts/run_hl_memory_zero_shot.py \
  --video-path /root/Users/lixiaotong/openpi/cross_task_datasets/video_block.mp4 \
  --instruction "Put the crescent and square blocks into their corresponding slots" \
  --language-memory "Task started." \
  --rollout-interval-sec 2 \
  --rollout-start-sec 0 \
  --rollout-end-sec 20 \
  --recent-step-sec 1 \
  --vlm-backend qwen2_5_vl \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --debug-dir cross_task_datasets/debug_rollout_block \
  --output-json cross_task_datasets/debug_rollout_block/summary.json
```

同样支持：

```bash
python scripts/eval_hl_memory_rollout.py --config-yaml <eval_hl_memory.yaml>
```

最小示例：

```yaml
dataset_dir: /path/to/hl_memory_val_local
local_vlm_ckpt_path: /path/to/local/hl-memory-checkpoint
vlm_backend: qwen2_5_vl
device: cuda
```

说明：

- `--config-yaml` 依赖当前环境安装 `pyyaml`
- CLI 显式传入的参数会覆盖 YAML 里的默认值

评估会自动跑四个 ablation：

- `no_memory`
- `language_memory_only`
- `keyframe_memory_only`
- `full`

主要指标：

- `subtask_exact_match`
- `subtask_normalized_match`
- `phase_accuracy`
- `target_query_accuracy`
- `goal_query_accuracy`
- `keyframe_precision`
- `keyframe_recall`
- `language_memory_similarity`
- `memory_drift`
- `event_accuracy`
- `episode_sequence_accuracy`

## Zero-shot 自定义视频推理

如果你想先不训练，直接看 HL zero-shot 在自己视频上的输出，可以用：

```bash
python scripts/run_hl_memory_zero_shot.py \
  --video-path /path/to/video.mp4 \
  --instruction "Put all the cans into the basket." \
  --language-memory "One can has already been placed into the basket." \
  --recent-end-sec 42 \
  --recent-step-sec 1 \
  --vlm-backend qwen2_5_vl \
  --vlm-hf-model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --debug-dir /tmp/hl_zero_shot_debug
```

视觉输入实际送入 Qwen 的方式：

- 脚本不会把原始 mp4 直接交给模型自由读取。
- 脚本先按秒抽帧，再 resize/pad 成两个定长 clip。
- 第一个 `video` 是 historical memory keyframes clip。
- 第二个 `video` 是 recent observation clip。
- 每个 clip 内位置按时间从旧到新排列。
- `keyframe_candidate_positions` 只能引用 recent clip 内的有效帧，且是 1-indexed。
- 如果 recent clip 只有 2 张有效帧，即使 padding 到 8 帧，合法位置也只有 `1,2`。

常用方式：

- 自动 recent clip：
  - 用 `--recent-end-sec` 和 `--recent-step-sec`
- 手工 recent clip：
  - `--recent-seconds "35,36,37,38,39,40,41,42"`
- 手工 memory keyframes：
  - `--memory-seconds "5,10,15,20"`
- 自动 memory keyframes：
  - 不传 `--memory-seconds`，默认从 recent clip 之前均匀抽样

输出是一个 JSON，包含：

- 选中的 `memory_seconds`
- 选中的 `recent_seconds`
- 原始模型输出 `raw_model_output`
- 模型原始解析结果 `model_prediction`
- `prediction.updated_language_memory`
- `prediction.current_subtask`
- `prediction.keyframe_candidate_positions`
- `language_memory_rule_applied`
- `keyframe_candidate_seconds`

如果传了 `--debug-dir`，脚本会把实际送入模型的 memory/recent 帧保存出来，方便你直接肉眼检查。

如果想对同一个视频按时间间隔持续 rollout，记录每次 memory 更新和关键帧，可以用：

```bash
python scripts/run_hl_memory_zero_shot.py \
  --video-path /path/to/video.mp4 \
  --instruction "Fold the shoebox" \
  --language-memory "Task started." \
  --rollout-interval-sec 2 \
  --rollout-start-sec 0 \
  --rollout-end-sec 20 \
  --recent-step-sec 1 \
  --vlm-backend qwen2_5_vl \
  --local-vlm-ckpt-path /path/to/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --debug-dir /tmp/hl_zero_shot_rollout \
  --output-json /tmp/hl_zero_shot_rollout/summary.json
```

rollout 模式会：

- 每隔 `--rollout-interval-sec` 秒跑一次 HL 推理。
- 用上一轮的 `prediction.updated_language_memory` 作为下一轮的 language memory。
- 如果模型把 memory 原样复制为 `Task started.` 或和上一轮完全相同，脚本会启用确定性 memory update rule：
  - 把 memory 改写成给下游 LL VLM 使用的四行上下文，而不是 debug log。
  - 不记录时间戳、frame id、每秒事件或 `target=...; goal=...` 这种工程字段。
  - 重复 subtask 不追加新行，只稳定更新当前 objective 和关键对象。
- `model_prediction` 保留模型原始 JSON 解析结果，`prediction` 是应用 memory rule 后用于下一轮 rollout 的结果。
- 把预测出的 recent keyframe candidates 映射成视频秒数，作为下一轮 memory clip 的候选 keyframes。
- keyframe memory 会过滤仍在 recent window 里可见的候选帧，只把已经退出 recent context 的帧作为 historical memory 输入。
- keyframe memory 会用 `--keyframe-merge-distance-sec` 做时间聚类，选每个 cluster 的中位代表帧，减少重复 memory。
- 在 `debug_dir/rollout_step_XXX/` 保存每轮实际送入模型的 memory/recent 帧。
- 在 `debug_dir/rollout_step_XXX/keyframe_candidates/` 保存每轮选中的关键帧。
- 在 `debug_dir/rollout.jsonl` 保存机器可读逐步记录。
- 在 `debug_dir/rollout_pretty.json` 额外保存易读版逐步记录。
- 也可以显式传 `--rollout-jsonl /path/to/rollout.jsonl` 和
  `--rollout-pretty-json /path/to/rollout_pretty.json`。

当前 `updated_language_memory` 约定固定为四行：

```text
Task progress: <one short sentence about completed/observed progress>
Current objective: <one short executable objective for the low-level policy>
Relevant objects: <comma-separated object/location phrases, or none>
Notes: <one short caution, spatial fact, or none>
```

这个字段是给 LL VLM/action policy 读的，不是给人调试的日志。调试细节应该看
`raw_model_output`、`model_prediction` 和 `rollout_pretty.json`。

## CrossTask 快速起步

先用 CrossTask 做 action-free HL 验证时，建议只用 **18 个 primary tasks**，也就是 `tasks_primary.txt` 加带边界标注的 `annotations/`。

CrossTask 官方 README 说明：

- 数据集包含 **83 个任务**
- 其中 **18 个 primary tasks** 有手工 temporal step boundary
- `videos_val.csv` 是论文里的验证划分
- 2022 年后视频不再依赖 YouTube，改为单独的视频打包下载

参考：

- CrossTask README: https://github.com/DmZhukov/CrossTask
- `crosstask_release.zip`: `https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip`
- 视频包: `https://www.rocq.inria.fr/cluster-willow/dzhukov/missing_videos.tar.gz`
- 可选 subtitles: `https://www.rocq.inria.fr/cluster-willow/dzhukov/crosstask-subtitles.tar.gz`

### CrossTask 导出脚本

新增脚本：

```bash
python scripts/export_hl_memory_crosstask.py --help
```

它会：

- 读取 `tasks_primary.txt`
- 读取 `videos.csv` / `videos_val.csv`
- 读取 `annotations/<task_id>_<video_id>.csv`
- 直接从视频里按秒抽帧
- 自动生成 `instruction / current_subtask / updated_language_memory / keyframe_candidate_positions`
- 导出标准 `samples.jsonl`

当前 CrossTask V1 约定：

- `instruction` = task title
- `current_subtask` = 当前 step 文本
- `phase` = 当前 step 文本
- `target_query` / `goal_query` 暂时留空
- `subtask_boundary` / `success` 事件由 segment 起止时间自动伪标注
- `recent clip` = 当前秒往前取固定窗口
- `memory clip` = 历史 keyframe 集合，按时间顺序排列

CrossTask 标注要点：

- `tasks_primary.txt` 定义的是 task 的 canonical ordered steps
- `annotations/<task>_<video>.csv` 只标这个具体视频里实际出现的 step segments
- 单个视频不保证覆盖 task 的全部步骤
- 因此 CrossTask 更适合先做：
  - `current_subtask`
  - `boundary / transition`
  - `keyframe candidate`
  - 模板化 `language memory`

### CrossTask 覆盖率检查与本地重切分

如果你本地只有一部分可用视频，先检查覆盖率：

```bash
python scripts/check_crosstask_video_coverage.py \
  --crosstask-release-dir "cross_task_datasets/crosstask_release" \
  --videos-root "cross_task_datasets/missing_videos" \
  --split train \
  --verify-decodable
```

这里会区分三类：

- `matched_local_records`: 文件名能匹配到
- `decodable_matched_records`: 真正能读帧的视频
- `corrupt_matched_records`: 名字匹配到但解码失败的视频

如果决定只在本地可用交集上做小规模实验，可以先重新切分：

```bash
python scripts/split_crosstask_matched_videos.py \
  --crosstask-release-dir "cross_task_datasets/crosstask_release" \
  --videos-root "cross_task_datasets/missing_videos" \
  --output-dir "cross_task_datasets/crosstask/matched_split" \
  --val-ratio 0.2 \
  --seed 0
```

默认会先做视频可解码检查，只基于 `decodable` 的本地视频做 split。

它会生成：

- `matched_split/train_records.csv`
- `matched_split/val_records.csv`
- `matched_split/split_summary.json`

然后导出脚本直接读取这个自定义 split：

```bash
python scripts/export_hl_memory_crosstask.py \
  --crosstask-release-dir "cross_task_datasets/crosstask_release" \
  --videos-root "cross_task_datasets/missing_videos" \
  --records-csv "cross_task_datasets/crosstask/matched_split/train_records.csv" \
  --output-dir "cross_task_datasets/crosstask/hl_memory_train_local" \
  --recent-frames-length 8 \
  --frame-subsample 1 \
  --memory-length 8 \
  --merge-distance 1 \
  --overwrite
```

验证集同理：

```bash
python scripts/export_hl_memory_crosstask.py \
  --crosstask-release-dir "cross_task_datasets/crosstask_release" \
  --videos-root "cross_task_datasets/missing_videos" \
  --records-csv "cross_task_datasets/crosstask/matched_split/val_records.csv" \
  --output-dir "cross_task_datasets/crosstask/hl_memory_val_local" \
  --recent-frames-length 8 \
  --frame-subsample 1 \
  --memory-length 8 \
  --merge-distance 1 \
  --overwrite
```

### 开发机下载与训练指令

下面这组命令假设你的开发机已经有：

- `python >= 3.11`
- `opencv-python`
- `torch`
- `transformers`
- 当前仓库代码

准备数据目录：

```bash
export DATA_ROOT=/path/to/data
mkdir -p "$DATA_ROOT/crosstask"
cd "$DATA_ROOT/crosstask"
```

下载 CrossTask 标注和视频：

```bash
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://www.rocq.inria.fr/cluster-willow/dzhukov/missing_videos.tar.gz
unzip crosstask_release.zip
tar -xzf missing_videos.tar.gz
```

建议先做一个 smoke export，只跑少量视频：

```bash
cd /path/to/openpi

python scripts/export_hl_memory_crosstask.py \
  --crosstask-release-dir "cross_task_datasets/crosstask_release" \
  --videos-root "cross_task_datasets/missing_videos" \
  --split train \
  --output-dir "cross_task_datasets/hl_memory_train_smoke" \
  --max-videos 8 \
  --recent-frames-length 8 \
  --frame-subsample 1 \
  --memory-length 8 \
  --merge-distance 1 \
  --overwrite
```

验证集也导一份：

```bash
python scripts/export_hl_memory_crosstask.py \
  --crosstask-release-dir "$DATA_ROOT/crosstask/crosstask_release" \
  --videos-root "$DATA_ROOT/crosstask/missing_videos" \
  --split val \
  --output-dir "$DATA_ROOT/crosstask/hl_memory_val_smoke" \
  --max-videos 8 \
  --recent-frames-length 8 \
  --frame-subsample 1 \
  --memory-length 8 \
  --merge-distance 1 \
  --overwrite
```

如果 smoke 没问题，再导完整 train / val：

```bash
python scripts/export_hl_memory_crosstask.py \
  --crosstask-release-dir "$DATA_ROOT/crosstask/crosstask_release" \
  --videos-root "$DATA_ROOT/crosstask/missing_videos" \
  --split train \
  --output-dir "$DATA_ROOT/crosstask/hl_memory_train" \
  --recent-frames-length 8 \
  --frame-subsample 1 \
  --memory-length 8 \
  --merge-distance 1 \
  --overwrite

python scripts/export_hl_memory_crosstask.py \
  --crosstask-release-dir "$DATA_ROOT/crosstask/crosstask_release" \
  --videos-root "$DATA_ROOT/crosstask/missing_videos" \
  --split val \
  --output-dir "$DATA_ROOT/crosstask/hl_memory_val" \
  --recent-frames-length 8 \
  --frame-subsample 1 \
  --memory-length 8 \
  --merge-distance 1 \
  --overwrite
```

训练：

```bash
python scripts/train_hl_memory.py \
  --dataset-dir "$DATA_ROOT/crosstask/hl_memory_train" \
  --output-dir "$DATA_ROOT/crosstask/hl_memory_ckpts_qwen25" \
  --vlm-backend qwen2_5_vl \
  --vlm-hf-model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-train-steps 1000 \
  --save-interval 200 \
  --log-interval 20
```

评估：

```bash
python scripts/eval_hl_memory_rollout.py \
  --dataset-dir "$DATA_ROOT/crosstask/hl_memory_val" \
  --model-path "$DATA_ROOT/crosstask/hl_memory_ckpts_qwen25/checkpoint-step-001000" \
  --vlm-backend qwen2_5_vl \
  --device cuda \
  --output-json "$DATA_ROOT/crosstask/hl_memory_val_metrics.json"
```

### CrossTask 一键 smoke

如果开发机已经把 CrossTask 数据下载并解压到同一个目录下，可以直接跑：

```bash
cd /path/to/openpi

DATA_ROOT=/path/to/data/crosstask \
DEVICE=cuda \
MODEL_BACKEND=qwen2_5_vl \
MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
TRAIN_STEPS=20 \
scripts/hl_memory_crosstask_smoke.sh
```

Qwen3.5 2B smoke：

```bash
DATA_ROOT=/path/to/data/crosstask \
DEVICE=cuda \
MODEL_BACKEND=qwen3_5_vl \
MODEL_VARIANT=qwen3_5_2b \
MODEL_ID=Qwen/Qwen3.5-2B \
TRAIN_STEPS=20 \
scripts/hl_memory_crosstask_smoke.sh
```

Qwen3.5 4B smoke：

```bash
DATA_ROOT=/path/to/data/crosstask \
DEVICE=cuda \
MODEL_BACKEND=qwen3_5_vl \
MODEL_VARIANT=qwen3_5_4b \
MODEL_ID=Qwen/Qwen3.5-4B \
TRAIN_STEPS=20 \
scripts/hl_memory_crosstask_smoke.sh
```

默认目录约定：

- `$DATA_ROOT/crosstask_release`
- `$DATA_ROOT/missing_videos`

默认输出：

- `$DATA_ROOT/hl_memory_train_smoke`
- `$DATA_ROOT/hl_memory_val_smoke`
- `$DATA_ROOT/hl_memory_ckpts_smoke`
- `$DATA_ROOT/hl_memory_val_smoke_metrics.json`

常用覆写参数：

```bash
SMOKE_VIDEOS=4
BATCH_SIZE=1
GRAD_ACCUM_STEPS=2
RECENT_FRAMES_LENGTH=8
FRAME_SUBSAMPLE=1
MEMORY_LENGTH=8
MERGE_DISTANCE=1
```
