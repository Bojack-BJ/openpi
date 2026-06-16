# HL Memory Eval And Rollout

## Eval

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

如果训练时用了 `--target-protocol memer_objective` 或 `--target-protocol subtask_keyframe`，eval 也必须显式传同一个协议：

```bash
python scripts/hl_memory/eval_hl_memory_rollout.py \
  --dataset-dir /root/Users/dataset/hl_memory/subtask_dense/20260116W001/val \
  --model-path /root/Users/checkpoints/hl_memory/subtask_dense_memer_objective_qwen35_lora/checkpoint-step-001500 \
  --target-protocol memer_objective \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --device cuda \
  --eval-modes full \
  --output-json /root/Users/eval/hl_memory/memer_objective/eval_metrics.json \
  --prediction-jsonl /root/Users/eval/hl_memory/memer_objective/full_predictions.jsonl
```

简化协议的主指标是 `objective_exact_match/objective_normalized_match` 和 `keyframe_precision/keyframe_recall`。progress、advance、language memory、target/goal 等指标仍会输出，但这些协议没有监督这些字段，不应作为主判断。`memer_objective` 额外训练 `horizon_current_objective`，主要用于学习短提前量；当前 eval 主表仍以 `current_objective` 为主。

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

## Video Inference

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
  --known-prior-next-step-require-completion \
  --known-prior-next-step-confirm-steps 2 \
  --known-prior-safe-skip-mode \
  --known-prior-skip-match-threshold 0.95 \
  --known-prior-skip-min-progress 0.8 \
  --known-prior-skip-min-stall-steps 2 \
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
- 不传 `--recent-step-sec` 时，rollout 会按固定采样频率抽 recent clip：`recent_step_sec = 1 / recent_sample_hz`。默认 `--recent-sample-hz 2.0 --recent-frames-length 8`，即包含当前帧并覆盖最近 `3.5s`。
- Qwen video metadata 的 `fps` 同样按实际 recent clip 采样频率推导，默认 `2.0Hz`；不要再把 HL clip 当作 1Hz 视频。如果要严格 2Hz 覆盖 4 秒，用 `--recent-sample-hz 2.0 --recent-frames-length 9`。
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
| `--known-prior-next-step-require-completion` | 开启后，相邻 prior step 的文本匹配不能单独触发推进，还必须满足模型输出 `should_advance_objective=true` 或 `subtask_progress >= --known-prior-advance-threshold`。safe-skip 证据不足时也不会降级推进一步。默认关闭以兼容旧行为。 |
| `--known-prior-next-step-confirm-steps` | completion gate 开启后，如果模型连续 N 轮高置信匹配同一个相邻下一步，也允许推进，避免模型已经识别到新视觉状态但无法继续报告旧 step progress 时卡死。默认 `0` 表示禁用；建议先用 `2`。只作用于相邻步骤，不放宽跨多步 safe-skip。 |
| `--known-prior-safe-skip-mode` | 开启受限追赶模式：普通文本匹配最多推进一步；跨步跳转需要更强 match 和完成/卡住证据，避免从 step 5 直接跳到 step 7。 |
| `--known-prior-skip-match-threshold` | safe-skip 模式下允许跨步追赶的更高文本匹配阈值，默认 `0.95`。 |
| `--known-prior-skip-min-progress` | safe-skip 模式下允许跨步追赶的 progress 下限，默认 `0.8`；`should_advance_objective=true` 也会视作完成证据。 |
| `--known-prior-skip-min-stall-steps` | safe-skip 模式下，如果 pointer 连续卡在同一 prior step 至少 N 轮，即使 progress 不高也允许高置信跨步追赶，默认 `2`。 |
| `--model-path` | 已训练 HL checkpoint 或 Hugging Face/local model 路径。 |
| `--local-vlm-ckpt-path` | 本地 VLM/checkpoint 路径；设置后优先覆盖 `--model-path`。 |
| `--vlm-backend` | VLM 后端，常用 `qwen2_5_vl` 或 `qwen3_5_vl`。 |
| `--vlm-variant` | Qwen3.5 变体，例如 `qwen3_5_2b`，也可用短名 `2b` / `4b` / `27b`。 |
| `--target-protocol` | 推理 prompt/输出协议，默认 `hl_v1`；`subtask_keyframe` 输出当前 objective + keyframes，`memer_objective` 输出当前 objective + horizon objective + keyframes。 |
| `--precision` | 推理精度，常用 `float16` 或 `bfloat16`；遇到 vision/cuDNN 问题先试 `float16`。 |
| `--device` | 推理设备，通常是 `cuda`；CPU 只适合小模型/调试。 |

clip 和 memory 参数：

| 参数 | 作用 |
| --- | --- |
| `--recent-end-sec` | 单次预测时 recent clip 的结束时间；不填时默认取视频末尾。 |
| `--recent-step-sec` | recent clip 抽帧间隔；最高优先级调试参数。不传时由 `1 / --recent-sample-hz` 推导。 |
| `--recent-sample-hz` | recent clip 固定采样频率，默认 `2.0Hz`；export/train/eval/rollout 应保持一致。 |
| `--training-fps` | HL 训练导出所基于的 LeRobot fps，默认 `20.0`。 |
| `--frame-subsample` | 旧兼容参数；固定窗口模式下不决定 recent clip 总时长。 |
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
| `--memory-input-mode` | 控制模型实际看到的 language memory：`full` 完整输入；`completed_only` 只输入 `Task progress`，隐藏 `Current objective` 等强提示；`empty` 完全不输入 memory。summary 每步会记录 `language_memory_input` 便于核对。 |
| `--output-json` | 保存 summary JSON。interval rollout 期间每完成一个推理 step 都会原子更新，结束后再写入包含 debug video 状态的最终版本。 |
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
