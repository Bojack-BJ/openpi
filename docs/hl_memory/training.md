# HL Memory Training

## Train

多任务 / batch pipeline 推荐直接用 `train_hl_memory_multitask.py`。它可以从 batch export 的 root 下自动发现多个 task 的 `train/` 目录，把所有 task 的 samples 合并成一个训练池：

```bash
export PYTHONPATH=/lumos-vePFS/suzhou/Users/lixiaotong/openpi/src
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_MODE=online

torchrun --standalone --nproc_per_node 8 scripts/hl_memory/train_hl_memory_multitask.py \
  --dataset-root /root/Users/dataset/hl_memory/subtask \
  --dataset-glob '*/train' \
  --val-dataset-root /root/Users/dataset/hl_memory/subtask \
  --val-dataset-glob '*/val' \
  --output-dir /root/Users/checkpoints/hl_memory/subtask_multitask_qwen35_lora_with_prior_new_sampling \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-4B \
  --precision bfloat16 \
  --training-fps 20 \
  --frame-subsample 5 \
  --recent-sample-hz 2.0 \
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
  --num-train-steps 6000 \
  --save-interval 200 \
  --log-interval 10 \
  --val-interval 100 \
  --val-batches 10 \
  --wandb-enabled \
  --wandb-project openpi-hl-memory \
  --wandb-run-name subtask-multitask-qwen35-4b-lora_with_prior_new_sampling
```

数据选择方式：

- `--dataset-root /path/to/root --dataset-glob '*/train'`：推荐 batch 训练入口，自动加载每个 task 的 train split。
- `--dataset-root /path/to/root --task-ids TASK_A TASK_B ...`：只训练指定 task id，脚本会自动解析成 `/path/to/root/TASK_A/train`、`/path/to/root/TASK_B/train`。也支持逗号写法：`--task-ids TASK_A,TASK_B`。
- `--val-dataset-root /path/to/root --val-task-ids TASK_A TASK_B ...`：只验证指定 task id；如果没传 `--val-task-ids`，但传了 `--task-ids`，val 会默认跟随同一批 task id 并解析成 `/path/to/root/<task_id>/val`。
- `--dataset-dir /path/to/task/train`：只训练单个导出目录，主要用于 smoke/debug。
- `--dataset-dirs ...` 或 `--dataset-dirs-json /path/to/list.json`：显式指定多个导出目录，适合排除坏任务或只跑子集。

`train_hl_memory_multitask.py` 支持 `torchrun` 多卡 DDP/FSDP。每张卡启动一个 rank，每个 rank 独立采样不同 HL samples、各自 forward/backward，然后在 optimizer step 前同步梯度。`--batch-size` 是每张卡的 micro batch；有效全局 batch 约等于 `batch_size * grad_accum_steps * nproc_per_node`。

当前脚本会把每个 rank 内的 `--batch-size` 个 samples 合成一个 VLM batch，一次 processor encode 和一次 model forward/backward。因此它同时支持“多卡 batch 并行”和“单卡 batch 内并行”。把 `--batch-size` 调大通常会提高 GPU 吞吐，但也会增加 peak activation memory；如果 OOM，优先降低 `--batch-size`，再用 `--grad-accum-steps` 保持全局 batch。

`--distributed-strategy ddp` 是默认值，每张卡各自保留一份完整模型，速度直接但显存压力最大。4B/27B 或 batch 内并行 OOM 时用 `--distributed-strategy fsdp`，FSDP 会 shard 参数、梯度和 optimizer states，显存更低，但通信和 checkpoint 保存更慢。FSDP 的参数 shard 数由 `torchrun --nproc_per_node` / world size 决定，不由 `--fsdp-min-num-params` 决定；`--fsdp-min-num-params` 只是 auto-wrap 阈值，控制多大的子模块会被单独包成一个 FSDP unit。阈值越小，wrap 越细，峰值显存通常越低但通信开销越高；阈值越大，wrap 越粗，速度可能更好但峰值显存更高。LoRA + FSDP 需要 `use_orig_params=True`，脚本内部已启用。FSDP flatten 同一个 unit 时要求浮点参数 dtype 一致；脚本会在 FSDP wrapping 前把 LoRA 等浮点参数对齐到 `--precision` 指定的 dtype，避免 bf16 base model + fp32 LoRA 混合报错。FSDP 路径不会在 grad accumulation 期间使用 `no_sync()`，因为 FSDP `no_sync()` 会保留未 shard 梯度、增加峰值显存；如果仍然 OOM，优先用 `--batch-size 1` 并增大 `--grad-accum-steps`，再把 `--fsdp-min-num-params` 降到 `50000000` 或 `20000000`，最后再考虑 `--fsdp-cpu-offload`，CPU offload 会明显变慢。

FSDP + LoRA 下，`--vision-tower-train-mode frozen/last_n/full` 改变的不只是 trainable 参数量，也会改变 autograd 图、FSDP communication/overlap、optimizer param groups 和 GPU 利用率。`frozen` 只会把 vision tower 参数设为 `requires_grad=False`，不会跳过 vision forward；如果 vision tower 仍被 FSDP auto-wrap，forward 仍可能触发参数 all-gather。因此 frozen 可能出现“计算少但通信/调度暴露更多”的低效状态，wall-clock 不一定比 full vision 更快。启动时看日志里的：

- `Vision tower train mode=... trainable=...`
- `Optimizer param groups: base_params=... vision_params=...`
- `FSDP report: ... vision_candidate_direct_wrap_modules=... vision_candidate_direct_wrap_params=...`

速度诊断可以开启轻量 profile。`--profile-steps N` 会在指定 step 上同步 CUDA 并打印 `data/forward/backward/optim/total` 分段耗时和显存；`--profile-trace-dir` 会导出 rank0 Chrome trace，用于在 Perfetto/Chrome tracing 里检查 FSDP all-gather、reduce-scatter、NCCL kernel 数量、CUDA idle gap。

```bash
torchrun --standalone --nproc_per_node 8 scripts/hl_memory/train_hl_memory_multitask.py \
  ... \
  --num-train-steps 120 \
  --val-interval 0 \
  --save-interval 120 \
  --log-interval 10 \
  --profile-start-step 60 \
  --profile-steps 3 \
  --profile-trace-dir /tmp/hl_memory_profiler/vision_frozen
```

开启 `--val-interval` 后，脚本会每隔 N 个 optimizer steps 从 val split 随机抽 `--val-batches` 个 batch 做 forward-only loss，并在 rank0 记录 `val/loss`、`val/time_s`、`val/batches_per_rank` 和 `val/effective_samples` 到 wandb。默认不启用 validation；多任务 batch pipeline 推荐显式设置 `--val-dataset-root ... --val-dataset-glob '*/val'`。

Loss 计算逻辑：输入序列是 `prompt + target JSON`，labels 会把 prompt tokens 和 padding tokens 置为 `-100`，只监督 target JSON token。Hugging Face causal LM loss 是所有未 mask target tokens 的平均 cross entropy。`grad_accum_steps` 内每个 micro batch 的 loss 会除以 accum steps 再 backward；日志里的 `train/loss` 也按 accum steps 做平均，因此不同 accum 配置下数值可比。DDP 下每个 rank 算本地 loss，日志再跨 rank 求平均。

多任务训练建议优先用 LoRA + `--language-memory-dropout` + `--step-prior-dropout`。原因是不同 task 的语言目标差异大，full finetune 更容易记住任务模板或破坏原 VLM 的通用视觉能力；step prior dropout 则避免模型只背 plan，不看 recent clip。`--num-train-steps` 是 optimizer steps，不是 epoch；实际见过的样本数约等于 `num_train_steps * global_batch_size`，需要按总 sample 数和任务数量估算。

训练协议由 `--target-protocol` 控制：

- `--target-protocol hl_v1`：默认完整 HL V1 target，监督 `task_progress/current_objective/subtask_progress/should_advance_objective/active_hand/relevant_objects/notes/keyframes/...`。
- `--target-protocol subtask_keyframe`：更纯的视觉时序 baseline，只监督 `{"current_objective": "...", "keyframe_candidate_positions": [...]}`。这里的 objective 是当前帧的 executable instruction，更适合作为 LL VLA 的语言目标；不监督 language memory、progress、advance、target/goal/notes。
- `--target-protocol memer_objective`：MEMER-style 简化 target，监督 `{"current_objective": "...", "horizon_current_objective": "...", "keyframe_candidate_positions": [...]}`。`current_objective` 是当前帧目标，`horizon_current_objective` 是 short-horizon 目标；如果 sample 没有 horizon 字段，horizon fallback 到 current。
- `--target-protocol objective_memory_state`：MEMER-style target，监督 `current_objective/horizon_current_objective/keyframes/updated_language_memory`，prompt 额外输入 `language_memory`。用于测试模型是否能消费并更新 compact memory。
- `--target-protocol objective_last_objective`：MEMER-style target，监督 `current_objective/horizon_current_objective/keyframes/last_objective`。不把 `last_objective` 放进 prompt；它是训练辅助输出，推理可忽略。
- `--target-protocol objective_prev_stage`：MEMER-style target，监督 `current_objective/horizon_current_objective/keyframes/previous_stage_objective`，prompt 额外输入 `previous_stage_objective`。用于测试模型是否能维护比 full memory 更轻的阶段状态。
- `--target-protocol keyframe_gated_memory`：单 pass keyframe-gated target，监督 `current_objective/horizon_current_objective/keyframe_candidate_positions/completed_objective`。runtime 只在 `completed_objective` 非空且有 keyframe candidate 时更新 completed-event log。
- `--target-protocol keyframe_gated_memory_two_pass`：two-pass typed target。训练时每个 sample 展开成 Pass A / Pass B 两条 VLM examples；Pass A 看 historical keyframes + 完整 recent window，只监督 current/horizon/keyframe proposal；Pass B 看 historical keyframes + candidate frames，只监督 canonical completed event，非 candidate recent frames 不进入 processor。训练时会对 GT proposal 做确定性位置扰动，并在 negative sample 上注入 false proposal，缩小 train/inference proposal exposure gap。Pass B prompt 不包含 GT current/horizon 文本。推理时 Pass A 无 candidate 会直接跳过 Pass B。
- `--target-protocol keyframe_gated_memory_typed_mask`：仅 Qwen2.5-VL。保持 JSON 字段顺序不变，但 4D mask 会阻止 horizon target rows attend teacher-forced `current_objective` target field，同时阻止 completed rows 直接 attend recent visual tokens。Horizon 仍可看完整 prompt、historical memory 和 recent visual source，因此该 setting 用于测试 horizon 是否真正从视觉和状态上下文预测，而不是学习 `GT current -> GT horizon` 映射。Qwen3.5 不支持这条 4D mask 路径；需要严格隔离时使用独立 pass/head。
- `--target-protocol known_prior_tracker`：监督 `current_objective/subtask_progress/should_advance_objective/keyframes`，用于已知 step prior 的状态机式对比实验；不建议作为当前主线。

Keyframe candidate 默认使用 event-band 监督，而不是只在 coarse segment 末尾的单帧给正例：

- `--keyframe-candidate-label-mode event_band`：只要 recent window 命中过渡点附近 band，就输出离 canonical keyframe 最近的一个 candidate position。
- `--keyframe-event-band-before-sec 1.0 --keyframe-event-band-after-sec 0.5`：默认 band，提升 H088 这类短 transition 的 recall。
- `--keyframe-candidate-label-mode canonical`：严格只监督 canonical keyframe，主要用于 ablation。
- `--keyframe-positive-sample-ratio 0.4`：可选 loader-level sampling balance，让每个 batch 里约 40% 样本带 keyframe candidate。
- `--keyframe-confirm-positive-sample-ratio 0.2`：仅用于 `keyframe_gated_memory_two_pass`，让约 20% source samples 是 canonical completion positives；该比例包含在上面的 40% proposal-positive 比例内。推荐三池比例为 canonical confirm 20%、event-band proposal-only 20%、negative 60%。
- `--two-pass-training-proposal-noise-probability 0.25`：canonical/event-band positive 的 Pass B candidate 位置扰动概率。Negative sample 始终给 Pass B 一个 deterministic false proposal 并监督空 completion，从而训练拒绝错误 Pass A proposal。
- `--two-pass-predict-loss-weight 1.0 --two-pass-confirm-loss-weight 1.0`：先分别对每条 Pass A/Pass B target 的 token loss 求均值，再按 pass 权重聚合。这样较短的 completion JSON 不会被较长的 Pass A JSON 按 token 数稀释。

`keyframe_label` 仍然是 canonical 单点，`keyframe_gated_memory.completed_objective` 也只在 canonical event 上非空；event band 只扩大 candidate proposal 的视觉监督，不会让 long-term memory 存入每个 band 帧。

### Keyframe History Auxiliary Loss

JSON token CE 只监督单个 recent window 的 `keyframe_candidate_positions`，不直接约束这些 proposals 经
merge/dedup 后是否形成正确、紧凑的 historical keyframe list。可以启用训练期 auxiliary head，让同一个
prompt hidden state额外预测：

- recent clip 内的 keyframe position distribution；
- 当前 window 是否命中 event band；
- 本次 event 对 history state 的更新类型：`reject/add/duplicate`；
- accepted timing 相对 canonical keyframe 的误差。

生成协议、target JSON 和 rollout state machine 均不改变。Auxiliary head 从 assistant target 开始前的最后一个
prompt hidden state读取特征，不读取 teacher-forced GT JSON。推荐起始配置：

```bash
  --keyframe-aux-position-loss-weight 0.5 \
  --keyframe-aux-event-loss-weight 0.5 \
  --keyframe-aux-timing-loss-weight 0.1 \
  --keyframe-aux-update-loss-weight 0.5 \
  --keyframe-aux-hidden-dim 512 \
  --keyframe-aux-timing-sigma-sec 0.5 \
  --keyframe-positive-sample-ratio 0.4
```

总 loss 为 language JSON CE 加四个加权 auxiliary losses。WandB 会记录
`loss_aux_position/event/timing/update/total` 和 `loss_language`。Position target 以 canonical frame 为中心
生成距离 soft label；history update target 根据当前 GT historical memory 判定 `reject/add/duplicate`。
Checkpoint 额外保存 `hl_keyframe_auxiliary.pt` 和 `hl_keyframe_auxiliary_config.json`。所有 auxiliary weight
保持 `0` 时完全禁用，不增加 hidden-state 输出和训练开销。

采样按每个 source sample 独立做概率选择，不按 microbatch 内数量取整。这个区别在
`batch_size=1` 时尤其重要：旧的 `round(batch_size * 0.4)` 会退化成 100% positive；
当前实现会跨 rank 和 grad-accum microsteps 在期望意义上保持 20% / 20% / 60%。

`objective_last_objective` 和 `objective_prev_stage` 不要求重新导出旧 `samples.jsonl`：loader 会按
`episode_index/step_index` 自动补 `last_objective` 和 `previous_stage_objective`。但如果要修正
`language_memory` 的历史累加内容，例如旧样本里一直是 `No completed subtask yet`，仍然需要重新 export dataset；
loader 不能从旧字符串里恢复正确的 completed-subtasks memory。

Context/state 训练方式：`objective_*` 协议不会改视频输入，核心 target 仍是 MEMER 的
`current_objective/horizon_current_objective/keyframes`，但会额外监督一个状态字段。`objective_memory_state`
在 prompt 里追加 `Completed-subtasks memory: ...`，并额外输出 `updated_language_memory`；
`objective_last_objective` 不把 last objective 放进 prompt，而是额外输出 `last_objective` 作为训练辅助目标，
推理时可忽略；`objective_prev_stage` 在 prompt 里追加 `Previous stage objective: ...`，并额外输出
`previous_stage_objective`，用于测试模型是否能维护一个比 full memory 更轻的阶段状态。

Proprio/state 输入默认关闭。打开后不是把 state 数字写进 prompt，而是把归一化 state 通过 MLP 投成 learned soft tokens，插入 Qwen text embedding 序列：

- `--proprio-enabled`：启用 proprio 输入；旧 samples 没有 `recent_robot_states` 会直接报错。
- `--proprio-token-mode per_frame`：每个 recent timestep 一个 state token。
- `--proprio-token-mode summary`：所有 recent states，包括 current，经 learned summary query 聚合成一个 state token。
- `--proprio-token-mode per_frame_plus_summary`：默认，同时给 summary token 和 per-frame tokens。
- `--proprio-dropout` 和 `--proprio-noise-std` 只在训练时生效，eval/rollout 不加噪声、不 dropout。

zero-shot runtime proprio 暂时未接入；在 `run_hl_memory_zero_shot.py` 里传 `--proprio-enabled` 会报错。等 runtime state JSON 格式确定后，再按 rollout `recent_seconds` 做最近邻对齐。

MEMER-style 训练示例：

```bash
torchrun --standalone --nproc_per_node 8 scripts/hl_memory/train_hl_memory_multitask.py \
  --dataset-root /root/Users/dataset/hl_memory/subtask_dense \
  --task-ids 20260126O014 20260207K038 20260328K086A \
  --dataset-glob '*/train' \
  --val-dataset-root /root/Users/dataset/hl_memory/subtask_dense \
  --val-task-ids 20260126O014 20260207K038 20260328K086A \
  --val-dataset-glob '*/val' \
  --output-dir /root/Users/checkpoints/hl_memory/subtask_dense_memer_objective_qwen35_lora \
  --target-protocol memer_objective \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-4B \
  --precision bfloat16 \
  --learning-rate 5e-6 \
  --lora-enabled \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --num-train-steps 1500 \
  --val-interval 100 \
  --val-batches 10
```

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
export WANDB_API_KEY="${WANDB_API_KEY}"
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

这里 `--batch-size` 是每张卡的 micro batch；有效全局 batch 约等于 `batch_size * grad_accum_steps * nproc_per_node`。训练进度条会显示 ETA、`s/it` / `it/s`、`data_s/it` 和 `step_s/it`。开启 wandb 后 rank0 会记录 `train/loss`、`time/data_s_per_it`、`time/step_s_per_it`、`time/data_fraction`、`train/lr`、可选的 `train/vision_lr` 和 `train/global_batch_size`。`--frame-cache-size` 是每个 rank 缓存的 resized frame 数；如果 `data_s/it` 占比高，可以适当增大，前提是 CPU 内存足够。

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

Vision tower 默认保持 frozen。Qwen3.5-VL 的视觉侧是 Qwen-style video vision tower：`Conv3D` patch embedding 把多帧视频切成时空 patch token，后面接 ViT block stack 和 rotary/position embedding，最后经过 spatial patch merger 把视觉 token 映射给语言模型。也就是说它不是简单“把多张图片拼成一张再当静态图”训练；但机器人鱼眼视角和手眼双视角仍然可能和预训练视频分布有 gap。

视觉适配有三种模式：

- `--vision-tower-train-mode frozen`：默认，显式冻结 vision tower；LoRA 训练时只训练语言侧 LoRA，full finetune 时也不会更新视觉侧，用作 baseline。
- `--vision-tower-train-mode last_n --vision-tower-unfreeze-last-n-layers N`：只解冻视觉 tower 最后 N 个 block，并解冻 `merger/patch_merger`；推荐先试 `N=2`，再试 `N=4`。
- `--vision-tower-train-mode full`：解冻整个视觉 tower；显存和过拟合风险最高，只在 last_n 明显有收益且仍不够时试。

推荐学习率：语言 LoRA 仍用 `--learning-rate 5e-6` 起步；vision 解冻时用 `--vision-tower-learning-rate` 单独控制视觉侧，last-N 先试 `2e-6`，full vision 先试 `1e-6` 或更低。没有传 `--vision-tower-learning-rate` 时，所有 trainable 参数沿用 `--learning-rate`。

视觉 frozen / last-N / full 对比建议固定 dropout，先去掉 text memory 和 step prior shortcut：

```bash
### A. Vision frozen baseline.
torchrun --standalone --nproc_per_node 8 scripts/hl_memory/train_hl_memory_multitask.py \
  --dataset-root /root/Users/dataset/hl_memory/subtask \
  --dataset-glob '*/train' \
  --val-dataset-root /root/Users/dataset/hl_memory/subtask \
  --val-dataset-glob '*/val' \
  --output-dir /root/Users/checkpoints/hl_memory/subtask_qwen35_visual_ablation_frozen \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-4B \
  --precision bfloat16 \
  --learning-rate 5e-6 \
  --lora-enabled \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --language-memory-dropout 1.0 \
  --step-prior-dropout 1.0 \
  --vision-tower-train-mode frozen \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --num-train-steps 6000 \
  --val-interval 100 \
  --val-batches 10
```

```bash
### B. Unfreeze final vision blocks. Start with N=2; try N=4 if stable.
torchrun --standalone --nproc_per_node 8 scripts/hl_memory/train_hl_memory_multitask.py \
  --dataset-root /root/Users/dataset/hl_memory/subtask \
  --dataset-glob '*/train' \
  --val-dataset-root /root/Users/dataset/hl_memory/subtask \
  --val-dataset-glob '*/val' \
  --output-dir /root/Users/checkpoints/hl_memory/subtask_qwen35_visual_ablation_last2 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-4B \
  --precision bfloat16 \
  --learning-rate 5e-6 \
  --vision-tower-learning-rate 2e-6 \
  --lora-enabled \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --language-memory-dropout 1.0 \
  --step-prior-dropout 1.0 \
  --vision-tower-train-mode last_n \
  --vision-tower-unfreeze-last-n-layers 2 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --num-train-steps 6000 \
  --val-interval 100 \
  --val-batches 10
```

```bash
### C. Full vision tower finetune. Use only if last_n helps but is not enough.
torchrun --standalone --nproc_per_node 8 scripts/hl_memory/train_hl_memory_multitask.py \
  --dataset-root /root/Users/dataset/hl_memory/subtask \
  --dataset-glob '*/train' \
  --val-dataset-root /root/Users/dataset/hl_memory/subtask \
  --val-dataset-glob '*/val' \
  --output-dir /root/Users/checkpoints/hl_memory/subtask_qwen35_visual_ablation_full \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_4b \
  --local-vlm-ckpt-path /root/Users/lixiaotong/Qwen3.5-4B \
  --precision bfloat16 \
  --learning-rate 5e-6 \
  --vision-tower-learning-rate 1e-6 \
  --lora-enabled \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --language-memory-dropout 1.0 \
  --step-prior-dropout 1.0 \
  --vision-tower-train-mode full \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --num-train-steps 6000 \
  --val-interval 100 \
  --val-batches 10
```

当前推荐 ablation 顺序：

1. `language_memory_dropout=1.0` + `step_prior_dropout=1.0`，分别跑 vision frozen / final 2 blocks / full vision。目的是确认模型是否能只靠 recent window + visual keyframes 学到进度。
2. 在第 1 步表现最好的 vision setting 上，把 `--step-prior-dropout` 从 `1.0` 降到 `0.3` 或 `0.0`，看 known step prior 是否提供稳定增益，而不是重新制造 shortcut。
3. 给 keyframe memory 加绝对时间戳，评估是否改善“短 subtask 一进入 memory 就过早切下一步”的时间感知问题。
4. 额外建议固定 keyframe label 策略和训练步数做对照。不要同时改 keyframe、dropout、vision 解冻和 target protocol，否则很难判断收益来源。
5. 额外建议保留一个 rollout-level 指标：advance precision/recall、平均提前/滞后秒数、空 keyframe 率。单看 val loss 很容易被 JSON/token 格式影响。

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
