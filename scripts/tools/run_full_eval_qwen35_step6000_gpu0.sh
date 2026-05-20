#!/usr/bin/env bash
set -euo pipefail

REPO=/root/Users/donggaoqi/openpi_vlm_finetune
GPU_ID="${GPU_ID:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-65000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO/hl_memory_ckpts/eval/full_qwen3_5_step6000_$STAMP"
LOG="$RUN_DIR/run.log"

mkdir -p "$RUN_DIR"
cd "$REPO"

echo "[wait] $(date) waiting for GPU${GPU_ID} free >= ${MIN_FREE_MB} MiB" | tee -a "$LOG"
while true; do
  free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')"
  echo "[wait] $(date) gpu=${GPU_ID} free_mb=${free_mb} util=${util}" >> "$LOG"
  if [[ "$free_mb" -ge "$MIN_FREE_MB" ]]; then
    break
  fi
  sleep 300
done

echo "[eval] $(date) starting full eval" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH=src /root/Users/miniconda3/envs/hl_qwen35/bin/python \
  scripts/hl_memory/eval_hl_memory_multitask.py \
  --dataset-root /root/Users/dataset/hl_memory/subtask \
  --dataset-glob '*/val' \
  --model-path hl_memory_ckpts/subtask_multitask_qwen3_5_2b_lora_ddp/checkpoint-step-006000 \
  --vlm-backend qwen3_5_vl \
  --vlm-variant qwen3_5_2b \
  --precision bfloat16 \
  --eval-modes full \
  --output-json "$RUN_DIR/metrics.json" \
  --predictions-jsonl "$RUN_DIR/predictions.jsonl" \
  2>&1 | tee -a "$LOG"

echo "[analyze] $(date) summarizing by task" | tee -a "$LOG"
/root/Users/miniconda3/envs/hl_qwen35/bin/python \
  scripts/hl_memory/analyze_hl_memory_eval_by_task.py \
  --predictions-jsonl "$RUN_DIR/predictions.jsonl" \
  --output-md "$RUN_DIR/task_summary.md" \
  --output-json "$RUN_DIR/task_summary.json" \
  2>&1 | tee -a "$LOG"

echo "[done] $(date) outputs in $RUN_DIR" | tee -a "$LOG"
