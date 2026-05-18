#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing crosstask_release and missing_videos.}"

MODEL_BACKEND="${MODEL_BACKEND:-qwen2_5_vl}"
MODEL_VARIANT="${MODEL_VARIANT:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
DEVICE="${DEVICE:-cuda}"
MODEL_VARIANT_ARGS=()
if [[ -n "$MODEL_VARIANT" ]]; then
  MODEL_VARIANT_ARGS=(--vlm-variant "$MODEL_VARIANT")
fi

SMOKE_VIDEOS="${SMOKE_VIDEOS:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
RECENT_FRAMES_LENGTH="${RECENT_FRAMES_LENGTH:-8}"
FRAME_SUBSAMPLE="${FRAME_SUBSAMPLE:-1}"
MEMORY_LENGTH="${MEMORY_LENGTH:-8}"
MERGE_DISTANCE="${MERGE_DISTANCE:-1}"

TRAIN_EXPORT_DIR="${TRAIN_EXPORT_DIR:-$DATA_ROOT/hl_memory_train_smoke}"
VAL_EXPORT_DIR="${VAL_EXPORT_DIR:-$DATA_ROOT/hl_memory_val_smoke}"
CKPT_DIR="${CKPT_DIR:-$DATA_ROOT/hl_memory_ckpts_smoke}"
METRICS_JSON="${METRICS_JSON:-$DATA_ROOT/hl_memory_val_smoke_metrics.json}"

cd "$REPO_ROOT"

echo "[1/4] Export CrossTask smoke train split"
uv run python scripts/export_hl_memory_crosstask.py \
  --crosstask-release-dir "$DATA_ROOT/crosstask_release" \
  --videos-root "$DATA_ROOT/missing_videos" \
  --split train \
  --output-dir "$TRAIN_EXPORT_DIR" \
  --max-videos "$SMOKE_VIDEOS" \
  --recent-frames-length "$RECENT_FRAMES_LENGTH" \
  --frame-subsample "$FRAME_SUBSAMPLE" \
  --memory-length "$MEMORY_LENGTH" \
  --merge-distance "$MERGE_DISTANCE" \
  --overwrite

echo "[2/4] Export CrossTask smoke val split"
uv run python scripts/export_hl_memory_crosstask.py \
  --crosstask-release-dir "$DATA_ROOT/crosstask_release" \
  --videos-root "$DATA_ROOT/missing_videos" \
  --split val \
  --output-dir "$VAL_EXPORT_DIR" \
  --max-videos "$SMOKE_VIDEOS" \
  --recent-frames-length "$RECENT_FRAMES_LENGTH" \
  --frame-subsample "$FRAME_SUBSAMPLE" \
  --memory-length "$MEMORY_LENGTH" \
  --merge-distance "$MERGE_DISTANCE" \
  --overwrite

echo "[3/4] Train HL memory smoke model"
uv run python scripts/train_hl_memory.py \
  --dataset-dir "$TRAIN_EXPORT_DIR" \
  --output-dir "$CKPT_DIR" \
  --vlm-backend "$MODEL_BACKEND" \
  "${MODEL_VARIANT_ARGS[@]}" \
  --vlm-hf-model-id "$MODEL_ID" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-train-steps "$TRAIN_STEPS" \
  --save-interval "$TRAIN_STEPS" \
  --log-interval 5

CHECKPOINT_PATH="$CKPT_DIR/checkpoint-step-$(printf '%06d' "$TRAIN_STEPS")"

echo "[4/4] Evaluate smoke checkpoint"
uv run python scripts/eval_hl_memory_rollout.py \
  --dataset-dir "$VAL_EXPORT_DIR" \
  --model-path "$CHECKPOINT_PATH" \
  --vlm-backend "$MODEL_BACKEND" \
  "${MODEL_VARIANT_ARGS[@]}" \
  --device "$DEVICE" \
  --output-json "$METRICS_JSON"

echo
echo "Smoke run complete."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Metrics:    $METRICS_JSON"
