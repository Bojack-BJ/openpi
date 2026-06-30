#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/lxt/Github/Lumos/openpi}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKBONE="${BACKBONE:-qwen3_5_4b}"
DATASET_ROOT="${DATASET_ROOT:-/root/Users/dataset/hl_memory/subtask_coarse}"
DATASET_GLOB="${DATASET_GLOB:-*/train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/checkpoints/hl_memory/qwen35_clean_context_matrix}"
CONTEXT_MODE="${CONTEXT_MODE:-baseline}"
TARGET_PROTOCOL="${TARGET_PROTOCOL:-memer_film_progress_two_pass}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29541}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
VISION_TOWER_LEARNING_RATE="${VISION_TOWER_LEARNING_RATE:-2e-6}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi-hl-memory}"

case "$BACKBONE" in
  qwen3_5_2b|qwen35_2b)
    VLM_BACKEND="${VLM_BACKEND:-qwen3_5_vl}"
    VLM_VARIANT="${VLM_VARIANT:-qwen3_5_2b}"
    LOCAL_VLM_CKPT_PATH="${LOCAL_VLM_CKPT_PATH:-/root/Users/lixiaotong/Qwen3.5-2B}"
    ;;
  qwen3_5_4b|qwen35_4b)
    VLM_BACKEND="${VLM_BACKEND:-qwen3_5_vl}"
    VLM_VARIANT="${VLM_VARIANT:-qwen3_5_4b}"
    LOCAL_VLM_CKPT_PATH="${LOCAL_VLM_CKPT_PATH:-/root/Users/lixiaotong/Qwen3.5-4B}"
    ;;
  *)
    echo "Unsupported BACKBONE=$BACKBONE" >&2
    exit 2
    ;;
esac

case "$CONTEXT_MODE" in
  baseline)
    EXTRA_ARGS=(
      --no-proprio-enabled
      --no-state-condition-enabled
    )
    RUN_TAG="baseline"
    ;;
  state_film)
    EXTRA_ARGS=(
      --no-proprio-enabled
      --state-condition-enabled
      --state-condition-mode film
    )
    RUN_TAG="state_film"
    ;;
  proprio_token)
    EXTRA_ARGS=(
      --proprio-enabled
      --proprio-token-mode per_frame_plus_summary
      --proprio-projector-mode joint
      --no-state-condition-enabled
    )
    RUN_TAG="proprio_token"
    ;;
  state_plus_proprio)
    EXTRA_ARGS=(
      --proprio-enabled
      --proprio-token-mode per_frame_plus_summary
      --proprio-projector-mode joint
      --state-condition-enabled
      --state-condition-mode film
    )
    RUN_TAG="state_plus_proprio"
    ;;
  *)
    echo "Unsupported CONTEXT_MODE=$CONTEXT_MODE" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${BACKBONE}_${RUN_TAG}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${BACKBONE}_${RUN_TAG}}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

WANDB_FLAG="--no-wandb-enabled"
if [[ "$WANDB_ENABLED" == "1" || "$WANDB_ENABLED" == "true" || "$WANDB_ENABLED" == "TRUE" ]]; then
  WANDB_FLAG="--wandb-enabled"
fi

cd "$REPO_ROOT"

"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  scripts/hl_memory/train_hl_memory_multitask.py \
  --dataset-root "$DATASET_ROOT" \
  --dataset-glob "$DATASET_GLOB" \
  --output-dir "$OUTPUT_DIR" \
  --vlm-backend "$VLM_BACKEND" \
  --vlm-variant "$VLM_VARIANT" \
  --local-vlm-ckpt-path "$LOCAL_VLM_CKPT_PATH" \
  --target-protocol "$TARGET_PROTOCOL" \
  --precision bfloat16 \
  --vision-tower-train-mode full \
  --learning-rate "$LEARNING_RATE" \
  --vision-tower-learning-rate "$VISION_TOWER_LEARNING_RATE" \
  --lora-enabled \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval 10 \
  "$WANDB_FLAG" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  "${EXTRA_ARGS[@]}"
