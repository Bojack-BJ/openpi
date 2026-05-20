#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/Users/donggaoqi/openpi_vlm_finetune}"
PYTHON_BIN="${PYTHON_BIN:-/root/Users/miniconda3/envs/hl_qwen35/bin/python}"

BACKBONE="${BACKBONE:-qwen3_5_2b}"
LOCAL_VLM_CKPT_PATH="${LOCAL_VLM_CKPT_PATH:-}"

usage() {
  cat <<'EOF'
Usage: run_train_hl_memory_multitask_ddp.sh [options]

Options:
  --backbone NAME        qwen3_5_2b, qwen3_5_4b, qwen3_5_27b, qwen2_5_3b, qwen2_5
  --ckpt PATH            Local HuggingFace VLM checkpoint path.
  --output-dir PATH      Checkpoint output directory.
  --dataset-root PATH    HL memory multitask dataset root.
  --dataset-glob GLOB    Dataset glob under dataset root, default */train.
  --gpus IDS             CUDA_VISIBLE_DEVICES, for example 0,1,2,3,4,5,6,7.
  --nproc N              Number of torchrun processes.
  --steps N              Training steps.
  --save-interval N      Save interval.
  --batch-size N         Per-rank batch size.
  --grad-accum N         Gradient accumulation steps.
  --wandb                Enable wandb.
  -h, --help             Show this help.

Environment variables with the same names are also supported.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone) BACKBONE="$2"; shift 2 ;;
    --ckpt|--local-vlm-ckpt-path) LOCAL_VLM_CKPT_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --dataset-glob) DATASET_GLOB="$2"; shift 2 ;;
    --gpus|--cuda-visible-devices) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --nproc|--nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --steps|--num-train-steps) NUM_TRAIN_STEPS="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum|--grad-accum-steps) GRAD_ACCUM_STEPS="$2"; shift 2 ;;
    --wandb) WANDB_ENABLED="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

DATASET_ROOT="${DATASET_ROOT:-/root/Users/dataset/hl_memory/subtask}"
DATASET_GLOB="${DATASET_GLOB:-*/train}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29531}"

case "$BACKBONE" in
  qwen2_5|qwen2_5_3b|qwen25|qwen25_3b)
    VLM_BACKEND="${VLM_BACKEND:-qwen2_5_vl}"
    VLM_VARIANT="${VLM_VARIANT:-qwen2_5_3b}"
    LOCAL_VLM_CKPT_PATH="${LOCAL_VLM_CKPT_PATH:-/root/Users/lixiaotong/Qwen2.5-VL-3B-Instruct}"
    BACKBONE_TAG="qwen2_5_3b"
    ;;
  qwen3_5|qwen3_5_2b|qwen35|qwen35_2b)
    VLM_BACKEND="${VLM_BACKEND:-qwen3_5_vl}"
    VLM_VARIANT="${VLM_VARIANT:-qwen3_5_2b}"
    LOCAL_VLM_CKPT_PATH="${LOCAL_VLM_CKPT_PATH:-/root/Users/lixiaotong/Qwen3.5-2B}"
    BACKBONE_TAG="qwen3_5_2b"
    ;;
  qwen3_5_4b|qwen35_4b)
    VLM_BACKEND="${VLM_BACKEND:-qwen3_5_vl}"
    VLM_VARIANT="${VLM_VARIANT:-qwen3_5_4b}"
    LOCAL_VLM_CKPT_PATH="${LOCAL_VLM_CKPT_PATH:-/root/Users/lixiaotong/Qwen3.5-4B}"
    BACKBONE_TAG="qwen3_5_4b"
    ;;
  qwen3_5_27b|qwen35_27b)
    VLM_BACKEND="${VLM_BACKEND:-qwen3_5_vl}"
    VLM_VARIANT="${VLM_VARIANT:-qwen3_5_27b}"
    LOCAL_VLM_CKPT_PATH="${LOCAL_VLM_CKPT_PATH:-/root/Users/lixiaotong/Qwen3.5-27B}"
    BACKBONE_TAG="qwen3_5_27b"
    ;;
  *)
    echo "Unsupported --backbone: $BACKBONE" >&2
    usage >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/hl_memory_ckpts/subtask_multitask_${BACKBONE_TAG}_lora_ddp}"
PRECISION="${PRECISION:-bfloat16}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-3000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
FRAME_CACHE_SIZE="${FRAME_CACHE_SIZE:-4096}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-openpi-hl-memory}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-subtask_multitask_${BACKBONE_TAG}_lora_ddp}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

cd "$REPO_ROOT"

echo "[train] repo=$REPO_ROOT"
echo "[train] python=$PYTHON_BIN"
echo "[train] cuda_visible_devices=$CUDA_VISIBLE_DEVICES nproc_per_node=$NPROC_PER_NODE"
echo "[train] dataset_root=$DATASET_ROOT dataset_glob=$DATASET_GLOB"
echo "[train] output_dir=$OUTPUT_DIR"
echo "[train] global_batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "[train] num_train_steps=$NUM_TRAIN_STEPS save_interval=$SAVE_INTERVAL"

WANDB_FLAG="--no-wandb-enabled"
if [[ "$WANDB_ENABLED" == "1" || "$WANDB_ENABLED" == "true" || "$WANDB_ENABLED" == "TRUE" ]]; then
  WANDB_FLAG="--wandb-enabled"
fi

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
  --precision "$PRECISION" \
  --lora-enabled \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --lora-target-modules "$LORA_TARGET_MODULES" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --frame-cache-size "$FRAME_CACHE_SIZE" \
  "$WANDB_FLAG" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME"
