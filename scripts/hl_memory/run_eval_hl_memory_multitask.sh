#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/Users/donggaoqi/openpi_vlm_finetune}"
PYTHON_BIN="${PYTHON_BIN:-/root/Users/miniconda3/envs/hl_qwen35/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/root/Users/dataset/hl_memory/subtask}"
DATASET_GLOB="${DATASET_GLOB:-*/val}"
BACKBONE="${BACKBONE:-qwen2_5}"
MODEL_PATH="${MODEL_PATH:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
MAX_TASKS="${MAX_TASKS:-}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-}"
EVAL_MODES="${EVAL_MODES:-sample_context}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/hl_memory_ckpts/eval}"

usage() {
  cat <<'EOF'
Usage: run_eval_hl_memory_multitask.sh --ckpt PATH [options]

Options:
  --ckpt PATH            LoRA checkpoint directory, e.g. checkpoint-step-001500.
  --backbone NAME        qwen2_5, qwen2_5_3b, qwen3_5_2b, qwen3_5_4b, qwen3_5_27b.
  --dataset-root PATH    HL memory dataset root.
  --dataset-glob GLOB    Dataset glob, default */val.
  --gpu ID               Single GPU id for eval.
  --max-samples N        Limit total samples for quick qualitative checks.
  --max-tasks N          Limit number of task dirs.
  --max-episodes-per-task N
  --eval-modes MODES     sample_context, full, or comma-separated ablations.
  --out-dir PATH         Directory for metrics and predictions.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt|--model-path) MODEL_PATH="$2"; shift 2 ;;
    --backbone) BACKBONE="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --dataset-glob) DATASET_GLOB="$2"; shift 2 ;;
    --gpu) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --max-tasks) MAX_TASKS="$2"; shift 2 ;;
    --max-episodes-per-task) MAX_EPISODES_PER_TASK="$2"; shift 2 ;;
    --eval-modes) EVAL_MODES="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "--ckpt is required." >&2
  usage >&2
  exit 2
fi

case "$BACKBONE" in
  qwen2_5|qwen2_5_3b|qwen25|qwen25_3b)
    VLM_BACKEND="qwen2_5_vl"
    VLM_VARIANT="qwen2_5_3b"
    BACKBONE_TAG="qwen2_5_3b"
    ;;
  qwen3_5|qwen3_5_2b|qwen35|qwen35_2b)
    VLM_BACKEND="qwen3_5_vl"
    VLM_VARIANT="qwen3_5_2b"
    BACKBONE_TAG="qwen3_5_2b"
    ;;
  qwen3_5_4b|qwen35_4b)
    VLM_BACKEND="qwen3_5_vl"
    VLM_VARIANT="qwen3_5_4b"
    BACKBONE_TAG="qwen3_5_4b"
    ;;
  qwen3_5_27b|qwen35_27b)
    VLM_BACKEND="qwen3_5_vl"
    VLM_VARIANT="qwen3_5_27b"
    BACKBONE_TAG="qwen3_5_27b"
    ;;
  *)
    echo "Unsupported --backbone: $BACKBONE" >&2
    usage >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

CKPT_NAME="$(basename "$MODEL_PATH")"
RUN_NAME="${BACKBONE_TAG}_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S)"
METRICS_JSON="$OUT_DIR/${RUN_NAME}_metrics.json"
PREDICTIONS_JSONL="$OUT_DIR/${RUN_NAME}_predictions.jsonl"

EXTRA_ARGS=()
if [[ -n "$MAX_TASKS" ]]; then
  EXTRA_ARGS+=(--max-tasks "$MAX_TASKS")
fi
if [[ -n "$MAX_EPISODES_PER_TASK" ]]; then
  EXTRA_ARGS+=(--max-episodes-per-task "$MAX_EPISODES_PER_TASK")
fi

echo "[eval] gpu=$CUDA_VISIBLE_DEVICES backbone=$BACKBONE_TAG"
echo "[eval] ckpt=$MODEL_PATH"
echo "[eval] dataset=$DATASET_ROOT/$DATASET_GLOB max_samples=$MAX_SAMPLES modes=$EVAL_MODES"
echo "[eval] metrics=$METRICS_JSON"
echo "[eval] predictions=$PREDICTIONS_JSONL"

"$PYTHON_BIN" scripts/hl_memory/eval_hl_memory_multitask.py \
  --dataset-root "$DATASET_ROOT" \
  --dataset-glob "$DATASET_GLOB" \
  --model-path "$MODEL_PATH" \
  --vlm-backend "$VLM_BACKEND" \
  --vlm-variant "$VLM_VARIANT" \
  --precision bfloat16 \
  --eval-modes "$EVAL_MODES" \
  --max-samples "$MAX_SAMPLES" \
  --output-json "$METRICS_JSON" \
  --predictions-jsonl "$PREDICTIONS_JSONL" \
  "${EXTRA_ARGS[@]}"
