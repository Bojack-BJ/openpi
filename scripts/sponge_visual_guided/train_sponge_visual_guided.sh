#!/usr/bin/env bash
set -e

# Usage:
#   bash scripts/sponge_visual_guided/train_sponge_visual_guided.sh qwen3_5
#   bash scripts/sponge_visual_guided/train_sponge_visual_guided.sh qwen2_5
#   bash scripts/sponge_visual_guided/train_sponge_visual_guided.sh pi0
#   bash scripts/sponge_visual_guided/train_sponge_visual_guided.sh pi05

variant=${1:-qwen3_5}

case "$variant" in
  pi0)    cfg=sponge_visual_guided_pi0 ;;
  pi05)   cfg=sponge_visual_guided_pi05 ;;
  qwen2_5 | qwen25) cfg=sponge_visual_guided_qwen2_5_3b_400m ;;
  qwen3_5 | qwen35) cfg=sponge_visual_guided_qwen3_5_2b_400m ;;
  *) echo "Unknown variant: $variant"; exit 2 ;;
esac

exp=${EXP_NAME:-$cfg}
repo=${REPO:-/root/Users/lixiaotong/openpi}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/root/Users/dataset/lerobot_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/root/Users/.cache/}
export WANDB_MODE=${WANDB_MODE:-online}
export PYTHONPATH="$repo/src:$repo/packages/openpi-client/src:$PYTHONPATH"

source /root/Users/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pi0_suzhou}"

cd "$repo"

# First run only, if assets/norm stats do not exist yet:
# python scripts/compute_norm_stats.py --config-name "$cfg" --num-workers 64

XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9} \
python scripts/train.py "$cfg" \
  --exp-name "$exp" \
  --project_name "${PROJECT_NAME:-umi-openpi}" \
  --fsdp_devices "${FSDP_DEVICES:-8}" \
  --jax-compilation-cache-dir "${JAX_COMPILATION_CACHE_DIR:-/root/Users/.cache/jax/openpi}" \
  --overwrite
