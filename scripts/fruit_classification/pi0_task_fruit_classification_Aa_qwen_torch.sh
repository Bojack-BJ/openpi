#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /root/Users/miniconda3/etc/profile.d/conda.sh
conda activate pi0_suzhou

REPO=/root/Users/lixiaotong/openpi
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$PYTHONPATH"


cfg=fruit_classification_Aa_qwen
exp="${exp:-$cfg}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/Users/dataset/lerobot_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/Users/.cache/}"
export WANDB_MODE="${WANDB_MODE:-online}"

cd /root/Users/lixiaotong/openpi

# python scripts/compute_norm_stats.py --config-name "$exp"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
  nproc_per_node="${NPROC_PER_NODE:-${#gpu_ids[@]}}"
else
  nproc_per_node="${NPROC_PER_NODE:-1}"
fi

torchrun \
  --standalone \
  --nnodes="${NNODES:-1}" \
  --nproc_per_node="$nproc_per_node" \
  scripts/train_pytorch.py "$cfg" \
  --exp-name "$exp" \
  --resume \
  --pytorch-training-precision "bfloat16" \
  --project_name "umi-openpi" \
  "$@"
