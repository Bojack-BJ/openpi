#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0,1
source /root/Users/miniconda3/etc/profile.d/conda.sh
conda activate pi0_suzhou

REPO=/root/Users/lixiaotong/openpi
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$PYTHONPATH"

cfg=Waste_sorting_Aa

# cp /home/liyang/.cache/openpi/big_vision/* /root/.cache/openpi/big_vision/

exp=original_torch   # 可单独覆写，默认与cfg一致

export HF_LEROBOT_HOME='/root/Users/dataset/lerobot_home'
export HF_DATASETS_CACHE="/root/Users/.cache/"
export WANDB_MODE=online

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
  --overwrite \
  --pytorch-training-precision "bfloat16" \
  --project_name "umi-openpi" \
  --fsdp_devices 2 \
  "$@"

