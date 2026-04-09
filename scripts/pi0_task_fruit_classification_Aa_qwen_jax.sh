#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7
source /root/Users/miniconda3/etc/profile.d/conda.sh
conda activate pi0_suzhou

REPO=/root/Users/lixiaotong/openpi
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$PYTHONPATH"

cfg=fruit_classification_Aa_qwen

# cp /home/liyang/.cache/openpi/big_vision/* /root/.cache/openpi/big_vision/

exp="fruit_classification_Aa_qwen_jax"

export HF_LEROBOT_HOME='/root/Users/dataset/lerobot_home'
export HF_DATASETS_CACHE="/root/Users/.cache/"
export WANDB_MODE=offline

cd /root/Users/lixiaotong/openpi

# python scripts/compute_norm_stats.py --config-name "$exp"
  
# XLA_PYTHON_CLIENT_PREALLOCATE=false \
# XLA_PYTHON_CLIENT_ALLOCATOR=platform \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py "$cfg" \
  --exp-name "$exp" \
  --project_name "umi-openpi" \
  --overwrite \
