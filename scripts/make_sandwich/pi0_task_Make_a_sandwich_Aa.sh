#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /root/Users/miniconda3/etc/profile.d/conda.sh
conda activate pi0_suzhou

REPO=/root/Users/lixiaotong/openpi
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$PYTHONPATH"

cfg=20260309K055Aa_Make_a_sandwich

# cp /home/liyang/.cache/openpi/big_vision/* /root/.cache/openpi/big_vision/

exp=original_jax   # 可单独覆写，默认与cfg一致

export HF_LEROBOT_HOME='/root/Users/dataset/lerobot_home'
export HF_DATASETS_CACHE="/root/Users/.cache/"
export WANDB_MODE=online

cd /root/Users/lixiaotong/openpi

# python scripts/compute_norm_stats.py --config-name "$exp"
  
# XLA_PYTHON_CLIENT_PREALLOCATE=false \
# XLA_PYTHON_CLIENT_ALLOCATOR=platform \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py "$cfg" \
  --exp-name "$exp" \
  --overwrite

