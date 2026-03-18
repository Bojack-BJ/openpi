#!/usr/bin/env bash
set -e

source /mnt/shared-storage-user/internvla/Users/liyang/env/miniconda3/etc/profile.d/conda.sh
conda activate f2

REPO=/mnt/shared-storage-user/internvla/Users/zhaxizhuoma/code/openpi
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$PYTHONPATH"

cfg=pi0_Fold_square_towels

# cp /home/liyang/.cache/openpi/big_vision/* /root/.cache/openpi/big_vision/

exp=${exp:-$cfg}   # 可单独覆写，默认与cfg一致

export HF_LEROBOT_HOME='/mnt/shared-storage-user/zhaxizhuoma/fastumi_data/lerobot_home'
export HF_DATASETS_CACHE="/mnt/shared-storage-user/internvla/Users/zhaxizhuoma/cache"
export WANDB_MODE=offline

cd /mnt/shared-storage-user/internvla/Users/zhaxizhuoma/code/openpi

python scripts/compute_norm_stats.py --config-name "$exp"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py "$cfg" \
  --exp-name "$exp" \
  --overwrite

