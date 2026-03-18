#!/usr/bin/env bash
set -e

source /root/code/miniconda3/etc/profile.d/conda.sh
conda activate openpi_dyh

REPO=fastumi/qwen_test
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$PYTHONPATH"

cfg=qwen

# cp /home/liyang/.cache/openpi/big_vision/* /root/.cache/openpi/big_vision/

exp=${exp:-$cfg}   # 可单独覆写，默认与cfg一致

export HF_LEROBOT_HOME='/root/.cache/huggingface/lerobot/'
export HF_DATASETS_CACHE="/root/.cache/"
export WANDB_MODE=offline

cd /lumos-vePFS/suda/duanyuhui/openpi_dyh

python scripts/compute_norm_stats.py --config-name "$exp"
  
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py "$cfg" \
  --exp-name "$exp" \
  --overwrite

