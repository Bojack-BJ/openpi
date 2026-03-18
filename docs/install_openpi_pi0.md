# OpenPI 安装说明（pi0 环境）

## 1. 创建并激活环境

```bash
conda create -n pi0 python=3.11 -y
conda activate pi0
```

## 2. 安装 `lerobot` 和 `dlimp`（固定 commit）

```bash
# 安装 lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
pip install .
cd ..

# 安装 dlimp
git clone https://github.com/kvablack/dlimp.git
cd dlimp
git checkout ad72ce3a9b414db2185bc0b38461d4101a65477a
pip install .
cd ..
```

说明：这一步可能出现 `pip` 依赖冲突警告（例如 `tensorflow/ml-dtypes/protobuf`），可先继续，后续以 `openpi` 安装结果为准。

## 3. 安装 `openpi-client`

```bash
cd openpi/packages/openpi-client
pip install .
cd ../..
```

## 4. 安装 `openpi` 主工程

```bash
cd openpi
pip install -e .
conda install -y ffmpeg=7.1.1 -c conda-forge
sudo ldconfig
```

## 5. 安装后测试

```bash
python scripts/compute_norm_stats.py --config-name debug
```

## 常见问题

### 1) 缺少 `chex`

报错示例：`ModuleNotFoundError: No module named 'chex'`

```bash
pip install chex==0.1.89
```

### 2) `datasets` 版本冲突（需补丁 lerobot）

该版本 lerobot 与 `datasets` 存在全区间不兼容问题（数据集由 `datasets 4.x` 写入）：

| datasets 版本 | 问题 |
|---|---|
| 2.x | 无法解析新 parquet schema，报 `TypeError: must be called with a dataclass type` |
| 3.x | 不认识 `List` feature type，报 `Feature type 'List' not found` |
| 4.x | `dataset["col"]` 返回 `Column` 对象，报 `torch.stack(): not a tuple of Tensors` |

**必须** 保持 `datasets 4.x` 并补丁已安装的 lerobot（共 4 处）：

```bash
# 确保 datasets 4.x
pip install "datasets>=4.0.0,<5.0.0"

# 修复已安装的 lerobot（不动项目源码）
LEROBOT_FILE=/root/code/miniconda3/envs/openpi_dyh/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py

sed -i \
  's/timestamps = torch\.stack(self\.hf_dataset\["timestamp"\])\.numpy()/timestamps = torch.tensor(self.hf_dataset["timestamp"]).numpy()/' \
  "$LEROBOT_FILE"

sed -i \
  's/episode_indices = torch\.stack(self\.hf_dataset\["episode_index"\])\.numpy()/episode_indices = torch.tensor(self.hf_dataset["episode_index"]).numpy()/' \
  "$LEROBOT_FILE"

sed -i \
  's/query_timestamps\[key\] = torch\.stack(timestamps)\.tolist()/query_timestamps[key] = torch.tensor(timestamps).tolist()/' \
  "$LEROBOT_FILE"

sed -i \
  's/key: torch\.stack(self\.hf_dataset\.select(q_idx)\[key\])/key: torch.stack([torch.as_tensor(x) for x in self.hf_dataset.select(q_idx)[key]])/' \
  "$LEROBOT_FILE"
```

### 3) 分片错误（batch 不能被 8 整除）

报错示例：
`ValueError: ... NamedSharding ... dimension 0 should be divisible by 8, but it is 2`

这不是安装失败，而是多卡分片与小 batch 不匹配。先用单卡验证：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/compute_norm_stats.py --config-name debug
```

### 4) `pip install -e .` 提示 `License file does not exist: LICENSE`

根因是 `pyproject.toml` 的 license 文件路径与仓库实际文件名不一致。确保仓库根目录存在 `LICENSE`，或在 `pyproject.toml` 中将 license 路径改为实际文件名（例如 `LICENSE.txt`）。
