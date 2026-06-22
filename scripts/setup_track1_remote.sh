#!/bin/bash
# 赛道一服务器端环境配置脚本
# 由本地 PowerShell 上传并执行

set -e

PROJECT_DIR="/root/jittor_competition"
CONDA_DIR="$HOME/miniconda3"
CUDA_DIR="/usr/local/cuda-12.4"

export DEBIAN_FRONTEND=noninteractive

echo "===== [1/9] 系统基础包 ====="
apt-get update -qq
apt-get install -y -qq build-essential wget git curl vim tmux

echo "===== [2/9] 安装 CUDA Toolkit 12.4 ====="
if ! command -v nvcc &> /dev/null; then
    cd /tmp
    rm -f cuda-keyring_1.1-1_all.deb
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4
else
    echo "nvcc 已存在，跳过 CUDA 安装"
fi

# 写入环境变量（幂等）
if ! grep -q "/usr/local/cuda-12.4/bin" "$HOME/.bashrc"; then
    echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> "$HOME/.bashrc"
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> "$HOME/.bashrc"
fi
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

nvcc --version

echo "===== [3/9] 安装 Miniconda ====="
if [ ! -d "$CONDA_DIR" ]; then
    cd /tmp
    rm -f Miniconda3-latest-Linux-x86_64.sh
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/bin/activate"

echo "===== [4/9] 创建 conda 环境 jittor_track1 ====="
if ! conda env list | grep -q "jittor_track1"; then
    conda create -n jittor_track1 python=3.10 -y
fi

echo "===== [5/9] 安装 Python 依赖 ====="
conda activate jittor_track1
pip install -q --upgrade pip
pip install -q jittor==1.3.9.13 pandas tqdm PyYAML einops networkx scikit-learn

echo "===== [6/9] 安装 JittorGeometric ====="
cd "$PROJECT_DIR/JittorGeometric"
pip install -q -e .

echo "===== [7/9] 建立 track1/data 软链接 ====="
cd "$PROJECT_DIR/track1"
rm -f data
ln -s ../data_A data

echo "===== [8/9] 验证 Jittor GPU ====="
cd "$PROJECT_DIR"
python - <<'PY'
import jittor as jt
jt.flags.use_cuda = 1
print('Jittor version:', jt.__version__)
print('CUDA available:', jt.has_cuda)
a = jt.array([1.0, 2.0, 3.0])
b = a * 2
print('GPU compute test:', b.data)
PY

echo "===== [9/9] baseline 5 epoch 验证（dataset1） ====="
cd "$PROJECT_DIR"
python craft_baseline/main.py \
  --dataset dataset1 \
  --data_dir track1/data \
  --save_dir track1/checkpoints \
  --output_dir track1/outputs \
  --epochs 5 \
  --batch_size 200 \
  --early_stop 3

echo ""
echo "===== 全部完成 ====="
echo "项目目录: $PROJECT_DIR"
echo "conda 环境: jittor_track1"
echo "验证结果在: $PROJECT_DIR/track1/outputs/"
