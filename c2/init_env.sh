#!/usr/bin/env bash
# AFAC c2 环境初始化：创建并配置 conda 环境 AFAC
# 用法： bash init_env.sh
set -e

ENV_NAME="AFAC"
PY_VER="3.11"

# 定位 conda
if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda，请先安装 Anaconda/Miniconda" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# 创建环境（已存在则跳过）
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "[init_env] 环境 ${ENV_NAME} 已存在，跳过创建"
else
  echo "[init_env] 创建 conda 环境 ${ENV_NAME} (python=${PY_VER})"
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

# 安装依赖
conda activate "${ENV_NAME}"
echo "[init_env] 安装 requirements.txt"
pip install -r "$(dirname "$0")/requirements.txt"

echo "[init_env] 完成。使用： conda activate ${ENV_NAME}"
