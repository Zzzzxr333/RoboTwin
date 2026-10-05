#!/usr/bin/env bash
# Source this file from a VS Code SSH terminal.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XDG_RUNTIME_DIR=/tmp/robotwin-runtime-root
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export PYTHONUNBUFFERED=1
export MAX_JOBS=6
export TORCH_CUDA_ARCH_LIST=8.6
cd /root/autodl-tmp/RoboTwin
