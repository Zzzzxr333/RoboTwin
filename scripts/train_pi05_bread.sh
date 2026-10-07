#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_LEROBOT_HOME=/root/autodl-tmp/pi05_data
export HF_DATASETS_CACHE=/root/autodl-tmp/pi05_data/hf_cache
export JAX_COMPILATION_CACHE_DIR=/opt/pi05/jax_cache
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
unset LD_LIBRARY_PATH
PY=/opt/pi05/.venv/bin/python
RUN_DIR="${RUN_DIR:-/opt/pi05/checkpoints/bread_lora_4090}"
STEPS="${STEPS:-1000}"
mkdir -p /opt/pi05/logs
test -f "$HF_LEROBOT_HOME/local/bread_square_100/conversion_manifest.json" || "$PY" scripts/prepare_pi05_training_data.py
extra=()
resume="${RESUME:-auto}"
if [[ "$resume" == 1 || ( "$resume" == auto && -d "$RUN_DIR" ) ]]; then extra+=(--resume); fi
"$PY" scripts/train_pi05_bread.py --steps "$STEPS" --batch-size "${BATCH_SIZE:-1}" \
    --run-dir "$RUN_DIR" "${extra[@]}" 2>&1 | tee "/opt/pi05/logs/$(basename "$RUN_DIR")_$(date +%Y%m%d_%H%M%S).log"
