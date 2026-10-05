#!/bin/bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/activate_autodl.sh"
set -u
cd "${ROOT}/XPolicyLab/policy/ACT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export ACT_ACTION_DIM="$(bash ../../utils/get_action_dim.sh "${ROOT}" aloha_agilex)"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
RUN_DIR="${RUN_DIR:-${ROOT}/XPolicyLab/policy/ACT/checkpoints/minimal_bread_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "${RUN_DIR}" ]]; then
    echo "Refusing to overwrite existing run: ${RUN_DIR}" >&2
    exit 1
fi
mkdir -p "${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" > "${ROOT}/setup_act_latest_run.txt"
python -u imitate_episodes.py \
  --bench_name bread_square_20 --task_name place_bread_basket \
  --ckpt_setting bread_square_20-place_bread_basket-aloha_agilex-joint \
  --ckpt_dir "${RUN_DIR}" --policy_class ACT --kl_weight 10 \
  --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
  --batch_size "${BATCH_SIZE}" --num_epochs "${NUM_EPOCHS}" \
  --lr 1e-5 --save_freq 10 --seed 0 2>&1 | tee "${RUN_DIR}/train.log"
