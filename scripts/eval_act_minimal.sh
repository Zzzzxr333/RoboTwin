#!/bin/bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/activate_autodl.sh"
set -u
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
RUN_DIR="${1:-$(cat "${ROOT}/setup_act_latest_run.txt")}"
PORT="${ACT_PORT:-18080}"
[[ -s "${RUN_DIR}/policy_last.ckpt" && -s "${RUN_DIR}/dataset_stats.pkl" ]]
export ACT_ACTION_DIM="$(bash XPolicyLab/utils/get_action_dim.sh "${ROOT}" aloha_agilex)"
LOG_DIR="${RUN_DIR}/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
# Refuse an occupied port so the evaluator cannot accidentally use another policy.
python - "${PORT}" <<'PY'
import socket,sys
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1',int(sys.argv[1])))
PY
python -u XPolicyLab/setup_policy_server.py \
  --config_path XPolicyLab/policy/ACT/deploy.yml \
  --overrides host=127.0.0.1 port="${PORT}" bench_name=RoboTwin \
  task_name=place_bread_basket ckpt_dir="${RUN_DIR}" ckpt_name="$(basename "${RUN_DIR}")" \
  env_cfg_type=aloha_agilex action_type=joint action_dim="${ACT_ACTION_DIM}" \
  temporal_agg=False seed=0 > "${LOG_DIR}/policy_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT
python - "${PORT}" "${SERVER_PID}" <<'PY'
import os,sys,time
from websockets.sync.client import connect
port,pid=map(int,sys.argv[1:])
for _ in range(120):
    os.kill(pid,0)
    try:
        with connect(f'ws://127.0.0.1:{port}',open_timeout=1):break
    except (OSError,TimeoutError):time.sleep(1)
else:raise TimeoutError('ACT policy server did not become ready')
PY
timeout 900 bash scripts/eval_policy.sh \
  --bench_name RoboTwin --task_name place_bread_basket \
  --task_config bread_square_20 --env_cfg_type aloha_agilex \
  --policy_name ACT --host 127.0.0.1 --port "${PORT}" --protocol ws \
  --device_id 0 --seed 100 --test_num 1 --expert_check true \
  --additional_info "action_type=joint,ckpt_name=$(basename "${RUN_DIR}")" \
  2>&1 | tee "${LOG_DIR}/eval.log"
printf 'Evaluation logs: %s\n' "${LOG_DIR}"
