#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/activate_autodl.sh
set -u
PYTHON_PI05="${PYTHON_PI05:-/opt/pi05/.venv/bin/python}"
PORT="${PORT:-18085}"
ROUNDS="${ROUNDS:-6}"
CHUNK="${CHUNK:-5}"
OUT="${OUT:-$ROOT/eval_result/pi05_demo/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
test -x "$PYTHON_PI05" || { echo 'Missing pi05 environment'; exit 1; }
test -f /root/autodl-tmp/checkpoints/pi05_base/assets/bread_square_100_demo/norm_stats.json || python scripts/prepare_pi05_demo_stats.py
python - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.55
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
# Keep RoboTwin's CUDA 12.1 library path out of the JAX process: JAX uses its wheel CUDA libraries.
env -u LD_LIBRARY_PATH "$PYTHON_PI05" XPolicyLab/setup_policy_server.py \
  --config_path XPolicyLab/policy/Pi_05/deploy.yml \
  --overrides host=127.0.0.1 port="$PORT" task_name=place_bread_basket \
  env_cfg_type=aloha_agilex action_type=joint eval_batch=False \
  model_path=/root/autodl-tmp/checkpoints/pi05_base \
  train_config_name=pi05_base_aloha_full_sim_arx-x5_seed_0 repo_id=bread_square_100_demo \
  ws_ping_timeout_s=None > "$OUT/policy.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo "Loading pi05. Logs: $OUT/policy.log"
python - "$PORT" "$SERVER_PID" "$OUT/policy.log" <<'PY'
import os, socket, sys, time
port, pid = map(int, sys.argv[1:3])
for _ in range(600):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit('Policy server exited; see ' + sys.argv[3])
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('Policy server startup timed out; see ' + sys.argv[3])
PY
python scripts/pi05_closed_loop_demo.py --port "$PORT" --rounds "$ROUNDS" --chunk "$CHUNK" \
  --seed "${SEED:-100}" --output "$OUT" 2>&1 | tee "$OUT/rollout.log"
echo "Video: $OUT/demo.mp4"
echo "Report: $OUT/summary.json"
