#!/usr/bin/env bash
set -eo pipefail
cd /root/autodl-tmp/RoboTwin
source scripts/activate_autodl.sh
set -u
OUT=/root/autodl-tmp/pi05_eval/step999_100seeds_v1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.55
export JAX_COMPILATION_CACHE_DIR=/opt/pi05/jax_eval_cache
export PYTHONPATH="$PWD:$PWD/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
export BREAD_DATA_VARIANT=default HF_HUB_OFFLINE=1
PORT=18086
env -u LD_LIBRARY_PATH /opt/pi05/.venv/bin/python scripts/pi05_eval_server.py \
  --checkpoint "$OUT/checkpoint" --port "$PORT" > "$OUT/policy.log" 2>&1 &
SERVER=$!
cleanup(){ kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT
python - "$PORT" "$SERVER" <<'PY'
import sys,os,time
from websockets.sync.client import connect
p,pid=map(int,sys.argv[1:])
for _ in range(600):
    os.kill(pid,0)
    try:
        with connect(f'ws://127.0.0.1:{p}',open_timeout=2):break
    except (OSError,TimeoutError):time.sleep(1)
else:raise RuntimeError('Policy startup timeout')
PY
python scripts/eval_pi05_batch.py --output "$OUT" --port "$PORT" --limit "${LIMIT:-100}" 2>&1 | tee -a "$OUT/evaluation.log"
if [[ "${LIMIT:-100}" == 100 ]]; then
  python scripts/finalize_pi05_eval.py | tee -a "$OUT/evaluation.log"
fi
