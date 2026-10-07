#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/activate_autodl.sh"
set -u
python "${ROOT}/scripts/prepare_act_demo_subset.py"
cd "${ROOT}/XPolicyLab/policy/ACT"
bash process_data.sh bread_square_20 place_bread_basket aloha_agilex joint --is_sim
