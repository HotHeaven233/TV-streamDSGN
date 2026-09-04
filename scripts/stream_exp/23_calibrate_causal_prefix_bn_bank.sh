#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
RESUME="${3:-0}"
EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"
BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${RAW_CKPT}"
mkdir -p "$(dirname "${BANK}")"

ARGS=()
if [ "${RESUME}" = "1" ]; then
  ARGS+=(--resume)
fi

banner "Calibrate 203 causal-prefix BN states | epoch=${EPOCH} | N=${N}"

python tools/calibrate_causal_prefix_bn_bank.py \
  --full_cfg "${FULL_CFG}" \
  --full_ckpt "${FULL_CKPT}" \
  --elastic_ckpt "${RAW_CKPT}" \
  --output_bank "${BANK}" \
  --batches_per_prefix "${N}" \
  --workers 4 \
  --save_every 5 \
  "${ARGS[@]}"

echo "[BANK] ${BANK}"

