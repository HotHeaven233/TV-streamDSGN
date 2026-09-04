#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
SCHEDULE="${3:-0.25,0.25,0.25,0.25,0.25,0.25}"
EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"
BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"
TAG="$(echo "${SCHEDULE}" | sed 's/1\.0/100/g; s/0\.75/075/g; s/0\.5/050/g; s/0\.25/025/g; s/,/_/g')"
OUT_CKPT="outputs/elastic_bev/${EXP_NAME}/prefix_profiles/e${EPOCH}_n${N}_${TAG}.pth"

require_file "${RAW_CKPT}"
require_file "${BANK}"
mkdir -p "$(dirname "${OUT_CKPT}")"

python tools/materialize_causal_prefix_bn_profile.py \
  --elastic_ckpt "${RAW_CKPT}" \
  --prefix_bn_bank "${BANK}" \
  --schedule "${SCHEDULE}" \
  --output_ckpt "${OUT_CKPT}"

echo "[PROFILE_CKPT] ${OUT_CKPT}"

