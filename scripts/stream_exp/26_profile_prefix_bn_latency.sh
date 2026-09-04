#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
BATCHES_PER_PREFIX="${2:-100}"
FRAMES="${3:-160}"
SCHEDULE="${4:-0.25,0.25,0.25,0.25,0.25,0.25}"
MODE="${5:-fused}"
EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

TAG="$(echo "${SCHEDULE}" | sed 's/1\.0/100/g; s/0\.75/075/g; s/0\.5/050/g; s/0\.25/025/g; s/,/_/g')"
PROFILE_CKPT="outputs/elastic_bev/${EXP_NAME}/prefix_profiles/e${EPOCH}_n${BATCHES_PER_PREFIX}_${TAG}.pth"
OUT_JSON="outputs/v4_prefix_bank_latency/e${EPOCH}_n${BATCHES_PER_PREFIX}/${MODE}_${TAG}.json"

if [ ! -f "${PROFILE_CKPT}" ]; then
    "${SCRIPT_DIR}/24_materialize_prefix_profile.sh" \
        "${EPOCH}" "${BATCHES_PER_PREFIX}" "${SCHEDULE}"
fi
require_file "${PROFILE_CKPT}"

python tools/profile_causal_prefix_bn_latency.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${PROFILE_CKPT}" \
    --schedule "${SCHEDULE}" \
    --mode "${MODE}" \
    --warmup 20 \
    --frames "${FRAMES}" \
    --output "${OUT_JSON}"

