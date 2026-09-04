#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
BATCHES_PER_PREFIX="${2:-100}"
HZ="${3:-10}"
EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${BATCHES_PER_PREFIX}.pth"
require_file "${BANK}"

PROFILES=(
    "0.75,0.75,0.75,0.75,0.75,0.75"
    "0.5,0.5,0.5,0.5,0.5,0.5"
    "0.25,0.25,0.25,0.25,0.25,0.25"
    "1.0,1.0,0.75,0.5,0.5,0.25"
    "1.0,1.0,1.0,0.75,0.5,0.25"
    "1.0,0.75,0.5,0.5,0.25,0.25"
    "0.75,0.5,0.5,0.25,0.25,0.25"
)

for SCHEDULE in "${PROFILES[@]}"; do
    TAG="$(echo "${SCHEDULE}" | sed 's/1\.0/100/g; s/0\.75/075/g; s/0\.5/050/g; s/0\.25/025/g; s/,/_/g')"
    PROFILE_CKPT="outputs/elastic_bev/${EXP_NAME}/prefix_profiles/e${EPOCH}_n${BATCHES_PER_PREFIX}_${TAG}.pth"
    OUT_DIR="outputs/v4_prefix_bank/e${EPOCH}_n${BATCHES_PER_PREFIX}/${TAG}"

    "${SCRIPT_DIR}/24_materialize_prefix_profile.sh" \
        "${EPOCH}" "${BATCHES_PER_PREFIX}" "${SCHEDULE}"

    python tools/test_elastic_stream_v4_bn.py \
        --full_cfg "${FULL_CFG}" \
        --full_ckpt "${FULL_CKPT}" \
        --elastic_ckpt "${PROFILE_CKPT}" \
        --input_hz "${HZ}" \
        --warmup 20 \
        --mode elastic_fixed \
        --fixed_schedule "${SCHEDULE}" \
        --output_dir "${OUT_DIR}"
done

