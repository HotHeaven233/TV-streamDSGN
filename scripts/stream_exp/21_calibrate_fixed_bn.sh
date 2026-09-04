#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
BATCHES="${2:-1000}"
SCHEDULE="${3:-0.25,0.25,0.25,0.25,0.25,0.25}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"

TAG="$(echo "${SCHEDULE}" | sed 's/1\.0/100/g; s/0\.75/075/g; s/0\.5/050/g; s/0\.25/025/g; s/,/_/g')"
OUT_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}_bncal_fixed_${TAG}.pth"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${RAW_CKPT}"

banner "Fixed-profile BN calibration | ${SCHEDULE}"

echo "[BASE]      ${FULL_CKPT}"
echo "[RAW]       ${RAW_CKPT}"
echo "[SCHEDULE]  ${SCHEDULE}"
echo "[BATCHES]   ${BATCHES}"
echo "[OUT]       ${OUT_CKPT}"
echo "[DATA]      train split only"

python tools/calibrate_elastic_v4_bn_fixed_profile.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${RAW_CKPT}" \
    --output_ckpt "${OUT_CKPT}" \
    --schedule "${SCHEDULE}" \
    --batches "${BATCHES}" \
    --workers 4

echo
echo "[DONE] calibrated checkpoint:"
echo "  ${OUT_CKPT}"
echo
echo "Test this exact profile with:"
echo "python tools/test_elastic_stream_v4_bn.py \\"
echo "  --full_cfg \"${FULL_CFG}\" \\"
echo "  --full_ckpt \"${FULL_CKPT}\" \\"
echo "  --elastic_ckpt \"${OUT_CKPT}\" \\"
echo "  --input_hz 10 --warmup 20 \\"
echo "  --mode elastic_fixed \\"
echo "  --fixed_schedule \"${SCHEDULE}\" \\"
echo "  --output_dir \"outputs/v4_bn_fixedcal_${TAG}\""

