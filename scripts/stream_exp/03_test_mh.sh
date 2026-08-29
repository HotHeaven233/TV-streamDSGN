#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

require_file "${FULL_CFG}"

START_EPOCH="${1:-1}"
END_EPOCH="${2:-5}"

banner "MH Residual Streaming Evaluation | epochs ${START_EPOCH}-${END_EPOCH}"

for E in $(seq "${START_EPOCH}" "${END_EPOCH}"); do

    CKPT="${MH_RETRAIN_CKPT_DIR}/checkpoint_epoch_${E}.pth"
    require_file "${CKPT}"

    OUT_DIR="outputs/${MH_RETRAIN_EXP_NAME}_stream_e${E}"
    mkdir -p "${OUT_DIR}"

    echo
    echo "============================================================"
    echo "STREAM TEST EPOCH ${E}"
    echo "CFG : ${FULL_CFG}"
    echo "CKPT: ${CKPT}"
    echo "OUT : ${OUT_DIR}"
    echo "============================================================"

    python tools/test_stream_buffer_timestamp.py \
        --cfg_file "${FULL_CFG}" \
        --ckpt "${CKPT}" \
        --input_hz 10 \
        --warmup 20 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/console.log"
done
