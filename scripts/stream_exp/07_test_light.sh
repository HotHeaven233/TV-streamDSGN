#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

require_file "${LIGHT_CFG}"
require_file "tools/test_stream_buffer_timestamp.py"

START_EPOCH="${1:-1}"
END_EPOCH="${2:-10}"

banner "Light-v1 Streaming Evaluation | epochs ${START_EPOCH}-${END_EPOCH}"

echo "[TEST] experiment = ${LIGHT_EXP_NAME}"
echo "[TEST] ckpt dir   = ${LIGHT_CKPT_DIR}"
echo "[TEST] input rate = 10 Hz"

for E in $(seq "${START_EPOCH}" "${END_EPOCH}"); do

    CKPT="${LIGHT_CKPT_DIR}/checkpoint_epoch_${E}.pth"

    require_file "${CKPT}"

    OUT_DIR="outputs/${LIGHT_EXP_NAME}_stream_e${E}"
    mkdir -p "${OUT_DIR}"

    echo
    echo "============================================================"
    echo "LIGHT STREAM TEST EPOCH ${E}"
    echo "CFG : ${LIGHT_CFG}"
    echo "CKPT: ${CKPT}"
    echo "OUT : ${OUT_DIR}"
    echo "============================================================"

    python tools/test_stream_buffer_timestamp.py \
        --cfg_file "${LIGHT_CFG}" \
        --ckpt "${CKPT}" \
        --input_hz 10 \
        --warmup 20 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/console.log"
done
