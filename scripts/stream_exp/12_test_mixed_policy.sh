#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

set -u


# ============================================================
# Args
# ============================================================

POLICY="${1:?Usage: $0 POLICY HZ [LIGHT_EPOCH]}"

HZ="${2:?Usage: $0 POLICY HZ [LIGHT_EPOCH]}"

LIGHT_EPOCH="${3:-10}"


# ============================================================
# Paths
# ============================================================

LIGHT_CKPT="${LIGHT_CKPT_DIR}/checkpoint_epoch_${LIGHT_EPOCH}.pth"

OUT_DIR="outputs/mixed_policy_streaming/${LIGHT_EXP_NAME}_e${LIGHT_EPOCH}/${HZ}hz/${POLICY}"


# ============================================================
# Checks
# ============================================================

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"

require_file "${LIGHT_CFG}"
require_file "${LIGHT_CKPT}"

require_file "tools/test_mixed_policy_streaming.py"


# ============================================================
# Run
# ============================================================

banner "Mixed Policy Streaming | ${POLICY} | ${HZ} Hz"

echo "[FULL]"
echo "  ${FULL_CKPT}"
echo

echo "[LIGHT]"
echo "  ${LIGHT_CKPT}"
echo

echo "[POLICY]"
echo "  ${POLICY}"
echo

echo "[HZ]"
echo "  ${HZ}"
echo

echo "[OUTPUT]"
echo "  ${OUT_DIR}"
echo

mkdir -p "${OUT_DIR}"

python tools/test_mixed_policy_streaming.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --light_cfg "${LIGHT_CFG}" \
    --light_ckpt "${LIGHT_CKPT}" \
    --policy "${POLICY}" \
    --input_hz "${HZ}" \
    --warmup 20 \
    --workers 0 \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/console.log"
