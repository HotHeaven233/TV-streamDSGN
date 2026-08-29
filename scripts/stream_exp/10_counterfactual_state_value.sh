#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# Conda activation already finished.
set -u


# ============================================================
# Experiment settings
# ============================================================

LIGHT_EPOCH="${1:-10}"

ANCHOR_STRIDE="${ANCHOR_STRIDE:-1}"

MAX_HORIZON="${MAX_HORIZON:-4}"

HISTORY_LEN=3

LIGHT_CKPT="${LIGHT_CKPT_DIR}/checkpoint_epoch_${LIGHT_EPOCH}.pth"

OUT_DIR="outputs/counterfactual_state_value/${LIGHT_EXP_NAME}_e${LIGHT_EPOCH}_stride${ANCHOR_STRIDE}"


# ============================================================
# Check
# ============================================================

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"

require_file "${LIGHT_CFG}"
require_file "${LIGHT_CKPT}"

require_file "tools/test_counterfactual_state_value.py"


# ============================================================
# Print
# ============================================================

banner "Counterfactual Temporal State Value"

echo "[FULL]"
echo "  cfg  = ${FULL_CFG}"
echo "  ckpt = ${FULL_CKPT}"
echo

echo "[LIGHT]"
echo "  cfg   = ${LIGHT_CFG}"
echo "  ckpt  = ${LIGHT_CKPT}"
echo "  epoch = ${LIGHT_EPOCH}"
echo

echo "[EXPERIMENT]"
echo "  history_len   = ${HISTORY_LEN}"
echo "  max_horizon   = ${MAX_HORIZON}"
echo "  anchor_stride = ${ANCHOR_STRIDE}"
echo

echo "[CAUSAL INTERVENTION]"
echo
echo "  common L memory -> H_t^F -> L -> L -> L -> L"
echo "  common L memory -> H_t^L -> L -> L -> L -> L"
echo
echo "  no timing"
echo "  no buffer"
echo "  no dropped frames"
echo "  identical future inputs"
echo

echo "[OUTPUT]"
echo "  ${OUT_DIR}"
echo


# ============================================================
# Run
# ============================================================

mkdir -p "${OUT_DIR}"

python tools/test_counterfactual_state_value.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --light_cfg "${LIGHT_CFG}" \
    --light_ckpt "${LIGHT_CKPT}" \
    --history_len "${HISTORY_LEN}" \
    --max_horizon "${MAX_HORIZON}" \
    --anchor_stride "${ANCHOR_STRIDE}" \
    --workers 0 \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/console.log"
