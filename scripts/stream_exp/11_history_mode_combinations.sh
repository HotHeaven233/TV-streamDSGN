#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# Conda activation finished.
set -u


# ============================================================
# Settings
# ============================================================

LIGHT_EPOCH="${1:-10}"

ANCHOR_STRIDE="${ANCHOR_STRIDE:-1}"

SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

LIGHT_CKPT="${LIGHT_CKPT_DIR}/checkpoint_epoch_${LIGHT_EPOCH}.pth"

OUT_DIR="outputs/history_mode_combinations/${LIGHT_EXP_NAME}_e${LIGHT_EPOCH}_stride${ANCHOR_STRIDE}"


# ============================================================
# Checks
# ============================================================

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"

require_file "${LIGHT_CFG}"
require_file "${LIGHT_CKPT}"

require_file "tools/test_counterfactual_state_value.py"
require_file "tools/test_history_mode_combinations.py"


# ============================================================
# Print
# ============================================================

banner "Cross-Mode History Compatibility"

echo "[FULL]"
echo "  cfg  = ${FULL_CFG}"
echo "  ckpt = ${FULL_CKPT}"
echo

echo "[LIGHT]"
echo "  cfg   = ${LIGHT_CFG}"
echo "  ckpt  = ${LIGHT_CKPT}"
echo "  epoch = ${LIGHT_EPOCH}"
echo

echo "[PROTOCOL]"
echo "  Current frame = ALWAYS Light"
echo
echo "  History order:"
echo "      [oldest, middle, newest]"
echo "      [t-3,    t-2,    t-1]"
echo
echo "  Patterns:"
echo "      LLL"
echo "      FLL"
echo "      LFL"
echo "      LLF"
echo "      FFL"
echo "      FLF"
echo "      LFF"
echo "      FFF"
echo
echo "  anchor_stride = ${ANCHOR_STRIDE}"
echo "  no timing"
echo "  no buffer"
echo "  no dropped frames"
echo

echo "[OUTPUT]"
echo "  ${OUT_DIR}"
echo


# ============================================================
# Optional prediction saving
# ============================================================

EXTRA_ARGS=()

if [ "${SAVE_PREDICTIONS}" = "1" ]; then
    EXTRA_ARGS+=(
        --save_predictions
    )
fi


# ============================================================
# Run
# ============================================================

mkdir -p "${OUT_DIR}"

python tools/test_history_mode_combinations.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --light_cfg "${LIGHT_CFG}" \
    --light_ckpt "${LIGHT_CKPT}" \
    --anchor_stride "${ANCHOR_STRIDE}" \
    --workers 0 \
    --output_dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/console.log"
