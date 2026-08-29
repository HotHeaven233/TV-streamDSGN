#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

set -u


# ============================================================
# Settings
# ============================================================

LIGHT_EPOCH="${LIGHT_EPOCH:-10}"

# 建议改成你 frequency sweep 的 crossover 周围。
HZ_LIST="${HZ_LIST:-35 40 45 50}"

POLICIES="${POLICIES:-F FFFL FFL FL FFLLL FLL FLLL FLLLL L}"

SKIP_EXISTING="${SKIP_EXISTING:-1}"


# ============================================================
# Paths
# ============================================================

LIGHT_CKPT="${LIGHT_CKPT_DIR}/checkpoint_epoch_${LIGHT_EPOCH}.pth"

ROOT="outputs/mixed_policy_streaming/${LIGHT_EXP_NAME}_e${LIGHT_EPOCH}"


# ============================================================
# Check
# ============================================================

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"

require_file "${LIGHT_CFG}"
require_file "${LIGHT_CKPT}"

require_file "tools/test_mixed_policy_streaming.py"


# ============================================================
# Print
# ============================================================

banner "Fixed Mixed-Policy Sweep"

echo "HZ_LIST:"
echo "  ${HZ_LIST}"
echo

echo "POLICIES:"
echo "  ${POLICIES}"
echo

echo "LIGHT_EPOCH:"
echo "  ${LIGHT_EPOCH}"
echo

echo "ROOT:"
echo "  ${ROOT}"
echo


# ============================================================
# Sweep
# ============================================================

for HZ in ${HZ_LIST}; do

    for POLICY in ${POLICIES}; do

        OUT_DIR="${ROOT}/${HZ}hz/${POLICY}"

        SUMMARY="${OUT_DIR}/summary.json"

        echo
        echo "======================================================================"
        echo "HZ=${HZ} | POLICY=${POLICY}"
        echo "======================================================================"

        if [[ "${SKIP_EXISTING}" = "1" && -f "${SUMMARY}" ]]; then
            echo "[SKIP] existing summary:"
            echo "       ${SUMMARY}"
            continue
        fi

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

    done

done


banner "Mixed Policy Sweep Finished"

echo "Results:"
echo "  ${ROOT}"
