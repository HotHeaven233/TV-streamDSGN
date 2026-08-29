#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# Conda activation is complete.
set -u


# ============================================================
# Configuration
# ============================================================

# Light checkpoint epoch.
# Usage:
#
#   ./scripts/stream_exp/09_sweep_full_light_hz.sh
#
# defaults to epoch 10.
#
# Or:
#
#   ./scripts/stream_exp/09_sweep_full_light_hz.sh 8
#
LIGHT_EPOCH="${1:-10}"

# Frequencies to evaluate.
#
# Can also override:
#
# HZ_LIST="30 35 40" ./scripts/stream_exp/09_sweep_full_light_hz.sh 10
#
HZ_LIST="${HZ_LIST:-10 20 30 35 40 45 50}"

WARMUP="${WARMUP:-20}"

# If OVERWRITE=1, existing output directories are removed before rerun.
OVERWRITE="${OVERWRITE:-0}"


# ============================================================
# Check paths
# ============================================================

LIGHT_CKPT="${LIGHT_CKPT_DIR}/checkpoint_epoch_${LIGHT_EPOCH}.pth"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"

require_file "${LIGHT_CFG}"
require_file "${LIGHT_CKPT}"

require_file "tools/test_stream_buffer_timestamp.py"


# ============================================================
# Output root
# ============================================================

SWEEP_ROOT="outputs/frequency_sweep"

FULL_ROOT="${SWEEP_ROOT}/full"
LIGHT_ROOT="${SWEEP_ROOT}/${LIGHT_EXP_NAME}_e${LIGHT_EPOCH}"

mkdir -p "${FULL_ROOT}"
mkdir -p "${LIGHT_ROOT}"


# ============================================================
# Experiment information
# ============================================================

banner "Full / Light Streaming Frequency Sweep"

echo "[SWEEP] frequencies  : ${HZ_LIST}"
echo "[SWEEP] warmup       : ${WARMUP}"
echo
echo "[FULL]"
echo "  cfg  = ${FULL_CFG}"
echo "  ckpt = ${FULL_CKPT}"
echo
echo "[LIGHT]"
echo "  cfg  = ${LIGHT_CFG}"
echo "  ckpt = ${LIGHT_CKPT}"
echo "  epoch= ${LIGHT_EPOCH}"
echo
echo "[OUTPUT]"
echo "  root = ${SWEEP_ROOT}"
echo


# ============================================================
# Helper
# ============================================================

prepare_output_dir() {
    local out_dir="$1"

    if [ -d "${out_dir}" ]; then

        if [ "${OVERWRITE}" = "1" ]; then
            echo "[INFO] Removing existing output:"
            echo "       ${out_dir}"
            rm -rf "${out_dir}"
        else
            echo
            echo "[ERROR] Output directory already exists:"
            echo "        ${out_dir}"
            echo
            echo "To rerun and overwrite:"
            echo
            echo "OVERWRITE=1 $0 ${LIGHT_EPOCH}"
            echo
            exit 1
        fi
    fi

    mkdir -p "${out_dir}"
}


run_stream_test() {
    local mode="$1"
    local hz="$2"
    local cfg="$3"
    local ckpt="$4"
    local out_dir="$5"

    prepare_output_dir "${out_dir}"

    echo
    echo "======================================================================"
    echo "${mode} | ${hz} Hz"
    echo "======================================================================"
    echo "CFG  : ${cfg}"
    echo "CKPT : ${ckpt}"
    echo "HZ   : ${hz}"
    echo "OUT  : ${out_dir}"
    echo "======================================================================"

    python tools/test_stream_buffer_timestamp.py \
        --cfg_file "${cfg}" \
        --ckpt "${ckpt}" \
        --input_hz "${hz}" \
        --warmup "${WARMUP}" \
        --output_dir "${out_dir}" \
        2>&1 | tee "${out_dir}/console.log"
}


# ============================================================
# Full sweep
# ============================================================

banner "Sweep FULL"

for HZ in ${HZ_LIST}; do

    OUT_DIR="${FULL_ROOT}/${HZ}hz"

    run_stream_test \
        "FULL" \
        "${HZ}" \
        "${FULL_CFG}" \
        "${FULL_CKPT}" \
        "${OUT_DIR}"

done


# ============================================================
# Light sweep
# ============================================================

banner "Sweep LIGHT | epoch ${LIGHT_EPOCH}"

for HZ in ${HZ_LIST}; do

    OUT_DIR="${LIGHT_ROOT}/${HZ}hz"

    run_stream_test \
        "LIGHT-e${LIGHT_EPOCH}" \
        "${HZ}" \
        "${LIGHT_CFG}" \
        "${LIGHT_CKPT}" \
        "${OUT_DIR}"

done


# ============================================================
# Finish
# ============================================================

banner "Frequency Sweep Finished"

echo "Full results:"
echo "  ${FULL_ROOT}"
echo
echo "Light results:"
echo "  ${LIGHT_ROOT}"
echo

echo "Completed frequencies:"
for HZ in ${HZ_LIST}; do
    echo "  ${HZ} Hz"
done
