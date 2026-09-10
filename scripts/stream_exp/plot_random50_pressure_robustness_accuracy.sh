#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR}/00_env.sh"

cd "${REPO_ROOT}"

HZ="${1:-35}"
TRACE_SEED="${2:-20260903}"
PRESSURE_FRACTION="${3:-0.5}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

PYTHON_SCRIPT="tools/plot_random50_pressure_robustness_accuracy.py"

OUT_DIR="outputs/paper_figures/random50_pressure_robustness_accuracy"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "[ERROR] missing: ${PYTHON_SCRIPT}"
    exit 2
fi

echo "============================================================"
echo "PLOT RANDOM50 PRESSURE ROBUSTNESS ACCURACY"
echo "============================================================"
echo "repo       : ${REPO_ROOT}"
echo "exp        : ${EXP_NAME}"
echo "Hz         : ${HZ}"
echo "pressures  : L1,L2,L3,L4"
echo "fraction   : ${PRESSURE_FRACTION}"
echo "trace seed : ${TRACE_SEED}"
echo "methods    :"
echo "  - Original StreamDSGN"
echo "  - Streamer-style StreamDSGN"
echo "  - MTD Three-Head"
echo "  - Transtreaming-style"
echo "  - TV-streamDSGN"
echo "figure     : Car / Pedestrian / Cyclist / Macro"
echo "line style : all solid"
echo "TV color   : red"
echo "Tran color : purple"
echo "output     : ${OUT_DIR}"
echo "============================================================"

python "${PYTHON_SCRIPT}" \
    --repo_root "${REPO_ROOT}" \
    --exp_name "${EXP_NAME}" \
    --hz "${HZ}" \
    --trace_seed "${TRACE_SEED}" \
    --pressure_fraction "${PRESSURE_FRACTION}" \
    --output_dir "${OUT_DIR}" \
    --dpi 400

echo
echo "============================================================"
echo "RANDOM50 PRESSURE ROBUSTNESS FIGURE COMPLETE"
echo "============================================================"
