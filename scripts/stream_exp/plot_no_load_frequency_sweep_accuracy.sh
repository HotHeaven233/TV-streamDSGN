#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR}/00_env.sh"

cd "${REPO_ROOT}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
OUT_DIR="outputs/paper_figures/frequency_sweep_accuracy"
PYTHON_SCRIPT="tools/plot_no_load_frequency_sweep_accuracy.py"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "[ERROR] missing: ${PYTHON_SCRIPT}"
    exit 2
fi

echo "============================================================"
echo "PLOT FORMAL NO-LOAD FREQUENCY SWEEP ACCURACY"
echo "============================================================"
echo "repo       : ${REPO_ROOT}"
echo "exp        : ${EXP_NAME}"
echo "frequencies: 35,40,45,50 Hz"
echo "methods    :"
echo "  - Original StreamDSGN"
echo "  - Streamer-style StreamDSGN"
echo "  - MTD Three-Head"
echo "  - Transtreaming-style"
echo "  - TV-streamDSGN"
echo "output     : ${OUT_DIR}"
echo "figure     : Car / Pedestrian / Cyclist / Macro"
echo "line style : all solid"
echo "============================================================"

python "${PYTHON_SCRIPT}" \
    --repo_root "${REPO_ROOT}" \
    --exp_name "${EXP_NAME}" \
    --output_dir "${OUT_DIR}" \
    --dpi 400

echo
echo "============================================================"
echo "FREQUENCY SWEEP ACCURACY FIGURE COMPLETE"
echo "============================================================"
