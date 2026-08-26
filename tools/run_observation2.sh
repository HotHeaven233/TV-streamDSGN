#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

ROI_SIZE="${ROI_SIZE:-160}"
AGE_CAP="${AGE_CAP:-8}"
CENTER_STRATEGY="${CENTER_STRATEGY:-oracle}"
PRINT_EVERY="${PRINT_EVERY:-200}"
FORCE="${FORCE:-0}"

OUT_ROOT="${OUT_ROOT:-outputs/single_roi}"

RUN_DIR="${RUN_DIR:-${OUT_ROOT}/observation2_staleness_${ROI_SIZE}}"
FIG_DIR="${FIG_DIR:-${OUT_ROOT}/observation2_fig2}"

STATS_JSON="${RUN_DIR}/observation2_staleness.json"

echo "============================================"
echo "Observation 2"
echo "Feature staleness under temporal reuse"
echo "============================================"
echo "repo            : ${ROOT_DIR}"
echo "python          : ${PYTHON_BIN}"
echo "ROI size        : ${ROI_SIZE}x${ROI_SIZE}"
echo "age cap         : ${AGE_CAP}"
echo "center strategy : ${CENTER_STRATEGY}"
echo "outside policy  : reuse"
echo "run directory   : ${RUN_DIR}"
echo "figure directory: ${FIG_DIR}"
echo "force           : ${FORCE}"
echo

if [[ "${FORCE}" != "1" && -f "${STATS_JSON}" ]]; then

    echo "[SKIP] Found existing staleness statistics:"
    echo "       ${STATS_JSON}"

else

    echo
    echo "============================================"
    echo "Running Observation-2 diagnostic"
    echo "============================================"

    mkdir -p "${RUN_DIR}"

    "${PYTHON_BIN}" -u \
        tools/eval_single_roi_bev_semantic.py \
        --center-strategy "${CENTER_STRATEGY}" \
        --outside-mode reuse \
        --roi-h "${ROI_SIZE}" \
        --roi-w "${ROI_SIZE}" \
        --age-cap "${AGE_CAP}" \
        --collect-staleness \
        --print-every "${PRINT_EVERY}" \
        --output "${RUN_DIR}" \
        2>&1 | tee "${RUN_DIR}/run.log"

    if [[ ! -f "${STATS_JSON}" ]]; then
        echo
        echo "[ERROR] Observation-2 statistics were not generated:"
        echo "        ${STATS_JSON}"
        exit 1
    fi

fi

echo
echo "============================================"
echo "Plotting Figure 2"
echo "============================================"

mkdir -p "${FIG_DIR}"

"${PYTHON_BIN}" -u \
    tools/plot_fig2.py \
    --input "${STATS_JSON}" \
    --outdir "${FIG_DIR}"

echo
echo "============================================"
echo "Observation 2 complete"
echo "============================================"
echo
echo "Raw statistics:"
echo "  ${STATS_JSON}"
echo
echo "Paper data:"
echo "  ${FIG_DIR}/observation2_exact.csv"
echo "  ${FIG_DIR}/observation2_grouped.csv"
echo "  ${FIG_DIR}/observation2_grouped_table.tex"
echo
echo "Figures:"
echo "  ${FIG_DIR}/fig2a_staleness_error.pdf"
echo "  ${FIG_DIR}/fig2a_staleness_error.png"
echo "  ${FIG_DIR}/fig2b_staleness_distribution.pdf"
echo "  ${FIG_DIR}/fig2b_staleness_distribution.png"
echo
