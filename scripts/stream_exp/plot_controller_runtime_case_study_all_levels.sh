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
WINDOW="${4:-50}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
PYTHON_SCRIPT="tools/plot_controller_runtime_case_study_all_levels.py"
OUT_DIR="outputs/paper_figures/controller_runtime_case_study_all_levels"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "[ERROR] missing: ${PYTHON_SCRIPT}"
    exit 2
fi

echo "============================================================"
echo "PLOT CONTROLLER / RUNTIME CASE STUDY (L1-L4)"
echo "============================================================"
echo "repo       : ${REPO_ROOT}"
echo "exp        : ${EXP_NAME}"
echo "Hz         : ${HZ}"
echo "levels     : L1,L2,L3,L4"
echo "fraction   : ${PRESSURE_FRACTION}"
echo "trace seed : ${TRACE_SEED}"
echo "window     : ${WINDOW}"
echo "policy     : prefer zero-miss windows automatically"
echo "output     : ${OUT_DIR}"
echo "============================================================"

python "${PYTHON_SCRIPT}" \
    --repo_root "${REPO_ROOT}" \
    --exp_name "${EXP_NAME}" \
    --hz "${HZ}" \
    --pressure_fraction "${PRESSURE_FRACTION}" \
    --trace_seed "${TRACE_SEED}" \
    --window "${WINDOW}" \
    --output_dir "${OUT_DIR}" \
    --dpi 400

echo
echo "============================================================"
echo "CONTROLLER / RUNTIME CASE STUDY ALL LEVELS COMPLETE"
echo "============================================================"
