#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)"

REPO_ROOT="$(
    cd "${SCRIPT_DIR}/../.." && pwd
)"

cd "${REPO_ROOT}"

source scripts/stream_exp/00_env.sh

HZ="${1:-35}"
TRACE_SEED="${2:-20260903}"
PRESSURE_FRACTION="${3:-0.5}"
WINDOW="${4:-50}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

INPUT_ROOT="outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}"

OUT_DIR="outputs/paper_figures/controller_runtime_case_study_all_levels"

PLOT_SCRIPT="tools/plot_controller_runtime_case_study_all_levels.py"

if [ ! -f "${PLOT_SCRIPT}" ]; then
    echo "[ERROR] missing plotting script:"
    echo "        ${PLOT_SCRIPT}"
    exit 2
fi

for LEVEL in L1 L2 L3 L4
do
    TIMELINE="${INPUT_ROOT}/L0_${LEVEL}_p${PRESSURE_FRACTION}/frame_timeline.csv"

    if [ ! -f "${TIMELINE}" ]; then
        echo "[ERROR] missing timeline:"
        echo "        ${TIMELINE}"
        echo
        echo "Run the TV random-contention evaluation first:"
        echo
        echo "./scripts/stream_exp/45_run_tv_stream3d_30hz_random50.sh \\"
        echo "    20 100 ${HZ} 80 'L1,L2,L3,L4' 0 0.25 \\"
        echo "    ${TRACE_SEED} ${PRESSURE_FRACTION}"
        exit 3
    fi
done

mkdir -p "${OUT_DIR}"

echo
echo "======================================================================"
echo "CONTROLLER RUNTIME CASE STUDY"
echo "======================================================================"
echo "Hz                : ${HZ}"
echo "trace seed        : ${TRACE_SEED}"
echo "pressure fraction : ${PRESSURE_FRACTION}"
echo "window            : ${WINDOW}"
echo "input root        : ${INPUT_ROOT}"
echo "output dir        : ${OUT_DIR}"
echo
echo "Latency curve:"
echo "  field           : arrival_to_finish_ms"
echo "  display label   : Response latency"
echo
echo "Terminology:"
echo "  Dropped frame   : unchanged"
echo "  3D RPN          : changed to 3D Voxel"
echo "======================================================================"
echo

EXTRA_ARGS=()

# 默认复用之前选好的窗口，保证修改图形后 case study 不变。
#
# 如需重新选择窗口：
#
#   FORCE_RESELECT=1 bash scripts/stream_exp/plot_controller_runtime_case_study_all_levels.sh ...
#
if [ "${FORCE_RESELECT:-0}" = "1" ]; then
    EXTRA_ARGS+=(
        --force-reselect
    )
fi

python "${PLOT_SCRIPT}" \
    --input-root "${INPUT_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --hz "${HZ}" \
    --trace-seed "${TRACE_SEED}" \
    --pressure-fraction "${PRESSURE_FRACTION}" \
    --window "${WINDOW}" \
    --dpi 400 \
    "${EXTRA_ARGS[@]}"

echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"
echo "Combined PNG:"
echo "  ${OUT_DIR}/runtime_behavior_analysis.png"
echo
echo "Combined PDF:"
echo "  ${OUT_DIR}/runtime_behavior_analysis.pdf"
echo
echo "Selected windows:"
echo "  ${OUT_DIR}/selected_window_L1.csv"
echo "  ${OUT_DIR}/selected_window_L2.csv"
echo "  ${OUT_DIR}/selected_window_L3.csv"
echo "  ${OUT_DIR}/selected_window_L4.csv"
echo "======================================================================"
