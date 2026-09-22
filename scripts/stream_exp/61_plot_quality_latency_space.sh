#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

source scripts/stream_exp/00_env.sh

EPOCH="${1:-20}"
N="${2:-100}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

INPUT_CSV="outputs/elastic_bev/${EXP_NAME}/all84_forward_profile/e${EPOCH}_n${N}/all84_quality_forward_latency.csv"

OUT_DIR="outputs/paper_figures/quality_latency_space"

require_file "${INPUT_CSV}"
require_file "tools/plot_quality_latency_space.py"

mkdir -p "${OUT_DIR}"

echo
echo "======================================================================"
echo "PAPER FIGURE | 84-SCHEDULE QUALITY--LATENCY SPACE"
echo "epoch       : ${EPOCH}"
echo "BN bank N   : ${N}"
echo "input CSV   : ${INPUT_CSV}"
echo "output dir  : ${OUT_DIR}"
echo "x-axis      : forward_total_ms_p99"
echo "y-axis      : reference_mean3d_moderate"
echo "======================================================================"

python tools/plot_quality_latency_space.py \
    --input-csv "${INPUT_CSV}" \
    --output-dir "${OUT_DIR}" \
    --dpi 400

echo
echo "[RESULT] ${OUT_DIR}/quality_latency_space.png"
echo "[RESULT] ${OUT_DIR}/quality_latency_pareto.csv"
