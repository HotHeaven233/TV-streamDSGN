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

TV_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

OUT_DIR="outputs/paper_figures/random50_35hz_table"

echo "============================================================"
echo "RANDOM50 MAIN-RESULT TABLE"
echo "============================================================"
echo "Hz       : ${HZ}"
echo "levels   : L1,L2,L3,L4"
echo "fraction : ${PRESSURE_FRACTION}"
echo "seed     : ${TRACE_SEED}"
echo "TV exp   : ${TV_EXP_NAME}"
echo "output   : ${OUT_DIR}"
echo "============================================================"

python tools/plot_random50_35hz_table.py \
    --repo_root "${REPO_ROOT}" \
    --hz "${HZ}" \
    --trace_seed "${TRACE_SEED}" \
    --pressure_fraction "${PRESSURE_FRACTION}" \
    --tv_exp_name "${TV_EXP_NAME}" \
    --output_dir "${OUT_DIR}" \
    --dpi 400

