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

OUT_DIR="outputs/paper_figures/tv_policy_ablation_table"

echo "============================================================"
echo "TV POLICY ABLATION TABLE"
echo "============================================================"
echo "Hz       : ${HZ}"
echo "levels   : L2,L3,L4"
echo "seed     : ${TRACE_SEED}"
echo "fraction : ${PRESSURE_FRACTION}"
echo "TV exp   : ${TV_EXP_NAME}"
echo "output   : ${OUT_DIR}"
echo "============================================================"

python tools/plot_tv_policy_ablation_table.py \
    --repo_root "${REPO_ROOT}" \
    --tv_exp_name "${TV_EXP_NAME}" \
    --hz "${HZ}" \
    --trace_seed "${TRACE_SEED}" \
    --pressure_fraction "${PRESSURE_FRACTION}" \
    --output_dir "${OUT_DIR}" \
    --dpi 400
