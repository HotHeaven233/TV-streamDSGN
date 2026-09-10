#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR}/00_env.sh"

cd "${REPO_ROOT}"

TV_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

OUT_DIR="outputs/paper_figures/no_load_frequency_sweep_table"

echo "============================================================"
echo "NO-LOAD FREQUENCY SWEEP TABLE"
echo "============================================================"
echo "repo   : ${REPO_ROOT}"
echo "TV exp : ${TV_EXP_NAME}"
echo "Hz     : 35,40,45,50"
echo "output : ${OUT_DIR}"
echo "============================================================"

python tools/plot_no_load_frequency_sweep_table.py \
    --repo_root "${REPO_ROOT}" \
    --tv_exp_name "${TV_EXP_NAME}" \
    --output_dir "${OUT_DIR}" \
    --dpi 400

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
