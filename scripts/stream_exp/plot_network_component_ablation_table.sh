#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR}/00_env.sh"

cd "${REPO_ROOT}"

ORIGINAL_ROOT="outputs/stream_buffer_timestamp/original_10hz"

ALL84_CSV="outputs/elastic_bev/elastic_bev_v4_bn_from_k3/all84_sap/e20_n100_10Hz/all84_sap_summary.csv"

OUT_DIR="outputs/paper_figures/network_component_ablation_table"

echo "============================================================"
echo "NETWORK COMPONENT / ELASTIC EXECUTION TABLE"
echo "============================================================"
echo "Original : ${ORIGINAL_ROOT}"
echo "All84    : ${ALL84_CSV}"
echo "Output   : ${OUT_DIR}"
echo "============================================================"

python tools/plot_network_component_ablation_table.py \
    --repo_root "${REPO_ROOT}" \
    --original_root "${ORIGINAL_ROOT}" \
    --all84_csv "${ALL84_CSV}" \
    --output_dir "${OUT_DIR}" \
    --dpi 400

