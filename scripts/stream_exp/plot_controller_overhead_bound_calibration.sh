#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR}/00_env.sh"

cd "${REPO_ROOT}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

AUDIT_DIR="outputs/elastic_bev/${EXP_NAME}/system_audit/controller_runtime_bounds_35hz"

OUT_DIR="outputs/paper_figures/controller_overhead_bound_calibration"

echo "============================================================"
echo "CONTROLLER OVERHEAD + BOUND CALIBRATION"
echo "============================================================"
echo "audit  : ${AUDIT_DIR}"
echo "output : ${OUT_DIR}"
echo "============================================================"

python tools/plot_controller_overhead_bound_calibration.py \
    --audit_dir "${AUDIT_DIR}" \
    --output_dir "${OUT_DIR}" \
    --dpi 400

