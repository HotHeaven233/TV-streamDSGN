#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
EPOCH="${1:-20}"
N="${2:-100}"

ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile_v6/e${EPOCH}_n${N}"
LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"
CTRL_CSV="${ROOT}/controller_remaining_latency_table.csv"
OUT_ROOT="${ROOT}/controller_stage1"

mkdir -p "${OUT_ROOT}"

python tools/audit_tv_stream3d_controller.py \
  --controller-csv "${CTRL_CSV}" \
  --levels-json "${LEVELS_JSON}" \
  --output-json "${OUT_ROOT}/controller_stage1_audit.json" \
  --decision-csv "${OUT_ROOT}/prefix_decision_audit.csv" \
  --frequencies "20,25,30,35,40"

echo
echo "[RESULT] ${OUT_ROOT}/controller_stage1_audit.json"
echo "[RESULT] ${OUT_ROOT}/prefix_decision_audit.csv"
