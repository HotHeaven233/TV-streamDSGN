#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
HZ="${3:-35}"

OVERHEAD_FRAMES="${4:-200}"
BOUND_FRAMES="${5:-100}"

RUNTIME_WARMUP="${6:-20}"
BOUND_WARMUP="${7:-10}"

GUARD="${8:-0.25}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

ELASTIC_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"

BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"

PROFILE_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile_v6/e${EPOCH}_n${N}"

CTRL_CSV="${PROFILE_ROOT}/controller_remaining_latency_table.csv"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

OUT="outputs/elastic_bev/${EXP_NAME}/system_audit/controller_runtime_bounds_${HZ}hz"

for f in \
    "${FULL_CFG}" \
    "${FULL_CKPT}" \
    "${ELASTIC_CKPT}" \
    "${BANK}" \
    "${CTRL_CSV}" \
    "${LEVELS_JSON}" \
    tools/audit_tv_controller_runtime_bounds.py
do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] missing: ${f}" >&2
        exit 2
    fi
done

rm -rf "${OUT}"
mkdir -p "${OUT}"

python tools/audit_tv_controller_runtime_bounds.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${ELASTIC_CKPT}" \
    --prefix_bn_bank "${BANK}" \
    --controller_csv "${CTRL_CSV}" \
    --levels_json "${LEVELS_JSON}" \
    --levels "L0,L1,L2,L3,L4" \
    --input_hz "${HZ}" \
    --control_guard_per_boundary_ms "${GUARD}" \
    --runtime_warmup_frames "${RUNTIME_WARMUP}" \
    --overhead_frames "${OVERHEAD_FRAMES}" \
    --bound_warmup_frames "${BOUND_WARMUP}" \
    --bound_frames "${BOUND_FRAMES}" \
    --workers 0 \
    --seed 1024 \
    --output_dir "${OUT}" \
    2>&1 | tee "${OUT}/run.log"

echo
echo "[PASS] controller runtime audit complete"
echo "summary = ${OUT}/summary.json"
