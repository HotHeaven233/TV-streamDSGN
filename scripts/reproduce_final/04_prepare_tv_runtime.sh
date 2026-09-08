#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
EPOCH=20
N=100

banner "STEP 04 | Build all deployment statistics for TV-Stream3D"

# 1) Causal-prefix BN bank: 203 states.
./scripts/stream_exp/23_calibrate_causal_prefix_bn_bank.sh \
    "${EPOCH}" "${N}" 0

# 2) Validate bank using representative profiles.
./scripts/stream_exp/27_validate_causal_prefix_bn_bank.sh \
    "${EPOCH}" "${N}" 10

# 3) Static quality Q(s) for all 84 monotonic schedules.
./scripts/stream_exp/28_test_all_84_profiles_sap.sh \
    "${EPOCH}" "${N}" 10 1 84

# 4) L0 forward/suffix latency for all 84 schedules.
./scripts/stream_exp/29_profile_all_84_forward.sh \
    "${EPOCH}" "${N}" 120 20 1 84

# 5) Calibrate L1-L4 contention states.
./scripts/stream_exp/38_calibrate_smooth_contention.sh \
    80 10

# 6) L1-L4 forward/suffix latency for all 84 schedules.
./scripts/stream_exp/40_profile_all_84_smooth_contention.sh \
    "${EPOCH}" "${N}" 80 10 1 84

# 7) Merge Q(s), L0 and L1-L4 profiles into controller tables.
ROOT="outputs/elastic_bev/${EXP_NAME}"
OUT="${ROOT}/all84_contention_profile_v6/e${EPOCH}_n${N}"

python tools/build_contention_latency_table_smooth.py \
    --levels-json \
    "${ROOT}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json" \
    --schedule-tsv \
    "${ROOT}/all84_sap/e${EPOCH}_n${N}_10Hz/schedules.tsv" \
    --quality-csv \
    "${ROOT}/all84_sap/e${EPOCH}_n${N}_10Hz/all84_sap_summary.csv" \
    --baseline-root \
    "${ROOT}/all84_forward_profile/e${EPOCH}_n${N}" \
    --contention-root \
    "${ROOT}/all84_contention_profile_v6/e${EPOCH}_n${N}" \
    --output-wide-csv \
    "${OUT}/all_levels_quality_forward_latency.csv" \
    --output-controller-csv \
    "${OUT}/controller_remaining_latency_table.csv" \
    --output-json \
    "${OUT}/contention_profile_summary.json"

# Strict completeness check: 5 levels * 84 paths; 7 controller checkpoints.
python - "${OUT}/contention_profile_summary.json" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
x = json.loads(p.read_text())

assert x["completed_wide_rows"] == 420, x
assert x["completed_controller_rows"] == 2940, x
assert x["missing"] == [], x

print("[PASS] wide rows       =", x["completed_wide_rows"])
print("[PASS] controller rows =", x["completed_controller_rows"])
print("[PASS] missing         =", x["missing"])
PY

# 8) Offline controller consistency audit.
./scripts/stream_exp/41_audit_tv_stream3d_controller.sh \
    "${EPOCH}" "${N}"

# 9) Real CUDA online smoke; still not a paper result.
./scripts/stream_exp/42_smoke_tv_stream3d_online.sh \
    "${EPOCH}" \
    "${N}" \
    33 \
    80 \
    100 \
    "L0 L1 L2 L3 L4" \
    0.25

echo "[PASS] TV-Stream3D runtime/controller preparation complete"
