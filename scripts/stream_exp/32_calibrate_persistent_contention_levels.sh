#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

PROBE_FRAMES="${1:-80}"
PROBE_WARMUP="${2:-10}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

CONTENT_DURATION_MS="${CONTENT_DURATION_MS:-100}"
CONTENT_START_DELAY_MS="${CONTENT_START_DELAY_MS:-5}"
CONTENT_THREADS="${CONTENT_THREADS:-256}"

# Nominal strength = persistent blocks / GPU SM count.
# It is deliberately NOT labeled as exact GPU-utilization percentage.
STRENGTHS="${STRENGTHS:-0.125 0.25 0.375 0.5 0.625 0.75 1.0 1.25 1.5}"
TARGETS="${TARGETS:-1.20,1.45,1.80,2.30}"

ROOT="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v3/persistent_window"
mkdir -p "${ROOT}/candidates"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "tools/persistent_cuda_contention.py"
require_file "tools/profile_fixed_prefix_probe_persistent.py"
require_file "tools/select_contention_levels_persistent.py"

echo "======================================================================"
echo "CALIBRATE FINITE PERSISTENT CUDA CONTENTION WINDOWS"
echo "duration ms  : ${CONTENT_DURATION_MS}"
echo "start delay  : ${CONTENT_START_DELAY_MS}"
echo "threads/block: ${CONTENT_THREADS}"
echo "strengths    : ${STRENGTHS}"
echo "targets      : ${TARGETS}"
echo "semantics    : one stable contention window per probe/forward"
echo "======================================================================"

# Compile once before loading the detector repeatedly.
python tools/persistent_cuda_contention.py

echo
echo "[L0] no contention"
python tools/profile_fixed_prefix_probe_persistent.py \
  --full_cfg "${FULL_CFG}" \
  --full_ckpt "${FULL_CKPT}" \
  --warmup "${PROBE_WARMUP}" \
  --frames "${PROBE_FRAMES}" \
  --output_json "${ROOT}/probe_L0.json"

for S in ${STRENGTHS}; do
  TAG="$(echo "${S}" | sed 's/\./p/g')"

  echo
  echo "[CANDIDATE] strength=${S}"

  python tools/profile_fixed_prefix_probe_persistent.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --warmup "${PROBE_WARMUP}" \
    --frames "${PROBE_FRAMES}" \
    --contention-strength "${S}" \
    --contention-duration-ms "${CONTENT_DURATION_MS}" \
    --contention-start-delay-ms "${CONTENT_START_DELAY_MS}" \
    --contention-threads "${CONTENT_THREADS}" \
    --output_json "${ROOT}/candidates/probe_s${TAG}.json"
done

python tools/select_contention_levels_persistent.py \
  --baseline "${ROOT}/probe_L0.json" \
  --candidate-dir "${ROOT}/candidates" \
  --duration-ms "${CONTENT_DURATION_MS}" \
  --start-delay-ms "${CONTENT_START_DELAY_MS}" \
  --threads "${CONTENT_THREADS}" \
  --targets "${TARGETS}" \
  --output "${ROOT}/contention_levels.json"

echo
echo "[RESULT] ${ROOT}/contention_levels.json"

