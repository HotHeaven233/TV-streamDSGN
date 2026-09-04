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

# Finer low-strength grid is intentional.  On a 128-SM GPU these correspond
# roughly to 4, 8, 12, 16, ... persistent blocks before rounding.
STRENGTHS="${STRENGTHS:-0.03125 0.0625 0.09375 0.125 0.1875 0.25 0.375 0.5 0.75 1.0 1.25}"
TARGETS="${TARGETS:-1.20,1.45,1.80,2.30}"

ROOT="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v4/persistent_window_dual_stream"
mkdir -p "${ROOT}/candidates"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "tools/persistent_cuda_contention.py"
require_file "tools/profile_fixed_prefix_probe_persistent.py"
require_file "tools/select_contention_levels_persistent.py"

echo "======================================================================"
echo "CALIBRATE PERSISTENT CONTENTION -- DUAL NON-DEFAULT CUDA STREAMS"
echo "duration ms  : ${CONTENT_DURATION_MS}"
echo "start delay  : ${CONTENT_START_DELAY_MS}"
echo "threads/block: ${CONTENT_THREADS}"
echo "strengths    : ${STRENGTHS}"
echo "targets      : ${TARGETS}"
echo "model stream : dedicated non-default stream"
echo "load stream  : explicit input-ready event dependency"
echo "======================================================================"

python tools/persistent_cuda_contention.py

echo
echo "[L0] no contention"
python tools/profile_fixed_prefix_probe_persistent.py \
  --full_cfg "${FULL_CFG}" \
  --full_ckpt "${FULL_CKPT}" \
  --warmup "${PROBE_WARMUP}" \
  --frames "${PROBE_FRAMES}" \
  --output_json "${ROOT}/probe_L0.json"

SUCCESS=0

for S in ${STRENGTHS}; do
  TAG="$(echo "${S}" | sed 's/\./p/g')"

  echo
  echo "[CANDIDATE] strength=${S}"

  set +e
  python tools/profile_fixed_prefix_probe_persistent.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --warmup "${PROBE_WARMUP}" \
    --frames "${PROBE_FRAMES}" \
    --contention-strength "${S}" \
    --contention-duration-ms "${CONTENT_DURATION_MS}" \
    --contention-start-delay-ms "${CONTENT_START_DELAY_MS}" \
    --contention-threads "${CONTENT_THREADS}" \
    --output_json "${ROOT}/candidates/probe_s${TAG}.json" \
    2>&1 | tee "${ROOT}/candidates/probe_s${TAG}.log"
  RC=${PIPESTATUS[0]}
  set -e

  if [ "${RC}" -eq 0 ]; then
    SUCCESS=$((SUCCESS + 1))
  else
    echo "[SKIP] strength=${S} failed/overran this window; continue scanning."
    rm -f "${ROOT}/candidates/probe_s${TAG}.json"
  fi
done

if [ "${SUCCESS}" -lt 4 ]; then
  echo "[ERROR] only ${SUCCESS} usable candidate strengths."
  echo "Try a longer CONTENT_DURATION_MS or a finer/lower STRENGTHS grid."
  exit 4
fi

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

