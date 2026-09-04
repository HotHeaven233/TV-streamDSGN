#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

PROBE_FRAMES="${1:-100}"
PROBE_WARMUP="${2:-20}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
GEMM_N="${GEMM_N:-4096}"
IDLE_MS="${IDLE_MS:-2.0}"
GEMM_DTYPE="${GEMM_DTYPE:-fp16}"
CANDIDATES="${CANDIDATES:-1 2 3 4 6 8 12 16 24 32}"
TARGETS="${TARGETS:-1.25,1.50,2.00,2.50}"

ROOT="outputs/elastic_bev/${EXP_NAME}/contention_calibration/gemm_n${GEMM_N}_idle${IDLE_MS}ms"
mkdir -p "${ROOT}/candidates"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "tools/gpu_gemm_contender.py"
require_file "tools/profile_fixed_prefix_probe.py"
require_file "tools/select_contention_levels.py"

CONT_PID=""
cleanup() {
  if [ -n "${CONT_PID}" ] && kill -0 "${CONT_PID}" 2>/dev/null; then
    kill "${CONT_PID}" 2>/dev/null || true
    wait "${CONT_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_contender() {
  local repeats="$1"
  local ready="${ROOT}/contender_ready_r${repeats}.txt"
  rm -f "${ready}"
  python tools/gpu_gemm_contender.py \
    --device "${CUDA_DEVICE:-0}" \
    --matrix-size "${GEMM_N}" \
    --busy-repeats "${repeats}" \
    --idle-ms "${IDLE_MS}" \
    --dtype "${GEMM_DTYPE}" \
    --ready-file "${ready}" \
    > "${ROOT}/contender_r${repeats}.log" 2>&1 &
  CONT_PID=$!

  for _ in $(seq 1 100); do
    if [ -f "${ready}" ]; then
      sleep 2
      return 0
    fi
    if ! kill -0 "${CONT_PID}" 2>/dev/null; then
      echo "[ERROR] contender exited early; see ${ROOT}/contender_r${repeats}.log"
      exit 2
    fi
    sleep 0.1
  done
  echo "[ERROR] contender readiness timeout"
  exit 3
}

stop_contender() {
  if [ -n "${CONT_PID}" ] && kill -0 "${CONT_PID}" 2>/dev/null; then
    kill "${CONT_PID}" 2>/dev/null || true
    wait "${CONT_PID}" 2>/dev/null || true
  fi
  CONT_PID=""
}

echo "======================================================================"
echo "CALIBRATE FIXED GEMM CONTENTION LEVELS"
echo "GEMM N      : ${GEMM_N}"
echo "idle ms     : ${IDLE_MS}"
echo "dtype       : ${GEMM_DTYPE}"
echo "candidates  : ${CANDIDATES}"
echo "targets     : ${TARGETS}"
echo "======================================================================"

echo "[L0] baseline probe"
python tools/profile_fixed_prefix_probe.py \
  --full_cfg "${FULL_CFG}" \
  --full_ckpt "${FULL_CKPT}" \
  --warmup "${PROBE_WARMUP}" \
  --frames "${PROBE_FRAMES}" \
  --output_json "${ROOT}/probe_L0.json"

for R in ${CANDIDATES}; do
  echo
  echo "[CANDIDATE] busy_repeats=${R}"
  start_contender "${R}"
  python tools/profile_fixed_prefix_probe.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --warmup "${PROBE_WARMUP}" \
    --frames "${PROBE_FRAMES}" \
    --output_json "${ROOT}/candidates/probe_r${R}.json"
  stop_contender
done

python tools/select_contention_levels.py \
  --baseline "${ROOT}/probe_L0.json" \
  --candidate-dir "${ROOT}/candidates" \
  --matrix-size "${GEMM_N}" \
  --idle-ms "${IDLE_MS}" \
  --dtype "${GEMM_DTYPE}" \
  --targets "${TARGETS}" \
  --output "${ROOT}/contention_levels.json"

echo
echo "[RESULT] ${ROOT}/contention_levels.json"

