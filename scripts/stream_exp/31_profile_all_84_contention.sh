#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
FRAMES="${3:-120}"
WARMUP="${4:-20}"
START_ID="${5:-1}"
END_ID="${6:-84}"
LEVELS_JSON="${7:-}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"
BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"
SAP_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_sap/e${EPOCH}_n${N}_10Hz"
SCHEDULE_TSV="${SAP_ROOT}/schedules.tsv"
QUALITY_CSV="${QUALITY_CSV:-${SAP_ROOT}/all84_sap_summary.csv}"
BASELINE_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_forward_profile/e${EPOCH}_n${N}"

if [ -z "${LEVELS_JSON}" ]; then
  LEVELS_JSON="$(find outputs/elastic_bev/${EXP_NAME}/contention_calibration -name contention_levels.json -print | sort | tail -n 1)"
fi
if [ -z "${LEVELS_JSON}" ]; then
  echo "[ERROR] contention_levels.json not found. Run 30_calibrate_gemm_contention_levels.sh first."
  exit 2
fi

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${RAW_CKPT}"
require_file "${BANK}"
require_file "${SCHEDULE_TSV}"
require_file "${QUALITY_CSV}"
require_file "${LEVELS_JSON}"
require_file "tools/gpu_gemm_contender.py"
require_file "tools/materialize_causal_prefix_bn_profile.py"
require_file "tools/profile_fixed_forward_components.py"
require_file "tools/build_contention_latency_table.py"

ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile/e${EPOCH}_n${N}"
TMP_DIR="${ROOT}/_tmp"
mkdir -p "${ROOT}" "${TMP_DIR}"

mapfile -t LEVEL_ROWS < <(python - "${LEVELS_JSON}" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
w=x['workload']
for l in x['levels']:
    if l['level']=='L0':
        continue
    print(f"{l['level']}\t{l['busy_repeats']}\t{w['matrix_size']}\t{w['idle_ms']}\t{w['dtype']}")
PY
)

CONT_PID=""
cleanup() {
  if [ -n "${CONT_PID}" ] && kill -0 "${CONT_PID}" 2>/dev/null; then
    kill "${CONT_PID}" 2>/dev/null || true
    wait "${CONT_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for ROW in "${LEVEL_ROWS[@]}"; do
  IFS=$'\t' read -r LEVEL REPEATS GEMM_N IDLE_MS GEMM_DTYPE <<< "${ROW}"
  LEVEL_ROOT="${ROOT}/${LEVEL}"
  mkdir -p "${LEVEL_ROOT}"
  READY="${LEVEL_ROOT}/contender_ready.txt"
  rm -f "${READY}"

  echo
  echo "======================================================================"
  echo "${LEVEL}: repeats=${REPEATS}, GEMM_N=${GEMM_N}, idle_ms=${IDLE_MS}"
  echo "profiles ${START_ID}..${END_ID}; frames=${FRAMES}; warmup=${WARMUP}"
  echo "======================================================================"

  python tools/gpu_gemm_contender.py \
    --device "${CUDA_DEVICE:-0}" \
    --matrix-size "${GEMM_N}" \
    --busy-repeats "${REPEATS}" \
    --idle-ms "${IDLE_MS}" \
    --dtype "${GEMM_DTYPE}" \
    --ready-file "${READY}" \
    > "${LEVEL_ROOT}/contender.log" 2>&1 &
  CONT_PID=$!

  for _ in $(seq 1 100); do
    if [ -f "${READY}" ]; then
      sleep 2
      break
    fi
    if ! kill -0 "${CONT_PID}" 2>/dev/null; then
      echo "[ERROR] contender died; see ${LEVEL_ROOT}/contender.log"
      exit 3
    fi
    sleep 0.1
  done
  require_file "${READY}"

  tail -n +2 "${SCHEDULE_TSV}" | while IFS=$'\t' read -r ID SCHEDULE TAG RES2 RES3 RES4 FPN STEREO RPN; do
    ID_NUM=$((10#${ID}))
    if [ "${ID_NUM}" -lt "${START_ID}" ] || [ "${ID_NUM}" -gt "${END_ID}" ]; then
      continue
    fi

    ID3="$(printf '%03d' "${ID_NUM}")"
    PROFILE_ROOT="${LEVEL_ROOT}/${ID3}_${TAG}"
    SUMMARY="${PROFILE_ROOT}/forward_profile_summary.json"
    RAW="${PROFILE_ROOT}/forward_profile_raw.csv"

    if [ -f "${SUMMARY}" ]; then
      echo "[SKIP ${LEVEL} ${ID3}/084] ${SCHEDULE}"
      continue
    fi

    mkdir -p "${PROFILE_ROOT}"
    TMP_CKPT="${TMP_DIR}/${LEVEL}_${ID3}_${TAG}.pth"

    python tools/materialize_causal_prefix_bn_profile.py \
      --elastic_ckpt "${RAW_CKPT}" \
      --prefix_bn_bank "${BANK}" \
      --schedule "${SCHEDULE}" \
      --output_ckpt "${TMP_CKPT}"

    set +e
    python tools/profile_fixed_forward_components.py \
      --full_cfg "${FULL_CFG}" \
      --full_ckpt "${FULL_CKPT}" \
      --elastic_ckpt "${TMP_CKPT}" \
      --schedule "${SCHEDULE}" \
      --warmup "${WARMUP}" \
      --frames "${FRAMES}" \
      --workers 0 \
      --raw_csv "${RAW}" \
      --summary_json "${SUMMARY}" \
      2>&1 | tee "${PROFILE_ROOT}/console.log"
    RC=${PIPESTATUS[0]}
    set -e

    if [ "${RC}" -ne 0 ]; then
      echo "[ERROR] ${LEVEL} profile ${ID3} failed; kept ${TMP_CKPT}"
      exit "${RC}"
    fi
    rm -f "${TMP_CKPT}"
  done

  kill "${CONT_PID}" 2>/dev/null || true
  wait "${CONT_PID}" 2>/dev/null || true
  CONT_PID=""
done

python tools/build_contention_latency_table.py \
  --levels-json "${LEVELS_JSON}" \
  --schedule-tsv "${SCHEDULE_TSV}" \
  --quality-csv "${QUALITY_CSV}" \
  --baseline-root "${BASELINE_ROOT}" \
  --contention-root "${ROOT}" \
  --output-wide-csv "${ROOT}/all_levels_quality_forward_latency.csv" \
  --output-controller-csv "${ROOT}/controller_remaining_latency_table.csv" \
  --output-json "${ROOT}/contention_profile_summary.json" \
  --allow-partial

echo
echo "[RESULT] ${ROOT}/all_levels_quality_forward_latency.csv"
echo "[RESULT] ${ROOT}/controller_remaining_latency_table.csv"
echo "[RESULT] ${ROOT}/contention_profile_summary.json"

