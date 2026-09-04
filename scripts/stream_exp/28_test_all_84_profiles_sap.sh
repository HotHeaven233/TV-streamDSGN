#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
HZ="${3:-10}"
START_ID="${4:-1}"
END_ID="${5:-84}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"
BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"

ROOT="outputs/elastic_bev/${EXP_NAME}/all84_sap/e${EPOCH}_n${N}_${HZ}Hz"
SCHEDULE_TSV="${ROOT}/schedules.tsv"
TMP_DIR="${ROOT}/_tmp"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${RAW_CKPT}"
require_file "${BANK}"
require_file "tools/generate_84_monotonic_profiles.py"
require_file "tools/materialize_causal_prefix_bn_profile.py"
require_file "tools/test_elastic_stream_v4_bn.py"
require_file "tools/summarize_84_profile_sap.py"

mkdir -p "${ROOT}" "${TMP_DIR}"

python tools/generate_84_monotonic_profiles.py \
  --output "${SCHEDULE_TSV}"

echo
echo "======================================================================"
echo "ALL-84 STATIC sAP"
echo "epoch       : ${EPOCH}"
echo "BN bank N   : ${N}"
echo "input Hz    : ${HZ}"
echo "profile IDs : ${START_ID}..${END_ID}"
echo "result root : ${ROOT}"
echo "======================================================================"

tail -n +2 "${SCHEDULE_TSV}" | while IFS=$'\t' read -r ID SCHEDULE TAG RES2 RES3 RES4 FPN STEREO RPN; do
  ID_NUM=$((10#${ID}))
  if [ "${ID_NUM}" -lt "${START_ID}" ] || [ "${ID_NUM}" -gt "${END_ID}" ]; then
    continue
  fi

  ID3="$(printf '%03d' "${ID_NUM}")"
  PROFILE_ROOT="${ROOT}/${ID3}_${TAG}"

  EXISTING_PAPER="$(find "${PROFILE_ROOT}" -type f -name paper_sap.txt -print -quit 2>/dev/null || true)"
  if [ -n "${EXISTING_PAPER}" ]; then
    echo "[SKIP ${ID3}/084] ${SCHEDULE} -> already complete"
    continue
  fi

  mkdir -p "${PROFILE_ROOT}"
  TMP_CKPT="${TMP_DIR}/profile_${ID3}_${TAG}.pth"

  echo
  echo "========================================================================"
  echo "[PROFILE ${ID3}/084] ${SCHEDULE}"
  echo "========================================================================"

  python tools/materialize_causal_prefix_bn_profile.py \
    --elastic_ckpt "${RAW_CKPT}" \
    --prefix_bn_bank "${BANK}" \
    --schedule "${SCHEDULE}" \
    --output_ckpt "${TMP_CKPT}"

  set +e
  python tools/test_elastic_stream_v4_bn.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${TMP_CKPT}" \
    --input_hz "${HZ}" \
    --warmup 20 \
    --mode elastic_fixed \
    --fixed_schedule "${SCHEDULE}" \
    --output_dir "${PROFILE_ROOT}" \
    2>&1 | tee "${PROFILE_ROOT}/console.log"
  TEST_RC=${PIPESTATUS[0]}
  set -e

  if [ "${TEST_RC}" -ne 0 ]; then
    echo "[ERROR] profile ${ID3} failed. Temporary checkpoint kept:"
    echo "        ${TMP_CKPT}"
    exit "${TEST_RC}"
  fi

  PAPER="$(find "${PROFILE_ROOT}" -type f -name paper_sap.txt -print -quit 2>/dev/null || true)"
  if [ -z "${PAPER}" ]; then
    echo "[ERROR] profile ${ID3} finished but paper_sap.txt was not found."
    echo "        Temporary checkpoint kept: ${TMP_CKPT}"
    exit 3
  fi

  rm -f "${TMP_CKPT}"
  echo "[OK ${ID3}/084] ${SCHEDULE}"
done

python tools/summarize_84_profile_sap.py \
  --schedule_tsv "${SCHEDULE_TSV}" \
  --result_root "${ROOT}" \
  --output_csv "${ROOT}/all84_sap_summary.csv" \
  --output_json "${ROOT}/all84_sap_summary.json"

echo
echo "[RESULT] ${ROOT}/all84_sap_summary.csv"
echo "[RESULT] ${ROOT}/all84_sap_summary.json"

