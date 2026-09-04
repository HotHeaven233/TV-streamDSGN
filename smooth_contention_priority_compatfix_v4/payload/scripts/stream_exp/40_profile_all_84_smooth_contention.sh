#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/00_env.sh"
EPOCH="${1:-20}"; N="${2:-100}"; FRAMES="${3:-80}"; WARMUP="${4:-10}"; START_ID="${5:-1}"; END_ID="${6:-84}"; EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
LEVELS_JSON="${7:-outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json}"; RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"; BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"; SCHEDULE_TSV="outputs/elastic_bev/${EXP_NAME}/all84_sap/e${EPOCH}_n${N}_10Hz/schedules.tsv"; ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile_v6/e${EPOCH}_n${N}"; TMP_DIR="${ROOT}/_tmp"; mkdir -p "${ROOT}" "${TMP_DIR}"
mapfile -t ROWS < <(python - "${LEVELS_JSON}" <<'PYLVL'
import json,sys
x=json.load(open(sys.argv[1])); w=x['workload']
for l in x['levels']:
    if l['level']!='L0': print(f"{l['level']}\t{l['strength']}\t{w['window_ms']}\t{w['slice_ms']}\t{w['start_delay_ms']}\t{w['threads']}")
PYLVL
)
for ROW in "${ROWS[@]}"; do IFS=$'\t' read -r LEVEL STRENGTH WINDOW SLICE DELAY THREADS <<< "${ROW}"; LROOT="${ROOT}/${LEVEL}"; mkdir -p "${LROOT}"; tail -n +2 "${SCHEDULE_TSV}"|while IFS=$'\t' read -r ID SCHEDULE TAG RES2 RES3 RES4 FPN STEREO RPN; do IDN=$((10#${ID})); [ "${IDN}" -lt "${START_ID}" ]&&continue; [ "${IDN}" -gt "${END_ID}" ]&&continue; ID3="$(printf '%03d' "${IDN}")"; PROOT="${LROOT}/${ID3}_${TAG}"; SUMMARY="${PROOT}/forward_profile_summary.json"; RAW="${PROOT}/forward_profile_raw.csv"; [ -f "${SUMMARY}" ]&&continue; mkdir -p "${PROOT}"; TMP="${TMP_DIR}/${LEVEL}_${ID3}.pth"; python tools/materialize_causal_prefix_bn_profile.py --elastic_ckpt "${RAW_CKPT}" --prefix_bn_bank "${BANK}" --schedule "${SCHEDULE}" --output_ckpt "${TMP}"; python tools/profile_fixed_forward_components.py --full_cfg "${FULL_CFG}" --full_ckpt "${FULL_CKPT}" --elastic_ckpt "${TMP}" --schedule "${SCHEDULE}" --warmup "${WARMUP}" --frames "${FRAMES}" --workers 0 --contention-strength "${STRENGTH}" --contention-window-ms "${WINDOW}" --contention-slice-ms "${SLICE}" --contention-start-delay-ms "${DELAY}" --contention-threads "${THREADS}" --raw_csv "${RAW}" --summary_json "${SUMMARY}" 2>&1|tee "${PROOT}/console.log"; rm -f "${TMP}"; done; done
