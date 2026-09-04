#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

ELASTIC_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v3}"
EPOCH="${1:-20}"
MODE="${2:-dynamic}"
HZ="${3:-10}"
FIXED_SCHEDULE="${4:-0.5,0.5,0.5,0.5,0.5,0.5}"
PROFILE_RUNS="${PROFILE_RUNS:-1}"
SAFETY="${ELASTIC_SAFETY:-1.12}"
DEADLINE_PERIODS="${ELASTIC_DEADLINE_PERIODS:-1.0}"

ELASTIC_CKPT="outputs/elastic_bev/${ELASTIC_EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"
OUT_DIR="outputs/elastic_bev_stream/${ELASTIC_EXP_NAME}_e${EPOCH}/${MODE}"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${ELASTIC_CKPT}"
require_file "tools/test_elastic_stream_v3.py"
require_file "pcdet/models/backbones_3d_stream/elastic_bev_branch_v3.py"

banner "Elastic-v3 K3 Streaming Test | ${MODE} | ${HZ} Hz | epoch ${EPOCH}"
echo "[BASE]          ${FULL_CKPT}"
echo "[ELASTIC]       ${ELASTIC_CKPT}"
echo "[MODE]          ${MODE}"
echo "[HZ]            ${HZ}"
echo "[FIXED]         ${FIXED_SCHEDULE}"
echo "[STAGES]        Res2,Res3,Res4,FPN,Stereo,RPN"

python tools/test_elastic_stream_v3.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${ELASTIC_CKPT}" \
    --input_hz "${HZ}" \
    --mode "${MODE}" \
    --fixed_schedule "${FIXED_SCHEDULE}" \
    --deadline_periods "${DEADLINE_PERIODS}" \
    --safety "${SAFETY}" \
    --warmup 8 \
    --profile_runs "${PROFILE_RUNS}" \
    --workers 0 \
    --output_dir "${OUT_DIR}"
