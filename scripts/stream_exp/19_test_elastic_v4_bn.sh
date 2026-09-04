#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

ELASTIC_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn}"
EPOCH="${1:-12}"
MODE="${2:-elastic_fixed}"
HZ="${3:-10}"
FIXED_SCHEDULE="${4:-0.5,0.5,0.5,0.5,0.5,0.5}"
DEADLINE_PERIODS="${ELASTIC_DEADLINE_PERIODS:-1.0}"

ELASTIC_CKPT="outputs/elastic_bev/${ELASTIC_EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}_bncal.pth"
OUT_DIR="outputs/elastic_bev_stream/${ELASTIC_EXP_NAME}_e${EPOCH}/${MODE}"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${ELASTIC_CKPT}"
require_file "tools/test_elastic_stream_v4_bn.py"
require_file "pcdet/models/backbones_3d_stream/elastic_bev_branch_v4_bn.py"
require_file "pcdet/models/backbones_3d_stream/elastic_v4_bn_hybrid.py"

banner "Elastic-v4-BN streaming test | fused BN | ${MODE} | ${HZ} Hz"
echo "[BASE]          ${FULL_CKPT}"
echo "[ELASTIC]       ${ELASTIC_CKPT}"
echo "[FIXED]         ${FIXED_SCHEDULE}"
echo "[DEPLOY]        native leading Full + materialized Conv/BN fusion"

python tools/test_elastic_stream_v4_bn.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${ELASTIC_CKPT}" \
    --input_hz "${HZ}" \
    --mode "${MODE}" \
    --fixed_schedule "${FIXED_SCHEDULE}" \
    --deadline_periods "${DEADLINE_PERIODS}" \
    --warmup 20 \
    --workers 0 \
    --output_dir "${OUT_DIR}"

