#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN
source scripts/stream_exp/00_env.sh

TS_CFG="${1:-configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat.yaml}"

if [ ! -f "${TS_CFG}" ]; then
    echo "[ERROR] missing config: ${TS_CFG}"
    exit 2
fi

if [ ! -f "${FULL_CKPT}" ]; then
    echo "[ERROR] missing init checkpoint: ${FULL_CKPT}"
    exit 2
fi

echo "================================================================================"
echo "TRUE TRANSTREAMING-STYLE STREAMDSGN TRAINING"
echo "================================================================================"
echo "CFG       : ${TS_CFG}"
echo "INIT CKPT : ${FULL_CKPT}"
echo
echo "Architecture:"
echo "  Stereo feature extractor"
echo "      -> TranstreamingBEVTAT / RTPE"
echo "      -> shared VANBackbone"
echo "      -> ONE shared StreamDetHead"
echo
echo "Training future horizons:"
echo "  +1, +2, +4, +8"
echo "================================================================================"

python tools/train.py \
    --cfg_file "${TS_CFG}" \
    --pretrained_model "${FULL_CKPT}" \
    --fix_random_seed \
    --workers 4 \
    --exp_name transtreaming_tat
