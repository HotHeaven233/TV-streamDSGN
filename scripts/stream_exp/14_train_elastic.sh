#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

ELASTIC_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v2}"
ELASTIC_EPOCHS="${ELASTIC_EPOCHS:-20}"
ELASTIC_WORKERS="${ELASTIC_WORKERS:-4}"
ELASTIC_LR="${ELASTIC_LR:-2e-4}"
ELASTIC_MIN_LR="${ELASTIC_MIN_LR:-2e-6}"
ELASTIC_WD="${ELASTIC_WD:-1e-4}"
ELASTIC_BEV_W="${ELASTIC_BEV_W:-2.0}"
ELASTIC_COS_W="${ELASTIC_COS_W:-0.20}"
ELASTIC_AUX_W="${ELASTIC_AUX_W:-0.35}"
ELASTIC_CANONICAL_HISTORY_PROB="${ELASTIC_CANONICAL_HISTORY_PROB:-0.60}"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "tools/train_elastic_bev.py"
require_file "pcdet/models/backbones_3d_stream/elastic_bev_branch.py"

OUT_DIR="outputs/elastic_bev/${ELASTIC_EXP_NAME}"
if compgen -G "${OUT_DIR}/ckpt/checkpoint_epoch_*.pth" > /dev/null; then
    echo "[ERROR] Existing Elastic-v2 checkpoints found: ${OUT_DIR}/ckpt"
    echo "Use another name, e.g. ELASTIC_EXP_NAME=elastic_bev_v2_b $0"
    exit 1
fi

banner "Train Elastic-v2 | Frozen K3 Full | Elastic Res2-4 + FPN + Stereo/RPN"
echo "[BASE CFG]      ${FULL_CFG}"
echo "[BASE CKPT]     ${FULL_CKPT}"
echo "[EXP]           ${ELASTIC_EXP_NAME}"
echo "[EPOCHS]        ${ELASTIC_EPOCHS}"
echo "[LR]            ${ELASTIC_LR} -> ${ELASTIC_MIN_LR}"
echo "[WEIGHT DECAY]  ${ELASTIC_WD}"
echo "[LOSS]          det=1.0 bev=${ELASTIC_BEV_W} cos=${ELASTIC_COS_W} aux=${ELASTIC_AUX_W}"
echo "[FIXED PREFIX]  original ResNet stem + layer1"
echo "[ELASTIC]       ResNet layer2/layer3/layer4 + FPN + Stereo3D + RPN3D"
echo "[WIDTHS]        25%, 50%, 75%, 100%; non-increasing across stages"
echo "[HISTORY]       choose 3 from t-5..t-1; [t-3,t-2,t-1] prob=${ELASTIC_CANONICAL_HISTORY_PROB}"
echo "[BASE]          all existing K3 parameters/buffers frozen and bitwise checked"

python tools/train_elastic_bev.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --epochs "${ELASTIC_EPOCHS}" \
    --workers "${ELASTIC_WORKERS}" \
    --exp_name "${ELASTIC_EXP_NAME}" \
    --lr "${ELASTIC_LR}" \
    --min_lr "${ELASTIC_MIN_LR}" \
    --weight_decay "${ELASTIC_WD}" \
    --det_weight 1.0 \
    --bev_weight "${ELASTIC_BEV_W}" \
    --cos_weight "${ELASTIC_COS_W}" \
    --aux_width_weight "${ELASTIC_AUX_W}" \
    --smooth_l1_beta 0.1 \
    --canonical_history_prob "${ELASTIC_CANONICAL_HISTORY_PROB}" \
    --warmup_ratio 0.05 \
    --grad_clip 5.0 \
    --log_interval 20 \
    --save_interval 1
