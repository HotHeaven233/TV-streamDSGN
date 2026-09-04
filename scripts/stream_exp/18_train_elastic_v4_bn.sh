#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

ELASTIC_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
ELASTIC_EPOCHS="${ELASTIC_EPOCHS:-20}"
ELASTIC_WORKERS="${ELASTIC_WORKERS:-4}"
ELASTIC_LR="${ELASTIC_LR:-2e-4}"
ELASTIC_MIN_LR="${ELASTIC_MIN_LR:-2e-6}"
ELASTIC_WD="${ELASTIC_WD:-1e-4}"

ELASTIC_PRE_BN_CAL_BATCHES="${ELASTIC_PRE_BN_CAL_BATCHES:-336}"
ELASTIC_FINAL_BN_CAL_BATCHES="${ELASTIC_FINAL_BN_CAL_BATCHES:-1660}"

ELASTIC_CANONICAL_HISTORY_PROB="${ELASTIC_CANONICAL_HISTORY_PROB:-0.60}"

ELASTIC_UNIFORM_P="${ELASTIC_UNIFORM_P:-0.40}"
ELASTIC_TRANSITION_P="${ELASTIC_TRANSITION_P:-0.30}"
ELASTIC_HARD_P="${ELASTIC_HARD_P:-0.20}"
ELASTIC_ANCHOR_P="${ELASTIC_ANCHOR_P:-0.10}"

ELASTIC_HISTORY_SAME_P="${ELASTIC_HISTORY_SAME_P:-0.40}"
ELASTIC_HISTORY_INDEP_P="${ELASTIC_HISTORY_INDEP_P:-0.40}"
ELASTIC_HISTORY_JUMP_P="${ELASTIC_HISTORY_JUMP_P:-0.20}"
ELASTIC_HISTORY_GRAD_SLOTS="${ELASTIC_HISTORY_GRAD_SLOTS:-1}"

ELASTIC_BEV_W="${ELASTIC_BEV_W:-1.5}"
ELASTIC_COS_W="${ELASTIC_COS_W:-0.15}"
ELASTIC_STAGE_W="${ELASTIC_STAGE_W:-0.50}"
ELASTIC_HISTORY_BEV_W="${ELASTIC_HISTORY_BEV_W:-0.75}"
ELASTIC_HISTORY_COS_W="${ELASTIC_HISTORY_COS_W:-0.10}"

# Fresh paper-style training:
# initialize the elastic branch ONLY from the frozen 3-history StreamDSGN/K3
# backbone loaded by --full_ckpt. Do not inherit any v2/v3 elastic checkpoint.
INIT_ARGS=()
echo "[INIT]          fresh elastic branch from frozen 3-history StreamDSGN/K3 Full"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "tools/train_elastic_bev_v4_bn.py"
require_file "tools/calibrate_elastic_v4_bn.py"
require_file "pcdet/models/backbones_3d_stream/elastic_bev_branch_v4_bn.py"
require_file "pcdet/models/backbones_3d_stream/elastic_v4_bn_hybrid.py"

OUT_DIR="outputs/elastic_bev/${ELASTIC_EXP_NAME}"
if compgen -G "${OUT_DIR}/ckpt/checkpoint_epoch_*.pth" > /dev/null; then
    echo "[ERROR] Existing v4-BN checkpoints found: ${OUT_DIR}/ckpt"
    echo "Use another ELASTIC_EXP_NAME or remove/resume explicitly."
    exit 1
fi

banner "Train Elastic-v4-BN FROM K3 | native Full prefix + fused-BN deployment"
echo "[BASE CFG]      ${FULL_CFG}"
echo "[BASE CKPT]     ${FULL_CKPT}"
echo "[ELASTIC INIT]  Full/K3 only; no v2/v3 elastic checkpoint"
echo "[EXP]           ${ELASTIC_EXP_NAME}"
echo "[EPOCHS]        ${ELASTIC_EPOCHS}"
echo "[LR]            ${ELASTIC_LR} -> ${ELASTIC_MIN_LR}"
echo "[PRE BN CAL]    ${ELASTIC_PRE_BN_CAL_BATCHES} batches"
echo "[FINAL BN CAL]  ${ELASTIC_FINAL_BN_CAL_BATCHES} val batches"
echo "[PROFILE]       uniform=${ELASTIC_UNIFORM_P} transition=${ELASTIC_TRANSITION_P} hard=${ELASTIC_HARD_P} anchor=${ELASTIC_ANCHOR_P}"
echo "[HISTORY]       same=${ELASTIC_HISTORY_SAME_P} independent=${ELASTIC_HISTORY_INDEP_P} jump=${ELASTIC_HISTORY_JUMP_P}; grad_slots=${ELASTIC_HISTORY_GRAD_SLOTS}"
echo "[NORM]          per-width BatchNorm during training; Conv/BN fused at deployment"
echo "[RUNTIME]       all leading 1.0 stages are native Full; train only 83 non-Full profiles"

python tools/train_elastic_bev_v4_bn.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --epochs "${ELASTIC_EPOCHS}" \
    --workers "${ELASTIC_WORKERS}" \
    --exp_name "${ELASTIC_EXP_NAME}" \
    --lr "${ELASTIC_LR}" \
    --min_lr "${ELASTIC_MIN_LR}" \
    --weight_decay "${ELASTIC_WD}" \
    --pre_bn_calibration_batches "${ELASTIC_PRE_BN_CAL_BATCHES}" \
    --det_weight 1.0 \
    --bev_weight "${ELASTIC_BEV_W}" \
    --cos_weight "${ELASTIC_COS_W}" \
    --stage_weight "${ELASTIC_STAGE_W}" \
    --stage_cos_ratio 0.10 \
    --history_bev_weight "${ELASTIC_HISTORY_BEV_W}" \
    --history_cos_weight "${ELASTIC_HISTORY_COS_W}" \
    --smooth_l1_beta 0.1 \
    --uniform_prob "${ELASTIC_UNIFORM_P}" \
    --transition_prob "${ELASTIC_TRANSITION_P}" \
    --hard_prob "${ELASTIC_HARD_P}" \
    --anchor_prob "${ELASTIC_ANCHOR_P}" \
    --canonical_history_prob "${ELASTIC_CANONICAL_HISTORY_PROB}" \
    --history_same_profile_prob "${ELASTIC_HISTORY_SAME_P}" \
    --history_independent_profile_prob "${ELASTIC_HISTORY_INDEP_P}" \
    --history_jump_profile_prob "${ELASTIC_HISTORY_JUMP_P}" \
    --history_grad_slots "${ELASTIC_HISTORY_GRAD_SLOTS}" \
    --warmup_ratio 0.05 \
    --grad_clip 5.0 \
    --log_interval 20 \
    --save_interval 1

RAW_CKPT="${OUT_DIR}/ckpt/checkpoint_epoch_${ELASTIC_EPOCHS}.pth"
CAL_CKPT="${OUT_DIR}/ckpt/checkpoint_epoch_${ELASTIC_EPOCHS}_bncal.pth"

require_file "${RAW_CKPT}"

banner "Final BN calibration | balanced 83-profile hybrid deployment distribution"

python tools/calibrate_elastic_v4_bn.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${RAW_CKPT}" \
    --output_ckpt "${CAL_CKPT}" \
    --batches "${ELASTIC_FINAL_BN_CAL_BATCHES}" \
    --workers "${ELASTIC_WORKERS}"

echo
echo "[DONE] Training checkpoint : ${RAW_CKPT}"
echo "[DONE] Deploy checkpoint   : ${CAL_CKPT}"
echo "[IMPORTANT] Use the *_bncal.pth checkpoint for sAP/latency tests."

