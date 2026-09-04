#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

ELASTIC_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v3}"
ELASTIC_EPOCHS="${ELASTIC_EPOCHS:-20}"
ELASTIC_WORKERS="${ELASTIC_WORKERS:-4}"
ELASTIC_LR="${ELASTIC_LR:-1e-4}"
ELASTIC_MIN_LR="${ELASTIC_MIN_LR:-1e-6}"
ELASTIC_WD="${ELASTIC_WD:-1e-4}"

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

# Reuse the learned v2 shared convolutional kernels when available. New v3
# GroupNorm and per-width BEV projections are initialized independently.
V2_INIT="${ELASTIC_INIT_CKPT:-outputs/elastic_bev/elastic_bev_v2/ckpt/checkpoint_epoch_20.pth}"
INIT_ARGS=()
if [ -f "${V2_INIT}" ]; then
    INIT_ARGS+=(--init_elastic_ckpt "${V2_INIT}")
    echo "[INIT]          partial shared-kernel init from ${V2_INIT}"
else
    echo "[INIT]          no prior elastic checkpoint; initialize from frozen Full"
fi

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "tools/train_elastic_bev_v3.py"
require_file "pcdet/models/backbones_3d_stream/elastic_bev_branch_v3.py"

OUT_DIR="outputs/elastic_bev/${ELASTIC_EXP_NAME}"
if compgen -G "${OUT_DIR}/ckpt/checkpoint_epoch_*.pth" > /dev/null; then
    echo "[ERROR] Existing Elastic-v3 checkpoints found: ${OUT_DIR}/ckpt"
    echo "Use another name or resume explicitly."
    exit 1
fi

banner "Train Elastic-v3 | Profile-space robust | Frozen K3"
echo "[BASE CFG]      ${FULL_CFG}"
echo "[BASE CKPT]     ${FULL_CKPT}"
echo "[EXP]           ${ELASTIC_EXP_NAME}"
echo "[EPOCHS]        ${ELASTIC_EPOCHS}"
echo "[LR]            ${ELASTIC_LR} -> ${ELASTIC_MIN_LR}"
echo "[PROFILE]       uniform=${ELASTIC_UNIFORM_P} transition=${ELASTIC_TRANSITION_P} hard=${ELASTIC_HARD_P} anchor=${ELASTIC_ANCHOR_P}"
echo "[HISTORY]       same=${ELASTIC_HISTORY_SAME_P} independent=${ELASTIC_HISTORY_INDEP_P} jump=${ELASTIC_HISTORY_JUMP_P}; grad_slots=${ELASTIC_HISTORY_GRAD_SLOTS}"
echo "[HISTORY TIME]  t-3,t-2,t-1 prob=${ELASTIC_CANONICAL_HISTORY_PROB}; otherwise 3 from t-5..t-1"
echo "[LOSS]          det=1.0 bev=${ELASTIC_BEV_W} cos=${ELASTIC_COS_W} stage=${ELASTIC_STAGE_W} hist_bev=${ELASTIC_HISTORY_BEV_W} hist_cos=${ELASTIC_HISTORY_COS_W}"
echo "[NORM]          per-width GroupNorm, no running statistics"
echo "[BEV PROJ]      independent 1x1 projection for 25/50/75/100%"
echo "[BASE]          K3 fusion/VAN/head frozen and bitwise checked"

python tools/train_elastic_bev_v3.py \
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
    --save_interval 1 \
    "${INIT_ARGS[@]}"
