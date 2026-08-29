#!/usr/bin/env bash

# ============================================================
# IMPORTANT:
# Conda activate.d scripts are not guaranteed to be compatible
# with bash "set -u" (nounset).
#
# Therefore temporarily disable nounset while activating Conda,
# then restore the caller's original nounset state.
# ============================================================

__STREAM_EXP_HAD_NOUNSET=0

case "$-" in
    *u*)
        __STREAM_EXP_HAD_NOUNSET=1
        ;;
esac

set +u
set -eo pipefail

# ============================================================
# Repository / Conda
# ============================================================

export REPO_ROOT="/data/jhb/workspace/streamDSGN"
export CONDA_ENV_PATH="/data/jhb/conda_envs/streamdsgn4090"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find conda.sh"
    exit 1
fi

conda activate "${CONDA_ENV_PATH}"

cd "${REPO_ROOT}"

# ============================================================
# Runtime environment
# ============================================================

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/mmdetection-v2.22.0:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export PYTHONUNBUFFERED=1

# ============================================================
# Stable experiment paths
# ============================================================

# Original StreamDSGN
export ORIGINAL_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl.yaml"
export ORIGINAL_CKPT="extra_data/checkpoint_epoch_20.pth"

# Full: K3 multi-history residual, epoch 5
export FULL_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual.yaml"

export FULL_CKPT="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual.mh3_residual_retrain/ckpt/checkpoint_epoch_5.pth"

# Light-v1
export LIGHT_CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-light_v1.yaml"

# Light training experiment.
# Can be overridden from command line:
#
# LIGHT_EXP_NAME=xxx ./scripts/stream_exp/06_train_light.sh
#
export LIGHT_EXP_NAME="${LIGHT_EXP_NAME:-light_distill_v1_retrain10}"

export LIGHT_CKPT_DIR="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-light_v1.${LIGHT_EXP_NAME}/ckpt"

# MH retraining experiment
export MH_RETRAIN_EXP_NAME="${MH_RETRAIN_EXP_NAME:-mh3_residual_retrain}"

export MH_RETRAIN_CKPT_DIR="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual.${MH_RETRAIN_EXP_NAME}/ckpt"

# ============================================================
# Helpers
# ============================================================

require_file() {
    local f="$1"

    if [ ! -f "$f" ]; then
        echo
        echo "[ERROR] Required file does not exist:"
        echo "        $f"
        echo
        exit 1
    fi
}

banner() {
    echo
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

# ============================================================
# Diagnostics
# ============================================================

echo "[ENV] repo        = ${REPO_ROOT}"
echo "[ENV] conda env   = ${CONDA_PREFIX:-unknown}"
echo "[ENV] python      = $(which python)"
echo "[ENV] python ver  = $(python --version 2>&1)"
echo "[ENV] CUDA device = ${CUDA_VISIBLE_DEVICES}"

# Restore nounset only if caller already had it enabled.
if [ "${__STREAM_EXP_HAD_NOUNSET}" -eq 1 ]; then
    set -u
fi

unset __STREAM_EXP_HAD_NOUNSET
