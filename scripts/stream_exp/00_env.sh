#!/usr/bin/env bash
# Portable experiment environment for StreamDSGN / TV-Stream3D reproduction.
#
# Usage:
#   1) Activate the intended Python environment yourself, OR
#   2) export STREAMDSGN_CONDA_ENV=/absolute/path/to/conda/env
#
# Optional:
#   export STREAMDSGN_REPO_ROOT=/path/to/streamDSGN
#   export STREAMDSGN_ORIGINAL_CKPT=/path/to/checkpoint_epoch_20.pth
#   export CUDA_VISIBLE_DEVICES=0

__STREAM_EXP_HAD_NOUNSET=0
case "$-" in
    *u*) __STREAM_EXP_HAD_NOUNSET=1 ;;
esac

set +u
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT="${STREAMDSGN_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ----------------------------------------------------------------------
# Optional machine-local configuration.
#
# This file is intentionally not committed to Git.
# It may define:
#   STREAMDSGN_REPO_ROOT
#   STREAMDSGN_CONDA_ENV
#   STREAMDSGN_ORIGINAL_CKPT
#   CUDA_VISIBLE_DEVICES
# ----------------------------------------------------------------------
LOCAL_ENV_FILE="${REPO_ROOT}/.streamdsgn_local_env.sh"

if [ -f "${LOCAL_ENV_FILE}" ]; then
    source "${LOCAL_ENV_FILE}"

    # The local file is allowed to override the repo path.
    export REPO_ROOT="${STREAMDSGN_REPO_ROOT:-${REPO_ROOT}}"
fi

# Optional conda activation. If not specified, keep the caller's environment.
if [ -n "${STREAMDSGN_CONDA_ENV:-}" ]; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        echo "[ERROR] STREAMDSGN_CONDA_ENV is set but conda cannot be initialized."
        exit 1
    fi

    conda activate "${STREAMDSGN_CONDA_ENV}"
fi

export CONDA_ENV_PATH="${STREAMDSGN_CONDA_ENV:-${CONDA_PREFIX:-}}"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/mmdetection-v2.22.0:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# ----------------------------------------------------------------------
# Stable experiment paths.
# ----------------------------------------------------------------------

# TRUE vanilla Original StreamDSGN.
export ORIGINAL_CFG="${STREAMDSGN_ORIGINAL_CFG:-configs/stream/kitti_models/stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl.yaml}"
export ORIGINAL_CKPT="${STREAMDSGN_ORIGINAL_CKPT:-extra_data/checkpoint_epoch_20.pth}"

# K3 multi-history residual model produced by reproduce step 02.
export FULL_CFG="${STREAMDSGN_FULL_CFG:-configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual.yaml}"
export MH_RETRAIN_EXP_NAME="${MH_RETRAIN_EXP_NAME:-mh3_residual_retrain}"
export MH_RETRAIN_CKPT_DIR="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual.${MH_RETRAIN_EXP_NAME}/ckpt"
export FULL_CKPT="${STREAMDSGN_FULL_CKPT:-${MH_RETRAIN_CKPT_DIR}/checkpoint_epoch_5.pth}"

# Legacy variables kept because older scripts source this file.
export LIGHT_CFG="${LIGHT_CFG:-configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-light_v1.yaml}"
export LIGHT_EXP_NAME="${LIGHT_EXP_NAME:-light_distill_v1_retrain10}"
export LIGHT_CKPT_DIR="outputs/stream_kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-light_v1.${LIGHT_EXP_NAME}/ckpt"

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

echo "[ENV] repo        = ${REPO_ROOT}"
echo "[ENV] conda env   = ${CONDA_PREFIX:-caller environment}"
echo "[ENV] python      = $(command -v python || true)"
echo "[ENV] python ver  = $(python --version 2>&1 || true)"
echo "[ENV] CUDA device = ${CUDA_VISIBLE_DEVICES}"
echo "[ENV] original    = ${ORIGINAL_CKPT}"

if [ "${__STREAM_EXP_HAD_NOUNSET}" -eq 1 ]; then
    set -u
fi
unset __STREAM_EXP_HAD_NOUNSET
