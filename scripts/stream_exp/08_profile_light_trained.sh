#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-10}"

CKPT="${LIGHT_CKPT_DIR}/checkpoint_epoch_${EPOCH}.pth"

require_file "${LIGHT_CFG}"
require_file "${CKPT}"
require_file "tools/profile_light_components.py"

OUT_DIR="outputs/profile_light_components/${LIGHT_EXP_NAME}_e${EPOCH}"
mkdir -p "${OUT_DIR}"

banner "Profile Trained Light-v1 | epoch ${EPOCH}"

echo "[PROFILE] cfg  = ${LIGHT_CFG}"
echo "[PROFILE] ckpt = ${CKPT}"

python tools/profile_light_components.py \
    --cfg_file "${LIGHT_CFG}" \
    --ckpt "${CKPT}" \
    --warmup 30 \
    --num_samples 500 \
    --skip_scene_prefix 3 \
    --workers 0 \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/console.log"
