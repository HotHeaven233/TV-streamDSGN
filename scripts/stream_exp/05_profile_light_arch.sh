#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

require_file "${LIGHT_CFG}"
require_file "${FULL_CKPT}"

OUT_DIR="outputs/profile_light_components/light_v1_arch"
mkdir -p "${OUT_DIR}"

banner "Profile Light-v1 Architecture | partial Full-e5 initialization"

python tools/profile_light_components.py \
    --cfg_file "${LIGHT_CFG}" \
    --ckpt "${FULL_CKPT}" \
    --warmup 30 \
    --num_samples 500 \
    --skip_scene_prefix 3 \
    --workers 0 \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/console.log"
