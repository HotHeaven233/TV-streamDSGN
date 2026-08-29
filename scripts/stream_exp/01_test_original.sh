#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

require_file "${ORIGINAL_CFG}"
require_file "${ORIGINAL_CKPT}"

OUT_DIR="outputs/stream_buffer_timestamp/original_10hz"
mkdir -p "${OUT_DIR}"

banner "Original StreamDSGN | capacity-1 streaming test | 10 Hz"

python tools/test_stream_buffer_timestamp.py \
    --cfg_file "${ORIGINAL_CFG}" \
    --ckpt "${ORIGINAL_CKPT}" \
    --input_hz 10 \
    --warmup 20 \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/console.log"
