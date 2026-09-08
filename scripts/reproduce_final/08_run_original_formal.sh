#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

banner "STEP 08 | TRUE vanilla Original StreamDSGN formal evaluation"

if [ "${ORIGINAL_CFG}" = "${FULL_CFG}" ] || \
   [ "${ORIGINAL_CKPT}" = "${FULL_CKPT}" ]; then
    echo "[ERROR] Original baseline identity is contaminated by FULL/K3."
    exit 20
fi

# A) L0 / no-load frequency sweep.
./scripts/stream_exp/48_run_original_streamdsgn_no_load.sh \
    "35,40,45,50" \
    80 \
    0 \
    20260903

# B) Random50 @ 35 Hz.
# TV is run first (step 07), so script 46 can byte-compare the trace.
./scripts/stream_exp/46_run_original_streamdsgn_30hz_random50.sh \
    35 \
    80 \
    "L1,L2,L3,L4" \
    0 \
    20260903 \
    0.5
