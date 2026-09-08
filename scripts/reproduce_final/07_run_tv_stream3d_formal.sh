#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

banner "STEP 07 | TV-Stream3D formal evaluation"

# A) L0 / no-load frequency sweep.
for HZ in 35 40 45 50
do
    ./scripts/stream_exp/44_run_tv_stream3d_30hz_levels.sh \
        20 \
        100 \
        "${HZ}" \
        80 \
        "L0" \
        0 \
        0.25
done

# B) Random50 @ 35 Hz, four pressure severities.
./scripts/stream_exp/45_run_tv_stream3d_30hz_random50.sh \
    20 \
    100 \
    35 \
    80 \
    "L1,L2,L3,L4" \
    0 \
    0.25 \
    20260903 \
    0.5
