#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

banner "STEP 10 | Transtreaming-style 3D | 15ep best epoch13 | formal"

require_file scripts/stream_exp/55_run_transtreaming_v2_best_no_load.sh
require_file scripts/stream_exp/56_run_transtreaming_v2_best_random50.sh

# A) L0 / no-load frequency sweep.
./scripts/stream_exp/55_run_transtreaming_v2_best_no_load.sh \
    "35,40,45,50" \
    80 \
    0 \
    20260903

# B) Random50 @ 35 Hz.
./scripts/stream_exp/56_run_transtreaming_v2_best_random50.sh \
    35 \
    80 \
    "L1,L2,L3,L4" \
    0 \
    20260903 \
    0.5
