#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

banner "STEP 02 | Train K3 multi-history residual from ORIGINAL checkpoint"

./scripts/stream_exp/02_train_mh.sh

require_file "${FULL_CKPT}"

# Validate exactly the checkpoint used by all downstream TV/Transtreaming steps.
./scripts/stream_exp/03_test_mh.sh 5 5

echo "[PASS] K3 Full checkpoint = ${FULL_CKPT}"
