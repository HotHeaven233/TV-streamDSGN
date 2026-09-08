#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

banner "STEP 01 | TRUE ORIGINAL StreamDSGN sanity"
./scripts/stream_exp/01_test_original.sh
