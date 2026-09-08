#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

banner "STEP 03 | Train final Elastic-v4-BN / TV-Stream3D model"

require_file "${FULL_CKPT}"

ELASTIC_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}" \
ELASTIC_EPOCHS=20 \
./scripts/stream_exp/18_train_elastic_v4_bn_final.sh

RAW="outputs/elastic_bev/${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}/ckpt/checkpoint_epoch_20.pth"
require_file "${RAW}"

echo "[PASS] Elastic checkpoint = ${RAW}"
