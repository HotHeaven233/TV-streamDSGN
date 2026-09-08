#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

banner "STEP 05 | Train TRUE MTD / DAM H2-H3 heads"

# This is the committed MTD training entry. It builds the H2/H3 configs,
# performs training smoke tests, freezes the original detector except the
# selected future head, and trains each head for five epochs.
#
# Args: EPOCHS WORKERS LR
./scripts/stream_exp/51_train_mtd_heads.sh \
    5 \
    4 \
    0.0002

H2="outputs/stream_kitti_models/stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl-mtd_h2.mtd_h2_head_only/ckpt/checkpoint_epoch_5.pth"
H3="outputs/stream_kitti_models/stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl-mtd_h3.mtd_h3_head_only/ckpt/checkpoint_epoch_5.pth"

require_file "${H2}"
require_file "${H3}"

echo "[PASS] MTD H2 = ${H2}"
echo "[PASS] MTD H3 = ${H3}"
