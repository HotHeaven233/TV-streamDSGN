#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN

source scripts/stream_exp/00_env.sh

TS_CFG="${1:-configs/stream/kitti_models/stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat.yaml}"

python tools/patch_transtreaming_dataset.py

python tools/build_transtreaming_config.py \
    --base_cfg "$FULL_CFG" \
    --output "$TS_CFG" \
    --epochs 5 \
    --lr 2e-4

python -m py_compile \
    pcdet/models/fusion_module/transtreaming_bev_tat.py \
    pcdet/models/detectors_stream/transtreaming_stream.py \
    pcdet/datasets/kitti_streaming/stereo_kitti_streaming.py \
    tools/build_transtreaming_config.py

echo
echo "============================================================"
echo "Transtreaming preparation OK"
echo "============================================================"
echo "BASE : $FULL_CFG"
echo "CFG  : $TS_CFG"
echo "CKPT : $FULL_CKPT"
echo

python - <<PY
import yaml

p = "$TS_CFG"

with open(p) as f:
    c = yaml.safe_load(f)

print("MODEL.NAME =", c["MODEL"]["NAME"])
print(
    "FUSION =",
    c["MODEL"]["FUSION_IN_SPATIAL_FEATURES"]["NAME"]
)
print(
    "P^F =",
    c["MODEL"]["TRANSTREAMING"]["TRAIN_FUTURE_STEPS"]
)
print(
    "P^P patterns =",
    c["MODEL"]["TRANSTREAMING"]["TRAIN_PAST_PATTERNS"]
)
print(
    "dense supervision =",
    c["MODEL"]["DENSE_HEAD"]["BOX3D_SUPERVISION"]
)
print(
    "dense history tag =",
    c["MODEL"]["DENSE_HEAD"]["HISTORY_TAG"]
)
PY
