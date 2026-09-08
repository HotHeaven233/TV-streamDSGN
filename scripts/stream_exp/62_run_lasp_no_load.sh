#!/usr/bin/env bash
set -euo pipefail

cd /data/jhb/workspace/streamDSGN
source scripts/stream_exp/00_env.sh

CFG="configs/stream/kitti_models/stream_dsgn_r18-token_prev_next-lasp_style_15ep.yaml"

CKPT="outputs/stream_kitti_models/stream_dsgn_r18-token_prev_next-lasp_style_15ep.lasp_style_15ep/ckpt/checkpoint_epoch_2.pth"

ROOT="outputs/stream_exp/lasp_style_final/no_load"

for F in \
    "$CFG" \
    "$CKPT" \
    tools/eval_lasp_stream.py
do
    [[ -f "$F" ]] || {
        echo "[ERROR] missing: $F" >&2
        exit 2
    }
done

sha256sum -c \
    outputs/lasp/final/checkpoint.sha256

for HZ in 35 40 45 50
do
    OUT="${ROOT}/${HZ}hz"

    echo
    echo "============================================================"
    echo "LASP NO-LOAD @ ${HZ} Hz"
    echo "============================================================"

    rm -rf "$OUT"

    python tools/eval_lasp_stream.py \
        --cfg "$CFG" \
        --ckpt "$CKPT" \
        --pressure_level L0 \
        --input_hz "$HZ" \
        --runtime_warmup_frames 80 \
        --max_frames 0 \
        --workers 0 \
        --seed 1024 \
        --output_dir "$OUT"

    [[ -f "$OUT/summary.json" ]] || {
        echo "[ERROR] missing summary: $OUT/summary.json" >&2
        exit 3
    }
done

echo
echo "[PASS] LASP no-load 35/40/45/50 complete"
