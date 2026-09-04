#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

FREQUENCIES="${1:-35,40,45,50}"
WARMUP="${2:-80}"
MAX_FRAMES="${3:-0}"
TRACE_SEED="${4:-20260903}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

EVAL="tools/eval_original_streamdsgn_no_load.py"

require_file "${ORIGINAL_CFG}"
require_file "${ORIGINAL_CKPT}"
require_file "${LEVELS_JSON}"
require_file "${EVAL}"

if [ "${ORIGINAL_CFG}" = "${FULL_CFG}" ] || [ "${ORIGINAL_CKPT}" = "${FULL_CKPT}" ]; then
    echo "[ERROR] ORIGINAL_* unexpectedly equals FULL_*"
    exit 20
fi

echo
echo "======================================================================"
echo "TRUE ORIGINAL StreamDSGN | NO-LOAD FREQUENCY SWEEP"
echo "======================================================================"
echo "cfg         : ${ORIGINAL_CFG}"
echo "ckpt        : ${ORIGINAL_CKPT}"
echo "frequencies : ${FREQUENCIES}"
echo "contention  : L0 only"
echo "timing      : forward-only"
echo "======================================================================"

IFS=',' read -r -a HZ_ARRAY <<< "${FREQUENCIES}"

for raw_hz in "${HZ_ARRAY[@]}"
do
    HZ="$(echo "${raw_hz}" | xargs)"

    if ! [[ "${HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "[ERROR] invalid Hz: ${HZ}"
        exit 3
    fi

    OUT_DIR="outputs/original_streamdsgn/formal_streaming_no_load/${HZ}Hz_forward_only/L0"

    mkdir -p "${OUT_DIR}"

    echo
    echo "======================================================================"
    echo "Original StreamDSGN | ${HZ} Hz | L0"
    echo "======================================================================"

    python "${EVAL}" \
        --cfg "${ORIGINAL_CFG}" \
        --ckpt "${ORIGINAL_CKPT}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level "L0" \
        --pressure_fraction 0.0 \
        --trace_seed "${TRACE_SEED}" \
        --input_hz "${HZ}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --workers 0 \
        --seed 1024 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"

    python - "${OUT_DIR}/summary.json" "${ORIGINAL_CFG}" "${ORIGINAL_CKPT}" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
expected_cfg = sys.argv[2]
expected_ckpt = sys.argv[3]

s = json.loads(p.read_text())

assert s["method"] == "Original StreamDSGN"
assert s["base_detector"] == "vanilla_streamdsgn"
assert s["model_cfg"] == expected_cfg, (
    s["model_cfg"],
    expected_cfg,
)
assert s["model_ckpt"] == expected_ckpt, (
    s["model_ckpt"],
    expected_ckpt,
)

levels = s["true_sensor_levels"]

assert set(levels) == {"L0"}, levels
assert int(levels["L0"]) == int(s["sensor_frames"])
assert s["no_load"] is True

print(
    "[IDENTITY CHECK PASS] "
    "true Original StreamDSGN / L0 only"
)
PY

done

# TRUE_ORIGINAL_NOLOAD_EOF
