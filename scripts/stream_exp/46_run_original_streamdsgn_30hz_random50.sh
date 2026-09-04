#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

HZ="${1:-30}"
WARMUP="${2:-80}"
PRESSURE_LEVELS="${3:-L1,L2,L3,L4}"
MAX_FRAMES="${4:-0}"
TRACE_SEED="${5:-20260903}"
PRESSURE_FRACTION="${6:-0.5}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

OUT_ROOT="outputs/original_streamdsgn/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}"

for f in \
    "${FULL_CFG}" \
    "${FULL_CKPT}" \
    "${LEVELS_JSON}" \
    "tools/eval_original_streamdsgn_30hz_random50.py" \
    "tools/eval_tv_stream3d_30hz_random50.py" \
    "tools/test_stream_buffer_timestamp.py" \
    "tools/test_tv_stream3d_online_forward.py" \
    "tools/smooth_cuda_contention.py"
do
    if [ ! -f "${f}" ]; then
        echo "[ERROR] missing: ${f}"
        exit 2
    fi
done

mkdir -p "${OUT_ROOT}"

IFS=',' read -r -a LEVEL_ARRAY <<< "${PRESSURE_LEVELS}"

for raw in "${LEVEL_ARRAY[@]}"
do
    pressure="$(echo "${raw}" | xargs)"

    case "${pressure}" in
        L1|L2|L3|L4)
            ;;
        *)
            echo "[ERROR] invalid pressure level: ${pressure}"
            exit 3
            ;;
    esac

    OUT_DIR="${OUT_ROOT}/L0_${pressure}_p${PRESSURE_FRACTION}"

    mkdir -p "${OUT_DIR}"

    echo
    echo "===================================================================================================="
    echo "ORIGINAL StreamDSGN RANDOM CONTENTION"
    echo "===================================================================================================="
    echo "cfg               : ${FULL_CFG}"
    echo "ckpt              : ${FULL_CKPT}"
    echo "input Hz          : ${HZ}"
    echo "trace             : L0 + ${pressure}"
    echo "pressure fraction : ${PRESSURE_FRACTION}"
    echo "trace seed        : ${TRACE_SEED}"
    echo "warmup            : ${WARMUP}"
    echo "max frames        : ${MAX_FRAMES}"
    echo "output            : ${OUT_DIR}"
    echo "===================================================================================================="

    python tools/eval_original_streamdsgn_30hz_random50.py \
        --cfg "${FULL_CFG}" \
        --ckpt "${FULL_CKPT}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level "${pressure}" \
        --pressure_fraction "${PRESSURE_FRACTION}" \
        --trace_seed "${TRACE_SEED}" \
        --input_hz "${HZ}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --workers 0 \
        --seed 1024 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"

    # ------------------------------------------------------------------
    # 如果 TV-Stream3D 对应 trace 已经存在，强制检查二者完全一致。
    # ------------------------------------------------------------------

    TV_TRACE="outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}/L0_${pressure}_p${PRESSURE_FRACTION}/contention_trace.csv"

    BASE_TRACE="${OUT_DIR}/contention_trace.csv"

    if [ -f "${TV_TRACE}" ]; then
        if cmp -s "${TV_TRACE}" "${BASE_TRACE}"; then
            echo "[TRACE CHECK PASS] Original and TV traces are byte-identical."
        else
            echo "[ERROR] contention trace differs from TV-Stream3D!"
            echo "TV      : ${TV_TRACE}"
            echo "Original: ${BASE_TRACE}"
            exit 10
        fi
    else
        echo "[WARN] TV trace not found; exact trace comparison skipped:"
        echo "       ${TV_TRACE}"
    fi
done


python - "${OUT_ROOT}" "${PRESSURE_LEVELS}" "${PRESSURE_FRACTION}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

levels = [
    x.strip()
    for x
    in sys.argv[2].split(",")
    if x.strip()
]

fraction = sys.argv[3]

rows = []

for level in levels:
    p = (
        root
        /
        f"L0_{level}_p{fraction}"
        /
        "summary.json"
    )

    if not p.exists():
        raise RuntimeError(
            f"missing summary: {p}"
        )

    s = json.loads(
        p.read_text()
    )

    q = s[
        "stream_sap_3d_moderate_R40"
    ]

    f = s[
        "forward_latency"
    ]

    rows.append({
        "pressure_level":
            level,

        "sensor_frames":
            s["sensor_frames"],

        "processed_frames":
            s["processed_frames"],

        "dropped_frames":
            s["dropped_frames"],

        "drop_rate":
            s["drop_rate"],

        "deadline_miss_count":
            s["deadline_miss_count"],

        "deadline_miss_rate":
            s["deadline_miss_rate"],

        "forward_p50_ms":
            f["p50_ms"],

        "forward_p90_ms":
            f["p90_ms"],

        "forward_p99_ms":
            f["p99_ms"],

        "Car_3d_mod_R40":
            q["Car"],

        "Pedestrian_3d_mod_R40":
            q["Pedestrian"],

        "Cyclist_3d_mod_R40":
            q["Cyclist"],

        "Macro_3d_mod_R40":
            q["Macro"],
    })

out = (
    root
    /
    "all_original_random50_summary.csv"
)

with out.open(
    "w",
    newline="",
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    w.writeheader()
    w.writerows(rows)

print()
print("=" * 110)
print("ORIGINAL StreamDSGN RANDOM-50 SUMMARY")
print("=" * 110)

for r in rows:
    print(
        f"L0/{r['pressure_level']} | "
        f"Car/Ped/Cyc/Macro="
        f"{r['Car_3d_mod_R40']:.4f}/"
        f"{r['Pedestrian_3d_mod_R40']:.4f}/"
        f"{r['Cyclist_3d_mod_R40']:.4f}/"
        f"{r['Macro_3d_mod_R40']:.4f} | "
        f"miss={100*r['deadline_miss_rate']:.3f}% | "
        f"drop={100*r['drop_rate']:.3f}% | "
        f"p99={r['forward_p99_ms']:.4f} ms"
    )

print(f"CSV = {out}")
print("=" * 110)
PY

# ORIGINAL_RANDOM50_RUN_EOF
