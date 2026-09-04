#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
HZ="${3:-30}"
WARMUP="${4:-80}"
PRESSURE_LEVELS="${5:-L1,L2,L3,L4}"
MAX_FRAMES="${6:-0}"
GUARD="${7:-0.25}"
TRACE_SEED="${8:-20260903}"
PRESSURE_FRACTION="${9:-0.5}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"

BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"

PROFILE_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile_v6/e${EPOCH}_n${N}"

CTRL_CSV="${PROFILE_ROOT}/controller_remaining_latency_table.csv"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

OUT_ROOT="outputs/elastic_bev/${EXP_NAME}/formal_streaming_random50/${HZ}Hz_forward_only_seed${TRACE_SEED}"

for f in \
    "${FULL_CFG}" \
    "${FULL_CKPT}" \
    "${RAW_CKPT}" \
    "${BANK}" \
    "${CTRL_CSV}" \
    "${LEVELS_JSON}" \
    "tools/eval_tv_stream3d_30hz_random50.py" \
    "tools/test_tv_stream3d_online_forward.py" \
    "tools/test_stream_buffer_timestamp.py" \
    "tools/tv_stream3d_controller.py" \
    "tools/tv_stream3d_causal_fused_runtime.py" \
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
    echo "TV-Stream3D RANDOM CONTENTION"
    echo "===================================================================================================="
    echo "input Hz          : ${HZ}"
    echo "trace             : L0 + ${pressure}"
    echo "pressure fraction : ${PRESSURE_FRACTION}"
    echo "trace seed        : ${TRACE_SEED}"
    echo "warmup            : ${WARMUP}"
    echo "max frames        : ${MAX_FRAMES} (0 = full dataset)"
    echo "guard             : ${GUARD} ms/boundary"
    echo "output            : ${OUT_DIR}"
    echo "===================================================================================================="

    python tools/eval_tv_stream3d_30hz_random50.py \
        --full_cfg "${FULL_CFG}" \
        --full_ckpt "${FULL_CKPT}" \
        --elastic_ckpt "${RAW_CKPT}" \
        --prefix_bn_bank "${BANK}" \
        --controller_csv "${CTRL_CSV}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level "${pressure}" \
        --pressure_fraction "${PRESSURE_FRACTION}" \
        --trace_seed "${TRACE_SEED}" \
        --input_hz "${HZ}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --workers 0 \
        --seed 1024 \
        --control_guard_per_boundary_ms "${GUARD}" \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"
done


python - "${OUT_ROOT}" "${PRESSURE_LEVELS}" "${PRESSURE_FRACTION}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

levels = [
    x.strip()
    for x in sys.argv[2].split(",")
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

    trace = s[
        "contention_trace"
    ]

    sensor_levels = s[
        "true_sensor_levels"
    ]

    processed_levels = s[
        "true_processed_levels"
    ]

    rows.append({
        "pressure_level":
            level,

        "pressure_fraction_target":
            trace["pressure_fraction_target"],

        "trace_seed":
            trace["trace_seed"],

        "trace_switch_rate":
            trace["trace_switch_rate"],

        "sensor_L0":
            sensor_levels.get(
                "L0",
                0,
            ),

        "sensor_pressure":
            sensor_levels.get(
                level,
                0,
            ),

        "processed_L0":
            processed_levels.get(
                "L0",
                0,
            ),

        "processed_pressure":
            processed_levels.get(
                level,
                0,
            ),

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

        "classifier_accuracy":
            s["classifier_accuracy"],

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
    "all_random50_summary.csv"
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
print("RANDOM-50 CONTENTION SUMMARY")
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
        f"cls={100*r['classifier_accuracy']:.2f}% | "
        f"p99={r['forward_p99_ms']:.4f} ms"
    )

print(f"CSV = {out}")
print("=" * 110)
PY

# TV_RANDOM50_RUN_EOF
