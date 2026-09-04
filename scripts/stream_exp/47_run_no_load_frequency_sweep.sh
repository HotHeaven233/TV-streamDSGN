#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
FREQUENCIES="${3:-35,40,45,50}"
WARMUP="${4:-80}"
MAX_FRAMES="${5:-0}"
GUARD="${6:-0.25}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"

BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"

PROFILE_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile_v6/e${EPOCH}_n${N}"

CTRL_CSV="${PROFILE_ROOT}/controller_remaining_latency_table.csv"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

TV_EVAL="tools/eval_tv_stream3d_30hz_levels.py"

ORI_EVAL="tools/eval_original_streamdsgn_no_load.py"

SUMMARY_ROOT="outputs/no_load_frequency_sweep"

mkdir -p "${SUMMARY_ROOT}"

for f in \
    "${ORIGINAL_CFG}" \
    "${ORIGINAL_CKPT}" \
    "${FULL_CFG}" \
    "${FULL_CKPT}" \
    "${RAW_CKPT}" \
    "${BANK}" \
    "${CTRL_CSV}" \
    "${LEVELS_JSON}" \
    "${TV_EVAL}" \
    "${ORI_EVAL}" \
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

IFS=',' read -r -a HZ_ARRAY <<< "${FREQUENCIES}"

for raw_hz in "${HZ_ARRAY[@]}"
do
    hz="$(echo "${raw_hz}" | xargs)"

    if ! [[ "${hz}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "[ERROR] invalid Hz: ${hz}"
        exit 3
    fi

    TV_OUT="outputs/elastic_bev/${EXP_NAME}/formal_streaming/${hz}Hz_forward_only/L0"

    ORI_OUT="outputs/original_streamdsgn/formal_streaming_no_load/${hz}Hz_forward_only/L0"

    mkdir -p "${TV_OUT}"
    mkdir -p "${ORI_OUT}"

    echo
    echo "===================================================================================================="
    echo "TV-Stream3D NO-LOAD FREQUENCY SWEEP"
    echo "===================================================================================================="
    echo "Hz          : ${hz}"
    echo "contention  : L0"
    echo "warmup      : ${WARMUP}"
    echo "max frames  : ${MAX_FRAMES}"
    echo "guard       : ${GUARD}"
    echo "output      : ${TV_OUT}"
    echo "===================================================================================================="

    python "${TV_EVAL}" \
        --full_cfg "${FULL_CFG}" \
        --full_ckpt "${FULL_CKPT}" \
        --elastic_ckpt "${RAW_CKPT}" \
        --prefix_bn_bank "${BANK}" \
        --controller_csv "${CTRL_CSV}" \
        --levels_json "${LEVELS_JSON}" \
        --true_level "L0" \
        --input_hz "${hz}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --workers 0 \
        --seed 1024 \
        --control_guard_per_boundary_ms "${GUARD}" \
        --output_dir "${TV_OUT}" \
        2>&1 | tee "${TV_OUT}/run.log"

    echo
    echo "===================================================================================================="
    echo "ORIGINAL StreamDSGN NO-LOAD FREQUENCY SWEEP"
    echo "===================================================================================================="
    echo "Hz          : ${hz}"
    echo "contention  : L0"
    echo "warmup      : ${WARMUP}"
    echo "max frames  : ${MAX_FRAMES}"
    echo "output      : ${ORI_OUT}"
    echo "===================================================================================================="

    python "${ORI_EVAL}" \
        --cfg "${ORIGINAL_CFG}" \
        --ckpt "${ORIGINAL_CKPT}" \
        --levels_json "${LEVELS_JSON}" \
        --pressure_level "L0" \
        --pressure_fraction 0.0 \
        --trace_seed 20260903 \
        --input_hz "${hz}" \
        --runtime_warmup_frames "${WARMUP}" \
        --max_frames "${MAX_FRAMES}" \
        --workers 0 \
        --seed 1024 \
        --output_dir "${ORI_OUT}" \
        2>&1 | tee "${ORI_OUT}/run.log"

    # Strictly verify that this Original run really had no contention.
    python - "${ORI_OUT}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])

if not p.exists():
    raise RuntimeError(
        f"missing summary: {p}"
    )

s = json.loads(
    p.read_text()
)

levels = s.get(
    "true_sensor_levels",
    {}
)

sensor_frames = int(
    s["sensor_frames"]
)

if set(levels.keys()) != {"L0"}:
    raise RuntimeError(
        f"NO-LOAD CHECK FAILED: {levels}"
    )

if int(levels["L0"]) != sensor_frames:
    raise RuntimeError(
        "NO-LOAD CHECK FAILED: "
        f"L0={levels['L0']} "
        f"sensor_frames={sensor_frames}"
    )

if not s.get("no_load", False):
    raise RuntimeError(
        "NO-LOAD metadata flag is false"
    )

print(
    "[NO-LOAD CHECK PASS] "
    f"all {sensor_frames} sensor frames are L0"
)
PY

done

python - \
    "${EXP_NAME}" \
    "${FREQUENCIES}" \
    "${SUMMARY_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

exp_name = sys.argv[1]

freqs = [
    x.strip()
    for x
    in sys.argv[2].split(",")
    if x.strip()
]

root = Path(
    sys.argv[3]
)

rows = []

for hz in freqs:
    paths = {
        "TV-Stream3D": (
            Path(
                "outputs/elastic_bev"
            )
            /
            exp_name
            /
            "formal_streaming"
            /
            f"{hz}Hz_forward_only"
            /
            "L0"
            /
            "summary.json"
        ),

        "Original StreamDSGN": (
            Path(
                "outputs/original_streamdsgn"
            )
            /
            "formal_streaming_no_load"
            /
            f"{hz}Hz_forward_only"
            /
            "L0"
            /
            "summary.json"
        ),
    }

    for method, p in paths.items():
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
            "Hz":
                float(hz),

            "period_ms":
                1000.0
                /
                float(hz),

            "method":
                method,

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
    "no_load_frequency_sweep_summary.csv"
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
print("=" * 126)
print("NO-LOAD FREQUENCY SWEEP SUMMARY")
print("=" * 126)

for hz in freqs:
    print(
        f"\n--- {hz} Hz "
        f"(T={1000.0/float(hz):.3f} ms) ---"
    )

    for r in rows:
        if (
            abs(
                r["Hz"]
                -
                float(hz)
            )
            >
            1e-9
        ):
            continue

        print(
            f"{r['method']:20s} | "
            f"Macro={r['Macro_3d_mod_R40']:.4f} | "
            f"Car/Ped/Cyc="
            f"{r['Car_3d_mod_R40']:.4f}/"
            f"{r['Pedestrian_3d_mod_R40']:.4f}/"
            f"{r['Cyclist_3d_mod_R40']:.4f} | "
            f"miss="
            f"{100*r['deadline_miss_rate']:.3f}% | "
            f"drop="
            f"{100*r['drop_rate']:.3f}% | "
            f"p99="
            f"{r['forward_p99_ms']:.4f} ms"
        )

print()
print(
    f"CSV = {out}"
)
print("=" * 126)
PY

# NO_LOAD_FREQUENCY_SWEEP_EOF
