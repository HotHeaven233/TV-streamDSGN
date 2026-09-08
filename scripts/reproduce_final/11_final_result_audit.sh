#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source scripts/stream_exp/00_env.sh

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"
SEED=20260903

banner "STEP 11 | Final result and exogenous-trace audit"

python - "${EXP_NAME}" "${SEED}" <<'PY'
import csv
import json
import sys
from pathlib import Path

exp = sys.argv[1]
seed = sys.argv[2]

methods = {
    "TV-Stream3D": {
        "nl": lambda hz: Path(
            f"outputs/elastic_bev/{exp}/formal_streaming/"
            f"{hz}Hz_forward_only/L0"
        ),
        "rnd": lambda lv: Path(
            f"outputs/elastic_bev/{exp}/formal_streaming_random50/"
            f"35Hz_forward_only_seed{seed}/L0_{lv}_p0.5"
        ),
    },
    "Original": {
        "nl": lambda hz: Path(
            f"outputs/original_streamdsgn/formal_streaming_no_load/"
            f"{hz}Hz_forward_only/L0"
        ),
        "rnd": lambda lv: Path(
            f"outputs/original_streamdsgn/formal_streaming_random50/"
            f"35Hz_forward_only_seed{seed}/L0_{lv}_p0.5"
        ),
    },
    "MTD": {
        "nl": lambda hz: Path(
            f"outputs/mtd_three_head/formal_streaming_no_load/"
            f"{hz}Hz_forward_only/L0"
        ),
        "rnd": lambda lv: Path(
            f"outputs/mtd_three_head/formal_streaming_random50/"
            f"35Hz_forward_only_seed{seed}/L0_{lv}_p0.5"
        ),
    },
    "Transtreaming": {
        "nl": lambda hz: Path(
            f"outputs/transtreaming_v2_best/formal_streaming_no_load/"
            f"{hz}Hz_forward_only/L0"
        ),
        "rnd": lambda lv: Path(
            f"outputs/transtreaming_v2_best/formal_streaming_random50/"
            f"35Hz_forward_only_seed{seed}/L0_{lv}_p0.5"
        ),
    },
}

def load_summary(d):
    p = d / "summary.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())

print()
print("=" * 118)
print("NO-LOAD | STRICT 3D AP_R40 MODERATE")
print("=" * 118)
print(
    f"{'Method':<18}{'Hz':>5}{'Car':>10}{'Ped':>10}{'Cyc':>10}"
    f"{'Macro':>10}{'Drop%':>10}{'Miss%':>10}{'p99ms':>10}"
)

for name, cfg in methods.items():
    for hz in (35, 40, 45, 50):
        s = load_summary(cfg["nl"](hz))
        q = s["stream_sap_3d_moderate_R40"]
        f = s["forward_latency"]
        print(
            f"{name:<18}{hz:>5}"
            f"{q['Car']:10.3f}{q['Pedestrian']:10.3f}"
            f"{q['Cyclist']:10.3f}{q['Macro']:10.3f}"
            f"{100*s['drop_rate']:10.3f}"
            f"{100*s['deadline_miss_rate']:10.3f}"
            f"{f['p99_ms']:10.3f}"
        )

print()
print("=" * 118)
print("RANDOM50 @ 35Hz | STRICT 3D AP_R40 MODERATE")
print("=" * 118)
print(
    f"{'Method':<18}{'Load':>8}{'Car':>10}{'Ped':>10}{'Cyc':>10}"
    f"{'Macro':>10}{'Drop%':>10}{'Miss%':>10}{'p99ms':>10}"
)

for name, cfg in methods.items():
    for lv in ("L1", "L2", "L3", "L4"):
        d = cfg["rnd"](lv)
        s = load_summary(d)
        q = s["stream_sap_3d_moderate_R40"]
        f = s["forward_latency"]
        print(
            f"{name:<18}{('L0/'+lv):>8}"
            f"{q['Car']:10.3f}{q['Pedestrian']:10.3f}"
            f"{q['Cyclist']:10.3f}{q['Macro']:10.3f}"
            f"{100*s['drop_rate']:10.3f}"
            f"{100*s['deadline_miss_rate']:10.3f}"
            f"{f['p99_ms']:10.3f}"
        )

# Exogenous contention trace must be byte-identical across methods.
print()
print("=" * 118)
print("RANDOM50 TRACE IDENTITY")
print("=" * 118)

for lv in ("L1", "L2", "L3", "L4"):
    paths = {
        name: cfg["rnd"](lv) / "contention_trace.csv"
        for name, cfg in methods.items()
    }
    for name, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(p)

    ref_name = "TV-Stream3D"
    ref = paths[ref_name].read_bytes()

    for name, p in paths.items():
        if p.read_bytes() != ref:
            raise RuntimeError(
                f"trace mismatch at {lv}: {ref_name} vs {name}\n"
                f"{paths[ref_name]}\n{p}"
            )

    print(f"[PASS] {lv}: all four contention_trace.csv files are byte-identical")

print()
print("[PASS] final result tree complete")
PY
