#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

EPOCH="${1:-20}"
N="${2:-100}"
HZ="${3:-35}"
WARMUP="${4:-80}"
FRAMES="${5:-100}"
LEVELS="${6:-L0 L1 L2 L3 L4}"

EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v4_bn_from_k3}"

RAW_CKPT="outputs/elastic_bev/${EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"
BANK="outputs/elastic_bev/${EXP_NAME}/bn_bank/causal_prefix_bank_e${EPOCH}_n${N}.pth"

PROFILE_ROOT="outputs/elastic_bev/${EXP_NAME}/all84_contention_profile_v6/e${EPOCH}_n${N}"
CTRL_CSV="${PROFILE_ROOT}/controller_remaining_latency_table.csv"

LEVELS_JSON="outputs/elastic_bev/${EXP_NAME}/contention_calibration_v6/smooth_microchain_priority/contention_levels.json"

OUT_ROOT="outputs/elastic_bev/${EXP_NAME}/tv_online_smoke/e${EPOCH}_n${N}_${HZ}Hz"
mkdir -p "${OUT_ROOT}"

for f in \
  "${FULL_CFG}" \
  "${FULL_CKPT}" \
  "${RAW_CKPT}" \
  "${BANK}" \
  "${CTRL_CSV}" \
  "${LEVELS_JSON}" \
  "tools/tv_stream3d_controller.py" \
  "tools/tv_stream3d_causal_fused_runtime.py" \
  "tools/test_tv_stream3d_online_forward.py" \
  "tools/smooth_cuda_contention.py"; do
  if [ ! -f "${f}" ]; then
    echo "[ERROR] missing: ${f}"
    exit 2
  fi
done

for L in ${LEVELS}; do
  echo
  echo "======================================================================"
  echo "ONLINE SMOKE ${L} @ ${HZ} Hz"
  echo "======================================================================"

  python tools/test_tv_stream3d_online_forward.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${RAW_CKPT}" \
    --prefix_bn_bank "${BANK}" \
    --controller_csv "${CTRL_CSV}" \
    --levels_json "${LEVELS_JSON}" \
    --expected_level "${L}" \
    --input_hz "${HZ}" \
    --warmup_frames "${WARMUP}" \
    --frames "${FRAMES}" \
    --workers 0 \
    --output_csv "${OUT_ROOT}/${L}_raw.csv" \
    --output_json "${OUT_ROOT}/${L}_summary.json"
done

python - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

print()
print("=" * 100)
print("TV-Stream3D ONLINE SMOKE SUMMARY")
print("=" * 100)

for p in sorted(root.glob("L*_summary.json")):
    d = json.loads(p.read_text())
    f = d["forward_ms"]
    print(
        f'{d["expected_level"]:>2s} '
        f'p50={f["p50"]:7.3f} '
        f'p99={f["p99"]:7.3f} '
        f'deadline={d["deadline_ms"]:7.3f} '
        f'miss={d["deadline_miss_rate"]:.3%} '
        f'infeasible_frames={d["decision_infeasible_frames"]:3d} '
        f'levels={d["observed_level_counts"]} '
        f'cache_new={d["new_fused_entries_during_measure"]}'
    )

print("=" * 100)
PY

echo
echo "[RESULT] ${OUT_ROOT}"
