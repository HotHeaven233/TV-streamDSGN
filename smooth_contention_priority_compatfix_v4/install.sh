#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/data/jhb/workspace/streamDSGN}"
cd "${REPO_ROOT}"

mkdir -p tools scripts/stream_exp
STAMP="$(date +%Y%m%d_%H%M%S)"

copy_one() {
  local src="$1"
  local dst="$2"
  if [ ! -f "${src}" ]; then
    echo "[ERROR] payload missing: ${src}"
    exit 2
  fi
  if [ -f "${dst}" ]; then
    cp -a "${dst}" "${dst}.bak_priority_hotfix_v2_${STAMP}"
  fi
  cp -a "${src}" "${dst}"
}

copy_one "${SCRIPT_DIR}/payload/tools/smooth_cuda_contention.py" "tools/smooth_cuda_contention.py"
copy_one "${SCRIPT_DIR}/payload/tools/profile_fixed_prefix_probe_smooth.py" "tools/profile_fixed_prefix_probe_smooth.py"
copy_one "${SCRIPT_DIR}/payload/tools/profile_fixed_forward_components.py" "tools/profile_fixed_forward_components.py"
copy_one "${SCRIPT_DIR}/payload/tools/test_smooth_contention_overlap.py" "tools/test_smooth_contention_overlap.py"
copy_one "${SCRIPT_DIR}/payload/tools/select_contention_levels_smooth.py" "tools/select_contention_levels_smooth.py"
copy_one "${SCRIPT_DIR}/payload/tools/build_contention_latency_table_smooth.py" "tools/build_contention_latency_table_smooth.py"

copy_one "${SCRIPT_DIR}/payload/scripts/stream_exp/38_calibrate_smooth_contention.sh" "scripts/stream_exp/38_calibrate_smooth_contention.sh"
copy_one "${SCRIPT_DIR}/payload/scripts/stream_exp/39_smoke_smooth_contention.sh" "scripts/stream_exp/39_smoke_smooth_contention.sh"
copy_one "${SCRIPT_DIR}/payload/scripts/stream_exp/40_profile_all_84_smooth_contention.sh" "scripts/stream_exp/40_profile_all_84_smooth_contention.sh"

chmod +x   scripts/stream_exp/38_calibrate_smooth_contention.sh   scripts/stream_exp/39_smoke_smooth_contention.sh   scripts/stream_exp/40_profile_all_84_smooth_contention.sh

python -m py_compile   tools/smooth_cuda_contention.py   tools/profile_fixed_prefix_probe_smooth.py   tools/profile_fixed_forward_components.py   tools/test_smooth_contention_overlap.py   tools/select_contention_levels_smooth.py   tools/build_contention_latency_table_smooth.py

bash -n scripts/stream_exp/38_calibrate_smooth_contention.sh
bash -n scripts/stream_exp/39_smoke_smooth_contention.sh
bash -n scripts/stream_exp/40_profile_all_84_smooth_contention.sh

echo "[DONE] smooth contention priority hotfix v2 installed."
echo "[NEXT] python tools/test_smooth_contention_overlap.py --strength 0.125 --slice-ms 0.05"
