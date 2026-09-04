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
    cp -a "${dst}" "${dst}.bak_tv_online_${STAMP}"
  fi

  cp -a "${src}" "${dst}"
}

copy_one "${SCRIPT_DIR}/payload/tools/tv_stream3d_controller.py" \
         "tools/tv_stream3d_controller.py"
copy_one "${SCRIPT_DIR}/payload/tools/tv_stream3d_causal_fused_runtime.py" \
         "tools/tv_stream3d_causal_fused_runtime.py"
copy_one "${SCRIPT_DIR}/payload/tools/test_tv_stream3d_online_forward.py" \
         "tools/test_tv_stream3d_online_forward.py"
copy_one "${SCRIPT_DIR}/payload/scripts/stream_exp/42_smoke_tv_stream3d_online.sh" \
         "scripts/stream_exp/42_smoke_tv_stream3d_online.sh"

chmod +x scripts/stream_exp/42_smoke_tv_stream3d_online.sh

python -m py_compile \
  tools/tv_stream3d_controller.py \
  tools/tv_stream3d_causal_fused_runtime.py \
  tools/test_tv_stream3d_online_forward.py

bash -n scripts/stream_exp/42_smoke_tv_stream3d_online.sh

echo "[DONE] TV-Stream3D online forward smoke v2 installed."
echo "[NEXT] ./scripts/stream_exp/42_smoke_tv_stream3d_online.sh 20 100 33 80 100 'L0 L4' 0.25"
