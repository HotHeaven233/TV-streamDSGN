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
    cp -a "${dst}" "${dst}.bak_tv_controller_${STAMP}"
  fi

  cp -a "${src}" "${dst}"
}

copy_one "${SCRIPT_DIR}/payload/tools/tv_stream3d_controller.py" \
         "tools/tv_stream3d_controller.py"
copy_one "${SCRIPT_DIR}/payload/tools/audit_tv_stream3d_controller.py" \
         "tools/audit_tv_stream3d_controller.py"
copy_one "${SCRIPT_DIR}/payload/scripts/stream_exp/41_audit_tv_stream3d_controller.sh" \
         "scripts/stream_exp/41_audit_tv_stream3d_controller.sh"

chmod +x scripts/stream_exp/41_audit_tv_stream3d_controller.sh

python -m py_compile \
  tools/tv_stream3d_controller.py \
  tools/audit_tv_stream3d_controller.py

bash -n scripts/stream_exp/41_audit_tv_stream3d_controller.sh

echo "[DONE] TV-Stream3D controller stage-1 installed."
echo "[NEXT] ./scripts/stream_exp/41_audit_tv_stream3d_controller.sh 20 100"
