#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

scene="${1:-}"
target_width="${2:-}"
target_height="${3:-}"

require_scene "${scene}"
if [[ -z "${target_width}" || -z "${target_height}" ]]; then
  echo "Usage: $0 <scene> <target_width> <target_height>" >&2
  exit 2
fi

python "${SCRIPT_DIR}/prepare_dimension_filtered_dataset.py" \
  --scene "${scene}" \
  --raw-dir "$(scene_raw_dir "${scene}")" \
  --processed-dir "$(scene_processed_dir "${scene}")" \
  --pattern "$(scene_pattern "${scene}")" \
  --expected "$(expected_count "${scene}")" \
  --width "${target_width}" \
  --height "${target_height}"
