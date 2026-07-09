#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

scene="${1:-}"
require_scene "${scene}"

raw_dir="$(scene_raw_dir "${scene}")"
processed_dir="$(scene_processed_dir "${scene}")"
images_dir="${processed_dir}/images"
pattern="$(scene_pattern "${scene}")"
expected="$(expected_count "${scene}")"
manifest="${processed_dir}/manifest.txt"

if [[ ! -d "${raw_dir}" ]]; then
  echo "Missing raw directory: ${raw_dir}" >&2
  exit 30
fi

mkdir -p "${images_dir}"

actual="$(count_scene_images "${scene}" "${raw_dir}")"
if [[ "${actual}" != "${expected}" ]]; then
  echo "Raw image count mismatch for ${scene}: ${actual}/${expected}" >&2
  exit 31
fi

while IFS= read -r src; do
  name="$(basename "${src}")"
  dst="${images_dir}/${name}"
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    continue
  fi
  if ln -s "${src}" "${dst}" 2>/dev/null; then
    :
  else
    cp "${src}" "${dst}"
  fi
done < <(find "${raw_dir}" -maxdepth 1 -type f -name "${pattern}" | sort)

prepared="$(find "${images_dir}" -maxdepth 1 \( -type f -o -type l \) -name "${pattern}" | wc -l)"
if [[ "${prepared}" != "${expected}" ]]; then
  echo "Prepared image count mismatch for ${scene}: ${prepared}/${expected}" >&2
  exit 32
fi

{
  echo "scene=${scene}"
  echo "raw_dir=${raw_dir}"
  echo "processed_dir=${processed_dir}"
  echo "images_dir=${images_dir}"
  echo "pattern=${pattern}"
  echo "expected_count=${expected}"
  echo "prepared_count=${prepared}"
  echo "created_at=$(date -Is)"
  echo
  find "${images_dir}" -maxdepth 1 \( -type f -o -type l \) -name "${pattern}" -printf "%f\n" | sort
} > "${manifest}"

echo "Prepared ${scene}: ${prepared}/${expected}"
echo "Manifest: ${manifest}"
