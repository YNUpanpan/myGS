#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

echo "== Project =="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo

echo "== System =="
hostname
uname -a
echo

echo "== GPU =="
require_command nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

echo "== CUDA =="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 4
else
  echo "nvcc not found in system PATH"
fi
echo

echo "== Tools =="
require_command git
require_command colmap
echo "git=$(command -v git)"
echo "colmap=$(command -v colmap)"
if command -v conda >/dev/null 2>&1; then
  echo "system conda=$(command -v conda)"
else
  echo "system conda not found; project conda expected at ${CONDA_ROOT}"
fi
echo

echo "== Data =="
for scene in "${VISIBLE_SCENE}" "${THERMAL_SCENE}"; do
  raw_dir="$(scene_raw_dir "${scene}")"
  actual="$(count_scene_images "${scene}" "${raw_dir}")"
  expected="$(expected_count "${scene}")"
  echo "${scene}: ${actual}/${expected} images in ${raw_dir}"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Image count mismatch for ${scene}" >&2
    exit 10
  fi
done
echo

echo "== Git =="
git -C "${PROJECT_ROOT}" status --short --branch
git -C "${PROJECT_ROOT}" remote -v || true
