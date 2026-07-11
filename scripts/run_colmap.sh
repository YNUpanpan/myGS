#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

scene="${1:-}"
require_scene "${scene}"

COLMAP_BIN="${COLMAP_BIN:-${TOOLS_ROOT}/colmap-3.9.1-cuda/bin/colmap}"
if [[ ! -x "${COLMAP_BIN}" ]]; then
  echo "Missing CUDA COLMAP binary: ${COLMAP_BIN}" >&2
  exit 44
fi

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
COLMAP_SIFT_EXTRACTION_USE_GPU="${COLMAP_SIFT_EXTRACTION_USE_GPU:-${COLMAP_SIFT_USE_GPU:-1}}"
COLMAP_SIFT_MATCHING_USE_GPU="${COLMAP_SIFT_MATCHING_USE_GPU:-${COLMAP_SIFT_USE_GPU:-1}}"
COLMAP_SIFT_NUM_THREADS="${COLMAP_SIFT_NUM_THREADS:-8}"
COLMAP_MATCHER="${COLMAP_MATCHER:-sequential}"
COLMAP_SEQUENTIAL_OVERLAP="${COLMAP_SEQUENTIAL_OVERLAP:-20}"
COLMAP_IMAGE_READER_SINGLE_CAMERA="${COLMAP_IMAGE_READER_SINGLE_CAMERA:-1}"

processed_dir="$(scene_processed_dir "${scene}")"
images_dir="${processed_dir}/images"
database_path="${processed_dir}/database.db"
distorted_dir="${processed_dir}/distorted"
sparse_dir="${processed_dir}/sparse"
log_file="$(log_path colmap "${scene}")"
pattern="$(scene_pattern "${scene}")"
expected="$(expected_count "${scene}")"

if [[ ! -d "${images_dir}" ]]; then
  echo "Missing images directory: ${images_dir}" >&2
  exit 40
fi

prepared="$(find "${images_dir}" -maxdepth 1 \( -type f -o -type l \) -name "${pattern}" | wc -l)"
if [[ "${prepared}" != "${expected}" ]]; then
  echo "Prepared image count mismatch for ${scene}: ${prepared}/${expected}" >&2
  exit 41
fi

mkdir -p "${distorted_dir}/sparse" "${sparse_dir}"

sparse_model_ready() {
  [[ -f "${sparse_dir}/0/cameras.bin" &&
    -f "${sparse_dir}/0/images.bin" &&
    -f "${sparse_dir}/0/points3D.bin" ]]
}

normalize_sparse_model() {
  if sparse_model_ready; then
    return 0
  fi

  if [[ -f "${sparse_dir}/cameras.bin" &&
    -f "${sparse_dir}/images.bin" &&
    -f "${sparse_dir}/points3D.bin" ]]; then
    mkdir -p "${sparse_dir}/0"
    mv "${sparse_dir}/cameras.bin" "${sparse_dir}/0/cameras.bin"
    mv "${sparse_dir}/images.bin" "${sparse_dir}/0/images.bin"
    mv "${sparse_dir}/points3D.bin" "${sparse_dir}/0/points3D.bin"
    echo "Normalized sparse model to ${sparse_dir}/0"
  fi
}

registered_image_count() {
  local model_path="$1"
  "${COLMAP_BIN}" model_analyzer --path "${model_path}" 2>&1 |
    awk -F': ' '/Registered images/ {print $2; exit}'
}

select_largest_sparse_model() {
  local best_path=""
  local best_count="-1"
  local model_path
  local count

  shopt -s nullglob
  for model_path in "${distorted_dir}/sparse"/*; do
    if [[ ! -d "${model_path}" ||
      ! -f "${model_path}/cameras.bin" ||
      ! -f "${model_path}/images.bin" ||
      ! -f "${model_path}/points3D.bin" ]]; then
      continue
    fi
    count="$(registered_image_count "${model_path}")"
    if [[ -n "${count}" && "${count}" -gt "${best_count}" ]]; then
      best_count="${count}"
      best_path="${model_path}"
    fi
  done
  shopt -u nullglob

  if [[ -z "${best_path}" ]]; then
    return 1
  fi

  echo "${best_path}"
}

{
  echo "COLMAP scene=${scene}"
  echo "images_dir=${images_dir}"
  echo "database_path=${database_path}"
  echo "prepared_images=${prepared}/${expected}"
  echo "COLMAP_BIN=${COLMAP_BIN}"
  echo "colmap_version=$("${COLMAP_BIN}" --help | head -n 2 | tr '\n' ' ')"
  echo "COLMAP_SIFT_EXTRACTION_USE_GPU=${COLMAP_SIFT_EXTRACTION_USE_GPU}"
  echo "COLMAP_SIFT_MATCHING_USE_GPU=${COLMAP_SIFT_MATCHING_USE_GPU}"
  echo "COLMAP_SIFT_NUM_THREADS=${COLMAP_SIFT_NUM_THREADS}"
  echo "COLMAP_MATCHER=${COLMAP_MATCHER}"
  echo "COLMAP_SEQUENTIAL_OVERLAP=${COLMAP_SEQUENTIAL_OVERLAP}"
  echo "COLMAP_IMAGE_READER_SINGLE_CAMERA=${COLMAP_IMAGE_READER_SINGLE_CAMERA}"
  echo "started_at=$(date -Is)"

  normalize_sparse_model
  if sparse_model_ready; then
    echo "Existing sparse model found: ${sparse_dir}/0"
  else
    "${COLMAP_BIN}" feature_extractor \
      --database_path "${database_path}" \
      --image_path "${images_dir}" \
      --ImageReader.single_camera "${COLMAP_IMAGE_READER_SINGLE_CAMERA}" \
      --SiftExtraction.use_gpu "${COLMAP_SIFT_EXTRACTION_USE_GPU}" \
      --SiftExtraction.num_threads "${COLMAP_SIFT_NUM_THREADS}"

    case "${COLMAP_MATCHER}" in
      sequential)
        "${COLMAP_BIN}" sequential_matcher \
          --database_path "${database_path}" \
          --SiftMatching.use_gpu "${COLMAP_SIFT_MATCHING_USE_GPU}" \
          --SiftMatching.num_threads "${COLMAP_SIFT_NUM_THREADS}" \
          --SequentialMatching.overlap "${COLMAP_SEQUENTIAL_OVERLAP}"
        ;;
      exhaustive)
        "${COLMAP_BIN}" exhaustive_matcher \
          --database_path "${database_path}" \
          --SiftMatching.use_gpu "${COLMAP_SIFT_MATCHING_USE_GPU}" \
          --SiftMatching.num_threads "${COLMAP_SIFT_NUM_THREADS}"
        ;;
      *)
        echo "Invalid COLMAP_MATCHER='${COLMAP_MATCHER}'. Expected 'sequential' or 'exhaustive'." >&2
        exit 43
        ;;
    esac

    "${COLMAP_BIN}" mapper \
      --database_path "${database_path}" \
      --image_path "${images_dir}" \
      --output_path "${distorted_dir}/sparse"

    if ! selected_sparse_model="$(select_largest_sparse_model)"; then
      echo "COLMAP mapper did not produce a valid sparse model under ${distorted_dir}/sparse" >&2
      exit 42
    fi
    selected_registered_count="$(registered_image_count "${selected_sparse_model}")"
    echo "selected_sparse_model=${selected_sparse_model}"
    echo "selected_registered_images=${selected_registered_count}"

    "${COLMAP_BIN}" image_undistorter \
      --image_path "${images_dir}" \
      --input_path "${selected_sparse_model}" \
      --output_path "${processed_dir}" \
      --output_type COLMAP

    normalize_sparse_model
  fi

  if ! sparse_model_ready; then
    echo "Expected sparse model files under ${sparse_dir}/0" >&2
    exit 45
  fi

  registered="$("${COLMAP_BIN}" model_analyzer --path "${sparse_dir}/0" 2>/dev/null | grep -E 'Registered images|Images' || true)"
  echo "registered_summary=${registered}"
  echo "finished_at=$(date -Is)"
} 2>&1 | tee "${log_file}"

echo "COLMAP log: ${log_file}"
