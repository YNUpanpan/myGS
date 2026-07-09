#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/pch/myGS"
CONFIG_FILE="${PROJECT_ROOT}/configs/scenes.env"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing config: ${CONFIG_FILE}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

timestamp() {
  date +"%Y%m%d-%H%M%S"
}

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "Missing required command: ${name}" >&2
    exit 3
  fi
}

require_scene() {
  local scene="$1"
  case "${scene}" in
    "${VISIBLE_SCENE}"|"${THERMAL_SCENE}") ;;
    *)
      echo "Invalid scene '${scene}'. Expected '${VISIBLE_SCENE}' or '${THERMAL_SCENE}'." >&2
      exit 4
      ;;
  esac
}

scene_pattern() {
  local scene="$1"
  require_scene "${scene}"
  if [[ "${scene}" == "${VISIBLE_SCENE}" ]]; then
    echo "${VISIBLE_PATTERN}"
  else
    echo "${THERMAL_PATTERN}"
  fi
}

expected_count() {
  local scene="$1"
  require_scene "${scene}"
  if [[ "${scene}" == "${VISIBLE_SCENE}" ]]; then
    echo "${VISIBLE_EXPECTED_COUNT}"
  else
    echo "${THERMAL_EXPECTED_COUNT}"
  fi
}

scene_raw_dir() {
  local scene="$1"
  require_scene "${scene}"
  echo "${RAW_ROOT}/${scene}"
}

scene_processed_dir() {
  local scene="$1"
  require_scene "${scene}"
  echo "${PROCESSED_ROOT}/${scene}"
}

scene_output_root() {
  local scene="$1"
  require_scene "${scene}"
  echo "${OUTPUT_ROOT}/${scene}"
}

log_path() {
  local step="$1"
  local scene="${2:-general}"
  mkdir -p "${LOG_ROOT}"
  echo "${LOG_ROOT}/$(timestamp)-${step}-${scene}.log"
}

count_scene_images() {
  local scene="$1"
  local dir="$2"
  local pattern
  pattern="$(scene_pattern "${scene}")"
  find "${dir}" -maxdepth 1 -type f -name "${pattern}" | wc -l
}
