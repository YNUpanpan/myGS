#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/pch/myGS"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/visible"
CONDA_BIN="${PROJECT_ROOT}/tools/miniconda3/bin/conda"
CONDA_ENV_NAME="mygs-3dgs-cu128"

run_dir="${1:-}"
if [[ -z "${run_dir}" ]]; then
  run_dir="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print | sort | tail -1)"
fi
test -n "${run_dir}"
run_dir="$(realpath -e "${run_dir}")"
case "${run_dir}" in
  "${OUTPUT_ROOT}"/*) ;;
  *) echo "Run directory is outside visible output root: ${run_dir}" >&2; exit 70 ;;
esac

echo "run_dir=${run_dir}"
status_file="${run_dir}/status.json"
if [[ -f "${status_file}" ]]; then
  "${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python -m json.tool "${status_file}"
fi
if [[ -f "${run_dir}/metrics.csv" ]]; then
  echo "metrics:"
  sed -n '1,200p' "${run_dir}/metrics.csv"
fi
if [[ -f "${run_dir}/train.pid" ]]; then
  pid="$(<"${run_dir}/train.pid")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "pid=${pid} alive=true"
  else
    echo "pid=${pid} alive=false"
  fi
fi
nvidia-smi --id=0 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

log_file=""
if [[ -f "${status_file}" ]]; then
  log_file="$("${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("log_file") or "")' "${status_file}")"
fi
if [[ -n "${log_file}" && -f "${log_file}" ]]; then
  echo "log_file=${log_file}"
  tail -n 30 "${log_file}"
fi
