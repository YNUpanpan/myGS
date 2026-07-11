#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

PATCH_FILE="${PROJECT_ROOT}/patches/gaussian-splatting/visible-monitored-training.patch"
MONITOR_MODULE="${PROJECT_ROOT}/scripts/visible_training_monitor.py"
CONDA_BIN="${CONDA_ROOT}/bin/conda"
DEFAULT_SCENE="night_visible"
TRAINING_SCENE="${MYGS_NIGHT_VISIBLE_SCENE:-${DEFAULT_SCENE}}"
require_scene "${TRAINING_SCENE}"
SOURCE_DIR="$(scene_processed_dir "${TRAINING_SCENE}")"
FORMAL_EVAL_ITERATIONS="5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000"

usage() {
  echo "Usage: $0 {smoke|start|resume <run_dir>}" >&2
}

ensure_no_active_night_visible_run() {
  local pid_file pid
  if [[ ! -d "$(scene_output_root "${TRAINING_SCENE}")" ]]; then
    return
  fi
  while IFS= read -r pid_file; do
    [[ -n "${pid_file}" ]] || continue
    pid="$(<"${pid_file}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "Night visible training process is already alive: pid=${pid} file=${pid_file}" >&2
      exit 61
    fi
  done < <(find "$(scene_output_root "${TRAINING_SCENE}")" -mindepth 2 -maxdepth 2 -type f -name train.pid -print)
}

ensure_monitor_patch() {
  test -f "${PATCH_FILE}"
  if git -C "${GAUSSIAN_SPLATTING_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    return
  fi
  if git -C "${GAUSSIAN_SPLATTING_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
    git -C "${GAUSSIAN_SPLATTING_DIR}" apply "${PATCH_FILE}"
    return
  fi
  echo "Monitoring patch is neither cleanly applied nor cleanly applicable." >&2
  exit 62
}

preflight() {
  local checkout_head prepared expected
  test -x "${CONDA_BIN}"
  test -f "${GAUSSIAN_SPLATTING_DIR}/train.py"
  test -f "${MONITOR_MODULE}"
  checkout_head="$(git -C "${GAUSSIAN_SPLATTING_DIR}" rev-parse --short HEAD)"
  test "${checkout_head}" = "54c035f"
  test -d "${SOURCE_DIR}/images"
  test -f "${SOURCE_DIR}/sparse/0/cameras.bin"
  test -f "${SOURCE_DIR}/sparse/0/images.bin"
  test -f "${SOURCE_DIR}/sparse/0/points3D.bin"
  expected="$(expected_count "${TRAINING_SCENE}")"
  prepared="$(find "${SOURCE_DIR}/images" -maxdepth 1 \( -type f -o -type l \) -name "$(scene_pattern "${TRAINING_SCENE}")" | wc -l)"
  test "${prepared}" = "${expected}"
  CUDA_VISIBLE_DEVICES=0 "${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" \
    python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1"
  ensure_monitor_patch
}

lpips_preflight() {
  (
    cd "${GAUSSIAN_SPLATTING_DIR}"
    CUDA_VISIBLE_DEVICES=0 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV_NAME}" \
      python -c 'import torch; from lpipsPyTorch.modules.lpips import LPIPS; model=LPIPS(net_type="vgg").cuda().eval(); value=model(torch.zeros(1,3,64,64,device="cuda"), torch.zeros(1,3,64,64,device="cuda")); assert torch.isfinite(value).all(); print("lpips-vgg-ok", float(value.mean()))'
  )
}

worker() {
  local output_dir="$1" log_file="$2" iterations="$3" eval_csv="$4" checkpoint="${5:-}"
  local rc
  local -a eval_iterations command
  IFS=',' read -r -a eval_iterations <<< "${eval_csv}"

  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH="${PROJECT_ROOT}/scripts"
  export MYGS_MONITOR_DIR="${output_dir}"
  export MYGS_EVAL_ITERATIONS="${eval_csv}"
  export MYGS_TRAIN_PID="$$"
  export MYGS_LOG_FILE="${log_file}"

  command=(
    python train.py
    -s "${SOURCE_DIR}"
    -m "${output_dir}"
    --eval
    --disable_viewer
    --iterations "${iterations}"
    --test_iterations "$((iterations + 1))"
    --save_iterations "${eval_iterations[@]}"
    --checkpoint_iterations "${eval_iterations[@]}"
  )
  if [[ -n "${checkpoint}" ]]; then
    command+=(--start_checkpoint "${checkpoint}")
  fi

  cd "${GAUSSIAN_SPLATTING_DIR}"
  set +e
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV_NAME}" "${command[@]}"
  rc=$?
  set -e
  "${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python "${MONITOR_MODULE}" finalize-exit \
    --run-dir "${output_dir}" --exit-code "${rc}"
  return "${rc}"
}

launch() {
  local output_dir="$1" log_file="$2" iterations="$3" eval_csv="$4" checkpoint="${5:-}"
  local rc
  mkdir -p "${output_dir}" "${LOG_ROOT}"
  touch "${log_file}"
  echo "$$" >"${output_dir}/train.pid"
  echo "foreground-ssh" >"${output_dir}/train.session"
  echo "run_id=$(basename "${output_dir}")"
  echo "output_dir=${output_dir}"
  echo "log_file=${log_file}"
  echo "pid=$$"
  echo "execution_mode=foreground-ssh"

  set +e
  worker "${output_dir}" "${log_file}" "${iterations}" "${eval_csv}" "${checkpoint}" >>"${log_file}" 2>&1
  rc=$?
  set -e
  return "${rc}"
}

new_run() {
  local kind="$1" iterations eval_csv run_id output_dir log_file
  if [[ "${kind}" == "smoke" ]]; then
    iterations=20
    eval_csv="10,20"
    run_id="smoke-$(timestamp)"
  else
    iterations=15000
    eval_csv="${FORMAL_EVAL_ITERATIONS}"
    run_id="$(timestamp)"
  fi
  output_dir="$(scene_output_root "${TRAINING_SCENE}")/${run_id}"
  log_file="$(log_path train "${TRAINING_SCENE}")"
  launch "${output_dir}" "${log_file}" "${iterations}" "${eval_csv}"
}

resume_run() {
  local requested="$1" run_dir pid checkpoint log_file
  run_dir="$(realpath -e "${requested}")"
  case "${run_dir}" in
    "$(scene_output_root "${TRAINING_SCENE}")"/*) ;;
    *) echo "Resume directory is outside ${TRAINING_SCENE} output root: ${run_dir}" >&2; exit 63 ;;
  esac
  test -f "${run_dir}/cfg_args"
  grep -Fq "source_path='${SOURCE_DIR}'" "${run_dir}/cfg_args"
  if [[ -f "${run_dir}/train.pid" ]]; then
    pid="$(<"${run_dir}/train.pid")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "Cannot resume a live run: pid=${pid}" >&2
      exit 64
    fi
  fi
  checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name 'chkpnt*.pth' -print | sort -V | tail -1)"
  test -n "${checkpoint}"
  log_file="$(log_path train "${TRAINING_SCENE}")"
  launch "${run_dir}" "${log_file}" 15000 "${FORMAL_EVAL_ITERATIONS}" "${checkpoint}"
}

mode="${1:-}"
case "${mode}" in
  _worker)
    shift
    worker_log="${2:?missing worker log path}"
    worker "$@" >>"${worker_log}" 2>&1
    ;;
  smoke|start)
    ensure_no_active_night_visible_run
    preflight
    lpips_preflight
    new_run "${mode}"
    ;;
  resume)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    ensure_no_active_night_visible_run
    preflight
    lpips_preflight
    resume_run "$2"
    ;;
  *)
    usage
    exit 2
    ;;
esac
