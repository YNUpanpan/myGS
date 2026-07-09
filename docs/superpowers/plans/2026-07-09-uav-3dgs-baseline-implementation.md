# UAV 3DGS Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 3DGS baseline pipeline for visible and thermal UAV scenes on `pch-5090`.

**Architecture:** The project is script-driven. Each shell script owns one stage: environment checks, local Miniconda installation, CUDA 12.8/PyTorch environment setup, official 3DGS checkout, dataset preparation, COLMAP, training, and run summaries. Raw data remains read-only; generated files stay under `processed/`, `outputs/`, and `logs/`.

**Tech Stack:** Bash, Git, Miniconda, Python, PyTorch cu128, COLMAP, official `graphdeco-inria/gaussian-splatting`, Ubuntu on `pch-5090`.

## Global Constraints

- Project root: `/home/pch/myGS`.
- SSH entry: `pch-5090`.
- Raw visible data: `/home/pch/myGS/datasets/uav_3dgs/raw/visible`, 339 `*_V.JPG` files.
- Raw thermal data: `/home/pch/myGS/datasets/uav_3dgs/raw/thermal`, 339 `*_T.JPG` files.
- Use project-local Conda at `/home/pch/myGS/tools/miniconda3`.
- Use CUDA 12.8 through the project Conda/PyTorch environment instead of changing system CUDA.
- Use official `graphdeco-inria/gaussian-splatting`.
- Run COLMAP independently for visible and thermal data.
- Do not modify, move, rename, or delete raw images.
- Do not use batch deletion commands or recursive directory deletion.
- GitHub remote: `git@github.com:YNUpanpan/myGS.git`.
- Git tracks engineering files only; raw data, processed data, logs, tools, and outputs remain ignored.

---

## File Structure

- Create `scripts/common.sh`: shared constants, timestamp helpers, scene validation, logging helpers, safe command checks.
- Create `scripts/check_environment.sh`: read-only server, GPU, CUDA, COLMAP, git, data-count, and repository checks.
- Create `scripts/install_miniconda.sh`: install Miniconda under `tools/miniconda3` if missing.
- Create `scripts/setup_3dgs_env.sh`: create/update Conda env `mygs-3dgs-cu128` with Python, PyTorch cu128, and 3DGS build dependencies.
- Create `scripts/fetch_3dgs.sh`: clone/update official `gaussian-splatting` into `tools/gaussian-splatting`.
- Create `scripts/prepare_dataset.sh`: prepare `processed/<scene>/images` with symlinks and a manifest.
- Create `scripts/run_colmap.sh`: run COLMAP for one scene and validate `sparse/0` outputs.
- Create `scripts/run_train.sh`: run official 3DGS training for one scene into `outputs/<scene>/<run_id>`.
- Create `scripts/summarize_runs.sh`: summarize environment, data, COLMAP, output, and git status.
- Create `configs/scenes.env`: central scene names and expected counts.
- Modify `README.md`: document command order.
- Modify `AGENTS.md`: append execution progress after each completed task.

---

### Task 1: Shared Configuration and Environment Check

**Files:**
- Create: `/home/pch/myGS/configs/scenes.env`
- Create: `/home/pch/myGS/scripts/common.sh`
- Create: `/home/pch/myGS/scripts/check_environment.sh`
- Modify: `/home/pch/myGS/README.md`
- Modify: `/home/pch/myGS/AGENTS.md`

**Interfaces:**
- Produces: `require_scene "$scene"`, `scene_raw_dir "$scene"`, `scene_processed_dir "$scene"`, `expected_count "$scene"`, `timestamp`, `log_path "$step" "$scene"`.
- Consumes: none.

- [ ] **Step 1: Create `configs/scenes.env`**

```bash
# Scene configuration for UAV 3DGS baseline.
PROJECT_ROOT="/home/pch/myGS"
DATASET_ROOT="${PROJECT_ROOT}/datasets/uav_3dgs"
RAW_ROOT="${DATASET_ROOT}/raw"
PROCESSED_ROOT="${DATASET_ROOT}/processed"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
LOG_ROOT="${PROJECT_ROOT}/logs"
TOOLS_ROOT="${PROJECT_ROOT}/tools"
CONDA_ROOT="${TOOLS_ROOT}/miniconda3"
CONDA_ENV_NAME="mygs-3dgs-cu128"
GAUSSIAN_SPLATTING_DIR="${TOOLS_ROOT}/gaussian-splatting"

VISIBLE_SCENE="visible"
THERMAL_SCENE="thermal"
VISIBLE_PATTERN="*_V.JPG"
THERMAL_PATTERN="*_T.JPG"
VISIBLE_EXPECTED_COUNT="339"
THERMAL_EXPECTED_COUNT="339"
```

- [ ] **Step 2: Create `scripts/common.sh`**

```bash
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
```

- [ ] **Step 3: Create `scripts/check_environment.sh`**

```bash
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
```

- [ ] **Step 4: Make scripts executable**

Run:

```bash
chmod +x /home/pch/myGS/scripts/common.sh /home/pch/myGS/scripts/check_environment.sh
```

Expected: no output.

- [ ] **Step 5: Run environment check**

Run:

```bash
cd /home/pch/myGS
bash scripts/check_environment.sh
```

Expected includes:

```text
visible: 339/339 images
thermal: 339/339 images
```

- [ ] **Step 6: Update docs and task log**

Append to `README.md`:

```markdown
## Baseline Commands

Run from `/home/pch/myGS`:

```bash
bash scripts/check_environment.sh
bash scripts/install_miniconda.sh
bash scripts/setup_3dgs_env.sh
bash scripts/fetch_3dgs.sh
bash scripts/prepare_dataset.sh visible
bash scripts/run_colmap.sh visible
bash scripts/run_train.sh visible
bash scripts/prepare_dataset.sh thermal
bash scripts/run_colmap.sh thermal
bash scripts/run_train.sh thermal
bash scripts/summarize_runs.sh
```
```

Append to `AGENTS.md` under the current task log:

```markdown
#### Implementation Progress

- Created shared scene configuration and environment check scripts.
- Verified visible and thermal raw image counts on `pch-5090`.
```

- [ ] **Step 7: Commit**

Run:

```bash
cd /home/pch/myGS
git add configs/scenes.env scripts/common.sh scripts/check_environment.sh README.md AGENTS.md
git commit -m "Add environment check scripts"
git push
```

Expected: commit succeeds and pushes to `origin/main`.

---

### Task 2: Project Miniconda and CUDA 12.8 Environment Scripts

**Files:**
- Create: `/home/pch/myGS/scripts/install_miniconda.sh`
- Create: `/home/pch/myGS/scripts/setup_3dgs_env.sh`
- Modify: `/home/pch/myGS/AGENTS.md`

**Interfaces:**
- Consumes: `scripts/common.sh` constants.
- Produces: project Conda root `/home/pch/myGS/tools/miniconda3` and environment `mygs-3dgs-cu128`.

- [ ] **Step 1: Create `scripts/install_miniconda.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

INSTALLER="${TOOLS_ROOT}/Miniconda3-latest-Linux-x86_64.sh"
URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

mkdir -p "${TOOLS_ROOT}"

if [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "Project conda already installed: ${CONDA_ROOT}/bin/conda"
  "${CONDA_ROOT}/bin/conda" --version
  exit 0
fi

if [[ ! -f "${INSTALLER}" ]]; then
  require_command curl
  echo "Downloading Miniconda installer to ${INSTALLER}"
  curl -L "${URL}" -o "${INSTALLER}"
fi

bash "${INSTALLER}" -b -p "${CONDA_ROOT}"
"${CONDA_ROOT}/bin/conda" config --set auto_activate_base false
"${CONDA_ROOT}/bin/conda" --version
```

- [ ] **Step 2: Create `scripts/setup_3dgs_env.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

CONDA_BIN="${CONDA_ROOT}/bin/conda"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Project conda not found. Run scripts/install_miniconda.sh first." >&2
  exit 20
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  "${CONDA_BIN}" create -y -n "${CONDA_ENV_NAME}" python=3.10 pip cmake ninja
fi

"${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
"${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
"${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python -m pip install tqdm plyfile opencv-python joblib scipy

"${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in PyTorch")
PY
```

- [ ] **Step 3: Make scripts executable**

Run:

```bash
chmod +x /home/pch/myGS/scripts/install_miniconda.sh /home/pch/myGS/scripts/setup_3dgs_env.sh
```

Expected: no output.

- [ ] **Step 4: Install Miniconda**

Run:

```bash
cd /home/pch/myGS
bash scripts/install_miniconda.sh
```

Expected includes:

```text
conda
```

- [ ] **Step 5: Create and verify CUDA environment**

Run:

```bash
cd /home/pch/myGS
bash scripts/setup_3dgs_env.sh
```

Expected includes:

```text
torch cuda 12.8
cuda available True
```

- [ ] **Step 6: Update `AGENTS.md`**

Append:

```markdown
- Installed project-local Miniconda at `/home/pch/myGS/tools/miniconda3`.
- Created Conda environment `mygs-3dgs-cu128`.
- Verified PyTorch CUDA availability in the project environment.
```

- [ ] **Step 7: Commit**

Run:

```bash
cd /home/pch/myGS
git add scripts/install_miniconda.sh scripts/setup_3dgs_env.sh AGENTS.md
git commit -m "Add project CUDA environment setup"
git push
```

Expected: commit succeeds and pushes to `origin/main`.

---

### Task 3: Fetch Official 3DGS Source

**Files:**
- Create: `/home/pch/myGS/scripts/fetch_3dgs.sh`
- Modify: `/home/pch/myGS/AGENTS.md`

**Interfaces:**
- Consumes: `GAUSSIAN_SPLATTING_DIR` from `scripts/common.sh`.
- Produces: official 3DGS checkout under `/home/pch/myGS/tools/gaussian-splatting`.

- [ ] **Step 1: Create `scripts/fetch_3dgs.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

require_command git
mkdir -p "${TOOLS_ROOT}"

if [[ -d "${GAUSSIAN_SPLATTING_DIR}/.git" ]]; then
  echo "3DGS checkout already exists: ${GAUSSIAN_SPLATTING_DIR}"
  git -C "${GAUSSIAN_SPLATTING_DIR}" status --short --branch
  git -C "${GAUSSIAN_SPLATTING_DIR}" submodule update --init --recursive
else
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "${GAUSSIAN_SPLATTING_DIR}"
fi

git -C "${GAUSSIAN_SPLATTING_DIR}" rev-parse --short HEAD
test -f "${GAUSSIAN_SPLATTING_DIR}/train.py"
```

- [ ] **Step 2: Make script executable**

Run:

```bash
chmod +x /home/pch/myGS/scripts/fetch_3dgs.sh
```

Expected: no output.

- [ ] **Step 3: Fetch 3DGS**

Run:

```bash
cd /home/pch/myGS
bash scripts/fetch_3dgs.sh
```

Expected: checkout exists and `train.py` is present.

- [ ] **Step 4: Install 3DGS Python packages**

Run:

```bash
cd /home/pch/myGS/tools/gaussian-splatting
/home/pch/myGS/tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m pip install -e submodules/diff-gaussian-rasterization
/home/pch/myGS/tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m pip install -e submodules/simple-knn
```

Expected: both editable installs complete successfully.

- [ ] **Step 5: Smoke test imports**

Run:

```bash
/home/pch/myGS/tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn
print("3dgs imports ok")
print(torch.cuda.get_device_name(0))
PY
```

Expected includes:

```text
3dgs imports ok
NVIDIA GeForce RTX 5090
```

- [ ] **Step 6: Update `AGENTS.md`**

Append:

```markdown
- Fetched official `graphdeco-inria/gaussian-splatting` into `tools/gaussian-splatting`.
- Built and import-tested 3DGS CUDA extension packages in `mygs-3dgs-cu128`.
```

- [ ] **Step 7: Commit**

Run:

```bash
cd /home/pch/myGS
git add scripts/fetch_3dgs.sh AGENTS.md
git commit -m "Add 3DGS source fetch script"
git push
```

Expected: commit succeeds and pushes to `origin/main`.

---

### Task 4: Dataset Preparation

**Files:**
- Create: `/home/pch/myGS/scripts/prepare_dataset.sh`
- Modify: `/home/pch/myGS/AGENTS.md`

**Interfaces:**
- Consumes: `scene_raw_dir`, `scene_processed_dir`, `scene_pattern`, `expected_count`.
- Produces: `datasets/uav_3dgs/processed/<scene>/images` and `manifest.txt`.

- [ ] **Step 1: Create `scripts/prepare_dataset.sh`**

```bash
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
```

- [ ] **Step 2: Make script executable**

Run:

```bash
chmod +x /home/pch/myGS/scripts/prepare_dataset.sh
```

Expected: no output.

- [ ] **Step 3: Prepare visible dataset**

Run:

```bash
cd /home/pch/myGS
bash scripts/prepare_dataset.sh visible
```

Expected:

```text
Prepared visible: 339/339
```

- [ ] **Step 4: Prepare thermal dataset**

Run:

```bash
cd /home/pch/myGS
bash scripts/prepare_dataset.sh thermal
```

Expected:

```text
Prepared thermal: 339/339
```

- [ ] **Step 5: Verify manifests**

Run:

```bash
grep -E 'scene=|prepared_count=' /home/pch/myGS/datasets/uav_3dgs/processed/visible/manifest.txt
grep -E 'scene=|prepared_count=' /home/pch/myGS/datasets/uav_3dgs/processed/thermal/manifest.txt
```

Expected includes:

```text
scene=visible
prepared_count=339
scene=thermal
prepared_count=339
```

- [ ] **Step 6: Update `AGENTS.md`**

Append:

```markdown
- Prepared processed image directories for visible and thermal scenes.
- Confirmed each processed scene has 339 image links or files.
```

- [ ] **Step 7: Commit**

Run:

```bash
cd /home/pch/myGS
git add scripts/prepare_dataset.sh AGENTS.md
git commit -m "Add dataset preparation script"
git push
```

Expected: commit succeeds and pushes to `origin/main`.

---

### Task 5: COLMAP Pipeline

**Files:**
- Create: `/home/pch/myGS/scripts/run_colmap.sh`
- Modify: `/home/pch/myGS/AGENTS.md`

**Interfaces:**
- Consumes: prepared scene directory with `images/`.
- Produces: COLMAP sparse model in `datasets/uav_3dgs/processed/<scene>/sparse/0`.

- [ ] **Step 1: Create `scripts/run_colmap.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

scene="${1:-}"
require_scene "${scene}"
require_command colmap

processed_dir="$(scene_processed_dir "${scene}")"
images_dir="${processed_dir}/images"
database_path="${processed_dir}/database.db"
distorted_dir="${processed_dir}/distorted"
sparse_dir="${processed_dir}/sparse"
log_file="$(log_path colmap "${scene}")"

if [[ ! -d "${images_dir}" ]]; then
  echo "Missing images directory: ${images_dir}" >&2
  exit 40
fi

mkdir -p "${distorted_dir}/sparse" "${sparse_dir}"

{
  echo "COLMAP scene=${scene}"
  echo "images_dir=${images_dir}"
  echo "database_path=${database_path}"
  echo "started_at=$(date -Is)"

  colmap feature_extractor \
    --database_path "${database_path}" \
    --image_path "${images_dir}" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1

  colmap exhaustive_matcher \
    --database_path "${database_path}" \
    --SiftMatching.use_gpu 1

  colmap mapper \
    --database_path "${database_path}" \
    --image_path "${images_dir}" \
    --output_path "${distorted_dir}/sparse"

  if [[ ! -d "${distorted_dir}/sparse/0" ]]; then
    echo "COLMAP mapper did not produce ${distorted_dir}/sparse/0" >&2
    exit 41
  fi

  colmap image_undistorter \
    --image_path "${images_dir}" \
    --input_path "${distorted_dir}/sparse/0" \
    --output_path "${processed_dir}" \
    --output_type COLMAP

  test -f "${sparse_dir}/0/cameras.bin"
  test -f "${sparse_dir}/0/images.bin"
  test -f "${sparse_dir}/0/points3D.bin"

  registered="$(colmap model_analyzer --path "${sparse_dir}/0" 2>/dev/null | grep -E 'Registered images|Images' || true)"
  echo "registered_summary=${registered}"
  echo "finished_at=$(date -Is)"
} 2>&1 | tee "${log_file}"

echo "COLMAP log: ${log_file}"
```

- [ ] **Step 2: Make script executable**

Run:

```bash
chmod +x /home/pch/myGS/scripts/run_colmap.sh
```

Expected: no output.

- [ ] **Step 3: Run visible COLMAP**

Run:

```bash
cd /home/pch/myGS
bash scripts/run_colmap.sh visible
```

Expected: exits 0 and creates:

```text
datasets/uav_3dgs/processed/visible/sparse/0/cameras.bin
datasets/uav_3dgs/processed/visible/sparse/0/images.bin
datasets/uav_3dgs/processed/visible/sparse/0/points3D.bin
```

- [ ] **Step 4: Run thermal COLMAP**

Run:

```bash
cd /home/pch/myGS
bash scripts/run_colmap.sh thermal
```

Expected: exits 0 and creates the same three sparse files under `processed/thermal/sparse/0`.

- [ ] **Step 5: Verify outputs**

Run:

```bash
test -f /home/pch/myGS/datasets/uav_3dgs/processed/visible/sparse/0/cameras.bin
test -f /home/pch/myGS/datasets/uav_3dgs/processed/thermal/sparse/0/cameras.bin
ls -1 /home/pch/myGS/logs/*-colmap-visible.log | tail -1
ls -1 /home/pch/myGS/logs/*-colmap-thermal.log | tail -1
```

Expected: both `test` commands exit 0 and log paths print.

- [ ] **Step 6: Update `AGENTS.md`**

Append:

```markdown
- Ran COLMAP independently for visible and thermal scenes.
- Verified each scene produced `sparse/0/cameras.bin`, `images.bin`, and `points3D.bin`.
- Recorded COLMAP log paths under `/home/pch/myGS/logs`.
```

- [ ] **Step 7: Commit**

Run:

```bash
cd /home/pch/myGS
git add scripts/run_colmap.sh AGENTS.md
git commit -m "Add COLMAP pipeline script"
git push
```

Expected: commit succeeds and pushes to `origin/main`.

---

### Task 6: 3DGS Training Pipeline

**Files:**
- Create: `/home/pch/myGS/scripts/run_train.sh`
- Modify: `/home/pch/myGS/AGENTS.md`

**Interfaces:**
- Consumes: scene directory with `images/` and `sparse/0`.
- Produces: 3DGS run directory under `outputs/<scene>/<run_id>`.

- [ ] **Step 1: Create `scripts/run_train.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

scene="${1:-}"
require_scene "${scene}"

CONDA_BIN="${CONDA_ROOT}/bin/conda"
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Project conda not found: ${CONDA_BIN}" >&2
  exit 50
fi

if [[ ! -f "${GAUSSIAN_SPLATTING_DIR}/train.py" ]]; then
  echo "3DGS train.py not found. Run scripts/fetch_3dgs.sh first." >&2
  exit 51
fi

source_dir="$(scene_processed_dir "${scene}")"
test -d "${source_dir}/images"
test -f "${source_dir}/sparse/0/cameras.bin"
test -f "${source_dir}/sparse/0/images.bin"
test -f "${source_dir}/sparse/0/points3D.bin"

run_id="$(timestamp)"
output_dir="$(scene_output_root "${scene}")/${run_id}"
log_file="$(log_path train "${scene}")"
mkdir -p "${output_dir}"

{
  echo "Training scene=${scene}"
  echo "source_dir=${source_dir}"
  echo "output_dir=${output_dir}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
  echo "started_at=$(date -Is)"

  cd "${GAUSSIAN_SPLATTING_DIR}"
  "${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python train.py \
    -s "${source_dir}" \
    -m "${output_dir}"

  echo "finished_at=$(date -Is)"
  find "${output_dir}" -path "*/point_cloud.ply" -print | sort | tail -5
} 2>&1 | tee "${log_file}"

if ! find "${output_dir}" -path "*/point_cloud.ply" -type f | grep -q .; then
  echo "No point_cloud.ply produced in ${output_dir}" >&2
  exit 52
fi

echo "Training output: ${output_dir}"
echo "Training log: ${log_file}"
```

- [ ] **Step 2: Make script executable**

Run:

```bash
chmod +x /home/pch/myGS/scripts/run_train.sh
```

Expected: no output.

- [ ] **Step 3: Run visible training**

Run:

```bash
cd /home/pch/myGS
CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh visible
```

Expected: exits 0 and prints `Training output: /home/pch/myGS/outputs/visible/<run_id>`.

- [ ] **Step 4: Run thermal training**

Run:

```bash
cd /home/pch/myGS
CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh thermal
```

Expected: exits 0 and prints `Training output: /home/pch/myGS/outputs/thermal/<run_id>`.

- [ ] **Step 5: Verify point clouds**

Run:

```bash
find /home/pch/myGS/outputs/visible -path "*/point_cloud.ply" -type f | tail -1
find /home/pch/myGS/outputs/thermal -path "*/point_cloud.ply" -type f | tail -1
```

Expected: one path prints for each scene.

- [ ] **Step 6: Update `AGENTS.md`**

Append:

```markdown
- Ran 3DGS baseline training for visible and thermal scenes.
- Verified each scene produced at least one `point_cloud.ply`.
- Recorded training output directories and log paths.
```

- [ ] **Step 7: Commit**

Run:

```bash
cd /home/pch/myGS
git add scripts/run_train.sh AGENTS.md
git commit -m "Add 3DGS training script"
git push
```

Expected: commit succeeds and pushes to `origin/main`.

---

### Task 7: Summary Script and Final Verification

**Files:**
- Create: `/home/pch/myGS/scripts/summarize_runs.sh`
- Modify: `/home/pch/myGS/README.md`
- Modify: `/home/pch/myGS/AGENTS.md`

**Interfaces:**
- Consumes: all prior scripts and generated outputs.
- Produces: terminal summary and updated documentation.

- [ ] **Step 1: Create `scripts/summarize_runs.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

echo "== Environment =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
  "${CONDA_ROOT}/bin/conda" run -n "${CONDA_ENV_NAME}" python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
fi
echo

echo "== Data =="
for scene in "${VISIBLE_SCENE}" "${THERMAL_SCENE}"; do
  raw_dir="$(scene_raw_dir "${scene}")"
  processed_dir="$(scene_processed_dir "${scene}")"
  echo "${scene}_raw=$(count_scene_images "${scene}" "${raw_dir}")"
  if [[ -d "${processed_dir}/images" ]]; then
    echo "${scene}_processed=$(find "${processed_dir}/images" -maxdepth 1 \( -type f -o -type l \) -name "$(scene_pattern "${scene}")" | wc -l)"
  else
    echo "${scene}_processed=missing"
  fi
done
echo

echo "== COLMAP =="
for scene in "${VISIBLE_SCENE}" "${THERMAL_SCENE}"; do
  sparse_dir="$(scene_processed_dir "${scene}")/sparse/0"
  if [[ -f "${sparse_dir}/cameras.bin" && -f "${sparse_dir}/images.bin" && -f "${sparse_dir}/points3D.bin" ]]; then
    echo "${scene}=sparse-ok"
  else
    echo "${scene}=sparse-missing"
  fi
done
echo

echo "== Training Outputs =="
for scene in "${VISIBLE_SCENE}" "${THERMAL_SCENE}"; do
  output_root="$(scene_output_root "${scene}")"
  latest_ply="$(find "${output_root}" -path "*/point_cloud.ply" -type f 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "${latest_ply}" ]]; then
    echo "${scene}_latest_point_cloud=${latest_ply}"
  else
    echo "${scene}_latest_point_cloud=missing"
  fi
done
echo

echo "== Git =="
git -C "${PROJECT_ROOT}" status --short --branch
git -C "${PROJECT_ROOT}" log --oneline -5
```

- [ ] **Step 2: Make script executable**

Run:

```bash
chmod +x /home/pch/myGS/scripts/summarize_runs.sh
```

Expected: no output.

- [ ] **Step 3: Run final summary**

Run:

```bash
cd /home/pch/myGS
bash scripts/summarize_runs.sh
```

Expected includes:

```text
visible_raw=339
thermal_raw=339
visible=sparse-ok
thermal=sparse-ok
```

Expected also includes non-missing point cloud paths after training.

- [ ] **Step 4: Update `README.md`**

Add:

```markdown
## Outputs

Training outputs are stored on the server and are intentionally ignored by git:

- `/home/pch/myGS/outputs/visible/<run_id>`
- `/home/pch/myGS/outputs/thermal/<run_id>`

Use `bash scripts/summarize_runs.sh` to inspect the latest environment, COLMAP, and training status.
```

- [ ] **Step 5: Update `AGENTS.md`**

Append:

```markdown
#### Final Baseline Status

- Environment, dataset preparation, COLMAP, and training scripts are present.
- Visible and thermal baseline runs have been summarized with `scripts/summarize_runs.sh`.
- GitHub remote `git@github.com:YNUpanpan/myGS.git` contains the engineering files only.
```

- [ ] **Step 6: Commit**

Run:

```bash
cd /home/pch/myGS
git add scripts/summarize_runs.sh README.md AGENTS.md
git commit -m "Add run summary script"
git push
```

Expected: commit succeeds and pushes to `origin/main`.

---

## Self-Review

Spec coverage:

- Server path, SSH entry, raw data paths, and expected counts are covered by Tasks 1 and 4.
- Project-local Miniconda and CUDA 12.8/PyTorch environment are covered by Task 2.
- Official 3DGS checkout is covered by Task 3.
- Independent visible and thermal COLMAP runs are covered by Task 5.
- Independent visible and thermal training runs are covered by Task 6.
- GitHub engineering-only tracking is covered by `.gitignore`, Task commits, and Task 7.
- `AGENTS.md` task logging is included in every task.
- Deletion constraints are preserved in global constraints and no task uses batch deletion.

Placeholder scan:

- The plan contains no placeholder or deferred implementation markers.

Type and interface consistency:

- Scene names are `visible` and `thermal` throughout.
- Shared helper names are defined in Task 1 and reused consistently in later scripts.
- All generated paths match `/home/pch/myGS` and the approved spec.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-09-uav-3dgs-baseline-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
