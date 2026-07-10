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
  "${CONDA_BIN}" create -y -n "${CONDA_ENV_NAME}" --override-channels -c conda-forge python=3.10 pip cmake ninja
fi

"${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
"${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python "${SCRIPT_DIR}/pip_ipv4.py" install --timeout 120 --retries 5 --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
"${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python -m pip install tqdm plyfile opencv-python joblib scipy pytest

"${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python -c "import torch; print('torch', torch.__version__); print('torch cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); raise SystemExit(0 if torch.cuda.is_available() else 'CUDA is not available in PyTorch')"
