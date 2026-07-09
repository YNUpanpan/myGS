#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

INSTALLER="${TOOLS_ROOT}/Miniconda3-latest-Linux-x86_64.sh"
URL="${MINICONDA_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

mkdir -p "${TOOLS_ROOT}"

if [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "Project conda already installed: ${CONDA_ROOT}/bin/conda"
  "${CONDA_ROOT}/bin/conda" --version
  exit 0
fi

if [[ -f "${INSTALLER}" ]]; then
  installer_size="$(stat -c '%s' "${INSTALLER}")"
else
  installer_size="0"
fi

if [[ "${installer_size}" -lt 104857600 ]]; then
  require_command curl
  echo "Downloading Miniconda installer to ${INSTALLER}"
  curl -L --fail "${URL}" -o "${INSTALLER}"
fi

bash "${INSTALLER}" -b -p "${CONDA_ROOT}"
"${CONDA_ROOT}/bin/conda" config --set auto_activate_base false
"${CONDA_ROOT}/bin/conda" --version
