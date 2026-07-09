#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

require_command git
mkdir -p "${TOOLS_ROOT}"
OFFICIAL_REPO_URL="git@github.com:graphdeco-inria/gaussian-splatting.git"
MIRROR_REPO_URL="https://gitclone.com/github.com/graphdeco-inria/gaussian-splatting.git"
DIFF_GAUSSIAN_URL="https://gitclone.com/github.com/graphdeco-inria/diff-gaussian-rasterization.git"
FUSED_SSIM_URL="git@github.com:rahul-goel/fused-ssim.git"

if [[ -d "${GAUSSIAN_SPLATTING_DIR}/.git" ]]; then
  echo "3DGS checkout already exists: ${GAUSSIAN_SPLATTING_DIR}"
  git -C "${GAUSSIAN_SPLATTING_DIR}" status --short --branch
  git -C "${GAUSSIAN_SPLATTING_DIR}" config submodule.submodules/diff-gaussian-rasterization.url "${DIFF_GAUSSIAN_URL}"
  git -C "${GAUSSIAN_SPLATTING_DIR}" config submodule.submodules/fused-ssim.url "${FUSED_SSIM_URL}"
  git -C "${GAUSSIAN_SPLATTING_DIR}" submodule update --init --depth 1 \
    submodules/diff-gaussian-rasterization \
    submodules/fused-ssim \
    submodules/simple-knn
else
  git clone --depth 1 "${MIRROR_REPO_URL}" "${GAUSSIAN_SPLATTING_DIR}"
  git -C "${GAUSSIAN_SPLATTING_DIR}" config submodule.submodules/diff-gaussian-rasterization.url "${DIFF_GAUSSIAN_URL}"
  git -C "${GAUSSIAN_SPLATTING_DIR}" config submodule.submodules/fused-ssim.url "${FUSED_SSIM_URL}"
  git -C "${GAUSSIAN_SPLATTING_DIR}" submodule update --init --depth 1 \
    submodules/diff-gaussian-rasterization \
    submodules/fused-ssim \
    submodules/simple-knn
fi

official_head="$(git ls-remote "${OFFICIAL_REPO_URL}" HEAD | awk '{print $1}')"
checkout_head="$(git -C "${GAUSSIAN_SPLATTING_DIR}" rev-parse HEAD)"
if [[ "${official_head}" != "${checkout_head}" ]]; then
  echo "3DGS mirror HEAD mismatch: official=${official_head}, checkout=${checkout_head}" >&2
  exit 61
fi

git -C "${GAUSSIAN_SPLATTING_DIR}" rev-parse --short HEAD
test -f "${GAUSSIAN_SPLATTING_DIR}/train.py"
