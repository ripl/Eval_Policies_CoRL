#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-official}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/cache/conda_pkgs}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${PROJECT_ROOT}/cache/apptainer}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${PROJECT_ROOT}/cache/singularity}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/artifacts/wandb}"

mkdir -p \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" \
  "${APPTAINER_CACHEDIR}" \
  "${SINGULARITY_CACHEDIR}" \
  "${WANDB_DIR}" \
  "${PROJECT_ROOT}/results" \
  "${PROJECT_ROOT}/artifacts"

COGACT_REPO="${COGACT_REPO:-${PROJECT_ROOT}/third_party/cogact}"
COGACT_SCRIPT="${COGACT_REPO}/sim_cogact/scripts/cogact_bridge.sh"

case "${MODE}" in
  official)
    CONFIG="${PROJECT_ROOT}/configs/simplerenv/cogact_base_official.yaml"
    ;;
  randomized)
    CONFIG="${PROJECT_ROOT}/configs/simplerenv/cogact_base_randomized.yaml"
    echo "Randomized-position injection is not implemented yet: ${CONFIG}" >&2
    exit 3
    ;;
  *)
    echo "Usage: $0 [official|randomized]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${COGACT_SCRIPT}" ]]; then
  echo "Missing CogACT eval script: ${COGACT_SCRIPT}" >&2
  echo "Add CogACT as a pinned submodule or set COGACT_REPO." >&2
  exit 2
fi

echo "Running ${MODE} config: ${CONFIG}"
exec bash "${COGACT_SCRIPT}"

