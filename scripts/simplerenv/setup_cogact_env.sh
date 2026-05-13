#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_cogact}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/cache/conda_pkgs}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/artifacts/wandb}"

mkdir -p \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" \
  "${WANDB_DIR}" \
  "${PROJECT_ROOT}/artifacts/setup" \
  "${PROJECT_ROOT}/logs/slurm" \
  "${PROJECT_ROOT}/envs"

cd "${PROJECT_ROOT}"
git submodule update --init --recursive third_party/cogact third_party/simpler_env

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip setuptools wheel

# SimplerEnv currently recommends numpy<2.0; 1.24.4 matches its README.
python -m pip install "numpy==1.24.4"

# Install CUDA-enabled PyTorch before editable packages so CogACT does not pull
# an arbitrary torch build from the default PyPI index.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch>=2.2.0" "torchvision>=0.16.0" torchaudio

python -m pip install -e "${PROJECT_ROOT}/third_party/simpler_env/ManiSkill2_real2sim"
python -m pip install -e "${PROJECT_ROOT}/third_party/simpler_env"
python -m pip install -e "${PROJECT_ROOT}/third_party/cogact"

# TensorFlow 2.15 is not NumPy-2 compatible. Recent OpenCV wheels can pull
# NumPy 2.x, so keep OpenCV on a NumPy-1-compatible build for this evaluator.
python -m pip install \
  "numpy==1.26.4" \
  "opencv-python==4.11.0.86" \
  transforms3d

python - <<'PY'
import importlib
for name in ["numpy", "cv2", "torch", "tensorflow", "simpler_env", "sim_cogact", "vla"]:
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'import-ok')}")
PY

python -m pip freeze > "${PROJECT_ROOT}/artifacts/setup/simplerenv_cogact_pip_freeze.txt"
conda list > "${PROJECT_ROOT}/artifacts/setup/simplerenv_cogact_conda_list.txt"
git submodule status --recursive > "${PROJECT_ROOT}/artifacts/setup/submodule_status.txt"

echo "Created/updated env: ${ENV_PREFIX}"
