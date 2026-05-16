#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_xvla_py310}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/cache/conda_pkgs}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/artifacts/wandb}"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" "${WANDB_DIR}" "${PROJECT_ROOT}/artifacts/setup" \
  "${PROJECT_ROOT}/logs/slurm" "${PROJECT_ROOT}/envs"

cd "${PROJECT_ROOT}"
git submodule update --init --recursive third_party/x_vla third_party/xvla_simpler_env

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip "setuptools==80.9.0" wheel
python -m pip install "numpy==1.26.3"
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.1.2" "torchvision==0.16.2" "torchaudio==2.1.2"

python -m pip install -e "${PROJECT_ROOT}/third_party/xvla_simpler_env/ManiSkill2_real2sim"
python -m pip install -e "${PROJECT_ROOT}/third_party/xvla_simpler_env"

python -m pip install \
  "numpy==1.26.3" "tensorflow==2.15.0" "opencv-python==4.11.0.86" \
  "transformers==4.51.3" "accelerate==1.2.1" "peft==0.17.1" \
  "safetensors==0.4.5" "fastapi" "uvicorn==0.34.3" \
  "json_numpy==2.1.0" "mediapy==1.2.4" "av==15.0.0" \
  "scipy==1.15.0" "einops==0.8.1" "timm==1.0.12" \
  "mmengine==0.10.5" "pyarrow==20.0.0" "h5py==3.12.1" \
  requests pillow transforms3d

python - <<'PY'
import importlib
for name in ["numpy", "torch", "tensorflow", "transformers", "fastapi", "uvicorn", "json_numpy", "simpler_env"]:
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'import-ok')}")
PY

python -m pip freeze > "${PROJECT_ROOT}/artifacts/setup/simplerenv_xvla_pip_freeze.txt"
conda list > "${PROJECT_ROOT}/artifacts/setup/simplerenv_xvla_conda_list.txt"
git submodule status --recursive > "${PROJECT_ROOT}/artifacts/setup/submodule_status.txt"

echo "Created/updated env: ${ENV_PREFIX}"
