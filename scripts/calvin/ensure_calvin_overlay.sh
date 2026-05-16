#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
CONDA_BASE="${CONDA_BASE:-/share/data/ripl/tianchong/conda}"
CALVIN_ENV="${CALVIN_ENV:-${CONDA_BASE}/envs/calvin}"
OVERLAY="${CALVIN_OVERLAY:-${PROJECT_ROOT}/envs/calvin_smoke_overlay}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"

mkdir -p "${OVERLAY}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" \
  "${PROJECT_ROOT}/logs/slurm" "${PROJECT_ROOT}/artifacts/setup"

lock="${OVERLAY}.lock"
exec 9>"${lock}"
flock 9

if [[ -f "${OVERLAY}/.calvin_smoke_overlay_ready" ]]; then
  echo "CALVIN smoke overlay already ready: ${OVERLAY}"
  exit 0
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CALVIN_ENV}"

python -m pip install --upgrade --target "${OVERLAY}" --no-deps \
  "json_numpy==2.1.0" \
  "ftfy==6.2.3" \
  "wcwidth==0.2.14" \
  "open_clip_torch==2.24.0" \
  "einops==0.8.1" \
  "einops-exts==0.0.4" \
  "braceexpand==0.1.7" \
  "webdataset==0.2.86" \
  "flamingo-pytorch==0.1.2" \
  "transformers==4.33.1" \
  "tokenizers==0.13.3"

python -m pip install --upgrade --target "${OVERLAY}" --no-deps \
  "git+https://github.com/openai/CLIP.git"

PYTHONPATH="${OVERLAY}:${PYTHONPATH:-}" python - <<'PY'
import importlib
for name in ["json_numpy", "clip", "open_clip", "transformers", "pyrender", "flamingo_pytorch"]:
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'import-ok')}")
PY

touch "${OVERLAY}/.calvin_smoke_overlay_ready"
python -m pip freeze > "${PROJECT_ROOT}/artifacts/setup/calvin_smoke_overlay_pip_freeze.txt"
echo "Created CALVIN smoke overlay: ${OVERLAY}"
