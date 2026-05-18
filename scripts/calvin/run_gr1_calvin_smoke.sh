#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
CONDA_BASE="${CONDA_BASE:-/share/data/ripl/tianchong/conda}"
CALVIN_ROOT="${CALVIN_ROOT:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right/CALVIN}"
DATASET_DIR="${DATASET_DIR:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right_cache/task_ABC_D}"
GR1_REPO="${GR1_REPO:-${PROJECT_ROOT}/third_party/gr1}"
NUM_SEQUENCES="${NUM_SEQUENCES:-50}"
EVAL_START="${EVAL_START:-0}"
EVAL_END="${EVAL_END:-${NUM_SEQUENCES}}"
RUN_TAG="${RUN_TAG:-gr1_abc_d_${EVAL_START}_${EVAL_END}seq_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/calvin/gr1_abc_d/${RUN_TAG}}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/gr1}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-3.3}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "${RESULTS_DIR}" "${CKPT_DIR}" "${PROJECT_ROOT}/logs/slurm"

bash "${PROJECT_ROOT}/scripts/calvin/ensure_calvin_overlay.sh"

download_if_missing() {
  local url="$1"
  local dst="$2"
  if [[ ! -s "${dst}" ]]; then
    curl -L --retry 5 --continue-at - --output "${dst}" "${url}"
  fi
}

download_if_missing \
  "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth" \
  "${CKPT_DIR}/mae_pretrain_vit_base.pth"
download_if_missing \
  "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr1_code_release/snapshot_ABC.pt" \
  "${CKPT_DIR}/snapshot_ABC.pt"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_BASE}/envs/calvin"

export PYTHONPATH="${PROJECT_ROOT}/envs/calvin_smoke_overlay:${GR1_REPO}:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"

{
  echo "run_tag=${RUN_TAG}"
  echo "policy=GR-1"
  echo "num_sequences=${NUM_SEQUENCES}"
  echo "eval_start=${EVAL_START}"
  echo "eval_end=${EVAL_END}"
  echo "dataset_dir=${DATASET_DIR}"
  echo "results_dir=${RESULTS_DIR}"
  echo "calvin_sequence_manifest=${CALVIN_SEQUENCE_MANIFEST:-}"
  echo "calvin_reset_bank=${CALVIN_RESET_BANK:-}"
  echo "calvin_reset_protocol=${CALVIN_RESET_PROTOCOL:-}"
  echo "hostname=$(hostname)"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  python --version
  python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
PY
  nvidia-smi -L || true
} > "${RESULTS_DIR}/metadata.txt" 2>&1

export CALVIN_RESET_EVAL_START="${EVAL_START}"
python -u "${PROJECT_ROOT}/scripts/calvin/gr1_smoke_eval.py" \
  --gr1-repo "${GR1_REPO}" \
  --calvin-root "${CALVIN_ROOT}" \
  --dataset-dir "${DATASET_DIR}" \
  --eval-dir "${RESULTS_DIR}" \
  --policy-ckpt "${CKPT_DIR}/snapshot_ABC.pt" \
  --mae-ckpt "${CKPT_DIR}/mae_pretrain_vit_base.pth" \
  --num-sequences "${NUM_SEQUENCES}" \
  --eval-start "${EVAL_START}" \
  --eval-end "${EVAL_END}" \
  --device 0
