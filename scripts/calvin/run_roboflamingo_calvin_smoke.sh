#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
CONDA_BASE="${CONDA_BASE:-/share/data/ripl/tianchong/conda}"
CALVIN_ENV="${CALVIN_ENV:-${CONDA_BASE}/envs/calvin}"
CALVIN_ROOT="${CALVIN_ROOT:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right/CALVIN}"
DATASET_DIR="${DATASET_DIR:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right_cache/task_ABC_D}"
RF_REPO="${RF_REPO:-${PROJECT_ROOT}/third_party/roboflamingo}"
NUM_SEQUENCES="${NUM_SEQUENCES:-50}"
EVAL_START="${EVAL_START:-0}"
EVAL_END="${EVAL_END:-${NUM_SEQUENCES}}"
RUN_TAG="${RUN_TAG:-roboflamingo_abc_d_${EVAL_START}_${EVAL_END}seq_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/calvin/roboflamingo_abc_d/${RUN_TAG}}"
MASTER_PORT="${MASTER_PORT:-$((19000 + (${SLURM_JOB_ID:-0} % 10000)))}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-3.3}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

mkdir -p "${RESULTS_DIR}" "${PROJECT_ROOT}/checkpoints/roboflamingo" "${PROJECT_ROOT}/logs/slurm"

bash "${PROJECT_ROOT}/scripts/calvin/ensure_calvin_overlay.sh"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CALVIN_ENV}"
export PYTHONPATH="${PROJECT_ROOT}/envs/calvin_smoke_overlay:${RF_REPO}:${RF_REPO}/open_flamingo:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"

ASSETS_JSON="${RESULTS_DIR}/asset_paths.json"
python - <<PY
import json, os
from pathlib import Path
from huggingface_hub import hf_hub_download
cache_dir = os.environ["HF_HOME"]
rf = hf_hub_download(
    "robovlms/RoboFlamingo",
    "ABC_D/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth",
    cache_dir=cache_dir,
)
of = hf_hub_download(
    "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
    "checkpoint.pt",
    cache_dir=cache_dir,
)
payload = {"roboflamingo_checkpoint": rf, "openflamingo_checkpoint": of}
Path("${ASSETS_JSON}").write_text(json.dumps(payload, indent=2) + "\\n")
print(json.dumps(payload, indent=2))
PY

RF_CKPT="$(python -c 'import json; print(json.load(open("'"${ASSETS_JSON}"'"))["roboflamingo_checkpoint"])')"
OF_CKPT="$(python -c 'import json; print(json.load(open("'"${ASSETS_JSON}"'"))["openflamingo_checkpoint"])')"

{
  echo "run_tag=${RUN_TAG}"
  echo "policy=RoboFlamingo"
  echo "num_sequences=${NUM_SEQUENCES}"
  echo "eval_start=${EVAL_START}"
  echo "eval_end=${EVAL_END}"
  echo "dataset_dir=${DATASET_DIR}"
  echo "results_dir=${RESULTS_DIR}"
  echo "calvin_sequence_manifest=${CALVIN_SEQUENCE_MANIFEST:-}"
  echo "calvin_reset_bank=${CALVIN_RESET_BANK:-}"
  echo "calvin_reset_protocol=${CALVIN_RESET_PROTOCOL:-}"
  echo "roboflamingo_checkpoint=${RF_CKPT}"
  echo "openflamingo_checkpoint=${OF_CKPT}"
  echo "hostname=$(hostname)"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  python --version
  python - <<'PY'
import torch, torchvision, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
PY
  nvidia-smi -L || true
} > "${RESULTS_DIR}/metadata.txt" 2>&1

export CALVIN_RESET_EVAL_START="${EVAL_START}"
torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/scripts/calvin/roboflamingo_smoke_eval.py" \
  --repo "${RF_REPO}" \
  --dataset-dir "${DATASET_DIR}" \
  --calvin-conf-path "${CALVIN_ROOT}/calvin_models/conf" \
  --eval-dir "${RESULTS_DIR}" \
  --checkpoint "${RF_CKPT}" \
  --openflamingo-checkpoint "${OF_CKPT}" \
  --num-sequences "${NUM_SEQUENCES}" \
  --eval-start "${EVAL_START}" \
  --eval-end "${EVAL_END}" \
  --precision "${PRECISION:-fp32}"
