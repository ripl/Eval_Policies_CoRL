#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 1
#SBATCH --mem=50G
#SBATCH -t 08:00:00
#SBATCH -J pi05_l2_full
#SBATCH --array=0-39
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

SUITES=(libero_spatial libero_object libero_goal libero_10)
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUITE_IDX=$((ARRAY_ID / 10))
TASK_ID=$((ARRAY_ID % 10))
export TASK_SUITE="${SUITES[${SUITE_IDX}]}"
export TASK_ID
export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export LEROBOT_REPO="${LEROBOT_REPO:-${PROJECT_ROOT}/third_party/lerobot}"
export LEROBOT_SRC="${LEROBOT_SRC:-${LEROBOT_REPO}/src}"
export ARTIFACT_NAME="${ARTIFACT_NAME:-init_state_layer2_4suite_2500_seed20260519}"
export INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/${ARTIFACT_NAME}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJECT_ROOT}/artifacts/libero/libero_config_layer2_4suite_2500_seed20260519}"
export CKPT_REPO="${CKPT_REPO:-lerobot/pi05_libero_finetuned}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/pi05-libero-finetuned}"
export TOKENIZER_REPO="${TOKENIZER_REPO:-Shakalaka/paligemma-3b-pt-224}"
export SIMVLA_OVERLAY="${SIMVLA_OVERLAY:-${PROJECT_ROOT}/envs/simvla_py310_overlay_tf4}"
export PI05_OVERLAY="${PI05_OVERLAY:-${PROJECT_ROOT}/envs/pi05_py310_overlay_v5}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
export RUN_GROUP="${RUN_GROUP:-layer2_full_${SLURM_ARRAY_JOB_ID:-manual}}"
export RESULT_ROOT="${PROJECT_ROOT}/results/libero/layer2_full/${RUN_GROUP}/pi05_lerobot/${TASK_SUITE}"
export RESULT_DIR="${RESULT_ROOT}/task_${TASK_ID}"
export EPISODE_CSV="${RESULT_DIR}/episodes.csv"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-250}"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda_pkgs"
export WANDB_DIR="${PROJECT_ROOT}/artifacts/wandb"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

mkdir -p "${RESULT_DIR}/logs" "${RESULT_DIR}/metadata" "${RESULT_ROOT}/shared" logs/slurm
cd "${PROJECT_ROOT}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "TASK_SUITE=${TASK_SUITE}"
echo "TASK_ID=${TASK_ID}"
echo "INIT_ROOT=${INIT_ROOT}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${PI05_OVERLAY}:${SIMVLA_OVERLAY}:${LEROBOT_SRC}:${PYTHONPATH:-}"

python - <<'PY'
from pathlib import Path
import os
import torch
import transformers
from transformers.models.siglip import check

for key in ["INIT_ROOT", "LEROBOT_SRC", "PI05_OVERLAY", "SIMVLA_OVERLAY", "CKPT_DIR"]:
    path = Path(os.environ[key])
    if not path.exists():
        raise SystemExit(f"missing required path {key}: {path}")
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit("OpenPI patched transformers check failed")
print("dependency validation ok")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY

if [[ "${ARRAY_ID}" == "0" ]]; then
  cp "${PROJECT_ROOT}/scripts/slurm/libero_layer2_pi05_full_array.sbatch" "${RESULT_ROOT}/shared/sbatch_template.sbatch" 2>/dev/null || true
  cp "${PROJECT_ROOT}/scripts/libero/run_pi05_goal_repro.py" "${RESULT_ROOT}/shared/run_pi05_goal_repro.py" 2>/dev/null || true
  cp "${LIBERO_CONFIG_PATH}/config.yaml" "${RESULT_ROOT}/shared/libero_config.yaml" 2>/dev/null || true
  cp "${INIT_ROOT}/MANIFEST.json" "${RESULT_ROOT}/shared/init_state_manifest.json" 2>/dev/null || true
  python -m pip freeze > "${RESULT_ROOT}/shared/pip_freeze.txt" 2>&1 || true
  conda list > "${RESULT_ROOT}/shared/conda_list.txt" 2>&1 || true
fi

{
  echo "run_group=${RUN_GROUP}"
  echo "policy=pi05_lerobot"
  echo "policy_tracker_id=869"
  echo "suite=${TASK_SUITE}"
  echo "task_id=${TASK_ID}"
  echo "checkpoint=${CKPT_DIR}"
  echo "tokenizer_repo=${TOKENIZER_REPO}"
  echo "n_action_steps=${N_ACTION_STEPS}"
  echo "init_root=${INIT_ROOT}"
  echo "episodes_per_task=${EPISODES_PER_TASK}"
  git -C "${PROJECT_ROOT}" rev-parse HEAD || true
  git -C "${PROJECT_ROOT}" status --short || true
  git -C "${LEROBOT_REPO}" rev-parse HEAD || true
  git -C "${LEROBOT_REPO}" status --short || true
  python --version
  echo "hostname=$(hostname)"
  echo "slurmd_nodename=${SLURMD_NODENAME:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "nproc=$(nproc --all 2>/dev/null || true)"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu
  fi
  if command -v free >/dev/null 2>&1; then
    free -h
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${RESULT_DIR}/metadata/metadata.txt" 2>&1

python -u "${PROJECT_ROOT}/scripts/libero/run_pi05_goal_repro.py" \
  --model-id "${CKPT_REPO}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --initial-states-path "${INIT_ROOT}" \
  --task-suite "${TASK_SUITE}" \
  --selected-task-id "${TASK_ID}" \
  --episode-start-idx 0 \
  --episode-end-idx "${EPISODES_PER_TASK}" \
  --max-total-episodes "${EPISODES_PER_TASK}" \
  --episode-results-path "${EPISODE_CSV}" \
  --summary-path "${RESULT_DIR}/summary.json" \
  --manifest-path "${RESULT_DIR}/manifest.json" \
  --policy-name pi05_lerobot \
  --policy-tracker-id 869 \
  --tokenizer-name "${TOKENIZER_REPO}" \
  --device cuda \
  --n-action-steps "${N_ACTION_STEPS}" \
  > "${RESULT_DIR}/logs/eval.log" 2>&1

python "${PROJECT_ROOT}/scripts/libero/validate_layer2_episode_csv.py" \
  --episode-csv "${EPISODE_CSV}" \
  --expected-rows "${EPISODES_PER_TASK}" \
  --policy-name pi05_lerobot \
  --suite "${TASK_SUITE}" \
  --summary-path "${RESULT_DIR}/smoke_validation_summary.json" \
  --require-no-errors \
  --require-no-videos
