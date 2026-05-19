#!/bin/bash
#SBATCH -p gpu
#SBATCH -C "48g&ampere"
#SBATCH --gpus=1
#SBATCH -c 1
#SBATCH --mem=50G
#SBATCH -t 03:00:00
#SBATCH -J pi05_l2_smoke
#SBATCH --array=0-3%4
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

SUITES=(libero_spatial libero_object libero_goal libero_10)
export TASK_SUITE="${SUITES[${SLURM_ARRAY_TASK_ID:-0}]}"
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
export RUN_GROUP="${RUN_GROUP:-layer2_smoke_${SLURM_ARRAY_JOB_ID:-manual}}"
export RESULT_ROOT="${PROJECT_ROOT}/results/libero/layer2_smokes/${RUN_GROUP}/pi05_lerobot/${TASK_SUITE}"
export RESULT_DIR="${RESULT_ROOT}/task_0"
export EPISODE_CSV="${RESULT_DIR}/episodes.csv"
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

mkdir -p "${RESULT_DIR}/logs" "${RESULT_DIR}/metadata" logs/slurm
cd "${PROJECT_ROOT}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${PI05_OVERLAY}:${SIMVLA_OVERLAY}:${LEROBOT_SRC}:${PYTHONPATH:-}"

python -u "${PROJECT_ROOT}/scripts/libero/run_pi05_goal_repro.py" \
  --model-id "${CKPT_REPO}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --initial-states-path "${INIT_ROOT}" \
  --task-suite "${TASK_SUITE}" \
  --selected-task-id 0 \
  --episode-start-idx 0 \
  --episode-end-idx 5 \
  --max-total-episodes 5 \
  --episode-results-path "${EPISODE_CSV}" \
  --summary-path "${RESULT_DIR}/summary.json" \
  --manifest-path "${RESULT_DIR}/manifest.json" \
  --policy-name pi05_lerobot \
  --policy-tracker-id 869 \
  --tokenizer-name "${TOKENIZER_REPO}" \
  --device cuda \
  --n-action-steps "${N_ACTION_STEPS}" \
  > "${RESULT_DIR}/logs/eval.log" 2>&1

python "${PROJECT_ROOT}/scripts/libero/validate_layer2_smoke_csv.py" \
  --episode-csv "${EPISODE_CSV}" \
  --expected-rows 5 \
  --policy-name pi05_lerobot \
  --suite "${TASK_SUITE}" \
  --summary-path "${RESULT_DIR}/smoke_validation_summary.json"
