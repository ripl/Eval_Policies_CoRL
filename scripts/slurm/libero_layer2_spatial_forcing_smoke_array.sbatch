#!/bin/bash
#SBATCH -p gpu
#SBATCH -C "48g&ampere"
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -J sf_l2_smoke
#SBATCH --array=0-3%4
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

SUITES=(libero_spatial libero_object libero_goal libero_10)
export TASK_SUITE="${SUITES[${SLURM_ARRAY_TASK_ID:-0}]}"
export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export SF_REPO="${SF_REPO:-/share/data/ripl/tianchong/projects/repro_policies/Spatial-Forcing/openvla-SF}"
export ARTIFACT_NAME="${ARTIFACT_NAME:-init_state_layer2_4suite_2500_seed20260519}"
export INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/${ARTIFACT_NAME}}"
if [[ -z "${CKPT_DIR:-}" ]]; then
  case "${TASK_SUITE}" in
    libero_spatial) CKPT_DIR="${PROJECT_ROOT}/checkpoints/spatial-forcing-7b-finetuned-libero-spatial" ;;
    libero_object) CKPT_DIR="${PROJECT_ROOT}/checkpoints/spatial-forcing-7b-finetuned-libero-object" ;;
    libero_goal) CKPT_DIR="${PROJECT_ROOT}/checkpoints/spatial-forcing-7b-finetuned-libero-goal" ;;
    libero_10) CKPT_DIR="${PROJECT_ROOT}/checkpoints/spatial-forcing-7b-finetuned-libero-10" ;;
    *) echo "unknown TASK_SUITE=${TASK_SUITE}" >&2; exit 1 ;;
  esac
fi
export CKPT_DIR
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJECT_ROOT}/artifacts/libero/libero_config_layer2_4suite_2500_seed20260519}"
export RUN_GROUP="${RUN_GROUP:-layer2_smoke_${SLURM_ARRAY_JOB_ID:-manual}}"
export RESULT_ROOT="${PROJECT_ROOT}/results/libero/layer2_smokes/${RUN_GROUP}/spatial_forcing/${TASK_SUITE}"
export RESULT_DIR="${RESULT_ROOT}/task_0"
export EPISODE_CSV="${RESULT_DIR}/episodes.csv"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda_pkgs"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

mkdir -p "${RESULT_DIR}/eval_logs" "${RESULT_DIR}/metadata" logs/slurm
cd "${SF_REPO}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${SF_REPO}:${PYTHONPATH:-}"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --task_suite_name "${TASK_SUITE}" \
  --initial_states_path "${INIT_ROOT}" \
  --num_trials_per_task 5 \
  --selected_task_id 0 \
  --episode_start_idx 0 \
  --episode_end_idx 5 \
  --policy_name spatial_forcing \
  --episode_results_path "${EPISODE_CSV}" \
  --local_log_dir "${RESULT_DIR}/eval_logs" \
  --run_id_note "${RUN_GROUP}--${TASK_SUITE}--task_0" \
  --save_videos False \
  --use_wandb False \
  --center_crop True

python "${PROJECT_ROOT}/scripts/libero/validate_layer2_smoke_csv.py" \
  --episode-csv "${EPISODE_CSV}" \
  --expected-rows 5 \
  --policy-name spatial_forcing \
  --suite "${TASK_SUITE}" \
  --summary-path "${RESULT_DIR}/summary.json"
