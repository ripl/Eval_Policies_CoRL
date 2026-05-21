#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -J sf_l2_full
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
export RUN_GROUP="${RUN_GROUP:-layer2_full_${SLURM_ARRAY_JOB_ID:-manual}}"
export RESULT_ROOT="${PROJECT_ROOT}/results/libero/layer2_full/${RUN_GROUP}/spatial_forcing/${TASK_SUITE}"
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

mkdir -p "${RESULT_DIR}/eval_logs" "${RESULT_DIR}/metadata" "${RESULT_ROOT}/shared" logs/slurm
cd "${SF_REPO}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "TASK_SUITE=${TASK_SUITE}"
echo "TASK_ID=${TASK_ID}"
echo "INIT_ROOT=${INIT_ROOT}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${SF_REPO}:${PYTHONPATH:-}"

if [[ "${ARRAY_ID}" == "0" ]]; then
  cp "${PROJECT_ROOT}/scripts/slurm/libero_layer2_spatial_forcing_full_array.sbatch" "${RESULT_ROOT}/shared/sbatch_template.sbatch" 2>/dev/null || true
  cp "${LIBERO_CONFIG_PATH}/config.yaml" "${RESULT_ROOT}/shared/libero_config.yaml"
  cp "${INIT_ROOT}/MANIFEST.json" "${RESULT_ROOT}/shared/init_state_manifest.json"
  python -m pip freeze > "${RESULT_ROOT}/shared/pip_freeze.txt"
  conda list > "${RESULT_ROOT}/shared/conda_list.txt"
fi

{
  echo "run_group=${RUN_GROUP}"
  echo "policy=spatial_forcing"
  echo "policy_tracker_id=438"
  echo "suite=${TASK_SUITE}"
  echo "task_id=${TASK_ID}"
  echo "checkpoint=${CKPT_DIR}"
  echo "init_root=${INIT_ROOT}"
  echo "episodes_per_task=${EPISODES_PER_TASK}"
  git -C "${PROJECT_ROOT}" rev-parse HEAD || true
  git -C "${PROJECT_ROOT}" status --short || true
  git -C "${SF_REPO}" rev-parse HEAD || true
  git -C "${SF_REPO}" status --short || true
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

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --task_suite_name "${TASK_SUITE}" \
  --initial_states_path "${INIT_ROOT}" \
  --num_trials_per_task "${EPISODES_PER_TASK}" \
  --selected_task_id "${TASK_ID}" \
  --episode_start_idx 0 \
  --episode_end_idx "${EPISODES_PER_TASK}" \
  --policy_name spatial_forcing \
  --episode_results_path "${EPISODE_CSV}" \
  --local_log_dir "${RESULT_DIR}/eval_logs" \
  --run_id_note "${RUN_GROUP}--${TASK_SUITE}--task_${TASK_ID}" \
  --save_videos False \
  --use_wandb False \
  --center_crop True

python "${PROJECT_ROOT}/scripts/libero/validate_layer2_episode_csv.py" \
  --episode-csv "${EPISODE_CSV}" \
  --expected-rows "${EPISODES_PER_TASK}" \
  --policy-name spatial_forcing \
  --suite "${TASK_SUITE}" \
  --summary-path "${RESULT_DIR}/summary.json" \
  --require-no-errors \
  --require-no-videos
