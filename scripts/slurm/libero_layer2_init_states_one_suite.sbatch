#!/bin/bash
#SBATCH -p gpu
#SBATCH -C "ampere"
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J libero_l2_suite
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

if [[ -z "${TASK_SUITE:-}" ]]; then
  echo "TASK_SUITE is required" >&2
  exit 1
fi

case "${TASK_SUITE}" in
  libero_spatial) SUITE_INDEX=0 ;;
  libero_object) SUITE_INDEX=1 ;;
  libero_goal) SUITE_INDEX=2 ;;
  libero_10) SUITE_INDEX=3 ;;
  *) echo "unknown TASK_SUITE=${TASK_SUITE}" >&2; exit 1 ;;
esac

export PEDR_ROOT="${PEDR_ROOT:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right}"
export LIBERO_ROOT="${LIBERO_ROOT:-${PEDR_ROOT}/LIBERO}"
export EPC_ROOT="${EPC_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export ARTIFACT_NAME="${ARTIFACT_NAME:-init_state_layer2_4suite_2500_seed20260519}"
export PART_ROOT="${PART_ROOT:-${LIBERO_ROOT}/libero/libero/${ARTIFACT_NAME}_parts}"
export OUT_ROOT="${OUT_ROOT:-${PART_ROOT}/${TASK_SUITE}}"
export BASE_SEED="${BASE_SEED:-20260519}"
export SUITE_SEED=$((BASE_SEED + SUITE_INDEX * 100000))
export PROJECT_SCRIPT="${PROJECT_SCRIPT:-${EPC_ROOT}/scripts/libero/generate_libero_layer2_init_states.py}"
export HF_HOME="${EPC_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${EPC_ROOT}/cache/torch"
export XDG_CACHE_HOME="${EPC_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${EPC_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${EPC_ROOT}/cache/conda_pkgs"
export MUJOCO_GL=egl

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

mkdir -p "${EPC_ROOT}/logs/slurm"
cd "${LIBERO_ROOT}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "TASK_SUITE=${TASK_SUITE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "SUITE_SEED=${SUITE_SEED}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_SCRIPT}" \
  --out-root "${OUT_ROOT}" \
  --suites "${TASK_SUITE}" \
  --num-states-per-task 250 \
  --seed "${SUITE_SEED}" \
  --env-img-res 256
