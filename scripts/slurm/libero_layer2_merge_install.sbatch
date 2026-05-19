#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J libero_l2_merge
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

export PEDR_ROOT="${PEDR_ROOT:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right}"
export LIBERO_ROOT="${LIBERO_ROOT:-${PEDR_ROOT}/LIBERO}"
export EPC_ROOT="${EPC_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export ARTIFACT_NAME="${ARTIFACT_NAME:-init_state_layer2_4suite_2500_seed20260519}"
export SOURCE_ROOT="${SOURCE_ROOT:-${LIBERO_ROOT}/libero/libero/${ARTIFACT_NAME}}"
export PART_ROOT="${PART_ROOT:-${LIBERO_ROOT}/libero/libero/${ARTIFACT_NAME}_parts}"
export ROLLOUT_ROOT="${ROLLOUT_ROOT:-${EPC_ROOT}/artifacts/libero/${ARTIFACT_NAME}}"
export CONFIG_DIR="${CONFIG_DIR:-${EPC_ROOT}/artifacts/libero/libero_config_layer2_4suite_2500_seed20260519}"
export BASE_CONFIG_DIR="${BASE_CONFIG_DIR:-${EPC_ROOT}/artifacts/libero/libero_config}"
export BASE_SEED="${BASE_SEED:-20260519}"
export PROJECT_SCRIPT="${PROJECT_SCRIPT:-${EPC_ROOT}/scripts/libero/merge_libero_layer2_init_states.py}"
export HF_HOME="${EPC_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${EPC_ROOT}/cache/torch"
export XDG_CACHE_HOME="${EPC_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${EPC_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${EPC_ROOT}/cache/conda_pkgs"

mkdir -p "${EPC_ROOT}/logs/slurm"
cd "${LIBERO_ROOT}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SOURCE_ROOT=${SOURCE_ROOT}"
echo "PART_ROOT=${PART_ROOT}"
echo "ROLLOUT_ROOT=${ROLLOUT_ROOT}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_SCRIPT}" \
  --final-root "${SOURCE_ROOT}" \
  --part-root "${PART_ROOT}" \
  --rollout-root "${ROLLOUT_ROOT}" \
  --config-dir "${CONFIG_DIR}" \
  --base-config-dir "${BASE_CONFIG_DIR}" \
  --num-states-per-task 250 \
  --seed "${BASE_SEED}"
