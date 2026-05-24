#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH --no-requeue
#SBATCH -J se_rep288
#SBATCH -o /share/data/ripl/tianchong/projects/Eval_Policies_CoRL/scratch/repeated_fixed_grid_calibration_288_20260523_v2/logs/slurm/%x-%j.out
#SBATCH -e /share/data/ripl/tianchong/projects/Eval_Policies_CoRL/scratch/repeated_fixed_grid_calibration_288_20260523_v2/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SCRATCH_DIR="${SCRATCH_DIR:-${PROJECT_ROOT}/scratch/repeated_fixed_grid_calibration_288_20260523_v2}"
POLICY="${POLICY:?Set POLICY to cogact or spatialvla}"
TASK="${TASK:?Set TASK to stack, carrot, spoon, or eggplant}"
RUN_ID="${RUN_ID:-repeated_fixed_grid_calibration_288_20260523_v2}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_DIR}/runs/${RUN_ID}}"
RESULTS_ROOT="${RESULTS_ROOT:-${RUN_ROOT}/${POLICY}/${TASK}}"
BASE_SEED="${BASE_SEED:-202605230}"
RESUME="${RESUME:-0}"

case "${POLICY}" in
  cogact)
    ENV_PREFIX="${COGACT_ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_cogact_py310_np126}"
    export PYTHONPATH="${PROJECT_ROOT}/third_party/cogact:${PROJECT_ROOT}/third_party/simpler_env:${PYTHONPATH:-}"
    ;;
  spatialvla)
    ENV_PREFIX="${SPATIALVLA_ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_spatialvla_py310}"
    SPATIALVLA_REPO="${SPATIALVLA_REPO:-${SCRATCH_DIR}/sources/simplerenv_openvla_ccfe380}"
    export SPATIALVLA_REPO
    export PYTHONPATH="${SPATIALVLA_REPO}:${PYTHONPATH:-}"
    ;;
  *)
    echo "Invalid POLICY=${POLICY}" >&2
    exit 2
    ;;
esac
case "${TASK}" in
  stack|carrot|spoon|eggplant) ;;
  *)
    echo "Invalid TASK=${TASK}" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_ROOT}"

export PROJECT_ROOT RUN_ID RESULTS_ROOT BASE_SEED RESUME POLICY TASK
export SCRATCH_DIR
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/cache/conda_pkgs}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${PROJECT_ROOT}/cache/apptainer}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${PROJECT_ROOT}/cache/singularity}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/artifacts/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing Python env at ${ENV_PREFIX}" >&2
  exit 2
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

{
  echo "node=$(hostname)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "SCRATCH_DIR=${SCRATCH_DIR}"
  echo "POLICY=${POLICY}"
  echo "TASK=${TASK}"
  echo "RUN_ID=${RUN_ID}"
  echo "RESULTS_ROOT=${RESULTS_ROOT}"
  echo "BASE_SEED=${BASE_SEED}"
  echo "RESUME=${RESUME}"
  echo "ENV_PREFIX=${ENV_PREFIX}"
}

python "${SCRATCH_DIR}/run_one_policy_task.py" \
  --policy "${POLICY}" \
  --task "${TASK}" \
  --run-id "${RUN_ID}" \
  --output-root "${RESULTS_ROOT}" \
  --base-seed "${BASE_SEED}"

python "${SCRATCH_DIR}/validate_job_csv.py" \
  --policy "${POLICY}" \
  --task "${TASK}" \
  --csv "${RESULTS_ROOT}/per_episode_results.csv" \
  --report "${RESULTS_ROOT}/validation_report.json"
