#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH -J se_p3_stack_100
#SBATCH --array=0-2
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm results artifacts cache

POLICIES=(cogact spatialvla xvla)

idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( idx < 0 || idx >= ${#POLICIES[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${idx}" >&2
  exit 2
fi

policy="${POLICIES[$idx]}"
export EPISODE_START="${EPISODE_START:-0}"
export EPISODE_END="${EPISODE_END:-100}"
export RUN_TAG="${RUN_TAG:-protocol3_stack_yellow_on_green_100_${policy}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
export RUN_PROTOCOL_NAME="widowx_protocol3_stack_yellow_on_green_protocol1_positions"
export ADDITIONAL_ENV_SAVE_TAGS="protocol3_yellow_on_green_protocol1_seed20260515"
export SIMPLERENV_PROTOCOL3_STACK_YELLOW_ON_GREEN=1

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "policy=${policy}"
echo "task=stack"
echo "RUN_TAG=${RUN_TAG}"
echo "RUN_PROTOCOL_NAME=${RUN_PROTOCOL_NAME}"
echo "EPISODE_START=${EPISODE_START}"
echo "EPISODE_END=${EPISODE_END}"

case "${policy}" in
  cogact)
    export TASK_FILTER=stack
    export RESULTS_DIR="${PROJECT_ROOT}/results/simplerenv/cogact_base/protocol3_stack_yellow_on_green_protocol1_positions/${RUN_TAG}"
    bash scripts/simplerenv/run_cogact_bridge_protocol1.sh
    ;;
  spatialvla)
    export TASK_FILTER=stack
    export RESULTS_DIR="${PROJECT_ROOT}/results/simplerenv/spatialvla_bridge/protocol3_stack_yellow_on_green_protocol1_positions/${RUN_TAG}"
    bash scripts/simplerenv/run_spatialvla_bridge_protocol1.sh
    ;;
  xvla)
    export TASK_LABEL=stack
    export RESULTS_DIR="${PROJECT_ROOT}/results/simplerenv/xvla_widowx/protocol3_stack_yellow_on_green_protocol1_positions/${RUN_TAG}"
    bash scripts/simplerenv/run_xvla_widowx_protocol1.sh
    ;;
  *)
    echo "Unknown policy: ${policy}" >&2
    exit 2
    ;;
esac
