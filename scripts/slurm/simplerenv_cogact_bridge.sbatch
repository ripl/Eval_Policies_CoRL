#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -J se_cogact
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
MODE="${MODE:-official}"

cd "${PROJECT_ROOT}"
mkdir -p logs/slurm results artifacts checkpoints cache envs

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODE=${MODE}"
echo "RUN_TAG=${RUN_TAG:-}"
echo "EPISODE_START=${EPISODE_START:-0}"
echo "EPISODE_END=${EPISODE_END:-24}"
echo "TASK_FILTER=${TASK_FILTER:-all}"

bash scripts/simplerenv/run_cogact_bridge.sh "${MODE}"
