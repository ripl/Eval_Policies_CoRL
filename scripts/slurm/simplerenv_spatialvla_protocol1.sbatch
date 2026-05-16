#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -J se_spatial_p1
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm results artifacts cache

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RUN_TAG=${RUN_TAG:-}"
echo "TASK_FILTER=${TASK_FILTER:-all}"
echo "EPISODE_START=${EPISODE_START:-0}"
echo "EPISODE_END=${EPISODE_END:-500}"

bash scripts/simplerenv/run_spatialvla_bridge_protocol1.sh
