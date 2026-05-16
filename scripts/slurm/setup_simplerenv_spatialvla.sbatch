#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 03:00:00
#SBATCH -J setup_se_spatial
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm envs cache artifacts/setup

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_PREFIX=${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_spatialvla_py310}"

bash scripts/simplerenv/setup_spatialvla_env.sh
