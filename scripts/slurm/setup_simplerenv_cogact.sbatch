#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -J setup_se_cogact
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm artifacts/setup cache envs

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ENV_PREFIX=${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_cogact}"

bash scripts/simplerenv/setup_cogact_env.sh
