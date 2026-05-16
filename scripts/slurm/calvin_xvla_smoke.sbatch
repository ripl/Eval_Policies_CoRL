#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -J calvin_xvla_50
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
cd "${PROJECT_ROOT}"
bash scripts/calvin/run_xvla_calvin_smoke.sh
