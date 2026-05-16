#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH -J robocasa_gr1_setup
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
cd "${PROJECT_ROOT}"
bash scripts/robocasa/setup_gr1_calibration_envs.sh
