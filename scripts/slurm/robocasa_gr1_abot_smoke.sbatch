#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 06:00:00
#SBATCH -J robocasa_abot_smoke
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
cd "${PROJECT_ROOT}"
export POLICY=abot
export RUN_GROUP="${RUN_GROUP:-smoke}"
export N_EPISODES="${N_EPISODES:-2}"
bash scripts/robocasa/run_gr1_policy_eval.sh
