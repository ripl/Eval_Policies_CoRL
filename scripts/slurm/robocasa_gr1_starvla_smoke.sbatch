#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -J robocasa_starvla_smoke
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
cd "${PROJECT_ROOT}"
export POLICY=starvla
export RUN_GROUP="${RUN_GROUP:-smoke}"
export N_EPISODES="${N_EPISODES:-2}"
bash scripts/robocasa/run_gr1_policy_eval.sh
