#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH --array=1-24%8
#SBATCH -J robocasa_starvla_calib
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
cd "${PROJECT_ROOT}"
export POLICY=starvla
export RUN_GROUP="${RUN_GROUP:-official_calibration}"
export N_EPISODES="${N_EPISODES:-50}"
export TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID}}"
bash scripts/robocasa/run_gr1_policy_eval.sh
