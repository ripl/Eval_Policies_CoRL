#!/bin/bash
#SBATCH -J repcal_xvla_task
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH -o /home-nfs/tianchong/repeated_fixed_grid_calibration_288_20260523_v2_%x_%j.out

set -Eeuo pipefail
SUPPORT_ROOT="${SUPPORT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/scratch/repeated_fixed_grid_calibration_288_20260523_v2/xvla_worker_a_support}"
TASK_ARG="${1:-${TASK:-}}"
if [[ -z "${TASK_ARG}" ]]; then
  echo "Usage: sbatch -J repcal_xvla_<task> ${SUPPORT_ROOT}/launch/run_xvla_task.sbatch {stack,carrot,spoon,eggplant}" >&2
  exit 2
fi
exec bash "${SUPPORT_ROOT}/run_one_task.sh" --task "${TASK_ARG}"
