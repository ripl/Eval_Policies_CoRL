#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=160G
#SBATCH -t 06:00:00
#SBATCH -J calvin_p1_50
#SBATCH --array=0-2
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
NUM_SEQUENCES="${NUM_SEQUENCES:-50}"
RESET_SEED="${RESET_SEED:-0}"
RESET_BANK="${CALVIN_RESET_BANK:-${RESET_BANK:-${PROJECT_ROOT}/results/calvin/reset_banks/abc_d_official_d_table_resets_${NUM_SEQUENCES}seq_seed${RESET_SEED}.npz}}"

cd "${PROJECT_ROOT}"
mkdir -p logs/slurm

if [[ ! -s "${RESET_BANK}" ]]; then
  echo "Missing reset bank: ${RESET_BANK}" >&2
  echo "Build it first with: sbatch scripts/slurm/calvin_train_pose_reset_bank.sbatch" >&2
  exit 2
fi

export NUM_SEQUENCES
export CALVIN_RESET_BANK="${RESET_BANK}"
export CALVIN_RESET_PROTOCOL="calvin_official_d_table_resets_v1"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    POLICY="xvla"
    RUN_SCRIPT="scripts/calvin/run_xvla_calvin_smoke.sh"
    ;;
  1)
    POLICY="gr1"
    RUN_SCRIPT="scripts/calvin/run_gr1_calvin_smoke.sh"
    ;;
  2)
    POLICY="roboflamingo"
    RUN_SCRIPT="scripts/calvin/run_roboflamingo_calvin_smoke.sh"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

export RUN_TAG="${RUN_TAG:-${POLICY}_abc_d_official_d_table_resets_${NUM_SEQUENCES}seq_seed${RESET_SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"

echo "policy=${POLICY}"
echo "run_tag=${RUN_TAG}"
echo "calvin_reset_bank=${CALVIN_RESET_BANK}"
echo "calvin_reset_protocol=${CALVIN_RESET_PROTOCOL}"

bash "${RUN_SCRIPT}"
