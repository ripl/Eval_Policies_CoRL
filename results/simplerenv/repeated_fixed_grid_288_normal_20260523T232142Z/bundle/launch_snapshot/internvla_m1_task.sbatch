#!/bin/bash
#SBATCH -J internvla_m1_rep288
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH --no-requeue
#SBATCH -o /share/data/ripl/tianchong/projects/Eval_Policies_CoRL/scratch/repeated_fixed_grid_calibration_288_20260523_v2/slurm/%x_%j.out
#SBATCH -e /share/data/ripl/tianchong/projects/Eval_Policies_CoRL/scratch/repeated_fixed_grid_calibration_288_20260523_v2/slurm/%x_%j.err

set -euo pipefail

ROOT="/share/data/ripl/tianchong/projects/Eval_Policies_CoRL"
SCRATCH="${SCRATCH:-${ROOT}/scratch/repeated_fixed_grid_calibration_288_20260523_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/simplerenv/repeated_fixed_grid_calibration_288_20260523_v2/internvla_m1_third_party_standard_horizon}"
SIM_PY="/share/data/ripl/tianchong/conda/envs/internvla_simplerenv_repro/bin/python"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

TASK="${1:-}"
if [[ -z "${TASK}" ]]; then
  echo "usage: sbatch ${SCRATCH}/internvla_m1_task.sbatch {stack|carrot|spoon|eggplant} [driver args...]" >&2
  exit 2
fi
shift || true
case "${TASK}" in
  stack|carrot|spoon|eggplant) ;;
  *) echo "unknown task: ${TASK}" >&2; exit 2 ;;
esac

PORT="${SERVER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"
mkdir -p "${OUTPUT_ROOT}/${TASK}/logs" "${OUTPUT_ROOT}/bundle/${TASK}"
cp "$0" "${OUTPUT_ROOT}/bundle/${TASK}/internvla_m1_task.sbatch"

HASH_ARGS=()
if [[ "${HASH_WEIGHTS:-1}" == "1" ]]; then
  HASH_ARGS+=(--hash-weights)
fi

"${SIM_PY}" -u "${SCRATCH}/internvla_m1_task_driver.py" \
  --task "${TASK}" \
  --output-root "${OUTPUT_ROOT}" \
  --port "${PORT}" \
  --no-skip-existing \
  "${HASH_ARGS[@]}" \
  "$@"
