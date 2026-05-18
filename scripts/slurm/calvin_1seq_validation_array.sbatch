#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 02:00:00
#SBATCH -J calvin_1seq_val
#SBATCH --array=0-5
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
EVAL_START="${EVAL_START:-0}"
EVAL_END="${EVAL_END:-1}"
RESET_SEED="${RESET_SEED:-0}"
SEQUENCE_WORKERS="${SEQUENCE_WORKERS:-4}"
SEQUENCE_MANIFEST="${CALVIN_SEQUENCE_MANIFEST:-${SEQUENCE_MANIFEST:-${PROJECT_ROOT}/results/calvin/sequence_manifests/abc_d_official_${NUM_SEQUENCES}seq_workers${SEQUENCE_WORKERS}.json}}"
RESET_BANK="${CALVIN_RESET_BANK:-${RESET_BANK:-${PROJECT_ROOT}/results/calvin/reset_banks/abc_d_official_d_table_resets_${NUM_SEQUENCES}seq_seed${RESET_SEED}.npz}}"
XVLA_EP_LEN="${XVLA_EP_LEN:-360}"

cd "${PROJECT_ROOT}"
mkdir -p logs/slurm

if [[ "${SLURM_CPUS_PER_TASK:-0}" -gt 4 ]]; then
  echo "Refusing to run with SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}; expected <=4" >&2
  exit 2
fi
if [[ ! -s "${SEQUENCE_MANIFEST}" ]]; then
  echo "Missing sequence manifest: ${SEQUENCE_MANIFEST}" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0) POLICY="xvla"; CONDITION="calibration"; RUN_SCRIPT="scripts/calvin/run_xvla_calvin_smoke.sh" ;;
  1) POLICY="gr1"; CONDITION="calibration"; RUN_SCRIPT="scripts/calvin/run_gr1_calvin_smoke.sh" ;;
  2) POLICY="roboflamingo"; CONDITION="calibration"; RUN_SCRIPT="scripts/calvin/run_roboflamingo_calvin_smoke.sh" ;;
  3) POLICY="xvla"; CONDITION="protocol1"; RUN_SCRIPT="scripts/calvin/run_xvla_calvin_smoke.sh" ;;
  4) POLICY="gr1"; CONDITION="protocol1"; RUN_SCRIPT="scripts/calvin/run_gr1_calvin_smoke.sh" ;;
  5) POLICY="roboflamingo"; CONDITION="protocol1"; RUN_SCRIPT="scripts/calvin/run_roboflamingo_calvin_smoke.sh" ;;
  *) echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

export NUM_SEQUENCES
export EVAL_START
export EVAL_END
export CALVIN_SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST}"
export XVLA_EP_LEN
export REQUIRE_XVLA_EP_LEN="${XVLA_EP_LEN}"

if [[ "${CONDITION}" == "protocol1" ]]; then
  if [[ ! -s "${RESET_BANK}" ]]; then
    echo "Missing reset bank: ${RESET_BANK}" >&2
    exit 2
  fi
  export CALVIN_RESET_BANK="${RESET_BANK}"
  export CALVIN_RESET_PROTOCOL="calvin_official_d_table_resets_v1"
  CONDITION_TAG="official_d_table_resets_seed${RESET_SEED}"
else
  unset CALVIN_RESET_BANK
  unset CALVIN_RESET_PROTOCOL
  CONDITION_TAG="calibration"
fi

export RUN_TAG="${RUN_TAG:-${POLICY}_abc_d_${CONDITION_TAG}_${NUM_SEQUENCES}seq_1seq_validation_${EVAL_START}_${EVAL_END}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"

echo "policy=${POLICY}"
echo "condition=${CONDITION}"
echo "eval_start=${EVAL_START}"
echo "eval_end=${EVAL_END}"
echo "run_tag=${RUN_TAG}"
echo "calvin_sequence_manifest=${CALVIN_SEQUENCE_MANIFEST}"
echo "calvin_reset_bank=${CALVIN_RESET_BANK:-}"
echo "xvla_ep_len=${XVLA_EP_LEN}"
echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

bash "${RUN_SCRIPT}"
