#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH --no-requeue
#SBATCH -J se_abcde_cs
#SBATCH -o /home-nfs/tianchong/simplerenv_protocol_abcde_full_%x_%j.out
#SBATCH -e /home-nfs/tianchong/simplerenv_protocol_abcde_full_%x_%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SUPPORT_ROOT="${SUPPORT_ROOT:?SUPPORT_ROOT required}"
POLICY="${POLICY:?POLICY required}"
CONDITION="${CONDITION:?CONDITION required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"

case "${POLICY}" in
  cogact) ENV_PREFIX="${COGACT_ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_cogact_py310_np126}" ;;
  spatialvla) ENV_PREFIX="${SPATIALVLA_ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_spatialvla_py310}" ;;
  *) echo "unsupported POLICY=${POLICY}" >&2; exit 2 ;;
esac

export PROJECT_ROOT SUPPORT_ROOT POLICY CONDITION RUN_ID OUTPUT_ROOT
export SIMPLERENV_PROTOCOL_CONDITION="${CONDITION}"
export SIMPLERENV_PROTOCOL_CONFIG="${PROJECT_ROOT}/configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json"
export SIMPLERENV_PROTOCOL_SHA256="1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "missing env python: ${ENV_PREFIX}/bin/python" >&2
  exit 2
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

python "${SUPPORT_ROOT}/run_cogact_spatialvla_condition.py" \
  --policy "${POLICY}" \
  --condition "${CONDITION}" \
  --run-id "${RUN_ID}" \
  --output-root "${OUTPUT_ROOT}"
