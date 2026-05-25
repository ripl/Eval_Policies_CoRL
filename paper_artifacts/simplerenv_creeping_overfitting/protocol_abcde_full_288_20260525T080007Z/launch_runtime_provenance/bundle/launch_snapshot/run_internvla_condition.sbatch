#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH --no-requeue
#SBATCH -J se_abcde_intern
#SBATCH -o /home-nfs/tianchong/simplerenv_protocol_abcde_full_%x_%j.out
#SBATCH -e /home-nfs/tianchong/simplerenv_protocol_abcde_full_%x_%j.err

set -Eeuo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SUPPORT_ROOT="${SUPPORT_ROOT:?SUPPORT_ROOT required}"
CONDITION="${CONDITION:?CONDITION required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"
PORT="${PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"
export PROJECT_ROOT SUPPORT_ROOT CONDITION RUN_ID OUTPUT_ROOT
export SIMPLERENV_PROTOCOL_CONDITION="${CONDITION}"
export SIMPLERENV_PROTOCOL_CONFIG="${PROJECT_ROOT}/configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json"
export SIMPLERENV_PROTOCOL_SHA256="1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

/share/data/ripl/tianchong/conda/envs/internvla_simplerenv_repro/bin/python -u "${SUPPORT_ROOT}/run_internvla_condition.py" \
  --condition "${CONDITION}" \
  --run-id "${RUN_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --support-root "${SUPPORT_ROOT}" \
  --port "${PORT}"
