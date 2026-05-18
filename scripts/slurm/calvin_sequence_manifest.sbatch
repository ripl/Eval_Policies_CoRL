#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J calvin_seq_manifest
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
CONDA_BASE="${CONDA_BASE:-/share/data/ripl/tianchong/conda}"
CALVIN_ROOT="${CALVIN_ROOT:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right/CALVIN}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
SEQUENCE_WORKERS="${SEQUENCE_WORKERS:-4}"
SEQUENCE_MANIFEST="${CALVIN_SEQUENCE_MANIFEST:-${SEQUENCE_MANIFEST:-${PROJECT_ROOT}/results/calvin/sequence_manifests/abc_d_official_${NUM_SEQUENCES}seq_workers${SEQUENCE_WORKERS}.json}}"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${SEQUENCE_MANIFEST}")" logs/slurm

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_BASE}/envs/calvin"
export PYTHONPATH="${PROJECT_ROOT}/envs/calvin_smoke_overlay:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"

python -u scripts/calvin/calvin_sequence_manifest.py \
  --calvin-root "${CALVIN_ROOT}" \
  --output "${SEQUENCE_MANIFEST}" \
  --num-sequences "${NUM_SEQUENCES}" \
  --sequence-workers "${SEQUENCE_WORKERS}"
