#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -J calvin_reset_bank
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
CONDA_BASE="${CONDA_BASE:-/share/data/ripl/tianchong/conda}"
CALVIN_ROOT="${CALVIN_ROOT:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right/CALVIN}"
DATASET_DIR="${DATASET_DIR:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right_cache/task_ABC_D}"
NUM_SEQUENCES="${NUM_SEQUENCES:-50}"
RESET_SEED="${RESET_SEED:-0}"
SEQUENCE_WORKERS="${SEQUENCE_WORKERS:-4}"
RESET_BANK="${RESET_BANK:-${PROJECT_ROOT}/results/calvin/reset_banks/abc_d_official_d_table_resets_${NUM_SEQUENCES}seq_seed${RESET_SEED}.npz}"
SEQUENCE_MANIFEST="${CALVIN_SEQUENCE_MANIFEST:-${SEQUENCE_MANIFEST:-}}"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${RESET_BANK}")" logs/slurm

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_BASE}/envs/calvin"
export PYTHONPATH="${PROJECT_ROOT}/envs/calvin_smoke_overlay:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"

MANIFEST_ARG=()
if [[ -n "${SEQUENCE_MANIFEST}" ]]; then
  if [[ ! -s "${SEQUENCE_MANIFEST}" ]]; then
    echo "Missing sequence manifest: ${SEQUENCE_MANIFEST}" >&2
    exit 2
  fi
  MANIFEST_ARG=(--sequence-manifest "${SEQUENCE_MANIFEST}")
fi

python -u scripts/calvin/calvin_reset_bank.py \
  --dataset-dir "${DATASET_DIR}" \
  --calvin-root "${CALVIN_ROOT}" \
  --output "${RESET_BANK}" \
  --num-sequences "${NUM_SEQUENCES}" \
  --seed "${RESET_SEED}" \
  --sequence-workers "${SEQUENCE_WORKERS}" \
  "${MANIFEST_ARG[@]}"

python - <<PY2
import json
import numpy as np
from pathlib import Path
path = Path("${RESET_BANK}")
data = np.load(path, allow_pickle=False)
meta = json.loads(str(data["metadata_json"].item()))
print(json.dumps({"reset_bank": str(path), "metadata": meta}, indent=2, sort_keys=True))
PY2
