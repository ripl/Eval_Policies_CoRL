#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J hif_goal_dl
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export REPO_ID="${REPO_ID:-minnielin/hifvla-libero-goal}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/hifvla-libero-goal}"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda_pkgs"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${CKPT_DIR}" "${PROJECT_ROOT}/logs/slurm"
cd "${PROJECT_ROOT}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "REPO_ID=${REPO_ID}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF

python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo_id = os.environ["REPO_ID"]
ckpt_dir = Path(os.environ["CKPT_DIR"])
info = HfApi().model_info(repo_id, revision="main")
snapshot_download(
    repo_id=repo_id,
    revision=info.sha,
    local_dir=ckpt_dir,
)
manifest = {
    "repo_id": repo_id,
    "revision": info.sha,
    "local_dir": str(ckpt_dir),
}
(ckpt_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

find "${CKPT_DIR}" -maxdepth 2 -type f | sort | sed -n '1,120p'
