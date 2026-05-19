#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -J sf_l2_stage
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda_pkgs"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_ROOT}/checkpoints" "${PROJECT_ROOT}/logs/slurm"
cd "${PROJECT_ROOT}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF

python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

project_root = Path(os.environ["PROJECT_ROOT"])
repos = {
    "libero_spatial": "haofuly/spatial-forcing-7b-finetuned-libero-spatial",
    "libero_object": "haofuly/spatial-forcing-7b-finetuned-libero-object",
    "libero_goal": "haofuly/spatial-forcing-7b-finetuned-libero-goal",
    "libero_10": "haofuly/spatial-forcing-7b-finetuned-libero-10",
}
api = HfApi()
records = []
for suite, repo_id in repos.items():
    local_dir = project_root / "checkpoints" / repo_id.split("/")[-1]
    info = api.model_info(repo_id, revision="main")
    required = [
        "config.json",
        "model.safetensors.index.json",
        "action_head--latest_checkpoint.pt",
        "proprio_projector--latest_checkpoint.pt",
    ]
    if all((local_dir / name).exists() for name in required):
        print(f"already staged {suite}: {local_dir}", flush=True)
    else:
        print(f"staging {suite}: {repo_id}@{info.sha} -> {local_dir}", flush=True)
        snapshot_download(repo_id=repo_id, revision=info.sha, local_dir=local_dir)
    missing = [name for name in required if not (local_dir / name).exists()]
    if missing:
        raise SystemExit(f"{repo_id} staged incompletely; missing {missing}")
    manifest = {
        "suite": suite,
        "repo_id": repo_id,
        "revision": info.sha,
        "local_dir": str(local_dir),
    }
    (local_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records.append(manifest)

out = project_root / "checkpoints" / "spatial_forcing_layer2_suite_checkpoints.json"
out.write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
PY
