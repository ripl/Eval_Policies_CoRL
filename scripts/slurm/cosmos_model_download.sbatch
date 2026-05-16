#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -J cosmos_dl
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export POLICY_REPO="${POLICY_REPO:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
export POLICY_REVISION="${POLICY_REVISION:-cb689ec0e3347c13667d70a78a3447388f5c3bb8}"
export BASE_REPO="${BASE_REPO:-nvidia/Cosmos-Predict2-2B-Video2World}"
export BASE_REVISION="${BASE_REVISION:-f50c09f5d8ab133a90cac3f4886a6471e9ba3f18}"
export POLICY_DIR="${POLICY_DIR:-${PROJECT_ROOT}/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B}"
export BASE_DIR="${BASE_DIR:-${PROJECT_ROOT}/checkpoints/Cosmos-Predict2-2B-Video2World}"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PYTHON="${PYTHON:-/share/data/ripl/tianchong/conda/envs/libero/bin/python}"

mkdir -p "${POLICY_DIR}" "${BASE_DIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" logs/slurm

echo "node=$(hostname)"
echo "date_utc=$(date -u --iso-8601=seconds)"
echo "POLICY_REPO=${POLICY_REPO}"
echo "POLICY_REVISION=${POLICY_REVISION}"
echo "POLICY_DIR=${POLICY_DIR}"
echo "BASE_REPO=${BASE_REPO}"
echo "BASE_REVISION=${BASE_REVISION}"
echo "BASE_DIR=${BASE_DIR}"

"${PYTHON}" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

policy_dir = Path(os.environ["POLICY_DIR"])
base_dir = Path(os.environ["BASE_DIR"])

snapshot_download(
    repo_id=os.environ["POLICY_REPO"],
    revision=os.environ["POLICY_REVISION"],
    local_dir=policy_dir,
    local_dir_use_symlinks=False,
)
required = [
    "Cosmos-Policy-LIBERO-Predict2-2B.pt",
    "libero_dataset_statistics.json",
    "libero_t5_embeddings.pkl",
]
missing = [name for name in required if not (policy_dir / name).exists()]
if missing:
    raise SystemExit(f"policy checkpoint download incomplete, missing: {missing}")
print(f"policy checkpoint staged at {policy_dir}")

try:
    snapshot_download(
        repo_id=os.environ["BASE_REPO"],
        revision=os.environ["BASE_REVISION"],
        local_dir=base_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["vae/**", "model_index.json", "config.json", "README.md"],
    )
except HfHubHTTPError as exc:
    raise SystemExit(
        "base Cosmos Predict2 VAE download failed; this repo is gated and no usable "
        f"Hugging Face token may be configured. Error: {exc}"
    ) from exc

if not (base_dir / "vae").is_dir():
    raise SystemExit(f"base VAE download incomplete: {base_dir / 'vae'} not found")
print(f"base VAE staged at {base_dir}")
PY
