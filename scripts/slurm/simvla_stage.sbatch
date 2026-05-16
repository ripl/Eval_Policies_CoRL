#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J simvla_stage
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export SIMVLA_REPO="${SIMVLA_REPO:-${PROJECT_ROOT}/third_party/simvla}"
export CKPT_REPO="${CKPT_REPO:-YuankaiLuo/SimVLA-LIBERO}"
export CKPT_REVISION="${CKPT_REVISION:-93dc4d90b0596c652ad2840ad743c62b9c4473fb}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/simvla-libero}"
export OVERLAY_DIR="${OVERLAY_DIR:-${PROJECT_ROOT}/envs/simvla_py310_overlay_tf4}"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda_pkgs"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${CKPT_DIR}" "${OVERLAY_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" logs/slurm
cd "${SIMVLA_REPO}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF

python -m pip install --upgrade --target "${OVERLAY_DIR}" \
  "transformers>=4.57.0,<5" \
  websockets \
  msgpack-numpy \
  openpi-client \
  hf_xet

export PYTHONPATH="${OVERLAY_DIR}:${SIMVLA_REPO}:${PYTHONPATH:-}"

python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ["CKPT_REPO"]
revision = os.environ["CKPT_REVISION"]
target = Path(os.environ["CKPT_DIR"])
path = snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=target,
    local_dir_use_symlinks=False,
)
required = ["config.json", "model.safetensors", "state.json"]
missing = [name for name in required if not (target / name).exists()]
if missing:
    raise SystemExit(f"checkpoint download missing files: {missing}")

manifest = {
    "checkpoint_hf_repo": repo,
    "checkpoint_hf_sha": revision,
    "checkpoint_dir": str(target),
    "snapshot_download_path": path,
    "overlay_dir": os.environ["OVERLAY_DIR"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
}
(target / "SIMVLA_STAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

python - <<'PY'
from packaging.version import Version
import transformers
import websockets
import msgpack_numpy
import openpi_client
import libero

if Version(transformers.__version__) < Version("4.57.0"):
    raise SystemExit(f"transformers too old: {transformers.__version__}")
print("transformers", transformers.__version__)
print("staging validation ok")
PY
