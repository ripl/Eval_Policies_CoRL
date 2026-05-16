#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -J simvla_goal_smoke
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export SIMVLA_REPO="${SIMVLA_REPO:-${PROJECT_ROOT}/third_party/simvla}"
export INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/init_state_goal_5000}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJECT_ROOT}/artifacts/libero/libero_config}"
export CKPT_REPO="${CKPT_REPO:-YuankaiLuo/SimVLA-LIBERO}"
export CKPT_REVISION="${CKPT_REVISION:-93dc4d90b0596c652ad2840ad743c62b9c4473fb}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/simvla-libero}"
export OVERLAY_DIR="${OVERLAY_DIR:-${PROJECT_ROOT}/envs/simvla_py310_overlay_tf4}"
export RUN_GROUP="${RUN_GROUP:-simvla_goal_5k_smoke_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULT_DIR="${PROJECT_ROOT}/results/libero/simvla_goal_5k/${RUN_GROUP}"
export EPISODE_CSV="${RESULT_DIR}/episodes.csv"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda_pkgs"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

mkdir -p "${RESULT_DIR}/logs" "${RESULT_DIR}/metadata" logs/slurm
cd "${SIMVLA_REPO}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SIMVLA_REPO=${SIMVLA_REPO}"
echo "INIT_ROOT=${INIT_ROOT}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "OVERLAY_DIR=${OVERLAY_DIR}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${OVERLAY_DIR}:${SIMVLA_REPO}:${PYTHONPATH:-}"

python - <<'PY'
from pathlib import Path
from packaging.version import Version
import os
import transformers
import websockets
import msgpack_numpy
import openpi_client
import libero
import torch

if Version(transformers.__version__) < Version("4.57.0"):
    raise SystemExit(f"transformers too old: {transformers.__version__}")
for path in [os.environ["CKPT_DIR"], os.environ["INIT_ROOT"], os.environ["OVERLAY_DIR"]]:
    if not Path(path).exists():
        raise SystemExit(f"missing required path: {path}")
print("dependency validation ok")
print("transformers", transformers.__version__)
print("torch", torch.__version__)
PY

{
  echo "run_group=${RUN_GROUP}"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "policy=simvla"
  echo "policy_tracker_id=798"
  echo "suite=libero_goal"
  echo "checkpoint_hf_repo=${CKPT_REPO}"
  echo "checkpoint_hf_sha=${CKPT_REVISION}"
  echo "checkpoint_dir=${CKPT_DIR}"
  echo "init_root=${INIT_ROOT}"
  echo "save_videos=False"
  git -C "${PROJECT_ROOT}" rev-parse HEAD || true
  git -C "${PROJECT_ROOT}" status --short || true
  git -C "${SIMVLA_REPO}" rev-parse HEAD || true
  git -C "${SIMVLA_REPO}" diff -- evaluation/libero/libero_client.py || true
  python --version
  python -m pip freeze
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${RESULT_DIR}/metadata/metadata.txt" 2>&1

PORT=$((8200 + (${SLURM_JOB_ID:-0} % 1000)))
python -u evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint "${CKPT_DIR}" \
  --norm_stats "${SIMVLA_REPO}/norm_stats/libero_norm.json" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  > "${RESULT_DIR}/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

PORT="${PORT}" SERVER_PID="${SERVER_PID}" python - <<'PY'
import os
import socket
import time

host = "127.0.0.1"
port = int(os.environ["PORT"])
server_pid = int(os.environ["SERVER_PID"])
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"server is listening on {host}:{port}")
            break
    except OSError:
        try:
            os.kill(server_pid, 0)
        except OSError:
            raise SystemExit("SimVLA server exited before opening the port")
        time.sleep(5)
else:
    raise SystemExit("timed out waiting for SimVLA server")
PY

python -u evaluation/libero/libero_client.py \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --client_type websocket \
  --task_suite libero_goal \
  --num_trials 5 \
  --initial_states_path "${INIT_ROOT}" \
  --selected_task_id 0 \
  --episode_start_idx 0 \
  --episode_end_idx 5 \
  --max_total_episodes 5 \
  --episode_results_path "${EPISODE_CSV}" \
  --policy_name simvla \
  --checkpoint_ref "${CKPT_DIR}@${CKPT_REVISION}" \
  --video_out "${RESULT_DIR}/videos" \
  --no_video \
  > "${RESULT_DIR}/logs/client.log" 2>&1

python - <<'PY'
import csv
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
episode_csv = Path(os.environ["EPISODE_CSV"])
rows = list(csv.DictReader(episode_csv.open()))
summary = {
    "run_group": os.environ["RUN_GROUP"],
    "policy_name": "simvla",
    "policy_tracker_id": 798,
    "suite": "libero_goal",
    "checkpoint_ref": f"{os.environ['CKPT_DIR']}@{os.environ['CKPT_REVISION']}",
    "checkpoint_hf_repo": os.environ["CKPT_REPO"],
    "checkpoint_hf_sha": os.environ["CKPT_REVISION"],
    "init_state_root": os.environ["INIT_ROOT"],
    "episode_csv": str(episode_csv),
    "rows": len(rows),
    "successes": sum(int(r["success"]) for r in rows),
    "videos": len(list(result_dir.rglob("*.mp4"))),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
}
manifest = {
    **summary,
    "init_state_manifest": str(Path(os.environ["INIT_ROOT"]) / "MANIFEST.json"),
    "libero_config_path": str(Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"),
    "save_videos": False,
    "command": "python evaluation/libero/libero_client.py ...",
    "metadata": str(result_dir / "metadata" / "metadata.txt"),
}
(result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if len(rows) != 5:
    raise SystemExit(f"expected 5 episode rows, got {len(rows)}")
if summary["videos"] != 0:
    raise SystemExit(f"expected no videos, found {summary['videos']}")
PY
