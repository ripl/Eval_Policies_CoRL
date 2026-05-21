#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -J simvla_l2_full
#SBATCH --array=0-39
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

SUITES=(libero_spatial libero_object libero_goal libero_10)
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUITE_IDX=$((ARRAY_ID / 10))
TASK_ID=$((ARRAY_ID % 10))
export TASK_SUITE="${SUITES[${SUITE_IDX}]}"
export TASK_ID
export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export SIMVLA_REPO="${SIMVLA_REPO:-${PROJECT_ROOT}/third_party/simvla}"
export ARTIFACT_NAME="${ARTIFACT_NAME:-init_state_layer2_4suite_2500_seed20260519}"
export INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/${ARTIFACT_NAME}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJECT_ROOT}/artifacts/libero/libero_config_layer2_4suite_2500_seed20260519}"
export CKPT_REPO="${CKPT_REPO:-YuankaiLuo/SimVLA-LIBERO}"
export CKPT_REVISION="${CKPT_REVISION:-93dc4d90b0596c652ad2840ad743c62b9c4473fb}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/simvla-libero}"
export OVERLAY_DIR="${OVERLAY_DIR:-${PROJECT_ROOT}/envs/simvla_py310_overlay_tf4}"
export RUN_GROUP="${RUN_GROUP:-layer2_full_${SLURM_ARRAY_JOB_ID:-manual}}"
export RESULT_ROOT="${PROJECT_ROOT}/results/libero/layer2_full/${RUN_GROUP}/simvla/${TASK_SUITE}"
export RESULT_DIR="${RESULT_ROOT}/task_${TASK_ID}"
export EPISODE_CSV="${RESULT_DIR}/episodes.csv"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-250}"
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

mkdir -p "${RESULT_DIR}/logs" "${RESULT_DIR}/metadata" "${RESULT_ROOT}/shared" logs/slurm
cd "${SIMVLA_REPO}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "TASK_SUITE=${TASK_SUITE}"
echo "TASK_ID=${TASK_ID}"
echo "INIT_ROOT=${INIT_ROOT}"
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

if [[ "${ARRAY_ID}" == "0" ]]; then
  cp "${PROJECT_ROOT}/scripts/slurm/libero_layer2_simvla_full_array.sbatch" "${RESULT_ROOT}/shared/sbatch_template.sbatch" 2>/dev/null || true
  cp "${LIBERO_CONFIG_PATH}/config.yaml" "${RESULT_ROOT}/shared/libero_config.yaml"
  cp "${INIT_ROOT}/MANIFEST.json" "${RESULT_ROOT}/shared/init_state_manifest.json"
  python -m pip freeze > "${RESULT_ROOT}/shared/pip_freeze.txt"
  conda list > "${RESULT_ROOT}/shared/conda_list.txt"
fi

{
  echo "run_group=${RUN_GROUP}"
  echo "policy=simvla"
  echo "policy_tracker_id=798"
  echo "suite=${TASK_SUITE}"
  echo "task_id=${TASK_ID}"
  echo "checkpoint=${CKPT_DIR}@${CKPT_REVISION}"
  echo "init_root=${INIT_ROOT}"
  echo "episodes_per_task=${EPISODES_PER_TASK}"
  git -C "${PROJECT_ROOT}" rev-parse HEAD || true
  git -C "${PROJECT_ROOT}" status --short || true
  git -C "${SIMVLA_REPO}" rev-parse HEAD || true
  git -C "${SIMVLA_REPO}" status --short || true
  python --version
  echo "hostname=$(hostname)"
  echo "slurmd_nodename=${SLURMD_NODENAME:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "nproc=$(nproc --all 2>/dev/null || true)"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu
  fi
  if command -v free >/dev/null 2>&1; then
    free -h
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${RESULT_DIR}/metadata/metadata.txt" 2>&1

PORT=$((8200 + (${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}} % 500) + ARRAY_ID))
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
  --task_suite "${TASK_SUITE}" \
  --num_trials "${EPISODES_PER_TASK}" \
  --initial_states_path "${INIT_ROOT}" \
  --selected_task_id "${TASK_ID}" \
  --episode_start_idx 0 \
  --episode_end_idx "${EPISODES_PER_TASK}" \
  --max_total_episodes "${EPISODES_PER_TASK}" \
  --episode_results_path "${EPISODE_CSV}" \
  --policy_name simvla \
  --checkpoint_ref "${CKPT_DIR}@${CKPT_REVISION}" \
  --video_out "${RESULT_DIR}/videos" \
  --no_video \
  > "${RESULT_DIR}/logs/client.log" 2>&1

python "${PROJECT_ROOT}/scripts/libero/validate_layer2_episode_csv.py" \
  --episode-csv "${EPISODE_CSV}" \
  --expected-rows "${EPISODES_PER_TASK}" \
  --policy-name simvla \
  --suite "${TASK_SUITE}" \
  --summary-path "${RESULT_DIR}/summary.json" \
  --require-no-errors \
  --require-no-videos
