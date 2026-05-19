#!/bin/bash
#SBATCH -p gpu
#SBATCH -C "48g&ampere"
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 03:00:00
#SBATCH -J simvla_l2_smoke
#SBATCH --array=0-3%4
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

SUITES=(libero_spatial libero_object libero_goal libero_10)
export TASK_SUITE="${SUITES[${SLURM_ARRAY_TASK_ID:-0}]}"
export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export SIMVLA_REPO="${SIMVLA_REPO:-${PROJECT_ROOT}/third_party/simvla}"
export ARTIFACT_NAME="${ARTIFACT_NAME:-init_state_layer2_4suite_2500_seed20260519}"
export INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/${ARTIFACT_NAME}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJECT_ROOT}/artifacts/libero/libero_config_layer2_4suite_2500_seed20260519}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/simvla-libero}"
export CKPT_REVISION="${CKPT_REVISION:-93dc4d90b0596c652ad2840ad743c62b9c4473fb}"
export OVERLAY_DIR="${OVERLAY_DIR:-${PROJECT_ROOT}/envs/simvla_py310_overlay_tf4}"
export RUN_GROUP="${RUN_GROUP:-layer2_smoke_${SLURM_ARRAY_JOB_ID:-manual}}"
export RESULT_ROOT="${PROJECT_ROOT}/results/libero/layer2_smokes/${RUN_GROUP}/simvla/${TASK_SUITE}"
export RESULT_DIR="${RESULT_ROOT}/task_0"
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

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${OVERLAY_DIR}:${SIMVLA_REPO}:${PYTHONPATH:-}"

PORT=$((8600 + (${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}} % 300) + ${SLURM_ARRAY_TASK_ID:-0}))
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

python "${PROJECT_ROOT}/scripts/libero/validate_layer2_smoke_csv.py" \
  --episode-csv "${EPISODE_CSV}" \
  --expected-rows 5 \
  --policy-name simvla \
  --suite "${TASK_SUITE}" \
  --summary-path "${RESULT_DIR}/summary.json"
