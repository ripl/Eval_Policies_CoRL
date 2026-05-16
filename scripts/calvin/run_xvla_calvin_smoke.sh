#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
CONDA_BASE="${CONDA_BASE:-/share/data/ripl/tianchong/conda}"
CALVIN_ENV="${CALVIN_ENV:-${CONDA_BASE}/envs/calvin}"
XVLA_ENV="${XVLA_ENV:-${PROJECT_ROOT}/envs/simplerenv_xvla_py310}"
CALVIN_ROOT="${CALVIN_ROOT:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right/CALVIN}"
DATASET_DIR="${DATASET_DIR:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right_cache/task_ABC_D}"
XVLA_REPO="${XVLA_REPO:-${PROJECT_ROOT}/third_party/x_vla}"
MODEL_PATH="${MODEL_PATH:-2toINF/X-VLA-Calvin-ABC_D}"
NUM_SEQUENCES="${NUM_SEQUENCES:-50}"
RUN_TAG="${RUN_TAG:-xvla_abc_d_${NUM_SEQUENCES}seq_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/calvin/xvla_abc_d/${RUN_TAG}}"
PORT="${PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 10000)))}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-3.3}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

mkdir -p "${RESULTS_DIR}/server" "${RESULTS_DIR}/client" "${RESULTS_DIR}/work" "${PROJECT_ROOT}/logs/slurm"

bash "${PROJECT_ROOT}/scripts/calvin/ensure_calvin_overlay.sh"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${XVLA_ENV}"

{
  echo "run_tag=${RUN_TAG}"
  echo "policy=X-VLA"
  echo "model_path=${MODEL_PATH}"
  echo "num_sequences=${NUM_SEQUENCES}"
  echo "dataset_dir=${DATASET_DIR}"
  echo "results_dir=${RESULTS_DIR}"
  echo "hostname=$(hostname)"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  python --version
  python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
PY
  nvidia-smi -L || true
} > "${RESULTS_DIR}/metadata.txt" 2>&1

cd "${XVLA_REPO}"
python deploy.py \
  --model_path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --disable_slurm \
  --output_dir "${RESULTS_DIR}/server" \
  > "${RESULTS_DIR}/server/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

python - <<PY
import socket, time, sys
host = "127.0.0.1"
port = int("${PORT}")
for _ in range(900):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex((host, port)) == 0:
            sys.exit(0)
    time.sleep(1)
raise SystemExit(f"server did not open {host}:{port}")
PY

ln -sfn "${DATASET_DIR}" "${RESULTS_DIR}/work/ABC_D"
cd "${RESULTS_DIR}/work"

conda activate "${CALVIN_ENV}"
export PYTHONPATH="${PROJECT_ROOT}/envs/calvin_smoke_overlay:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"
python -u "${PROJECT_ROOT}/scripts/calvin/xvla_calvin_client_no_video.py" \
  --source-client "${XVLA_REPO}/evaluation/calvin/calvin_client.py" \
  --server_ip 127.0.0.1 \
  --server_port "${PORT}" \
  --output_dir "${RESULTS_DIR}/client" \
  --eval_start 0 \
  --eval_end "${NUM_SEQUENCES}" \
  > "${RESULTS_DIR}/client/client.log" 2>&1

python - <<PY
import json, re
from pathlib import Path
log = Path("${RESULTS_DIR}/client/log.txt")
summary = {"policy": "X-VLA", "num_sequences": int("${NUM_SEQUENCES}"), "log": str(log)}
if log.exists() and log.read_text().strip():
    line = log.read_text().strip().splitlines()[-1]
    vals = [float(x) / 100.0 for x in re.findall(r": ([0-9.]+)%", line)]
    summary.update({"last_line": line, "stage_success": vals[:5], "avg_task_completed": vals[5] if len(vals) > 5 else sum(vals[:5])})
Path("${RESULTS_DIR}/summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
print(summary)
PY
