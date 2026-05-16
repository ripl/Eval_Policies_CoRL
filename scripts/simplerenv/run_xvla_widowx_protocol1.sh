#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TAG="${RUN_TAG:-protocol1_xvla_${TASK_LABEL:-spoon}_$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_xvla_py310}"
MODEL_PATH="${MODEL_PATH:-2toINF/X-VLA-WidowX}"
TASK_LABEL="${TASK_LABEL:-spoon}"
EPISODE_START="${EPISODE_START:-0}"
EPISODE_END="${EPISODE_END:-500}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/simplerenv/xvla_widowx/protocol1_random_positions/${RUN_TAG}}"
PORT="${PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 10000)))}"
PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-${PROJECT_ROOT}/configs/simplerenv/randomized_positions/widowx_protocol1_seed20260515_n500.json}"
PROTOCOL_SHA256_FILE="${PROTOCOL_SHA256_FILE:-${PROTOCOL_CONFIG}.sha256}"
RUN_PROTOCOL_NAME="${RUN_PROTOCOL_NAME:-widowx_protocol1_random_positions}"

case "${TASK_LABEL}" in
  stack|blocks|carrot|spoon)
    TASK_MAX_STEPS="${TASK_MAX_STEPS:-60}"
    ;;
  *)
    echo "Protocol 1 supports TASK_LABEL=stack|blocks|carrot|spoon, got ${TASK_LABEL}" >&2
    exit 2
    ;;
esac
CLIENT_MAX_STEPS="${CLIENT_MAX_STEPS:-${TASK_MAX_STEPS}}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/cache/conda_pkgs}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/artifacts/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

X_VLA_REPO="${PROJECT_ROOT}/third_party/x_vla"
SIMPLER_ENV_REPO="${PROJECT_ROOT}/third_party/xvla_simpler_env"
SERVER_DIR="${RESULTS_DIR}/server"
CLIENT_DIR="${RESULTS_DIR}/videos"
METADATA_DIR="${PROJECT_ROOT}/artifacts/simplerenv/${RUN_TAG}"

if [[ ! -f "${PROTOCOL_CONFIG}" ]]; then
  echo "Missing protocol config: ${PROTOCOL_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${PROTOCOL_SHA256_FILE}" ]]; then
  echo "Missing protocol checksum: ${PROTOCOL_SHA256_FILE}" >&2
  exit 2
fi
export SIMPLERENV_PROTOCOL_CONFIG="${PROTOCOL_CONFIG}"
export SIMPLERENV_PROTOCOL_SHA256="$(awk '{print $1}' "${PROTOCOL_SHA256_FILE}")"
export X_VLA_REPO

mkdir -p "${RESULTS_DIR}" "${SERVER_DIR}" "${CLIENT_DIR}" "${METADATA_DIR}" "${PROJECT_ROOT}/logs/slurm"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing env at ${ENV_PREFIX}. Run scripts/simplerenv/setup_xvla_env.sh first." >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

export PYTHONPATH="${PROJECT_ROOT}/scripts/simplerenv:${X_VLA_REPO}:${SIMPLER_ENV_REPO}:${PYTHONPATH:-}"

{
  echo "run_tag=${RUN_TAG}"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "project_root=${PROJECT_ROOT}"
  echo "policy=xvla"
  echo "protocol=${RUN_PROTOCOL_NAME}"
  echo "protocol_config=${SIMPLERENV_PROTOCOL_CONFIG}"
  echo "protocol_sha256=${SIMPLERENV_PROTOCOL_SHA256}"
  echo "protocol3_stack_yellow_on_green=${SIMPLERENV_PROTOCOL3_STACK_YELLOW_ON_GREEN:-0}"
  echo "model_path=${MODEL_PATH}"
  echo "task_label=${TASK_LABEL}"
  echo "task_max_steps=${TASK_MAX_STEPS}"
  echo "client_max_steps=${CLIENT_MAX_STEPS}"
  echo "episode_start=${EPISODE_START}"
  echo "episode_end=${EPISODE_END}"
  echo "results_dir=${RESULTS_DIR}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "cpu_model=$(lscpu | awk -F: '/Model name/ {gsub(/^[ \t]+/, \"\", $2); print $2; exit}')"
  echo "cpu_arch=$(uname -m)"
  echo "cpu_count=$(nproc)"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu
  fi
  git -C "${PROJECT_ROOT}" rev-parse HEAD
  git -C "${PROJECT_ROOT}" submodule status --recursive
  python --version
  python -m pip freeze
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${METADATA_DIR}/metadata.txt" 2>&1

cd "${X_VLA_REPO}"
python deploy.py \
  --model_path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --disable_slurm \
  --output_dir "${SERVER_DIR}" \
  > "${SERVER_DIR}/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

python - <<PY
import socket, sys, time
host = "127.0.0.1"
port = int("${PORT}")
for _ in range(900):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex((host, port)) == 0:
            sys.exit(0)
    time.sleep(1)
print(f"server did not open {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

python "${PROJECT_ROOT}/scripts/simplerenv/xvla_widowx_protocol_client.py" \
  --server_ip 127.0.0.1 \
  --server_port "${PORT}" \
  --output_dir "${CLIENT_DIR}" \
  --task-label "${TASK_LABEL}" \
  --episode-start "${EPISODE_START}" \
  --episode-end "${EPISODE_END}" \
  --max-steps "${CLIENT_MAX_STEPS}"

python - <<PY
import json
from pathlib import Path

path = Path("${CLIENT_DIR}/widowx_results.txt")
if not path.exists():
    raise SystemExit(f"missing result file: {path}")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
errors = [r for r in rows if "error" in r]
missing_done = [r for r in rows if "done" not in r]
expected = int("${EPISODE_END}") - int("${EPISODE_START}")
if len(rows) != expected or errors or missing_done:
    raise SystemExit(
        f"invalid X-VLA results: rows={len(rows)} expected={expected} "
        f"errors={len(errors)} missing_done={len(missing_done)}"
    )
standard_horizon = int("${TASK_MAX_STEPS}")
released_success = sum(1 for r in rows if bool(r["done"]))
standard_success = sum(
    1 for r in rows
    if bool(r["done"]) and int(r.get("steps", 10**9)) <= standard_horizon
)
late_successes = [
    {"proc_id": r.get("proc_id"), "steps": r.get("steps")}
    for r in rows
    if bool(r["done"]) and int(r.get("steps", 10**9)) > standard_horizon
]
summary = {
    "policy": "xvla",
    "protocol": "${RUN_PROTOCOL_NAME}",
    "task": "${TASK_LABEL}",
    "episode_start": int("${EPISODE_START}"),
    "episode_end": int("${EPISODE_END}"),
    "standard_horizon": standard_horizon,
    "client_max_steps": int("${CLIENT_MAX_STEPS}"),
    "success_standard_horizon": standard_success,
    "success_rate_standard_horizon": standard_success / len(rows),
    "success_client_horizon": released_success,
    "success_rate_client_horizon": released_success / len(rows),
    "total": len(rows),
    "late_successes": late_successes,
}
out = Path("${RESULTS_DIR}/summary.json")
out.write_text(json.dumps(summary, indent=2) + "\\n")
print(summary)
PY
