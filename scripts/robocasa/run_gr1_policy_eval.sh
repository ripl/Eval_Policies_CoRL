#!/bin/bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export ROBOCASA_REPO="${ROBOCASA_REPO:-${PROJECT_ROOT}/third_party/robocasa-gr1-tabletop-tasks}"
export STARVLA_REPO="${STARVLA_REPO:-${PROJECT_ROOT}/third_party/starVLA}"
export ABOT_REPO="${ABOT_REPO:-${PROJECT_ROOT}/third_party/ABot-Manipulation}"

export SIM_ENV="${SIM_ENV:-${PROJECT_ROOT}/envs/robocasa_gr1_sim}"
export STARVLA_ENV="${STARVLA_ENV:-${PROJECT_ROOT}/envs/robocasa_gr1_starvla}"
export ABOT_ENV="${ABOT_ENV:-${PROJECT_ROOT}/envs/robocasa_gr1_abot}"

export ROBOCASA_CKPT_ROOT="${ROBOCASA_CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/robocasa}"
export STARVLA_CKPT_PATH="${STARVLA_CKPT_PATH:-${ROBOCASA_CKPT_ROOT}/starvla_qwen3_vl_oft_robocasa/checkpoints/steps_90000_pytorch_model.pt}"
export ABOT_CKPT_PATH="${ABOT_CKPT_PATH:-${ROBOCASA_CKPT_ROOT}/abot_m0_robocasa/checkpoints/steps_50000_pytorch_model.pt}"

export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda_pkgs"
export WANDB_DIR="${PROJECT_ROOT}/artifacts/wandb"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

usage() {
  cat >&2 <<'EOF'
Usage:
  POLICY=starvla|abot [TASK_INDEX=1..24|ENV_NAME=...] [N_EPISODES=2] bash scripts/robocasa/run_gr1_policy_eval.sh

Runs one RoboCasa-GR1 task with the released policy websocket server and the
policy repo's official simulation_env.py client.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  [[ -e "${path}" ]] || fail "missing required path: ${path}"
}

choose_port() {
  python - <<'PY'
import os
import socket

job_id = int(os.environ.get("SLURM_JOB_ID") or 0)
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID") or 0)
base = job_id * 37 + task_id * 101 + os.getpid()
for offset in range(20000):
    port = 20000 + ((base + offset) % 20000)
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    raise SystemExit(0)
raise SystemExit("no free local port found in 20000..39999")
PY
}

POLICY="${1:-${POLICY:-}}"
[[ -n "${POLICY}" ]] || { usage; fail "POLICY is required"; }

case "${POLICY}" in
  starvla)
    POLICY_REPO="${STARVLA_REPO}"
    POLICY_ENV="${STARVLA_ENV}"
    CKPT_PATH="${CKPT_PATH:-${STARVLA_CKPT_PATH}}"
    SERVER_EXTRA_ARGS=(--use_bf16)
    ;;
  abot)
    POLICY_REPO="${ABOT_REPO}"
    POLICY_ENV="${ABOT_ENV}"
    CKPT_PATH="${CKPT_PATH:-${ABOT_CKPT_PATH}}"
    SERVER_EXTRA_ARGS=(--use_bf16)
    ;;
  *)
    usage
    fail "unknown POLICY: ${POLICY}"
    ;;
esac

cd "${PROJECT_ROOT}"
TASK_FILE="${TASK_FILE:-${PROJECT_ROOT}/scripts/robocasa/gr1_tasks.txt}"
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ -n "${ENV_NAME:-}" && -n "${TASK_INDEX}" ]]; then
  fail "set only one of ENV_NAME or TASK_INDEX"
fi

if [[ -n "${TASK_INDEX}" ]]; then
  [[ "${TASK_INDEX}" =~ ^[0-9]+$ ]] || fail "TASK_INDEX must be an integer: ${TASK_INDEX}"
  ENV_NAME="$(sed -n "${TASK_INDEX}p" "${TASK_FILE}")"
  [[ -n "${ENV_NAME}" ]] || fail "TASK_INDEX ${TASK_INDEX} not found in ${TASK_FILE}"
else
  ENV_NAME="${ENV_NAME:-gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env}"
fi

grep -Fxq "${ENV_NAME}" "${TASK_FILE}" || fail "ENV_NAME is not in official GR1 task file: ${ENV_NAME}"

N_EPISODES="${N_EPISODES:-2}"
N_ENVS="${N_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-12}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
PORT="${PORT:-$(choose_port)}"
RUN_GROUP="${RUN_GROUP:-official_calibration}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_job${SLURM_JOB_ID:-manual}_${POLICY}_task${TASK_INDEX:-custom}_n${N_EPISODES}}"
ENV_SLUG="$(printf '%s' "${ENV_NAME}" | tr '/:' '__' | tr -cd 'A-Za-z0-9_.-')"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/results/robocasa/gr1_calibration/${RUN_GROUP}/${RUN_ID}/${ENV_SLUG}}"
VIDEO_DIR="${VIDEO_DIR:-${RESULT_DIR}/videos}"
SERVER_LOG="${RESULT_DIR}/server.log"
SIM_LOG="${RESULT_DIR}/simulation.log"
META_FILE="${RESULT_DIR}/metadata.txt"

require_path "${ROBOCASA_REPO}/.git"
require_path "${POLICY_REPO}/.git"
require_path "${SIM_ENV}/bin/python"
require_path "${POLICY_ENV}/bin/python"
require_path "${CKPT_PATH}"
mkdir -p "${RESULT_DIR}" "${VIDEO_DIR}" "${PROJECT_ROOT}/logs/slurm"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

{
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "policy=${POLICY}"
  echo "env_name=${ENV_NAME}"
  echo "task_index=${TASK_INDEX:-}"
  echo "n_episodes=${N_EPISODES}"
  echo "n_envs=${N_ENVS}"
  echo "max_episode_steps=${MAX_EPISODE_STEPS}"
  echo "n_action_steps=${N_ACTION_STEPS}"
  echo "port=${PORT}"
  echo "ckpt_path=${CKPT_PATH}"
  echo "policy_repo=${POLICY_REPO}"
  git -C "${POLICY_REPO}" rev-parse HEAD
  git -C "${POLICY_REPO}" status --short
  echo "robocasa_repo=${ROBOCASA_REPO}"
  git -C "${ROBOCASA_REPO}" rev-parse HEAD
  git -C "${ROBOCASA_REPO}" status --short
  nvidia-smi -L || true
} > "${META_FILE}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "starting ${POLICY} server on port ${PORT}"
conda activate "${POLICY_ENV}"
cd "${POLICY_REPO}"
PYTHONPATH="${POLICY_REPO}:${PYTHONPATH:-}" \
  python deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${PORT}" \
    --idle_timeout -1 \
    "${SERVER_EXTRA_ARGS[@]}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
conda deactivate

ready=0
for ((elapsed = 0; elapsed <= SERVER_READY_TIMEOUT; elapsed += 5)); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -120 "${SERVER_LOG}" >&2 || true
    fail "policy server exited before becoming ready"
  fi
  if python - "${PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
  then
    ready=1
    break
  fi
  sleep 5
done

[[ "${ready}" == "1" ]] || { tail -120 "${SERVER_LOG}" >&2 || true; fail "policy server did not open port ${PORT} within ${SERVER_READY_TIMEOUT}s"; }

echo "server ready; starting simulator for ${ENV_NAME}"
conda activate "${SIM_ENV}"
cd "${POLICY_REPO}"
ABOT_SKIP_FRAMEWORK_AUTOIMPORT=1 \
PYTHONPATH="${POLICY_REPO}:${ROBOCASA_REPO}:${PYTHONPATH:-}" \
  python examples/Robocasa_tabletop/eval_files/simulation_env.py \
    --args.env_name "${ENV_NAME}" \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.n_episodes "${N_EPISODES}" \
    --args.n_envs "${N_ENVS}" \
    --args.max_episode_steps "${MAX_EPISODE_STEPS}" \
    --args.n_action_steps "${N_ACTION_STEPS}" \
    --args.video_out_path "${VIDEO_DIR}" \
    --args.pretrained_path "${CKPT_PATH}" \
    > "${SIM_LOG}" 2>&1
conda deactivate

{
  echo "result_dir=${RESULT_DIR}"
  grep -E "Running |Results for|Success rate|Collecting " "${SIM_LOG}" || true
} | tee "${RESULT_DIR}/summary.txt"

echo "robocasa gr1 ${POLICY} eval complete: ${RESULT_DIR}"
