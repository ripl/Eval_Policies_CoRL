#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${PROJECT_ROOT}/scratch/repeated_fixed_grid_calibration_288_20260523_v2/xvla_worker_a_support}"
BIN_DIR="${SUPPORT_ROOT}/bin"
POLICY="xvla"
TASK=""
DRY_RUN=0
RESUME=0
CONTINUE_ON_ERROR=0
RUN_TAG="${RUN_TAG:-}"
RESULT_BASE="${RESULT_BASE:-${SUPPORT_ROOT}/runs}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_xvla_py310}"
X_VLA_REPO="${PROJECT_ROOT}/third_party/x_vla"
SIMPLER_ENV_REPO="${PROJECT_ROOT}/third_party/xvla_simpler_env"
MODEL_PATH="${MODEL_PATH:-2toINF/X-VLA-WidowX}"
PORT="${PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 10000)))}"
EPISODE_TIMEOUT_SECONDS="${EPISODE_TIMEOUT_SECONDS:-1800}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

usage() {
  cat <<'EOF'
Usage: bash run_one_task.sh --task {stack,carrot,spoon,eggplant} [--dry-run] [--resume] [--continue-on-error] [--run-tag TAG]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume) RESUME=1; shift ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${TASK}" ]]; then
  usage >&2
  exit 2
fi

python3 "${BIN_DIR}/preflight.py" --task "${TASK}" --env-prefix "${ENV_PREFIX}" --model-path "${MODEL_PATH}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run=1"
  echo "policy=${POLICY}"
  echo "task=${TASK}"
  echo "normal_command=bash ${SUPPORT_ROOT}/run_one_task.sh --task ${TASK}"
  echo "sbatch_command=sbatch -J repcal_xvla_${TASK} ${SUPPORT_ROOT}/launch/run_xvla_task.sbatch ${TASK}"
  exit 0
fi

if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG="xvla_${TASK}_$(date -u +%Y%m%dT%H%M%SZ)_job${SLURM_JOB_ID:-manual}_$$"
fi
RESULT_ROOT="${RESULT_BASE}/${RUN_TAG}"
LOG_DIR="${RESULT_ROOT}/logs"
MANIFEST="${RESULT_ROOT}/manifest.csv"
RESULT_CSV="${RESULT_ROOT}/per_episode_results.csv"
JOB_ID="${SLURM_JOB_ID:-manual}"

if [[ "${RESUME}" == "0" && -e "${RESULT_ROOT}" ]]; then
  echo "ERROR: result root already exists; pass --resume only if this is an intentional resume: ${RESULT_ROOT}" >&2
  exit 2
fi
if [[ "${RESUME}" == "1" && ! -d "${RESULT_ROOT}" ]]; then
  echo "ERROR: --resume requested but result root does not exist: ${RESULT_ROOT}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${RESULT_ROOT}/episodes" "${RESULT_ROOT}/metadata"
exec > >(tee -a "${LOG_DIR}/driver.log") 2>&1

python3 "${BIN_DIR}/preflight.py" \
  --task "${TASK}" \
  --env-prefix "${ENV_PREFIX}" \
  --model-path "${MODEL_PATH}" \
  --output-json "${RESULT_ROOT}/metadata/preflight.json"
SERVER_MODEL_PATH="$(python3 -c 'import json, sys; data=json.load(open(sys.argv[1])); ident=data["model_identity"]; print(ident.get("snapshot_path") or ident.get("local_path") or ident["model_path"])' "${RESULT_ROOT}/metadata/preflight.json")"

HORIZON="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from protocol import horizon_for; print(horizon_for(sys.argv[2]))' "${BIN_DIR}" "${TASK}")"
SERVER_SEED="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from protocol import server_start_seed; print(server_start_seed(sys.argv[2]))' "${BIN_DIR}" "${TASK}")"

if [[ "${RESUME}" == "0" ]]; then
  python3 "${BIN_DIR}/write_manifest.py" --task "${TASK}" --output "${MANIFEST}"
  python3 "${BIN_DIR}/init_results_csv.py" --output "${RESULT_CSV}"
else
  python3 "${BIN_DIR}/validate_task_results.py" \
    --task "${TASK}" \
    --results "${RESULT_CSV}" \
    --manifest "${MANIFEST}" \
    --report "${RESULT_ROOT}/validation_resume_precheck.json" \
    --allow-partial
fi

cat > "${RESULT_ROOT}/seed_control_caveat.txt" <<EOF
Rollout seed formula: 20260523 + task_index*10000 + repeat_id*100 + official_episode_id.
For each X-VLA client rollout subprocess, PYTHONHASHSEED, Python random, NumPy, and torch seeds are set to the manifest seed before the official client module is loaded.
The X-VLA model server is started once for the task job with startup seed ${SERVER_SEED}.
The server API does not expose a per-rollout RNG reset or seed field, so server-side torch sampling is not independently reset to each manifest seed. Treat the manifest seed as controlling the client/simulator subprocess and as an identifier for the rollout, not as full server RNG control.
EOF

{
  echo "policy=${POLICY}"
  echo "task=${TASK}"
  echo "horizon=${HORIZON}"
  echo "run_tag=${RUN_TAG}"
  echo "result_root=${RESULT_ROOT}"
  echo "job_id=${JOB_ID}"
  echo "node=$(hostname)"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "model_path=${MODEL_PATH}"
  echo "server_model_path=${SERVER_MODEL_PATH}"
  echo "x_vla_repo=${X_VLA_REPO}"
  echo "xvla_simpler_env=${SIMPLER_ENV_REPO}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
} > "${RESULT_ROOT}/metadata/runtime_metadata.txt"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
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

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
export X_VLA_REPO
export PYTHONPATH="${BIN_DIR}:${X_VLA_REPO}:${SIMPLER_ENV_REPO}:${PYTHONPATH:-}"

{
  which python
  python --version
  python -m pip freeze
} > "${RESULT_ROOT}/metadata/python_environment.txt" 2>&1

cd "${X_VLA_REPO}"
PYTHONHASHSEED="${SERVER_SEED}" EVAL_ROLLOUT_SEED="${SERVER_SEED}" \
python "${BIN_DIR}/seeded_runpy.py" \
  --seed "${SERVER_SEED}" \
  --script "${X_VLA_REPO}/deploy.py" \
  -- --model_path "${SERVER_MODEL_PATH}" --host 127.0.0.1 --port "${PORT}" --disable_slurm --output_dir "${RESULT_ROOT}/server" \
  > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python - "${PORT}" <<'PYSOCK'
import socket
import sys
import time
port = int(sys.argv[1])
for _ in range(900):
    sock = socket.socket()
    sock.settimeout(1.0)
    try:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            sys.exit(0)
    finally:
        sock.close()
    time.sleep(1)
print(f"server did not open 127.0.0.1:{port}", file=sys.stderr)
sys.exit(1)
PYSOCK

for repeat_id in $(seq 0 11); do
  for episode_id in $(seq 0 23); do
    set +e
    python "${BIN_DIR}/row_status.py" --csv "${RESULT_CSV}" --official-episode-id "${episode_id}" --repeat-id "${repeat_id}"
    row_status=$?
    set -e
    if [[ "${row_status}" == "0" ]]; then
      if [[ "${RESUME}" != "1" ]]; then
        echo "ERROR: existing row in non-resume run repeat=${repeat_id} episode=${episode_id}" >&2
        exit 3
      fi
      echo "resume_skip repeat=${repeat_id} episode=${episode_id}"
      continue
    elif [[ "${row_status}" != "1" ]]; then
      echo "ERROR: duplicate or invalid existing row repeat=${repeat_id} episode=${episode_id}" >&2
      exit 3
    fi

    seed="$(python "${BIN_DIR}/protocol_cli.py" seed "${TASK}" "${repeat_id}" "${episode_id}")"
    episode_dir="${RESULT_ROOT}/episodes/repeat_${repeat_id}/episode_${episode_id}"
    if [[ -e "${episode_dir}" ]]; then
      echo "ERROR: stale episode output without a result row: ${episode_dir}" >&2
      exit 3
    fi
    mkdir -p "$(dirname "${episode_dir}")"
    stdout_log="${LOG_DIR}/repeat_${repeat_id}_episode_${episode_id}.stdout"
    stderr_log="${LOG_DIR}/repeat_${repeat_id}_episode_${episode_id}.stderr"
    set +e
    PYTHONHASHSEED="${seed}" EVAL_ROLLOUT_SEED="${seed}" \
    timeout "${EPISODE_TIMEOUT_SECONDS}" \
      python "${BIN_DIR}/seeded_xvla_rollout.py" \
        --task "${TASK}" \
        --official-episode-id "${episode_id}" \
        --seed "${seed}" \
        --server-ip 127.0.0.1 \
        --server-port "${PORT}" \
        --output-dir "${episode_dir}" \
      > "${stdout_log}" 2> "${stderr_log}"
    status=$?
    set -e
    timeout_flag=0
    if [[ "${status}" == "124" ]]; then
      timeout_flag=1
    fi
    append_args=(
      --csv "${RESULT_CSV}"
      --policy "${POLICY}"
      --task "${TASK}"
      --official-episode-id "${episode_id}"
      --repeat-id "${repeat_id}"
      --seed "${seed}"
      --episode-dir "${episode_dir}"
      --exit-code "${status}"
      --timeout "${timeout_flag}"
      --job-id "${JOB_ID}"
      --stderr-log "${stderr_log}"
    )
    if [[ "${CONTINUE_ON_ERROR}" == "0" ]]; then
      append_args+=(--fail-on-error)
    fi
    python "${BIN_DIR}/append_result_row.py" "${append_args[@]}"
  done
done

python "${BIN_DIR}/validate_task_results.py" \
  --task "${TASK}" \
  --results "${RESULT_CSV}" \
  --manifest "${MANIFEST}" \
  --report "${RESULT_ROOT}/validation_report.json"
