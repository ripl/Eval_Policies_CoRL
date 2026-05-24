#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

TASK="${1:?task required}"
POLICY="dexbotic"
PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${PROJECT_ROOT}/scratch/repeated_fixed_grid_calibration_288_20260523_v2/dexbotic_support}"
BIN_DIR="${SUPPORT_ROOT}/bin"
RESULT_BASE="${RESULT_BASE:-${SUPPORT_ROOT}/runs}"
RUN_TAG="${RUN_TAG:-dexbotic_${TASK}_$(date -u +%Y%m%dT%H%M%SZ)_job${SLURM_JOB_ID:-manual}_$$}"
RESULT_ROOT="${RESULT_BASE}/${RUN_TAG}"
RUN_DIR="${RESULT_ROOT}/run"
LOG_DIR="${RESULT_ROOT}/logs"
MANIFEST="${RESULT_ROOT}/manifest.csv"
RESULT_CSV="${RESULT_ROOT}/per_episode_results.csv"
CACHE_ROOT="${PROJECT_ROOT}/cache/dexbotic_official_images"
DEXBOTIC_SRC="${PROJECT_ROOT}/third_party/dexbotic"
BENCHMARK_SRC="${PROJECT_ROOT}/scratch/strict_calibrations_20260523/sources/dexbotic-benchmark_54a6c749_simpler4ab717_clean_20260523T0152Z"
DEXBOTIC_MODEL_ID="${DEXBOTIC_MODEL_ID:-Dexmal/simpler-db-memvla}"
DEXBOTIC_SERVER_IMAGE="${DEXBOTIC_SERVER_IMAGE:-docker://dexmal/dexbotic@sha256:7851cf8ed236dc18b5df3df6e8ea8ff5a543d154c03ac637a6dc6bd4e9eda654}"
DEXBOTIC_BENCHMARK_IMAGE="${DEXBOTIC_BENCHMARK_IMAGE:-docker://dexmal/dexbotic_benchmark@sha256:2e6938be25991c43e5261f91a14abcfaad68d6f1a664643ef69a2e3628b60fef}"
DEXBOTIC_PORT="${DEXBOTIC_PORT:-$((7891 + (${SLURM_JOB_ID:-0} % 800)))}"
EPISODE_TIMEOUT_SECONDS="${EPISODE_TIMEOUT_SECONDS:-2400}"
NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-${PROJECT_ROOT}/results/simplerenv/strict_calibrations_20260523/dexbotic_official_workspace_strict_20260523T015147Z/run/norm_stats.json}"
EXPECTED_NORM_STATS_SHA256="ddbeb68786543c68f8fa198c33fd9a265025d35c5f7ebba144d6ec9c655693e6"
HORIZON="$(python3 "${BIN_DIR}/seed_utils.py" horizon "${TASK}")"
SERVER_SEED="$(python3 "${BIN_DIR}/seed_utils.py" server-seed "${POLICY}" "${TASK}")"
JOB_ID="${SLURM_JOB_ID:-manual}"
NORM_STATS_HOST="${RUN_DIR}/norm_stats.json"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

if [[ -e "${RESULT_ROOT}" ]]; then
  echo "ERROR: result root already exists: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${RUN_DIR}/container_home" "${RESULT_ROOT}/episodes" "${RESULT_ROOT}/metadata" \
  "${CACHE_ROOT}"/{hf,torch,xdg,pip,apptainer,wandb,matplotlib,triton}
exec > >(tee -a "${LOG_DIR}/driver.log") 2>&1

python3 "${BIN_DIR}/preflight.py" --policy "${POLICY}" --task "${TASK}"
python3 "${BIN_DIR}/write_manifest.py" --policy "${POLICY}" --task "${TASK}" --output "${MANIFEST}"
python3 "${BIN_DIR}/init_results_csv.py" --output "${RESULT_CSV}"

if [[ ! -s "${NORM_STATS_SOURCE}" ]]; then
  echo "ERROR: strict norm_stats file missing or empty: ${NORM_STATS_SOURCE}" >&2
  exit 4
fi
actual_norm_sha="$(sha256sum "${NORM_STATS_SOURCE}" | awk '{print $1}')"
if [[ "${actual_norm_sha}" != "${EXPECTED_NORM_STATS_SHA256}" ]]; then
  echo "ERROR: strict norm_stats sha256 mismatch: ${actual_norm_sha} != ${EXPECTED_NORM_STATS_SHA256}" >&2
  exit 4
fi
cp "${NORM_STATS_SOURCE}" "${NORM_STATS_HOST}"
sha256sum "${NORM_STATS_HOST}" > "${RESULT_ROOT}/metadata/norm_stats.sha256"

cat > "${RESULT_ROOT}/seed_control_caveat.txt" <<EOF
Rollout seed formula: 20260523 + policy_index*1000000 + task_index*10000 + repeat_id*100 + official_episode_id.
For each Dexbotic client/evaluation subprocess, PYTHONHASHSEED, Python random, NumPy, and torch seeds are set to the manifest seed before evaluation/run_simpler_evaluation.py is executed.
The Dexbotic model server is started once for the task job with startup seed ${SERVER_SEED}.
The server API exposes episode_first_frame for memory reset but does not expose a per-rollout RNG seed reset. Server-side diffusion sampling therefore follows the single server process RNG stream rather than being independently reset to each manifest seed.
Treat the CSV seed column as a host/client/repeat identifier and client/simulator seed, not as full model RNG control.
Validation requires 288 '** reset memory **' server-log entries, one per one-episode evaluation call.
EOF

{
  echo "policy=Dexbotic / DB-MemVLA"
  echo "task=${TASK}"
  echo "horizon=${HORIZON}"
  echo "run_tag=${RUN_TAG}"
  echo "result_root=${RESULT_ROOT}"
  echo "job_id=${JOB_ID}"
  echo "node=$(hostname)"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "server_image=${DEXBOTIC_SERVER_IMAGE}"
  echo "benchmark_image=${DEXBOTIC_BENCHMARK_IMAGE}"
  echo "model_id=${DEXBOTIC_MODEL_ID}"
  echo "norm_stats_source=${NORM_STATS_SOURCE}"
  echo "norm_stats_sha256=${actual_norm_sha}"
  echo "server_restart_strategy=single_server_per_task"
  echo "seed_column_semantics=host/client/repeat identifier and client/simulator seed; not full model-server RNG control"
  echo "dexbotic_src=${DEXBOTIC_SRC}"
  echo "benchmark_src=${BENCHMARK_SRC}"
  echo "dexbotic_commit=$(git -C "${DEXBOTIC_SRC}" rev-parse HEAD 2>/dev/null || true)"
  echo "benchmark_commit=$(git -C "${BENCHMARK_SRC}" rev-parse HEAD 2>/dev/null || true)"
  echo "simpler_commit=$(git -C "${BENCHMARK_SRC}/simpler" rev-parse HEAD 2>/dev/null || true)"
  echo "maniskill2_real2sim_commit=$(git -C "${BENCHMARK_SRC}/simpler/ManiSkill2_real2sim" rev-parse HEAD 2>/dev/null || true)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
} > "${RESULT_ROOT}/metadata/runtime_metadata.txt" 2>&1

{
  echo "dexbotic_status_start"
  git -C "${DEXBOTIC_SRC}" status --short
  echo "dexbotic_status_end"
  echo "benchmark_status_start"
  git -C "${BENCHMARK_SRC}" status --short
  echo "benchmark_status_end"
  echo "simpler_status_start"
  git -C "${BENCHMARK_SRC}/simpler" status --short
  echo "simpler_status_end"
  echo "maniskill2_real2sim_status_start"
  git -C "${BENCHMARK_SRC}/simpler/ManiSkill2_real2sim" status --short
  echo "maniskill2_real2sim_status_end"
} > "${RESULT_ROOT}/metadata/code_status.txt"

find "${SUPPORT_ROOT}" -type f \
  \( -path "*/runs/*" -o -path "*/__pycache__/*" \) -prune -o \
  -type f -print | sort | xargs sha256sum > "${RESULT_ROOT}/metadata/support_files.sha256"

export HF_HOME="${CACHE_ROOT}/hf"
export HF_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"
export SINGULARITY_CACHEDIR="${CACHE_ROOT}/apptainer"
export WANDB_DIR="${CACHE_ROOT}/wandb"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0
export TRANSFORMERS_NO_FLAX=1

apptainer inspect "${DEXBOTIC_SERVER_IMAGE}" > "${RESULT_ROOT}/metadata/server_image_inspect.txt" 2>&1 || true
apptainer inspect "${DEXBOTIC_BENCHMARK_IMAGE}" > "${RESULT_ROOT}/metadata/benchmark_image_inspect.txt" 2>&1 || true

apptainer exec --nv --cleanenv \
  --bind "${DEXBOTIC_SRC}:/mnt/dexbotic" \
  --bind "${SUPPORT_ROOT}:${SUPPORT_ROOT}" \
  --bind "${RUN_DIR}:/dex_run" \
  --bind "${CACHE_ROOT}:${CACHE_ROOT}" \
  --env "HF_HOME=${HF_HOME}" \
  --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
  --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
  --env "TORCH_HOME=${TORCH_HOME}" \
  --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
  --env "WANDB_DIR=${WANDB_DIR}" \
  --env "MPLCONFIGDIR=${MPLCONFIGDIR}" \
  --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "PYTHONHASHSEED=${SERVER_SEED}" \
  --env "EVAL_ROLLOUT_SEED=${SERVER_SEED}" \
  --env "DEXBOTIC_SERVER_SEED=${SERVER_SEED}" \
  --env "DEXBOTIC_MODEL_ID=${DEXBOTIC_MODEL_ID}" \
  --env "DEXBOTIC_PORT=${DEXBOTIC_PORT}" \
  --env "DEXBOTIC_NORM_STATS=/dex_run/norm_stats.json" \
  "${DEXBOTIC_SERVER_IMAGE}" \
  bash -lc 'set -Eeuo pipefail; source /opt/conda/etc/profile.d/conda.sh; conda activate dexbotic; export PYTHONPATH=/mnt/dexbotic:'"${BIN_DIR}"':${PYTHONPATH:-}; cd /mnt/dexbotic; python -u '"${BIN_DIR}"'/launch_memvla_server_seeded.py' \
  > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 360); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Dexbotic server exited before becoming ready" >&2
    tail -240 "${LOG_DIR}/server.log" || true
    exit 1
  fi
  if python3 - "${DEXBOTIC_PORT}" <<'PYSOCK'
import socket
import sys
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PYSOCK
  then
    READY=1
    break
  fi
  sleep 5
done
if [[ "${READY}" != "1" ]]; then
  echo "Dexbotic server did not open port ${DEXBOTIC_PORT} in time" >&2
  tail -240 "${LOG_DIR}/server.log" || true
  exit 1
fi

for repeat_id in $(seq 0 11); do
  for episode_id in $(seq 0 23); do
    seed="$(python3 "${BIN_DIR}/seed_utils.py" seed "${POLICY}" "${TASK}" "${repeat_id}" "${episode_id}")"
    episode_dir="${RESULT_ROOT}/episodes/repeat_${repeat_id}/episode_${episode_id}"
    if [[ -e "${episode_dir}" ]]; then
      echo "ERROR: refusing to overwrite episode dir ${episode_dir}" >&2
      exit 3
    fi
    mkdir -p "${episode_dir}/config" "${episode_dir}/eval"
    config_path="${episode_dir}/config/dexbotic_${TASK}_repeat_${repeat_id}_episode_${episode_id}.yaml"
    python3 "${BIN_DIR}/make_dexbotic_config.py" \
      --task "${TASK}" \
      --official-episode-id "${episode_id}" \
      --repeat-id "${repeat_id}" \
      --output-dir "${episode_dir}/eval" \
      --base-url "http://127.0.0.1:${DEXBOTIC_PORT}" \
      --model-id "${DEXBOTIC_MODEL_ID}" \
      --config-out "${config_path}"
    set +e
    timeout "${EPISODE_TIMEOUT_SECONDS}" \
      apptainer exec --nv --cleanenv --pwd /workspace \
        --home "${RUN_DIR}/container_home:/home/tianchong" \
        --bind "${BENCHMARK_SRC}:/workspace" \
        --bind "${SUPPORT_ROOT}:${SUPPORT_ROOT}" \
        --bind "${RESULT_ROOT}:${RESULT_ROOT}" \
        --bind "${CACHE_ROOT}:${CACHE_ROOT}" \
        --env "HF_HOME=${HF_HOME}" \
        --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
        --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
        --env "TORCH_HOME=${TORCH_HOME}" \
        --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
        --env "WANDB_DIR=${WANDB_DIR}" \
        --env "MPLCONFIGDIR=${MPLCONFIGDIR}" \
        --env "MUJOCO_GL=${MUJOCO_GL}" \
        --env "TOKENIZERS_PARALLELISM=false" \
        --env "VK_ICD_FILENAMES=${VK_ICD_FILENAMES}" \
        --env "PYTHONHASHSEED=${seed}" \
        --env "EVAL_ROLLOUT_SEED=${seed}" \
        --env "SCRATCH_BIN=${BIN_DIR}" \
        --env "CONFIG_PATH=${config_path}" \
        "${DEXBOTIC_BENCHMARK_IMAGE}" \
        bash -lc 'set -Eeuo pipefail; source /opt/conda/etc/profile.d/conda.sh; conda activate simpler_env; export PYTHONPATH=/workspace:'"${BIN_DIR}"':${PYTHONPATH:-}; cd /workspace; python "${SCRATCH_BIN}/seeded_runpy.py" --seed "${EVAL_ROLLOUT_SEED}" --script /workspace/evaluation/run_simpler_evaluation.py -- --config "${CONFIG_PATH}"' \
      > "${LOG_DIR}/repeat_${repeat_id}_episode_${episode_id}.stdout" \
      2> "${LOG_DIR}/repeat_${repeat_id}_episode_${episode_id}.stderr"
    status=$?
    set -e
    timeout_flag=0
    if [[ "${status}" == "124" ]]; then timeout_flag=1; fi
    python3 "${BIN_DIR}/append_result_row.py" \
      --csv "${RESULT_CSV}" \
      --source dexbotic \
      --policy "${POLICY}" \
      --task "${TASK}" \
      --official-episode-id "${episode_id}" \
      --repeat-id "${repeat_id}" \
      --seed "${seed}" \
      --episode-dir "${episode_dir}" \
      --exit-code "${status}" \
      --timeout "${timeout_flag}" \
      --job-id "${JOB_ID}"
  done
done

python3 "${BIN_DIR}/validate_results.py" \
  --policy "${POLICY}" \
  --task "${TASK}" \
  --results "${RESULT_CSV}" \
  --manifest "${MANIFEST}" \
  --report "${RESULT_ROOT}/validation_report.json" \
  --server-log "${LOG_DIR}/server.log" \
  --expected-reset-count 288
