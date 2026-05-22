#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -J libero_off_cal
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
POLICY="${POLICY:?set POLICY to spatial_forcing, simvla, or pi05_lerobot}"
TASK_SUITE="${TASK_SUITE:?set TASK_SUITE to one LIBERO suite}"
ARTIFACT_NAME="${ARTIFACT_NAME:-init_state_official_4suite_50}"
INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/${ARTIFACT_NAME}}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJECT_ROOT}/artifacts/libero/libero_config_official_4suite_50}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
RESULT_FAMILY="${RESULT_FAMILY:-official_calibration}"
RUN_GROUP="${RUN_GROUP:-official_calibration_${SLURM_JOB_ID:-manual}}"

case "${TASK_SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *) echo "unknown TASK_SUITE=${TASK_SUITE}" >&2; exit 1 ;;
esac
case "${POLICY}" in
  spatial_forcing) POLICY_DIR="spatial_forcing"; POLICY_NAME="spatial_forcing"; POLICY_TRACKER_ID="438" ;;
  simvla) POLICY_DIR="simvla"; POLICY_NAME="simvla"; POLICY_TRACKER_ID="798" ;;
  pi05_lerobot) POLICY_DIR="pi05_lerobot"; POLICY_NAME="pi05_lerobot"; POLICY_TRACKER_ID="869" ;;
  *) echo "unknown POLICY=${POLICY}" >&2; exit 1 ;;
esac
if ! [[ "${EPISODES_PER_TASK}" =~ ^[0-9]+$ ]] || (( EPISODES_PER_TASK < 1 || EPISODES_PER_TASK > 50 )); then
  echo "EPISODES_PER_TASK must be in 1..50 for official init files, got ${EPISODES_PER_TASK}" >&2
  exit 1
fi

SUITE_RESULT_ROOT="${PROJECT_ROOT}/results/libero/${RESULT_FAMILY}/${RUN_GROUP}/${POLICY_DIR}/${TASK_SUITE}"

export PROJECT_ROOT POLICY TASK_SUITE INIT_ROOT LIBERO_CONFIG_PATH EPISODES_PER_TASK SUITE_RESULT_ROOT
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

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

for path in "${INIT_ROOT}" "${INIT_ROOT}/MANIFEST.json" "${INIT_ROOT}/${TASK_SUITE}" "${LIBERO_CONFIG_PATH}/config.yaml"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required calibration path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${SUITE_RESULT_ROOT}/shared" logs/slurm

if ! command -v conda >/dev/null 2>&1; then
  export PATH="/share/data/ripl/tianchong/conda/bin:${PATH}"
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF

cp "${PROJECT_ROOT}/scripts/slurm/libero_official_calibration_policy_suite.sbatch" "${SUITE_RESULT_ROOT}/shared/sbatch_template.sbatch" 2>/dev/null || true
cp "${LIBERO_CONFIG_PATH}/config.yaml" "${SUITE_RESULT_ROOT}/shared/libero_config.yaml"
cp "${INIT_ROOT}/MANIFEST.json" "${SUITE_RESULT_ROOT}/shared/init_state_manifest.json"
python -m pip freeze > "${SUITE_RESULT_ROOT}/shared/pip_freeze.txt" 2>&1 || true
conda list > "${SUITE_RESULT_ROOT}/shared/conda_list.txt" 2>&1 || true

set_task_paths() {
  TASK_ID="$1"
  if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 9 )); then
    echo "TASK_ID must be in 0..9, got ${TASK_ID}" >&2
    exit 1
  fi
  RESULT_DIR="${SUITE_RESULT_ROOT}/task_${TASK_ID}"
  EPISODE_CSV="${RESULT_DIR}/episodes.csv"
  export TASK_ID RESULT_DIR EPISODE_CSV
  if [[ -e "${EPISODE_CSV}" ]]; then
    echo "refusing to overwrite existing episode CSV: ${EPISODE_CSV}" >&2
    exit 1
  fi
  mkdir -p "${RESULT_DIR}/logs" "${RESULT_DIR}/metadata"
}

write_metadata() {
  local policy_repo="${1:-}"
  {
    echo "run_group=${RUN_GROUP}"
    echo "result_family=${RESULT_FAMILY}"
    echo "policy=${POLICY_NAME}"
    echo "policy_tracker_id=${POLICY_TRACKER_ID}"
    echo "suite=${TASK_SUITE}"
    echo "task_id=${TASK_ID}"
    echo "init_root=${INIT_ROOT}"
    echo "libero_config_path=${LIBERO_CONFIG_PATH}"
    echo "episodes_per_task=${EPISODES_PER_TASK}"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "hostname=$(hostname)"
    echo "slurmd_nodename=${SLURMD_NODENAME:-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "date_utc=$(date -u --iso-8601=seconds)"
    git -C "${PROJECT_ROOT}" rev-parse HEAD || true
    git -C "${PROJECT_ROOT}" status --short || true
    if [[ -n "${policy_repo}" ]]; then
      git -C "${policy_repo}" rev-parse HEAD || true
      git -C "${policy_repo}" status --short || true
    fi
    python --version
    echo "nproc=$(nproc --all 2>/dev/null || true)"
    if command -v lscpu >/dev/null 2>&1; then lscpu; fi
    if command -v free >/dev/null 2>&1; then free -h; fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L
      nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
    fi
  } > "${RESULT_DIR}/metadata/metadata.txt" 2>&1
}

validate_csv() {
  local summary_path="$1"
  python "${PROJECT_ROOT}/scripts/libero/validate_layer2_episode_csv.py" \
    --episode-csv "${EPISODE_CSV}" \
    --expected-rows "${EPISODES_PER_TASK}" \
    --policy-name "${POLICY_NAME}" \
    --suite "${TASK_SUITE}" \
    --summary-path "${summary_path}" \
    --require-no-errors \
    --require-no-videos
}

run_spatial_forcing_task() {
  local sf_repo="${SF_REPO:-${PROJECT_ROOT}/third_party/spatial_forcing_libero_adapter/openvla-SF}"
  local ckpt_dir="${CKPT_DIR:-}"
  if [[ -z "${ckpt_dir}" ]]; then
    case "${TASK_SUITE}" in
      libero_spatial) ckpt_dir="${PROJECT_ROOT}/checkpoints/spatial-forcing-7b-finetuned-libero-spatial" ;;
      libero_object) ckpt_dir="${PROJECT_ROOT}/checkpoints/spatial-forcing-7b-finetuned-libero-object" ;;
      libero_goal) ckpt_dir="${PROJECT_ROOT}/checkpoints/spatial-forcing-7b-finetuned-libero-goal" ;;
      libero_10) ckpt_dir="${PROJECT_ROOT}/checkpoints/spatial-forcing-7b-finetuned-libero-10" ;;
    esac
  fi
  [[ -d "${sf_repo}" ]] || { echo "missing SF_REPO=${sf_repo}" >&2; exit 1; }
  [[ -d "${ckpt_dir}" ]] || { echo "missing CKPT_DIR=${ckpt_dir}" >&2; exit 1; }
  export PYTHONPATH="${sf_repo}:${PYTHONPATH:-}"
  write_metadata "${sf_repo}"
  cd "${sf_repo}"
  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${ckpt_dir}" \
    --task_suite_name "${TASK_SUITE}" \
    --initial_states_path "${INIT_ROOT}" \
    --num_trials_per_task "${EPISODES_PER_TASK}" \
    --selected_task_id "${TASK_ID}" \
    --episode_start_idx 0 \
    --episode_end_idx "${EPISODES_PER_TASK}" \
    --policy_name spatial_forcing \
    --episode_results_path "${EPISODE_CSV}" \
    --local_log_dir "${RESULT_DIR}/logs" \
    --run_id_note "${RUN_GROUP}--${TASK_SUITE}--task_${TASK_ID}" \
    --save_videos False \
    --use_wandb False \
    --center_crop True
  validate_csv "${RESULT_DIR}/summary.json"
}

run_simvla_task() {
  local simvla_repo="${SIMVLA_REPO:-${PROJECT_ROOT}/third_party/simvla}"
  local ckpt_dir="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/simvla-libero}"
  local ckpt_revision="${CKPT_REVISION:-93dc4d90b0596c652ad2840ad743c62b9c4473fb}"
  local overlay_dir="${OVERLAY_DIR:-${PROJECT_ROOT}/envs/simvla_py310_overlay_tf4}"
  [[ -d "${simvla_repo}" ]] || { echo "missing SIMVLA_REPO=${simvla_repo}" >&2; exit 1; }
  [[ -d "${ckpt_dir}" ]] || { echo "missing CKPT_DIR=${ckpt_dir}" >&2; exit 1; }
  [[ -d "${overlay_dir}" ]] || { echo "missing OVERLAY_DIR=${overlay_dir}" >&2; exit 1; }
  export PYTHONPATH="${overlay_dir}:${simvla_repo}:${PYTHONPATH:-}"
  write_metadata "${simvla_repo}"
  cd "${simvla_repo}"
  python - <<'PY'
from packaging.version import Version
import transformers
import torch

if Version(transformers.__version__) < Version("4.57.0"):
    raise SystemExit(f"transformers too old: {transformers.__version__}")
print("dependency validation ok")
print("transformers", transformers.__version__)
print("torch", torch.__version__)
PY
  local port_block=$(( (${SLURM_JOB_ID:-0} % 4000) * 10 ))
  local port=$((20000 + port_block + TASK_ID))
  python -u evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint "${ckpt_dir}" \
    --norm_stats "${simvla_repo}/norm_stats/libero_norm.json" \
    --host 127.0.0.1 \
    --port "${port}" \
    > "${RESULT_DIR}/logs/server.log" 2>&1 &
  local server_pid=$!
  trap 'kill ${server_pid} 2>/dev/null || true' EXIT
  PORT="${port}" SERVER_PID="${server_pid}" python - <<'PY'
import os
import socket
import time

host = "127.0.0.1"
port = int(os.environ["PORT"])
server_pid = int(os.environ["SERVER_PID"])
deadline = time.time() + 900
while time.time() < deadline:
    try:
        os.kill(server_pid, 0)
    except OSError:
        raise SystemExit("SimVLA server exited before opening the port")
    try:
        with socket.create_connection((host, port), timeout=2):
            try:
                os.kill(server_pid, 0)
            except OSError:
                raise SystemExit("SimVLA server exited before opening the port")
            print(f"server is listening on {host}:{port}")
            break
    except OSError:
        time.sleep(5)
else:
    raise SystemExit("timed out waiting for SimVLA server")
PY
  python -u evaluation/libero/libero_client.py \
    --host 127.0.0.1 \
    --port "${port}" \
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
    --checkpoint_ref "${ckpt_dir}@${ckpt_revision}" \
    --video_out "${RESULT_DIR}/videos" \
    --no_video \
    > "${RESULT_DIR}/logs/client.log" 2>&1
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT
  validate_csv "${RESULT_DIR}/summary.json"
}

run_pi05_task() {
  local lerobot_repo="${LEROBOT_REPO:-${PROJECT_ROOT}/third_party/lerobot}"
  local lerobot_src="${LEROBOT_SRC:-${lerobot_repo}/src}"
  local ckpt_repo="${CKPT_REPO:-lerobot/pi05_libero_finetuned}"
  local ckpt_dir="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/pi05-libero-finetuned}"
  local tokenizer_repo="${TOKENIZER_REPO:-Shakalaka/paligemma-3b-pt-224}"
  local simvla_overlay="${SIMVLA_OVERLAY:-${PROJECT_ROOT}/envs/simvla_py310_overlay_tf4}"
  local pi05_overlay="${PI05_OVERLAY:-${PROJECT_ROOT}/envs/pi05_py310_overlay_v5}"
  local n_action_steps="${N_ACTION_STEPS:-10}"
  [[ -d "${lerobot_src}" ]] || { echo "missing LEROBOT_SRC=${lerobot_src}" >&2; exit 1; }
  [[ -d "${ckpt_dir}" ]] || { echo "missing CKPT_DIR=${ckpt_dir}" >&2; exit 1; }
  [[ -d "${pi05_overlay}" ]] || { echo "missing PI05_OVERLAY=${pi05_overlay}" >&2; exit 1; }
  [[ -d "${simvla_overlay}" ]] || { echo "missing SIMVLA_OVERLAY=${simvla_overlay}" >&2; exit 1; }
  export PYTHONPATH="${pi05_overlay}:${simvla_overlay}:${lerobot_src}:${PYTHONPATH:-}"
  write_metadata "${lerobot_repo}"
  cd "${PROJECT_ROOT}"
  python - <<'PY'
import torch
import transformers
from transformers.models.siglip import check

if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit("OpenPI patched transformers check failed")
print("dependency validation ok")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY
  python -u "${PROJECT_ROOT}/scripts/libero/run_pi05_goal_repro.py" \
    --model-id "${ckpt_repo}" \
    --checkpoint-dir "${ckpt_dir}" \
    --initial-states-path "${INIT_ROOT}" \
    --task-suite "${TASK_SUITE}" \
    --selected-task-id "${TASK_ID}" \
    --episode-start-idx 0 \
    --episode-end-idx "${EPISODES_PER_TASK}" \
    --max-total-episodes "${EPISODES_PER_TASK}" \
    --episode-results-path "${EPISODE_CSV}" \
    --summary-path "${RESULT_DIR}/summary.json" \
    --manifest-path "${RESULT_DIR}/manifest.json" \
    --policy-name pi05_lerobot \
    --policy-tracker-id 869 \
    --tokenizer-name "${tokenizer_repo}" \
    --device cuda \
    --n-action-steps "${n_action_steps}" \
    > "${RESULT_DIR}/logs/eval.log" 2>&1
  validate_csv "${RESULT_DIR}/validation_summary.json"
}

aggregate_suite() {
  python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["SUITE_RESULT_ROOT"])
rows = []
for task_id in range(10):
    path = root / f"task_{task_id}" / "episodes.csv"
    if not path.exists():
        raise SystemExit(f"missing task CSV: {path}")
    task_rows = list(csv.DictReader(path.open()))
    if len(task_rows) != int(os.environ["EPISODES_PER_TASK"]):
        raise SystemExit(f"{path} has {len(task_rows)} rows")
    rows.extend(task_rows)
errors = [row for row in rows if row.get("error_type")]
if errors:
    raise SystemExit(f"found {len(errors)} rows with error_type")
summary = {
    "policy_name": os.environ["POLICY"],
    "suite": os.environ["TASK_SUITE"],
    "run_group": os.environ["RUN_GROUP"],
    "tasks": 10,
    "rows": len(rows),
    "successes": sum(int(row["success"]) for row in rows),
    "errors": len(errors),
    "success_rate": sum(int(row["success"]) for row in rows) / len(rows),
    "validation_status": "passed",
}
(root / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
}

echo "node=$(hostname)"
echo "POLICY=${POLICY}"
echo "TASK_SUITE=${TASK_SUITE}"
echo "EPISODES_PER_TASK=${EPISODES_PER_TASK}"
echo "RUN_GROUP=${RUN_GROUP}"
echo "SUITE_RESULT_ROOT=${SUITE_RESULT_ROOT}"
echo "INIT_ROOT=${INIT_ROOT}"
echo "date_utc=$(date -u --iso-8601=seconds)"

for task_id in 0 1 2 3 4 5 6 7 8 9; do
  echo "=== task_${task_id} start $(date -u --iso-8601=seconds) ==="
  set_task_paths "${task_id}"
  case "${POLICY}" in
    spatial_forcing) run_spatial_forcing_task ;;
    simvla) run_simvla_task ;;
    pi05_lerobot) run_pi05_task ;;
  esac
  echo "=== task_${task_id} done $(date -u --iso-8601=seconds) ==="
done

aggregate_suite
