#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TAG="${RUN_TAG:-protocol1_spatialvla_$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_spatialvla_py310}"
CKPT_PATH="${CKPT_PATH:-IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge}"
TASK_FILTER="${TASK_FILTER:-all}"
EPISODE_START="${EPISODE_START:-0}"
EPISODE_END="${EPISODE_END:-500}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/simplerenv/spatialvla_bridge/protocol1_random_positions/${RUN_TAG}}"
PROTOCOL_CONFIG="${PROTOCOL_CONFIG:-${PROJECT_ROOT}/configs/simplerenv/randomized_positions/widowx_protocol1_seed20260515_n500.json}"
PROTOCOL_SHA256_FILE="${PROTOCOL_SHA256_FILE:-${PROTOCOL_CONFIG}.sha256}"
RUN_PROTOCOL_NAME="${RUN_PROTOCOL_NAME:-widowx_protocol1_random_positions}"
ADDITIONAL_ENV_SAVE_TAGS="${ADDITIONAL_ENV_SAVE_TAGS:-protocol1_seed20260515}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/cache/conda_pkgs}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/artifacts/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

SPATIAL_REPO="${PROJECT_ROOT}/third_party/simplerenv_openvla"
SPATIAL_MAIN="${SPATIAL_REPO}/simpler_env/main_inference.py"
PROTOCOL_MAIN="${PROJECT_ROOT}/scripts/simplerenv/main_inference_with_protocol.py"
OVERLAY_DIR="${SPATIAL_REPO}/ManiSkill2_real2sim/data/real_inpainting"
METADATA_DIR="${PROJECT_ROOT}/artifacts/simplerenv/${RUN_TAG}"

case "${TASK_FILTER}" in
  all|stack|carrot|spoon) ;;
  *) echo "Protocol 1 supports TASK_FILTER=all|stack|carrot|spoon, got ${TASK_FILTER}" >&2; exit 2 ;;
esac

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
export SIMPLERENV_TARGET_MAIN="${SPATIAL_MAIN}"

mkdir -p "${RESULTS_DIR}" "${METADATA_DIR}" "${PROJECT_ROOT}/cache" "${PROJECT_ROOT}/logs/slurm"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing env at ${ENV_PREFIX}. Run scripts/simplerenv/setup_spatialvla_env.sh first." >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

export PYTHONPATH="${PROJECT_ROOT}/scripts/simplerenv:${SPATIAL_REPO}:${PYTHONPATH:-}"

{
  echo "run_tag=${RUN_TAG}"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "project_root=${PROJECT_ROOT}"
  echo "policy=spatialvla"
  echo "protocol=${RUN_PROTOCOL_NAME}"
  echo "protocol_config=${SIMPLERENV_PROTOCOL_CONFIG}"
  echo "protocol_sha256=${SIMPLERENV_PROTOCOL_SHA256}"
  echo "protocol3_stack_yellow_on_green=${SIMPLERENV_PROTOCOL3_STACK_YELLOW_ON_GREEN:-0}"
  echo "ckpt_path=${CKPT_PATH}"
  echo "task_filter=${TASK_FILTER}"
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

run_task() {
  local label="$1"
  local env_name="$2"

  if [[ "${TASK_FILTER}" != "all" && "${TASK_FILTER}" != "${label}" ]]; then
    return 0
  fi

  echo "=== SpatialVLA Protocol 1 ${label}: ${env_name} episodes ${EPISODE_START}:${EPISODE_END} ==="
  python "${PROTOCOL_MAIN}" \
    --policy-model spatialvla --ckpt-path "${CKPT_PATH}" \
    --robot widowx --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
    --env-name "${env_name}" --scene-name bridge_table_1_v1 \
    --rgb-overlay-path "${OVERLAY_DIR}/bridge_real_eval_1.png" \
    --robot-init-x-range 0.147 0.147 1 \
    --robot-init-y-range 0.028 0.028 1 \
    --obj-variation-mode episode --obj-episode-range "${EPISODE_START}" "${EPISODE_END}" \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --additional-env-save-tags "${ADDITIONAL_ENV_SAVE_TAGS}" \
    --logging-dir "${RESULTS_DIR}/videos"
}

run_task stack StackGreenCubeOnYellowCubeBakedTexInScene-v0
run_task carrot PutCarrotOnPlateInScene-v0
run_task spoon PutSpoonOnTableClothInScene-v0

python - <<PY
import json
from pathlib import Path

root = Path("${RESULTS_DIR}/videos")
mp4s = list(root.rglob("*.mp4"))
success = sum(1 for p in mp4s if p.name.lower().startswith("success_"))
task_envs = {
    "stack": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
    "carrot": "PutCarrotOnPlateInScene-v0",
    "spoon": "PutSpoonOnTableClothInScene-v0",
}
by_task = {}
for label, env_name in task_envs.items():
    task_mp4s = [p for p in mp4s if env_name in str(p)]
    if task_mp4s:
        task_success = sum(1 for p in task_mp4s if p.name.lower().startswith("success_"))
        by_task[label] = {
            "success": task_success,
            "total": len(task_mp4s),
            "success_rate": task_success / len(task_mp4s),
        }
task_count = 3 if "${TASK_FILTER}" == "all" else 1
expected = task_count * (int("${EPISODE_END}") - int("${EPISODE_START}"))
summary = {
    "policy": "spatialvla",
    "protocol": "${RUN_PROTOCOL_NAME}",
    "task_filter": "${TASK_FILTER}",
    "episode_start": int("${EPISODE_START}"),
    "episode_end": int("${EPISODE_END}"),
    "success": success,
    "total": len(mp4s),
    "success_rate": success / len(mp4s) if mp4s else None,
    "by_task": by_task,
    "result_root": str(root),
}
Path("${RESULTS_DIR}/summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
print(summary)
if len(mp4s) != expected:
    raise SystemExit(f"expected {expected} videos, found {len(mp4s)}")
PY
