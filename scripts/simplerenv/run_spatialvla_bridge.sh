#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TAG="${RUN_TAG:-spatialvla_bridge_eggplant_$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_spatialvla_py310}"
CKPT_PATH="${CKPT_PATH:-IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge}"
TASK_FILTER="${TASK_FILTER:-all}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/simplerenv/spatialvla_bridge/official_fixed_grid/${RUN_TAG}}"

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
MAIN="${SPATIAL_REPO}/simpler_env/main_inference.py"
OVERLAY_DIR="${SPATIAL_REPO}/ManiSkill2_real2sim/data/real_inpainting"
METADATA_DIR="${PROJECT_ROOT}/artifacts/simplerenv/${RUN_TAG}"

mkdir -p "${RESULTS_DIR}" "${METADATA_DIR}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing env at ${ENV_PREFIX}. Run scripts/simplerenv/setup_spatialvla_env.sh first." >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

export PYTHONPATH="${SPATIAL_REPO}:${PYTHONPATH:-}"

{
  echo "run_tag=${RUN_TAG}"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "project_root=${PROJECT_ROOT}"
  echo "policy=spatialvla"
  echo "ckpt_path=${CKPT_PATH}"
  echo "task_filter=${TASK_FILTER}"
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
  local scene_name="$3"
  local robot="$4"
  local overlay="$5"
  local init_x="$6"
  local init_y="$7"
  local max_steps="$8"

  if [[ "${TASK_FILTER}" != "all" && "${TASK_FILTER}" != "${label}" ]]; then
    return 0
  fi

  echo "=== SpatialVLA ${label}: ${env_name} episodes 0:24 ==="
  python "${MAIN}" \
    --policy-model spatialvla --ckpt-path "${CKPT_PATH}" \
    --robot "${robot}" --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps "${max_steps}" \
    --env-name "${env_name}" --scene-name "${scene_name}" \
    --rgb-overlay-path "${overlay}" \
    --robot-init-x-range "${init_x}" "${init_x}" 1 \
    --robot-init-y-range "${init_y}" "${init_y}" 1 \
    --obj-variation-mode episode --obj-episode-range 0 24 \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --logging-dir "${RESULTS_DIR}/videos"
}

run_task stack StackGreenCubeOnYellowCubeBakedTexInScene-v0 bridge_table_1_v1 widowx \
  "${OVERLAY_DIR}/bridge_real_eval_1.png" 0.147 0.028 60
run_task carrot PutCarrotOnPlateInScene-v0 bridge_table_1_v1 widowx \
  "${OVERLAY_DIR}/bridge_real_eval_1.png" 0.147 0.028 60
run_task spoon PutSpoonOnTableClothInScene-v0 bridge_table_1_v1 widowx \
  "${OVERLAY_DIR}/bridge_real_eval_1.png" 0.147 0.028 60
run_task eggplant PutEggplantInBasketScene-v0 bridge_table_1_v2 widowx_sink_camera_setup \
  "${OVERLAY_DIR}/bridge_sink.png" 0.127 0.06 120

python - <<PY
import json
from pathlib import Path
root = Path("${RESULTS_DIR}/videos")
mp4s = list(root.rglob("*.mp4"))
success = sum(1 for p in mp4s if p.name.lower().startswith("success_"))
by_task = {}
task_envs = {
    "stack": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
    "carrot": "PutCarrotOnPlateInScene-v0",
    "spoon": "PutSpoonOnTableClothInScene-v0",
    "eggplant": "PutEggplantInBasketScene-v0",
}
for label, env_name in task_envs.items():
    task_mp4s = [p for p in mp4s if env_name in str(p)]
    if task_mp4s:
        task_success = sum(1 for p in task_mp4s if p.name.lower().startswith("success_"))
        by_task[label] = {
            "success": task_success,
            "total": len(task_mp4s),
            "success_rate": task_success / len(task_mp4s),
        }
summary = {
    "policy": "spatialvla",
    "task_filter": "${TASK_FILTER}",
    "success": success,
    "total": len(mp4s),
    "success_rate": success / len(mp4s) if mp4s else None,
    "by_task": by_task,
    "result_root": str(root),
}
Path("${RESULTS_DIR}/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(summary)
expected = 96 if "${TASK_FILTER}" == "all" else 24
if len(mp4s) != expected:
    raise SystemExit(f"expected {expected} videos, found {len(mp4s)}")
PY
