#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-official}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_cogact_py310_np126}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/cache/conda_pkgs}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${PROJECT_ROOT}/cache/apptainer}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${PROJECT_ROOT}/cache/singularity}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/artifacts/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" \
  "${APPTAINER_CACHEDIR}" \
  "${SINGULARITY_CACHEDIR}" \
  "${WANDB_DIR}" \
  "${PROJECT_ROOT}/results" \
  "${PROJECT_ROOT}/artifacts" \
  "${PROJECT_ROOT}/artifacts/simplerenv/${RUN_TAG}"

COGACT_REPO="${COGACT_REPO:-${PROJECT_ROOT}/third_party/cogact}"
SIMPLER_ENV_REPO="${SIMPLER_ENV_REPO:-${PROJECT_ROOT}/third_party/simpler_env}"
COGACT_MAIN="${PROJECT_ROOT}/scripts/simplerenv/main_inference_cogact.py"
SIMPLER_OVERLAY_DIR="${SIMPLER_ENV_REPO}/ManiSkill2_real2sim/data/real_inpainting"
CKPT_PATH="${CKPT_PATH:-CogACT/CogACT-Base}"
EPISODE_START="${EPISODE_START:-0}"
EPISODE_END="${EPISODE_END:-24}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/simplerenv/cogact_base/official_fixed_grid/${RUN_TAG}}"
TASK_FILTER="${TASK_FILTER:-all}"

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_PREFIX}"
else
  echo "Missing env at ${ENV_PREFIX}. Run scripts/simplerenv/setup_cogact_env.sh first." >&2
  exit 2
fi

export PYTHONPATH="${COGACT_REPO}:${SIMPLER_ENV_REPO}:${PYTHONPATH:-}"

case "${MODE}" in
  official)
    CONFIG="${PROJECT_ROOT}/configs/simplerenv/cogact_base_official.yaml"
    ;;
  randomized)
    CONFIG="${PROJECT_ROOT}/configs/simplerenv/cogact_base_randomized.yaml"
    echo "Randomized-position injection is not implemented yet: ${CONFIG}" >&2
    exit 3
    ;;
  *)
    echo "Usage: $0 [official|randomized]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${COGACT_MAIN}" ]]; then
  echo "Missing CogACT inference wrapper: ${COGACT_MAIN}" >&2
  exit 2
fi
if [[ ! -d "${SIMPLER_ENV_REPO}/simpler_env" ]]; then
  echo "Missing SimplerEnv checkout: ${SIMPLER_ENV_REPO}" >&2
  echo "Initialize submodules with scripts/simplerenv/setup_cogact_env.sh." >&2
  exit 2
fi
if [[ ! -d "${COGACT_REPO}/sim_cogact" ]]; then
  echo "Missing CogACT checkout: ${COGACT_REPO}" >&2
  echo "Initialize submodules with scripts/simplerenv/setup_cogact_env.sh." >&2
  exit 2
fi

echo "Running ${MODE} config: ${CONFIG}"
mkdir -p "${RESULTS_DIR}"

{
  echo "run_tag=${RUN_TAG}"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "project_root=${PROJECT_ROOT}"
  echo "mode=${MODE}"
  echo "ckpt_path=${CKPT_PATH}"
  echo "episode_start=${EPISODE_START}"
  echo "episode_end=${EPISODE_END}"
  echo "task_filter=${TASK_FILTER}"
  echo "results_dir=${RESULTS_DIR}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  git -C "${PROJECT_ROOT}" rev-parse HEAD
  git -C "${PROJECT_ROOT}" submodule status --recursive
  python --version
  python -m pip freeze
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${PROJECT_ROOT}/artifacts/simplerenv/${RUN_TAG}/metadata.txt" 2>&1

run_task() {
  local label="$1"
  local env_name="$2"
  local scene_name="$3"
  local robot="$4"
  local rgb_overlay_path="$5"
  local robot_init_x="$6"
  local robot_init_y="$7"

  if [[ "${TASK_FILTER}" != "all" && "${TASK_FILTER}" != "${label}" ]]; then
    return 0
  fi

  echo "=== ${label}: ${env_name} episodes ${EPISODE_START}:${EPISODE_END} ==="
  python "${COGACT_MAIN}" \
    --policy-model cogact --ckpt-path "${CKPT_PATH}" \
    --robot "${robot}" --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
    --env-name "${env_name}" --scene-name "${scene_name}" \
    --rgb-overlay-path "${rgb_overlay_path}" \
    --robot-init-x "${robot_init_x}" "${robot_init_x}" 1 \
    --robot-init-y "${robot_init_y}" "${robot_init_y}" 1 \
    --obj-variation-mode episode --obj-episode-range "${EPISODE_START}" "${EPISODE_END}" \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --logging-dir "${RESULTS_DIR}/videos"
}

run_task stack \
  StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
  bridge_table_1_v1 \
  widowx \
  "${SIMPLER_OVERLAY_DIR}/bridge_real_eval_1.png" \
  0.147 \
  0.028

run_task carrot \
  PutCarrotOnPlateInScene-v0 \
  bridge_table_1_v1 \
  widowx \
  "${SIMPLER_OVERLAY_DIR}/bridge_real_eval_1.png" \
  0.147 \
  0.028

run_task spoon \
  PutSpoonOnTableClothInScene-v0 \
  bridge_table_1_v1 \
  widowx \
  "${SIMPLER_OVERLAY_DIR}/bridge_real_eval_1.png" \
  0.147 \
  0.028

run_task eggplant \
  PutEggplantInBasketScene-v0 \
  bridge_table_1_v2 \
  widowx_sink_camera_setup \
  "${SIMPLER_OVERLAY_DIR}/bridge_sink.png" \
  0.127 \
  0.06
