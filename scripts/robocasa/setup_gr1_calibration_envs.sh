#!/bin/bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export ROBOCASA_REPO="${ROBOCASA_REPO:-${PROJECT_ROOT}/third_party/robocasa-gr1-tabletop-tasks}"
export ROBOSUITE_REPO="${ROBOSUITE_REPO:-${PROJECT_ROOT}/third_party/robosuite}"
export ROBOSUITE_MODELS_REPO="${ROBOSUITE_MODELS_REPO:-${PROJECT_ROOT}/third_party/robosuite_models}"
export STARVLA_REPO="${STARVLA_REPO:-${PROJECT_ROOT}/third_party/starVLA}"
export ABOT_REPO="${ABOT_REPO:-${PROJECT_ROOT}/third_party/ABot-Manipulation}"
export VGGT_REPO="${VGGT_REPO:-${PROJECT_ROOT}/third_party/vggt}"

export SIM_ENV="${SIM_ENV:-${PROJECT_ROOT}/envs/robocasa_gr1_sim}"
export STARVLA_ENV="${STARVLA_ENV:-${PROJECT_ROOT}/envs/robocasa_gr1_starvla}"
export ABOT_ENV="${ABOT_ENV:-${PROJECT_ROOT}/envs/robocasa_gr1_abot}"

export ROBOCASA_CKPT_ROOT="${ROBOCASA_CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/robocasa}"
export STARVLA_CKPT_DIR="${STARVLA_CKPT_DIR:-${ROBOCASA_CKPT_ROOT}/starvla_qwen3_vl_oft_robocasa}"
export ABOT_CKPT_DIR="${ABOT_CKPT_DIR:-${ROBOCASA_CKPT_ROOT}/abot_m0_robocasa}"
export STARVLA_CKPT_PATH="${STARVLA_CKPT_PATH:-${STARVLA_CKPT_DIR}/checkpoints/steps_90000_pytorch_model.pt}"
export ABOT_CKPT_PATH="${ABOT_CKPT_PATH:-${ABOT_CKPT_DIR}/checkpoints/steps_50000_pytorch_model.pt}"
export QWEN3_VL_DIR="${QWEN3_VL_DIR:-${ROBOCASA_CKPT_ROOT}/Qwen3-VL-4B-Instruct}"
export QWEN3_VL_ACTION_DIR="${QWEN3_VL_ACTION_DIR:-${ROBOCASA_CKPT_ROOT}/Qwen3-VL-4B-Instruct-Action}"

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
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
export FLASH_ATTN_WHEEL_PATH="${FLASH_ATTN_WHEEL_PATH:-${PROJECT_ROOT}/cache/pip/wheels/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

mkdir -p \
  "${ROBOCASA_CKPT_ROOT}" \
  "${PROJECT_ROOT}/cache" \
  "${PROJECT_ROOT}/artifacts/setup/robocasa_gr1" \
  "${PROJECT_ROOT}/logs/slurm"

cd "${PROJECT_ROOT}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "date_utc=$(date -u --iso-8601=seconds)"

for required in "${ROBOCASA_REPO}" "${ROBOSUITE_REPO}" "${ROBOSUITE_MODELS_REPO}" "${STARVLA_REPO}" "${ABOT_REPO}" "${VGGT_REPO}"; do
  if [[ ! -d "${required}/.git" ]]; then
    echo "missing required repo: ${required}" >&2
    exit 2
  fi
done

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

create_env_if_missing() {
  local env_path="$1"
  if [[ ! -x "${env_path}/bin/python" ]]; then
    conda create -y -p "${env_path}" python=3.10
  fi
}

install_sim_env() {
  create_env_if_missing "${SIM_ENV}"
  conda activate "${SIM_ENV}"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install "gymnasium>=0.29,<1.0" tyro websockets msgpack msgpack-numpy imageio-ffmpeg "pydantic>=2,<3" matplotlib av rich omegaconf torchvision
  python -m pip install -e "${ROBOSUITE_REPO}"
  python -m pip install -e "${ROBOSUITE_MODELS_REPO}"
  python -m pip install "mink==0.0.5"
  python -m pip install -e "${ROBOCASA_REPO}"
  python -m pip freeze > "${PROJECT_ROOT}/artifacts/setup/robocasa_gr1/sim_pip_freeze.txt"
  python "${ROBOCASA_REPO}/robocasa/scripts/download_tabletop_assets.py" -y
  PYTHONPATH="${ROBOCASA_REPO}:${PYTHONPATH:-}" python - <<'PY'
import gymnasium as gym
import robocasa  # noqa: F401
import robosuite
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

env_name = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
env = gym.make(env_name, enable_render=False)
obs, info = env.reset(seed=123)
raw_env = env.unwrapped.env
print("sim validation ok")
print("robocasa", robocasa.__version__)
print("robosuite", robosuite.__version__)
print("layout_id", getattr(raw_env, "layout_id", None))
print("style_id", getattr(raw_env, "style_id", None))
print("obs_keys", sorted(list(obs.keys()))[:8])
env.close()
PY
  conda deactivate
}

install_policy_env() {
  local env_path="$1"
  local repo="$2"
  local needs_vggt="$3"
  create_env_if_missing "${env_path}"
  conda activate "${env_path}"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0" "torchvision==0.21.0"
  python -m pip install -r "${repo}/requirements.txt"
  if [[ "${needs_vggt}" == "yes" ]]; then
    python -m pip install -e "${VGGT_REPO}"
    python -m pip install "ninja" "packaging"
    if ! python -c 'import flash_attn' >/dev/null 2>&1; then
      mkdir -p "$(dirname "${FLASH_ATTN_WHEEL_PATH}")"
      curl -L --fail --retry 8 --retry-all-errors --connect-timeout 20 \
        --output "${FLASH_ATTN_WHEEL_PATH}.tmp" "${FLASH_ATTN_WHEEL_URL}"
      mv "${FLASH_ATTN_WHEEL_PATH}.tmp" "${FLASH_ATTN_WHEEL_PATH}"
      python -m pip install "${FLASH_ATTN_WHEEL_PATH}"
    fi
  fi
  python -m pip install -e "${repo}"
  python -m pip freeze > "${PROJECT_ROOT}/artifacts/setup/robocasa_gr1/$(basename "${env_path}")_pip_freeze.txt"
  conda deactivate
}

download_checkpoints() {
  conda activate "${STARVLA_ENV}"
  python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="StarVLA/Qwen3-VL-OFT-Robocasa",
    local_dir=os.environ["STARVLA_CKPT_DIR"],
    local_dir_use_symlinks=False,
    allow_patterns=["config.yaml", "dataset_statistics.json", "checkpoints/steps_90000_pytorch_model.pt"],
)
snapshot_download(
    repo_id="acvlab/ABot-M0-Robocasa",
    local_dir=os.environ["ABOT_CKPT_DIR"],
    local_dir_use_symlinks=False,
    allow_patterns=["config.yaml", "dataset_statistics.json", "checkpoints/*", "*.pt", "*.safetensors"],
)
print("checkpoint snapshots downloaded")
PY
  conda deactivate
}

patch_qwen_config_paths() {
  conda activate "${ABOT_ENV}"
  python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

qwen_dir = Path(os.environ["QWEN3_VL_DIR"])
qwen_action_dir = Path(os.environ["QWEN3_VL_ACTION_DIR"])
legacy_qwen_dir = Path(os.environ["ROBOCASA_CKPT_ROOT"]) / "qwen3_vl_4b_instruct"
if not qwen_dir.exists():
    if legacy_qwen_dir.exists():
        qwen_dir.symlink_to(legacy_qwen_dir, target_is_directory=True)
        print(f"linked Qwen base model to {qwen_dir}")
    else:
        snapshot_download(
            repo_id="Qwen/Qwen3-VL-4B-Instruct",
            local_dir=str(qwen_dir),
            local_dir_use_symlinks=False,
        )

if not qwen_action_dir.exists():
    snapshot_download(
        repo_id="StarVLA/Qwen3-VL-4B-Instruct-Action",
        local_dir=str(qwen_action_dir),
        local_dir_use_symlinks=False,
    )

for label, ckpt_dir_key, base_dir in (
    ("StarVLA", "STARVLA_CKPT_DIR", qwen_dir),
    ("ABot", "ABOT_CKPT_DIR", qwen_action_dir),
):
    cfg_path = Path(os.environ[ckpt_dir_key]) / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"{label} config missing: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    base_vlm = cfg.framework.qwenvl.base_vlm
    if str(base_vlm) != str(base_dir):
        cfg.framework.qwenvl.base_vlm = str(base_dir)
        OmegaConf.save(cfg, cfg_path)
        print(f"patched {label} base_vlm from {base_vlm} to {base_dir}")
    else:
        print(f"{label} base_vlm already patched: {base_vlm}")
    if label == "ABot" and cfg.framework.get("use_vggt", None) is not False:
        cfg.framework.use_vggt = False
        OmegaConf.save(cfg, cfg_path)
        print("patched ABot use_vggt to false for released RoboCasa checkpoint")
PY
  conda deactivate
}

patch_abot_framework_autoimport_guard() {
  python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["ABOT_REPO"]) / "ABot/model/framework/__init__.py"
text = path.read_text()
if 'ABOT_SKIP_FRAMEWORK_AUTOIMPORT' in text:
    print(f"ABot framework auto-import guard already present: {path}")
    raise SystemExit(0)

text = text.replace(
    "import pkgutil\nimport importlib\n",
    "import pkgutil\nimport importlib\nimport os\n",
)
old = "# Auto-import all framework submodules to trigger registration\nif pkg_path is not None:\n"
new = (
    "# Auto-import all framework submodules to trigger registration.\n"
    "# The RoboCasa simulator imports share_tools only for normalization stats; allow\n"
    "# that path to skip heavy model registration imports.\n"
    'if pkg_path is not None and os.environ.get("ABOT_SKIP_FRAMEWORK_AUTOIMPORT") != "1":\n'
)
if old not in text:
    raise SystemExit(f"could not find ABot framework auto-import block in {path}")
path.write_text(text.replace(old, new))
print(f"patched ABot framework auto-import guard: {path}")
PY
}

write_manifest() {
  local manifest="${PROJECT_ROOT}/artifacts/setup/robocasa_gr1/setup_manifest.txt"
  {
    echo "date_utc=$(date -u --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "sim_env=${SIM_ENV}"
    echo "starvla_env=${STARVLA_ENV}"
    echo "abot_env=${ABOT_ENV}"
    echo "starvla_ckpt_path=${STARVLA_CKPT_PATH}"
    echo "abot_ckpt_dir=${ABOT_CKPT_DIR}"
    echo "abot_ckpt_path_expected=${ABOT_CKPT_PATH}"
    for repo in "${ROBOCASA_REPO}" "${ROBOSUITE_REPO}" "${ROBOSUITE_MODELS_REPO}" "${STARVLA_REPO}" "${ABOT_REPO}" "${VGGT_REPO}"; do
      echo "repo=${repo}"
      git -C "${repo}" remote -v | head -2 || true
      git -C "${repo}" rev-parse HEAD || true
      git -C "${repo}" status --short || true
    done
  } > "${manifest}"
  cat "${manifest}"
}

install_sim_env
install_policy_env "${STARVLA_ENV}" "${STARVLA_REPO}" no
install_policy_env "${ABOT_ENV}" "${ABOT_REPO}" yes
download_checkpoints
patch_qwen_config_paths
patch_abot_framework_autoimport_guard
write_manifest

echo "robocasa gr1 setup complete"
