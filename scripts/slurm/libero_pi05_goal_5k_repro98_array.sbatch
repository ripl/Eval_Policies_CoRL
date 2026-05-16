#!/bin/bash
#SBATCH -p gpu
#SBATCH -C "48g&ampere"
#SBATCH --gpus=1
#SBATCH -c 1
#SBATCH --mem=50G
#SBATCH -t 08:00:00
#SBATCH -J pi05_5k_repro98
#SBATCH --array=0-9%4
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export LEROBOT_REPO="${LEROBOT_REPO:-${PROJECT_ROOT}/third_party/lerobot}"
export LEROBOT_SRC="${LEROBOT_SRC:-${LEROBOT_REPO}/src}"
export INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/init_state_goal_5000}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJECT_ROOT}/artifacts/libero/libero_config}"
export CKPT_REPO="${CKPT_REPO:-lerobot/pi05_libero_finetuned}"
export CKPT_REVISION="${CKPT_REVISION:-}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/pi05-libero-finetuned}"
export TOKENIZER_REPO="${TOKENIZER_REPO:-Shakalaka/paligemma-3b-pt-224}"
export SIMVLA_OVERLAY="${SIMVLA_OVERLAY:-${PROJECT_ROOT}/envs/simvla_py310_overlay_tf4}"
export PI05_OVERLAY="${PI05_OVERLAY:-${PROJECT_ROOT}/envs/pi05_py310_overlay_v5}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
export RUN_GROUP="${RUN_GROUP:-pi05_goal_5k_repro98_${SLURM_ARRAY_JOB_ID:-manual}}"
export TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export RESULT_ROOT="${PROJECT_ROOT}/results/libero/pi05_goal_5k_repro98/${RUN_GROUP}"
export RESULT_DIR="${RESULT_ROOT}/task_${TASK_ID}"
export EPISODE_CSV="${RESULT_DIR}/episodes.csv"
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

mkdir -p "${RESULT_DIR}/logs" "${RESULT_DIR}/metadata" "${RESULT_ROOT}/shared" logs/slurm
cd "${PROJECT_ROOT}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "LEROBOT_REPO=${LEROBOT_REPO}"
echo "INIT_ROOT=${INIT_ROOT}"
echo "CKPT_REPO=${CKPT_REPO}"
echo "CKPT_REVISION=${CKPT_REVISION}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "TOKENIZER_REPO=${TOKENIZER_REPO}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "PI05_OVERLAY=${PI05_OVERLAY}"
echo "SIMVLA_OVERLAY=${SIMVLA_OVERLAY}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF

mkdir -p "${PI05_OVERLAY}"
if [[ ! -f "${PI05_OVERLAY}/.pi05_deps_ready" ]]; then
  python -m pip install --upgrade --target "${PI05_OVERLAY}" \
    "numpy==1.26.4" \
    "av>=15.0.0,<16.0.0" \
    "Pillow>=10.0.0,<13.0.0" \
    "datasets>=4.0.0,<5.0.0" \
    "huggingface-hub==0.36.2" \
    "tokenizers>=0.21,<0.22" \
    "safetensors>=0.4.3,<1.0.0" \
    "gymnasium>=1.1.1,<2.0.0" \
    "pyserial>=3.5,<4.0" \
    "draccus==0.10.0" \
    "jsonlines>=4.0.0,<5.0.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "termcolor>=2.4.0,<4.0.0" \
    "packaging>=24.2,<26.0" \
    "scipy>=1.10.1,<1.15"
  python -m pip install --upgrade --target "${PI05_OVERLAY}" --no-deps \
    "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
  touch "${PI05_OVERLAY}/.pi05_deps_ready"
fi

export PYTHONPATH="${PI05_OVERLAY}:${SIMVLA_OVERLAY}:${LEROBOT_SRC}:${PYTHONPATH:-}"

python - <<'PY'
from pathlib import Path
import os
import datasets
import gymnasium
import serial
import torch
import transformers
import libero
import av
from PIL import Image
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
import lerobot.policies.pi05.processor_pi05
from transformers.models.siglip import check

for key in ["INIT_ROOT", "LEROBOT_SRC", "PI05_OVERLAY", "SIMVLA_OVERLAY"]:
    path = Path(os.environ[key])
    if not path.exists():
        raise SystemExit(f"missing required path {key}: {path}")
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit("OpenPI patched transformers check failed")
print("dependency validation ok")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("gymnasium", gymnasium.__version__)
print("av", av.__version__)
PY

if [[ "${TASK_ID}" == "0" ]]; then
  cp "${PROJECT_ROOT}/scripts/slurm/libero_pi05_goal_5k_repro98_array.sbatch" "${RESULT_ROOT}/shared/sbatch_template.sbatch" 2>/dev/null || true
  cp "${PROJECT_ROOT}/scripts/libero/run_pi05_goal_repro.py" "${RESULT_ROOT}/shared/run_pi05_goal_repro.py" 2>/dev/null || true
  cp "${LIBERO_CONFIG_PATH}/config.yaml" "${RESULT_ROOT}/shared/libero_config.yaml" 2>/dev/null || true
  cp "${INIT_ROOT}/MANIFEST.json" "${RESULT_ROOT}/shared/init_state_manifest.json" 2>/dev/null || true
  python -m pip freeze > "${RESULT_ROOT}/shared/pip_freeze.txt" 2>&1 || true
  conda list > "${RESULT_ROOT}/shared/conda_list.txt" 2>&1 || true
fi

{
  echo "run_group=${RUN_GROUP}"
  echo "task_id=${TASK_ID}"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "policy=pi05_lerobot"
  echo "policy_tracker_id=869"
  echo "suite=libero_goal"
  echo "checkpoint_hf_repo=${CKPT_REPO}"
  echo "checkpoint_hf_revision=${CKPT_REVISION}"
  echo "checkpoint_dir=${CKPT_DIR}"
  echo "tokenizer_repo=${TOKENIZER_REPO}"
  echo "n_action_steps=${N_ACTION_STEPS}"
  echo "init_root=${INIT_ROOT}"
  echo "episodes_per_task=500"
  echo "save_videos=False"
  git -C "${PROJECT_ROOT}" rev-parse HEAD || true
  git -C "${PROJECT_ROOT}" status --short || true
  git -C "${LEROBOT_REPO}" rev-parse HEAD || true
  git -C "${LEROBOT_REPO}" status --short || true
  python --version
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${RESULT_DIR}/metadata/metadata.txt" 2>&1

python -u "${PROJECT_ROOT}/scripts/libero/run_pi05_goal_repro.py" \
  --model-id "${CKPT_REPO}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --initial-states-path "${INIT_ROOT}" \
  --task-suite libero_goal \
  --selected-task-id "${TASK_ID}" \
  --episode-start-idx 0 \
  --episode-end-idx 500 \
  --max-total-episodes 500 \
  --episode-results-path "${EPISODE_CSV}" \
  --summary-path "${RESULT_DIR}/summary.json" \
  --manifest-path "${RESULT_DIR}/manifest.json" \
  --policy-name pi05_lerobot \
  --policy-tracker-id 869 \
  --tokenizer-name "${TOKENIZER_REPO}" \
  --device cuda \
  --n-action-steps "${N_ACTION_STEPS}" \
  > "${RESULT_DIR}/logs/eval.log" 2>&1

python - <<'PY'
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
manifest_path = result_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["metadata"] = str(result_dir / "metadata" / "metadata.txt")
manifest["libero_config_path"] = str(Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml")
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
