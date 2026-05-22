#!/bin/bash
#SBATCH -p gpu
#SBATCH -C "48g&ampere"
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -J sf_goal_5k
#SBATCH --array=0-9%4
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export SF_REPO="${SF_REPO:-${PROJECT_ROOT}/third_party/spatial_forcing_libero_adapter/openvla-SF}"
export INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/init_state_goal_5000}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/spatial-forcing-7b-finetuned-libero-goal}"
export RUN_GROUP="${RUN_GROUP:-sf_goal_5k_full_${SLURM_ARRAY_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)}"
export TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export RESULT_ROOT="${PROJECT_ROOT}/results/libero/spatial_forcing_goal_5k/${RUN_GROUP}"
export RESULT_DIR="${RESULT_ROOT}/task_${TASK_ID}"
export EPISODE_CSV="${RESULT_DIR}/episodes.csv"
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/artifacts/libero/libero_config"
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

mkdir -p "${RESULT_DIR}/eval_logs" "${RESULT_DIR}/metadata" "${RESULT_ROOT}/shared" logs/slurm
cd "${SF_REPO}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SF_REPO=${SF_REPO}"
echo "INIT_ROOT=${INIT_ROOT}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "EPISODE_CSV=${EPISODE_CSV}"
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${SF_REPO}:${PYTHONPATH:-}"

if [[ "${TASK_ID}" == "0" ]]; then
  cp "${PROJECT_ROOT}/scripts/slurm/libero_spatial_forcing_goal_5k_array.sbatch" "${RESULT_ROOT}/shared/sbatch_template.sbatch" 2>/dev/null || true
  cp "${LIBERO_CONFIG_PATH}/config.yaml" "${RESULT_ROOT}/shared/libero_config.yaml"
  cp "${INIT_ROOT}/MANIFEST.json" "${RESULT_ROOT}/shared/init_state_manifest.json"
  python -m pip freeze > "${RESULT_ROOT}/shared/pip_freeze.txt"
  conda list > "${RESULT_ROOT}/shared/conda_list.txt"
fi

{
  echo "run_group=${RUN_GROUP}"
  echo "task_id=${TASK_ID}"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "policy=spatial_forcing"
  echo "suite=libero_goal"
  echo "checkpoint=${CKPT_DIR}"
  echo "init_root=${INIT_ROOT}"
  echo "episodes_per_task=500"
  echo "save_videos=False"
  git -C "${PROJECT_ROOT}" rev-parse HEAD || true
  git -C "${PROJECT_ROOT}" status --short || true
  git -C "${SF_REPO}" rev-parse HEAD || true
  git -C "${SF_REPO}" diff -- experiments/robot/libero/run_libero_eval.py experiments/robot/openvla_utils.py || true
  python --version
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${RESULT_DIR}/metadata/metadata.txt" 2>&1

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --task_suite_name libero_goal \
  --initial_states_path "${INIT_ROOT}" \
  --num_trials_per_task 500 \
  --selected_task_id "${TASK_ID}" \
  --episode_start_idx 0 \
  --episode_end_idx 500 \
  --policy_name spatial_forcing \
  --episode_results_path "${EPISODE_CSV}" \
  --local_log_dir "${RESULT_DIR}/eval_logs" \
  --run_id_note "${RUN_GROUP}--task_${TASK_ID}" \
  --save_videos False \
  --use_wandb False \
  --center_crop True

python - <<'PY'
import csv
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
episode_csv = Path(os.environ["EPISODE_CSV"])
rows = list(csv.DictReader(episode_csv.open()))
summary = {
    "run_group": os.environ["RUN_GROUP"],
    "policy_name": "spatial_forcing",
    "policy_tracker_id": 438,
    "suite": "libero_goal",
    "task_id": int(os.environ["TASK_ID"]),
    "checkpoint_ref": os.environ["CKPT_DIR"],
    "init_state_root": os.environ["INIT_ROOT"],
    "episode_csv": str(episode_csv),
    "rows": len(rows),
    "successes": sum(int(r["success"]) for r in rows),
    "videos": len(list(result_dir.rglob("*.mp4"))),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
}
manifest = {
    **summary,
    "checkpoint_hf_repo": "haofuly/spatial-forcing-7b-finetuned-libero-goal",
    "checkpoint_hf_sha": "c0093e70524ad781ae6fe644618f501af32a184d",
    "init_state_manifest": str(Path(os.environ["INIT_ROOT"]) / "MANIFEST.json"),
    "libero_config_path": str(Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"),
    "save_videos": False,
    "command": "python experiments/robot/libero/run_libero_eval.py ...",
    "metadata": str(result_dir / "metadata" / "metadata.txt"),
}
(result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if len(rows) != 500:
    raise SystemExit(f"expected 500 episode rows, got {len(rows)}")
if summary["videos"] != 0:
    raise SystemExit(f"expected no videos, found {summary['videos']}")
PY
