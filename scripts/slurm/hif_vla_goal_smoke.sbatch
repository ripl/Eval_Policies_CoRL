#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=100G
#SBATCH -t 02:00:00
#SBATCH -J hif_goal_smoke
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export HIF_REPO="${HIF_REPO:-${PROJECT_ROOT}/third_party/hif_vla}"
export HIF_OVERLAY="${HIF_OVERLAY:-${PROJECT_ROOT}/envs/hif_vla_overlay}"
export INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/init_state_goal_5000}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/hifvla-libero-goal}"
export RUN_GROUP="${RUN_GROUP:-hif_vla_goal_smoke_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULT_ROOT="${PROJECT_ROOT}/results/libero/hif_vla_goal_5k"
export RESULT_DIR="${RESULT_ROOT}/${RUN_GROUP}"
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

mkdir -p "${RESULT_DIR}/eval_logs" "${RESULT_DIR}/metadata" "${PROJECT_ROOT}/logs/slurm"
cd "${HIF_REPO}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "HIF_REPO=${HIF_REPO}"
echo "HIF_OVERLAY=${HIF_OVERLAY}"
echo "INIT_ROOT=${INIT_ROOT}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "EPISODE_CSV=${EPISODE_CSV}"
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${HIF_OVERLAY}:${HIF_REPO}:${PYTHONPATH:-}"

python - <<'PY'
from pathlib import Path
import os

if not Path(os.environ["HIF_OVERLAY"]).exists():
    raise SystemExit(f"missing HIF overlay: {os.environ['HIF_OVERLAY']}")
import av
print("av", av.__version__)
PY

{
  echo "run_group=${RUN_GROUP}"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "policy=hif_vla"
  echo "suite=libero_goal"
  echo "checkpoint=${CKPT_DIR}"
  echo "init_root=${INIT_ROOT}"
  echo "hif_overlay=${HIF_OVERLAY}"
  echo "episodes=5"
  echo "save_videos=False"
  git -C "${HIF_REPO}" rev-parse HEAD || true
  git -C "${HIF_REPO}" diff -- experiments/robot/libero/run_libero_eval.py || true
  python --version
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${RESULT_DIR}/metadata/metadata.txt" 2>&1

python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --task_suite_name libero_goal \
  --history_length 8 \
  --initial_states_path "${INIT_ROOT}" \
  --num_trials_per_task 500 \
  --selected_task_id 0 \
  --episode_start_idx 0 \
  --episode_end_idx 5 \
  --max_total_episodes 5 \
  --policy_name hif_vla \
  --episode_results_path "${EPISODE_CSV}" \
  --local_log_dir "${RESULT_DIR}/eval_logs" \
  --run_id_note "${RUN_GROUP}" \
  --save_videos False \
  --use_wandb False

python - <<'PY'
import csv
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
episode_csv = Path(os.environ["EPISODE_CSV"])
rows = list(csv.DictReader(episode_csv.open()))
checkpoint_manifest = Path(os.environ["CKPT_DIR"]) / "download_manifest.json"
checkpoint_meta = json.loads(checkpoint_manifest.read_text()) if checkpoint_manifest.exists() else {}
summary = {
    "run_group": os.environ["RUN_GROUP"],
    "policy_name": "hif_vla",
    "policy_tracker_id": 271,
    "suite": "libero_goal",
    "checkpoint_ref": os.environ["CKPT_DIR"],
    "checkpoint_hf_repo": checkpoint_meta.get("repo_id", "minnielin/hifvla-libero-goal"),
    "checkpoint_hf_sha": checkpoint_meta.get("revision", ""),
    "init_state_root": os.environ["INIT_ROOT"],
    "episode_csv": str(episode_csv),
    "rows": len(rows),
    "successes": sum(int(r["success"]) for r in rows),
    "errors": sum(1 for r in rows if r["error_type"]),
    "videos": len(list(result_dir.rglob("*.mp4"))),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
}
manifest = {
    **summary,
    "init_state_manifest": str(Path(os.environ["INIT_ROOT"]) / "MANIFEST.json"),
    "libero_config_path": str(Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"),
    "save_videos": False,
    "command": "python experiments/robot/libero/run_libero_eval.py ...",
    "metadata": str(result_dir / "metadata" / "metadata.txt"),
}
(result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if len(rows) != 5:
    raise SystemExit(f"expected 5 episode rows, got {len(rows)}")
if summary["errors"]:
    raise SystemExit(f"expected no episode errors, got {summary['errors']}")
if summary["videos"] != 0:
    raise SystemExit(f"expected no videos, found {summary['videos']}")
PY
