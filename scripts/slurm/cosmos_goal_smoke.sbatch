#!/bin/bash
#SBATCH -p gpu
#SBATCH -C "48g&ampere"
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -J cosmos_goal_smoke
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export ALPHABRAIN_REPO="${ALPHABRAIN_REPO:-${PROJECT_ROOT}/third_party/alphabrain}"
export INIT_ROOT="${INIT_ROOT:-${PROJECT_ROOT}/artifacts/libero/init_state_goal_5000}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B}"
export PRETRAINED_DIR="${PRETRAINED_DIR:-${PROJECT_ROOT}/checkpoints/Cosmos-Predict2-2B-Video2World}"
export RUN_GROUP="${RUN_GROUP:-cosmos_goal_smoke_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULT_DIR="${PROJECT_ROOT}/results/libero/cosmos_goal_5k/${RUN_GROUP}"
export EPISODE_CSV="${RESULT_DIR}/episodes.csv"
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/artifacts/libero/libero_config"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda_pkgs"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PORT="${PORT:-10093}"
export PYTHON="${PYTHON:-/share/data/ripl/tianchong/conda/envs/libero/bin/python}"

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

mkdir -p "${RESULT_DIR}/metadata" "${RESULT_DIR}/videos" logs/slurm
cd "${ALPHABRAIN_REPO}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "ALPHABRAIN_REPO=${ALPHABRAIN_REPO}"
echo "INIT_ROOT=${INIT_ROOT}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "PRETRAINED_DIR=${PRETRAINED_DIR}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "EPISODE_CSV=${EPISODE_CSV}"
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "date_utc=$(date -u --iso-8601=seconds)"

test -f "${CKPT_DIR}/Cosmos-Policy-LIBERO-Predict2-2B.pt"
test -f "${CKPT_DIR}/libero_dataset_statistics.json"
test -f "${CKPT_DIR}/libero_t5_embeddings.pkl"
test -d "${PRETRAINED_DIR}/vae"
test -f "${INIT_ROOT}/MANIFEST.json"

"${PYTHON}" - <<'PY'
missing = []
for mod in ["tyro", "websockets", "msgpack", "diffusers", "transformer_engine"]:
    try:
        __import__(mod)
    except Exception as exc:
        missing.append(f"{mod}: {exc}")
if missing:
    raise SystemExit("Cosmos runtime dependencies missing:\n" + "\n".join(missing))
PY

{
  echo "run_group=${RUN_GROUP}"
  echo "date_utc=$(date -u --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "policy=cosmos_policy"
  echo "suite=libero_goal"
  echo "checkpoint=${CKPT_DIR}"
  echo "checkpoint_hf_repo=nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
  echo "checkpoint_hf_sha=cb689ec0e3347c13667d70a78a3447388f5c3bb8"
  echo "base_hf_repo=nvidia/Cosmos-Predict2-2B-Video2World"
  echo "base_hf_sha=f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"
  echo "init_root=${INIT_ROOT}"
  echo "save_videos=False"
  git -C "${ALPHABRAIN_REPO}" rev-parse HEAD || true
  git -C "${ALPHABRAIN_REPO}" diff -- benchmarks/LIBERO/eval/eval_libero_cosmos.py || true
  "${PYTHON}" --version
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  fi
} > "${RESULT_DIR}/metadata/metadata.txt" 2>&1

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" deployment/model_server/server_policy_cosmos.py \
  --ckpt_dir "${CKPT_DIR}" \
  --pretrained_dir "${PRETRAINED_DIR}" \
  --port "${PORT}" \
  > "${RESULT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

WAITED=0
while ! "${PYTHON}" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${PORT})); s.close()" 2>/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "server exited before becoming ready" >&2
    tail -80 "${RESULT_DIR}/server.log" >&2 || true
    exit 1
  fi
  if [[ "${WAITED}" -ge 900 ]]; then
    echo "server did not become ready after ${WAITED}s" >&2
    tail -80 "${RESULT_DIR}/server.log" >&2 || true
    exit 1
  fi
  sleep 3
  WAITED=$((WAITED + 3))
done

"${PYTHON}" benchmarks/LIBERO/eval/eval_libero_cosmos.py \
  --ckpt_dir "${CKPT_DIR}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --task_suite_name libero_goal \
  --num_trials_per_task 5 \
  --selected_task_id 0 \
  --episode_start_idx 0 \
  --episode_end_idx 5 \
  --max_total_episodes 5 \
  --initial_states_path "${INIT_ROOT}" \
  --episode_results_path "${EPISODE_CSV}" \
  --policy_name cosmos_policy \
  --checkpoint_ref "nvidia/Cosmos-Policy-LIBERO-Predict2-2B@cb689ec0e3347c13667d70a78a3447388f5c3bb8" \
  --save_videos False \
  --video_out_path "${RESULT_DIR}/videos" \
  --job_name "${RUN_GROUP}" \
  2>&1 | tee "${RESULT_DIR}/eval.log"

"${PYTHON}" - <<'PY'
import csv
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
episode_csv = Path(os.environ["EPISODE_CSV"])
rows = list(csv.DictReader(episode_csv.open()))
summary = {
    "run_group": os.environ["RUN_GROUP"],
    "policy_name": "cosmos_policy",
    "policy_tracker_id": 592,
    "suite": "libero_goal",
    "checkpoint_ref": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B@cb689ec0e3347c13667d70a78a3447388f5c3bb8",
    "base_ref": "nvidia/Cosmos-Predict2-2B-Video2World@f50c09f5d8ab133a90cac3f4886a6471e9ba3f18",
    "init_state_root": os.environ["INIT_ROOT"],
    "episode_csv": str(episode_csv),
    "rows": len(rows),
    "successes": sum(int(r["success"]) for r in rows),
    "videos": len(list(result_dir.rglob("*.mp4"))),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
}
manifest = {
    **summary,
    "init_state_manifest": str(Path(os.environ["INIT_ROOT"]) / "MANIFEST.json"),
    "libero_config_path": str(Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"),
    "save_videos": False,
    "command": "python benchmarks/LIBERO/eval/eval_libero_cosmos.py ...",
    "metadata": str(result_dir / "metadata" / "metadata.txt"),
}
(result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if len(rows) != 5:
    raise SystemExit(f"expected 5 episode rows, got {len(rows)}")
if summary["videos"] != 0:
    raise SystemExit(f"expected no videos, found {summary['videos']}")
PY
