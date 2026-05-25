#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SUPPORT_ROOT="${SUPPORT_ROOT:?SUPPORT_ROOT required}"
CONDITION="${CONDITION:?CONDITION required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/simplerenv_xvla_py310}"
MODEL_PATH="${MODEL_PATH:-2toINF/X-VLA-WidowX}"
PORT="${PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 10000)))}"

export PROJECT_ROOT SUPPORT_ROOT CONDITION RUN_ID OUTPUT_ROOT
export ENV_PREFIX MODEL_PATH
export SIMPLERENV_PROTOCOL_CONDITION="${CONDITION}"
export SIMPLERENV_PROTOCOL_CONFIG="${PROJECT_ROOT}/configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json"
export SIMPLERENV_PROTOCOL_SHA256="1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"
export RUN_PROTOCOL_NAME="simplerenv_protocol_abcde_stack_v1"
export X_VLA_REPO="${PROJECT_ROOT}/third_party/x_vla"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/cache/conda_pkgs}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/artifacts/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "missing env python: ${ENV_PREFIX}/bin/python" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/server" "${OUTPUT_ROOT}/videos" "${OUTPUT_ROOT}/logs"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
export PYTHONPATH="${SUPPORT_ROOT}:${PROJECT_ROOT}/scripts/simplerenv:${PROJECT_ROOT}/third_party/x_vla:${PROJECT_ROOT}/third_party/xvla_simpler_env:${PYTHONPATH:-}"

python - <<'PY'
import os
from pathlib import Path
from protocol_abcde_common import EPISODE_IDS, write_manifest, load_protocol_config, write_runtime_metadata
load_protocol_config()
root = Path(os.environ["OUTPUT_ROOT"])
write_manifest(root / "manifest.csv", "xvla", os.environ["CONDITION"], os.environ["RUN_ID"])
(root / "episode_ids.txt").write_text("\n".join(str(i) for i in EPISODE_IDS) + "\n")
write_runtime_metadata(
    root / "runtime_metadata.json",
    policy="xvla",
    condition=os.environ["CONDITION"],
    run_id=os.environ["RUN_ID"],
    checkpoint_identity=os.environ["MODEL_PATH"],
    extra={
        "policy_model": "xvla",
        "x_vla_repo": os.environ["X_VLA_REPO"],
        "env_prefix": os.environ["ENV_PREFIX"],
    },
)
PY

cd "${X_VLA_REPO}"
python deploy.py \
  --model_path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --disable_slurm \
  --output_dir "${OUTPUT_ROOT}/server" \
  > "${OUTPUT_ROOT}/server/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

python - <<PY
import socket, sys, time
host = "127.0.0.1"; port = int("${PORT}")
for _ in range(900):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex((host, port)) == 0:
            sys.exit(0)
    time.sleep(1)
print(f"server did not open {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

python "${PROJECT_ROOT}/scripts/simplerenv/xvla_widowx_protocol_client.py" \
  --server_ip 127.0.0.1 \
  --server_port "${PORT}" \
  --output_dir "${OUTPUT_ROOT}/videos" \
  --task-label stack \
  --episode-ids-file "${OUTPUT_ROOT}/episode_ids.txt" \
  --require-explicit-episode-ids \
  --max-steps 60

python - <<'PY'
import json
import os
from pathlib import Path
from protocol_abcde_common import EPISODE_IDS, append_result, base_row, validate_results_csv

root = Path(os.environ["OUTPUT_ROOT"])
policy = "xvla"
condition = os.environ["CONDITION"]
run_id = os.environ["RUN_ID"]
result_path = root / "videos/widowx_results.txt"
rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
if len(rows) != len(EPISODE_IDS):
    raise SystemExit(f"expected 288 X-VLA result rows, found {len(rows)}")
csv_path = root / "per_episode_results.csv"
for expected_id, source in zip(EPISODE_IDS, rows):
    if int(source.get("proc_id", -1)) != expected_id:
        raise SystemExit(f"X-VLA proc_id mismatch: {source.get('proc_id')} != {expected_id}")
    row = base_row(policy, condition, expected_id, run_id)
    row["success"] = int(bool(source.get("done", False)))
    row["steps"] = source.get("steps", "")
    row["video_path"] = source.get("output", "")
    if "error" in source:
        row["error"] = str(source["error"])
    append_result(csv_path, row)
validate_results_csv(csv_path, policy, condition, root / "validation_report.json")
PY
