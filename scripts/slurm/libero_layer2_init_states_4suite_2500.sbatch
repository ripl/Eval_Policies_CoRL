#!/bin/bash
#SBATCH -p gpu
#SBATCH -C "48g&ampere"
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -J libero_l2_init
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

export PEDR_ROOT="${PEDR_ROOT:-/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right}"
export LIBERO_ROOT="${LIBERO_ROOT:-${PEDR_ROOT}/LIBERO}"
export EPC_ROOT="${EPC_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
export ARTIFACT_NAME="${ARTIFACT_NAME:-init_state_layer2_4suite_2500_seed20260519}"
export SOURCE_ROOT="${SOURCE_ROOT:-${LIBERO_ROOT}/libero/libero/${ARTIFACT_NAME}}"
export ROLLOUT_ROOT="${ROLLOUT_ROOT:-${EPC_ROOT}/artifacts/libero/${ARTIFACT_NAME}}"
export CONFIG_DIR="${CONFIG_DIR:-${EPC_ROOT}/artifacts/libero/libero_config_layer2_4suite_2500_seed20260519}"
export BASE_CONFIG_DIR="${BASE_CONFIG_DIR:-${EPC_ROOT}/artifacts/libero/libero_config}"
export BASE_SEED="${BASE_SEED:-20260519}"
export PROJECT_SCRIPT="${PROJECT_SCRIPT:-${EPC_ROOT}/scripts/libero/generate_libero_layer2_init_states.py}"
export HF_HOME="${EPC_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${EPC_ROOT}/cache/torch"
export XDG_CACHE_HOME="${EPC_ROOT}/cache/xdg"
export PIP_CACHE_DIR="${EPC_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${EPC_ROOT}/cache/conda_pkgs"
export MUJOCO_GL=egl

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

mkdir -p "${EPC_ROOT}/logs/slurm"
cd "${LIBERO_ROOT}"

echo "node=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SOURCE_ROOT=${SOURCE_ROOT}"
echo "ROLLOUT_ROOT=${ROLLOUT_ROOT}"
echo "CONFIG_DIR=${CONFIG_DIR}"
echo "BASE_SEED=${BASE_SEED}"
echo "date_utc=$(date -u --iso-8601=seconds)"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_SCRIPT}" \
  --out-root "${SOURCE_ROOT}" \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --num-states-per-task 250 \
  --seed "${BASE_SEED}" \
  --env-img-res 256

if [[ -e "${ROLLOUT_ROOT}" ]]; then
  if find "${ROLLOUT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty rollout root: ${ROLLOUT_ROOT}" >&2
    exit 1
  fi
fi
mkdir -p "$(dirname "${ROLLOUT_ROOT}")"
rsync -a "${SOURCE_ROOT}/" "${ROLLOUT_ROOT}/"

mkdir -p "${CONFIG_DIR}"
python - <<'PY'
import os
from pathlib import Path

base = Path(os.environ["BASE_CONFIG_DIR"]) / "config.yaml"
dst = Path(os.environ["CONFIG_DIR"]) / "config.yaml"
rollout_root = os.environ["ROLLOUT_ROOT"]
text = base.read_text().splitlines()
out = []
replaced = False
for line in text:
    if line.startswith("init_states:"):
        out.append(f"init_states: {rollout_root}")
        replaced = True
    else:
        out.append(line)
if not replaced:
    raise SystemExit("base config had no init_states line")
dst.write_text("\n".join(out) + "\n")
print(dst)
PY

python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE_ROOT"]) / "MANIFEST.json"
rollout = Path(os.environ["ROLLOUT_ROOT"]) / "MANIFEST.json"
source_manifest = json.loads(source.read_text())
rollout_manifest = json.loads(rollout.read_text())
if source_manifest["total_states"] != 10000 or rollout_manifest["total_states"] != 10000:
    raise SystemExit("manifest total_states check failed")
for root in [Path(os.environ["SOURCE_ROOT"]), Path(os.environ["ROLLOUT_ROOT"])]:
    suites = sorted(p.name for p in root.iterdir() if p.is_dir())
    if suites != ["libero_10", "libero_goal", "libero_object", "libero_spatial"]:
        raise SystemExit(f"unexpected suites under {root}: {suites}")
print("artifact install validation passed")
PY
