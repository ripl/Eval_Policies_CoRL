#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 00:10:00
#SBATCH --no-requeue
#SBATCH -J se_abcde_dex_asset_preflight
#SBATCH -o /home-nfs/tianchong/simplerenv_protocol_abcde_asset_preflight_%x_%j.out
#SBATCH -e /home-nfs/tianchong/simplerenv_protocol_abcde_asset_preflight_%x_%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SUPPORT_ROOT="${SUPPORT_ROOT:?SUPPORT_ROOT required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT required}"
PREFLIGHT_OUTPUT_ROOT="${PREFLIGHT_OUTPUT_ROOT:?PREFLIGHT_OUTPUT_ROOT required}"
DEXBOTIC_BENCHMARK_IMAGE="${DEXBOTIC_BENCHMARK_IMAGE:-docker://dexmal/dexbotic_benchmark@sha256:2e6938be25991c43e5261f91a14abcfaad68d6f1a664643ef69a2e3628b60fef}"
BENCHMARK_SRC="${BENCHMARK_SRC:-${PROJECT_ROOT}/scratch/strict_calibrations_20260523/sources/dexbotic-benchmark_54a6c749_simpler4ab717_clean_20260523T0152Z}"
CACHE_ROOT="${PROJECT_ROOT}/cache/dexbotic_official_images"
DEXBOTIC_MS2_REAL2SIM_ASSET_DIR="${DEXBOTIC_MS2_REAL2SIM_ASSET_DIR:-/workspace/simpler/ManiSkill2_real2sim/data}"

mkdir -p "${PREFLIGHT_OUTPUT_ROOT}" "${CACHE_ROOT}"/{hf,torch,xdg,apptainer,wandb,matplotlib,triton}

export HF_HOME="${CACHE_ROOT}/hf"
export HF_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"
export SINGULARITY_CACHEDIR="${CACHE_ROOT}/apptainer"
export WANDB_DIR="${CACHE_ROOT}/wandb"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0
export TRANSFORMERS_NO_FLAX=1
export SIMPLERENV_PROTOCOL_CONFIG="${PROJECT_ROOT}/configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json"
export SIMPLERENV_PROTOCOL_SHA256="1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"

apptainer exec --nv --cleanenv --pwd /workspace \
  --bind "${BENCHMARK_SRC}:/workspace" \
  --bind "${SUPPORT_ROOT}:${SUPPORT_ROOT}" \
  --bind "${PROJECT_ROOT}/scripts/simplerenv:${PROJECT_ROOT}/scripts/simplerenv" \
  --bind "${PROJECT_ROOT}/configs:${PROJECT_ROOT}/configs" \
  --bind "${RUN_ROOT}:${RUN_ROOT}" \
  --bind "${CACHE_ROOT}:${CACHE_ROOT}" \
  --env "HF_HOME=${HF_HOME}" \
  --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
  --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
  --env "TORCH_HOME=${TORCH_HOME}" \
  --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
  --env "WANDB_DIR=${WANDB_DIR}" \
  --env "MPLCONFIGDIR=${MPLCONFIGDIR}" \
  --env "MUJOCO_GL=${MUJOCO_GL}" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "VK_ICD_FILENAMES=${VK_ICD_FILENAMES}" \
  --env "MS2_REAL2SIM_ASSET_DIR=${DEXBOTIC_MS2_REAL2SIM_ASSET_DIR}" \
  --env "PYTHONPATH=${SUPPORT_ROOT}:${PROJECT_ROOT}/scripts/simplerenv:/workspace:${PYTHONPATH:-}" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  --env "SIMPLERENV_PROTOCOL_CONFIG=${SIMPLERENV_PROTOCOL_CONFIG}" \
  --env "SIMPLERENV_PROTOCOL_SHA256=${SIMPLERENV_PROTOCOL_SHA256}" \
  "${DEXBOTIC_BENCHMARK_IMAGE}" \
  bash -lc 'set -Eeuo pipefail; source /opt/conda/etc/profile.d/conda.sh; conda activate simpler_env; cd /workspace; python '"${SUPPORT_ROOT}"'/check_dexbotic_runtime_assets.py --report '"${PREFLIGHT_OUTPUT_ROOT}"'/dexbotic_runtime_asset_preflight.json --asset-dir '"${DEXBOTIC_MS2_REAL2SIM_ASSET_DIR}"' --protocol-config '"${SIMPLERENV_PROTOCOL_CONFIG}"
