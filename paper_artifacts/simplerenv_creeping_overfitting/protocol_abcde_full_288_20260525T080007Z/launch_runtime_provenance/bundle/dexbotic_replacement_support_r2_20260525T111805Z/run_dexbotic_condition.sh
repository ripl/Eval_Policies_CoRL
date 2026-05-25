#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SUPPORT_ROOT="${SUPPORT_ROOT:?SUPPORT_ROOT required}"
CONDITION="${CONDITION:?CONDITION required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"
DEXBOTIC_MODEL_ID="${DEXBOTIC_MODEL_ID:-Dexmal/simpler-db-memvla}"
DEXBOTIC_SERVER_IMAGE="${DEXBOTIC_SERVER_IMAGE:-docker://dexmal/dexbotic@sha256:7851cf8ed236dc18b5df3df6e8ea8ff5a543d154c03ac637a6dc6bd4e9eda654}"
DEXBOTIC_BENCHMARK_IMAGE="${DEXBOTIC_BENCHMARK_IMAGE:-docker://dexmal/dexbotic_benchmark@sha256:2e6938be25991c43e5261f91a14abcfaad68d6f1a664643ef69a2e3628b60fef}"
DEXBOTIC_SRC="${PROJECT_ROOT}/third_party/dexbotic"
BENCHMARK_SRC="${BENCHMARK_SRC:-${PROJECT_ROOT}/scratch/strict_calibrations_20260523/sources/dexbotic-benchmark_54a6c749_simpler4ab717_clean_20260523T0152Z}"
CACHE_ROOT="${PROJECT_ROOT}/cache/dexbotic_official_images"
DEXBOTIC_SERVER_LAUNCHER="${SUPPORT_ROOT}/bin/launch_memvla_server_seeded.py"
DEXBOTIC_NORM_STATS_SOURCE="${DEXBOTIC_NORM_STATS_SOURCE:-${PROJECT_ROOT}/results/simplerenv/strict_calibrations_20260523/dexbotic_official_workspace_strict_20260523T015147Z/run/norm_stats.json}"
DEXBOTIC_NORM_STATS_SHA256="${DEXBOTIC_NORM_STATS_SHA256:-ddbeb68786543c68f8fa198c33fd9a265025d35c5f7ebba144d6ec9c655693e6}"
DEXBOTIC_MS2_REAL2SIM_ASSET_DIR="${DEXBOTIC_MS2_REAL2SIM_ASSET_DIR:-/workspace/simpler/ManiSkill2_real2sim/data}"
PORT="${PORT:-$((7891 + (${SLURM_JOB_ID:-0} % 800)))}"
SERVER_SEED="$(PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 - <<PY
from protocol_abcde_common import server_seed_for
print(server_seed_for("dexbotic", "${CONDITION}"))
PY
)"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to reuse OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/run/container_home" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/eval" "${OUTPUT_ROOT}/metadata" "${CACHE_ROOT}"/{hf,torch,xdg,pip,apptainer,wandb,matplotlib,triton}

if [[ ! -f "${DEXBOTIC_SERVER_LAUNCHER}" ]]; then
  echo "missing Dexbotic server launcher: ${DEXBOTIC_SERVER_LAUNCHER}" >&2
  exit 4
fi
if [[ ! -s "${DEXBOTIC_NORM_STATS_SOURCE}" ]]; then
  echo "missing Dexbotic norm_stats: ${DEXBOTIC_NORM_STATS_SOURCE}" >&2
  exit 4
fi
actual_norm_sha="$(sha256sum "${DEXBOTIC_NORM_STATS_SOURCE}" | awk '{print $1}')"
if [[ "${actual_norm_sha}" != "${DEXBOTIC_NORM_STATS_SHA256}" ]]; then
  echo "Dexbotic norm_stats SHA mismatch: ${actual_norm_sha} != ${DEXBOTIC_NORM_STATS_SHA256}" >&2
  exit 4
fi
cp "${DEXBOTIC_NORM_STATS_SOURCE}" "${OUTPUT_ROOT}/run/norm_stats.json"
sha256sum "${OUTPUT_ROOT}/run/norm_stats.json" > "${OUTPUT_ROOT}/metadata/norm_stats.sha256"

export PROJECT_ROOT SUPPORT_ROOT CONDITION RUN_ID OUTPUT_ROOT DEXBOTIC_MODEL_ID DEXBOTIC_SERVER_IMAGE DEXBOTIC_BENCHMARK_IMAGE DEXBOTIC_SRC BENCHMARK_SRC
export DEXBOTIC_SERVER_LAUNCHER DEXBOTIC_NORM_STATS_SOURCE DEXBOTIC_NORM_STATS_SHA256 DEXBOTIC_MS2_REAL2SIM_ASSET_DIR
export SIMPLERENV_PROTOCOL_CONDITION="${CONDITION}"
export SIMPLERENV_PROTOCOL_CONFIG="${PROJECT_ROOT}/configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json"
export SIMPLERENV_PROTOCOL_SHA256="1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"
export HF_HOME="${CACHE_ROOT}/hf"
export HF_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
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
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0
export TRANSFORMERS_NO_FLAX=1

PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 - <<'PY'
import os
from pathlib import Path
from protocol_abcde_common import load_protocol_config, write_manifest, write_runtime_metadata
root = Path(os.environ["OUTPUT_ROOT"])
load_protocol_config()
write_manifest(root / "manifest.csv", "dexbotic", os.environ["CONDITION"], os.environ["RUN_ID"])
write_runtime_metadata(
    root / "runtime_metadata.json",
    policy="dexbotic",
    condition=os.environ["CONDITION"],
    run_id=os.environ["RUN_ID"],
    checkpoint_identity=os.environ["DEXBOTIC_MODEL_ID"],
    extra={
        "policy_model": "db-memvla",
        "dexbotic_src": os.environ["DEXBOTIC_SRC"],
        "benchmark_src": os.environ["BENCHMARK_SRC"],
        "server_container_image": os.environ["DEXBOTIC_SERVER_IMAGE"],
        "benchmark_container_image": os.environ["DEXBOTIC_BENCHMARK_IMAGE"],
        "server_launcher": os.environ["DEXBOTIC_SERVER_LAUNCHER"],
        "norm_stats_source": os.environ["DEXBOTIC_NORM_STATS_SOURCE"],
        "norm_stats_sha256": os.environ["DEXBOTIC_NORM_STATS_SHA256"],
        "ms2_real2sim_asset_dir": os.environ["DEXBOTIC_MS2_REAL2SIM_ASSET_DIR"],
    },
)
PY

CONFIG_PATH="${OUTPUT_ROOT}/run/dexbotic_protocol_abcde_${CONDITION}.yaml"
PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 "${SUPPORT_ROOT}/make_dexbotic_abcde_config.py" \
  --output-dir "${OUTPUT_ROOT}/eval" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model-id "${DEXBOTIC_MODEL_ID}" \
  --config-out "${CONFIG_PATH}"

apptainer exec --nv --cleanenv \
  --bind "${DEXBOTIC_SRC}:/mnt/dexbotic" \
  --bind "${SUPPORT_ROOT}:${SUPPORT_ROOT}" \
  --bind "${OUTPUT_ROOT}/run:/dex_run" \
  --bind "${CACHE_ROOT}:${CACHE_ROOT}" \
  --env "HF_HOME=${HF_HOME}" \
  --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
  --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
  --env "TORCH_HOME=${TORCH_HOME}" \
  --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
  --env "WANDB_DIR=${WANDB_DIR}" \
  --env "MPLCONFIGDIR=${MPLCONFIGDIR}" \
  --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "PYTHONHASHSEED=${SERVER_SEED}" \
  --env "EVAL_ROLLOUT_SEED=${SERVER_SEED}" \
  --env "DEXBOTIC_SERVER_SEED=${SERVER_SEED}" \
  --env "DEXBOTIC_MODEL_ID=${DEXBOTIC_MODEL_ID}" \
  --env "DEXBOTIC_PORT=${PORT}" \
  --env "DEXBOTIC_NORM_STATS=/dex_run/norm_stats.json" \
  "${DEXBOTIC_SERVER_IMAGE}" \
  bash -lc 'set -Eeuo pipefail; source /opt/conda/etc/profile.d/conda.sh; conda activate dexbotic; export PYTHONPATH=/mnt/dexbotic:'"${SUPPORT_ROOT}"':'"${SUPPORT_ROOT}"'/bin:${PYTHONPATH:-}; cd /mnt/dexbotic; python -u '"${DEXBOTIC_SERVER_LAUNCHER}" \
  > "${OUTPUT_ROOT}/logs/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 360); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Dexbotic server exited before ready" >&2
    tail -240 "${OUTPUT_ROOT}/logs/server.log" || true
    exit 1
  fi
  if python3 - "${PORT}" <<'PYSOCK'
import socket, sys
s = socket.socket(); s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PYSOCK
  then READY=1; break; fi
  sleep 5
done
if [[ "${READY}" != "1" ]]; then
  echo "Dexbotic server did not open port ${PORT}" >&2
  exit 1
fi

apptainer exec --nv --cleanenv --pwd /workspace \
  --home "${OUTPUT_ROOT}/run/container_home:/home/tianchong" \
  --bind "${BENCHMARK_SRC}:/workspace" \
  --bind "${SUPPORT_ROOT}:${SUPPORT_ROOT}" \
  --bind "${PROJECT_ROOT}/scripts/simplerenv:${PROJECT_ROOT}/scripts/simplerenv" \
  --bind "${PROJECT_ROOT}/configs:${PROJECT_ROOT}/configs" \
  --bind "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
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
  --env "PYTHONHASHSEED=${SERVER_SEED}" \
  --env "EVAL_ROLLOUT_SEED=${SERVER_SEED}" \
  --env "SIMPLERENV_PROTOCOL_CONDITION=${CONDITION}" \
  --env "SIMPLERENV_PROTOCOL_CONFIG=${SIMPLERENV_PROTOCOL_CONFIG}" \
  --env "SIMPLERENV_PROTOCOL_SHA256=${SIMPLERENV_PROTOCOL_SHA256}" \
  --env "PYTHONPATH=${SUPPORT_ROOT}:${PROJECT_ROOT}/scripts/simplerenv:/workspace:${PYTHONPATH:-}" \
  "${DEXBOTIC_BENCHMARK_IMAGE}" \
  bash -lc 'set -Eeuo pipefail; source /opt/conda/etc/profile.d/conda.sh; conda activate simpler_env; cd /workspace; python '"${SUPPORT_ROOT}"'/seeded_runpy_with_protocol.py --seed "${EVAL_ROLLOUT_SEED}" --script /workspace/evaluation/run_simpler_evaluation.py -- --config '"${CONFIG_PATH}" \
  > "${OUTPUT_ROOT}/logs/eval.stdout" \
  2> "${OUTPUT_ROOT}/logs/eval.stderr"

PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 - <<'PY'
import json
import os
from pathlib import Path
from protocol_abcde_common import EPISODE_IDS, HORIZON, append_result, base_row, count_video_steps, validate_results_csv

root = Path(os.environ["OUTPUT_ROOT"])
policy = "dexbotic"
condition = os.environ["CONDITION"]
run_id = os.environ["RUN_ID"]
result_files = sorted((root / "eval").glob("**/results.json"))
if len(result_files) != 1:
    raise SystemExit(f"expected one Dexbotic results.json, found {len(result_files)}")
data = json.loads(result_files[0].read_text())
successes = data.get("success_array")
if not isinstance(successes, list) or len(successes) != len(EPISODE_IDS):
    raise SystemExit(f"expected 288 Dexbotic success entries, got {None if successes is None else len(successes)}")

def configured_horizon(payload):
    config = payload.get("config") if isinstance(payload, dict) else {}
    try:
        horizon = int((config or {}).get("max_episode_steps", HORIZON))
    except Exception as exc:
        raise SystemExit(f"Dexbotic results.json has invalid max_episode_steps: {exc}") from exc
    if horizon != HORIZON:
        raise SystemExit(f"Dexbotic runtime horizon mismatch: {horizon} != {HORIZON}")
    return horizon

runtime_horizon = configured_horizon(data)

def int_steps(value, *, source):
    try:
        steps = int(value)
    except Exception:
        return None
    if steps < 0 or steps > HORIZON:
        raise SystemExit(f"Dexbotic {source} steps outside 0..{HORIZON}: {steps}")
    return steps

def result_json_steps(payload, episode_id):
    for key in ("steps_array", "step_array", "episode_steps", "step_counts", "elapsed_steps"):
        values = payload.get(key)
        if isinstance(values, list) and len(values) == len(EPISODE_IDS):
            steps = int_steps(values[episode_id], source=f"results.json:{key}")
            if steps is not None:
                return steps, f"results_json_{key}"
    return None, ""

csv_path = root / "per_episode_results.csv"
steps_source_counts = {}
fallback_episodes = []
for episode_id, success in zip(EPISODE_IDS, successes):
    videos = sorted((root / "eval").glob(f"**/*obj_episode_{episode_id}_*.mp4"))
    if len(videos) != 1:
        raise SystemExit(f"expected one video for episode {episode_id}, found {len(videos)}")
    steps, steps_source = result_json_steps(data, episode_id)
    if steps is None:
        steps = count_video_steps(videos[0], subtract_initial_frame=True)
        steps_source = "video_frame_count_minus_initial" if steps is not None else ""
    if steps is None:
        steps = runtime_horizon
        steps_source = "runtime_config_horizon_fallback"
        fallback_episodes.append(episode_id)
    if steps < 0 or steps > HORIZON:
        raise SystemExit(f"Dexbotic steps outside 0..{HORIZON} for episode {episode_id}: {steps}")
    steps_source_counts[steps_source] = steps_source_counts.get(steps_source, 0) + 1
    row = base_row(policy, condition, episode_id, run_id)
    row["success"] = int(bool(success))
    row["steps"] = steps
    row["video_path"] = str(videos[0])
    append_result(csv_path, row)
(root / "steps_metadata.json").write_text(json.dumps({
    "steps_source_counts": steps_source_counts,
    "runtime_horizon_fallback_episode_ids": fallback_episodes,
    "runtime_horizon_fallback_caveat": (
        "Fallback steps equal the configured horizon from Dexbotic results.json and are not success timing; "
        "they are used only when per-episode result step fields and video frame counts are unavailable."
    ),
}, indent=2, sort_keys=True) + "\n")
validate_results_csv(csv_path, policy, condition, root / "validation_report.json")
PY
