#!/bin/bash
#SBATCH -p gpu
#SBATCH -C 48g
#SBATCH --exclude=g0,g7
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=160G
#SBATCH -t 08:00:00
#SBATCH -J calvin_l2fresh
#SBATCH --array=0-23%6
#SBATCH -o logs/slurm/%x-%A_%a.out
#SBATCH -e logs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
NUM_SEQUENCES=1000
NUM_CHUNKS=4
CHUNK_SIZE=250
CURATED_PACKAGE_DIR="${PROJECT_ROOT}/paper_artifacts/calvin_creeping_overfitting/layer2_fresh_1000_20260526"
MANIFEST_LIST="${MANIFEST_LIST:-${CURATED_PACKAGE_DIR}/layer2_fresh_manifest_list.json}"
RUN_KIND="${RUN_KIND:-paper}"
if [[ -n "${XVLA_EP_LEN+x}" ]]; then
  if [[ "${XVLA_EP_LEN}" != "360" ]]; then
    echo "Layer 2 fresh X-VLA must use XVLA_EP_LEN=360; inherited XVLA_EP_LEN=${XVLA_EP_LEN}" >&2
    exit 2
  fi
fi
XVLA_EP_LEN=360
LAYER2_RESULTS_ROOT="${LAYER2_RESULTS_ROOT:-${PROJECT_ROOT}/results/calvin/layer2_fresh_1000/${RUN_KIND}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"
mkdir -p logs/slurm

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

case "${RUN_KIND}" in
  paper|smoke|debug) ;;
  *) echo "RUN_KIND must be paper, smoke, or debug; got ${RUN_KIND}" >&2; exit 2 ;;
esac

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required; submit this file as the declared 0-23 array." >&2
  exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-0}" -gt 4 ]]; then
  echo "Refusing to run with SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}; expected <=4" >&2
  exit 2
fi
if [[ ! -s "${MANIFEST_LIST}" ]]; then
  echo "Missing Worker A Layer 2 manifest list: ${MANIFEST_LIST}" >&2
  echo "Refusing to fall back to ignored results/calvin/sequence_manifests files." >&2
  exit 2
fi

for forbidden in CALVIN_RESET_BANK RESET_BANK CALVIN_RESET_PROTOCOL; do
  if [[ -n "${!forbidden-}" ]]; then
    echo "Layer 2 fresh runs must not set ${forbidden}; refusing to launch." >&2
    exit 2
  fi
done
unset CALVIN_RESET_BANK RESET_BANK CALVIN_RESET_PROTOCOL

TASK_ID="${SLURM_ARRAY_TASK_ID}"
GROUP_ID=$((TASK_ID / NUM_CHUNKS))
CHUNK_ID=$((TASK_ID % NUM_CHUNKS))
MANIFEST_IDX=$((GROUP_ID / 3))
POLICY_IDX=$((GROUP_ID % 3))

case "${MANIFEST_IDX}" in
  0) CONDITION_TAG="layer2_fresh_seed2026052601" ;;
  1) CONDITION_TAG="layer2_fresh_seed2026052602" ;;
  *) echo "Unexpected MANIFEST_IDX=${MANIFEST_IDX} from task ${TASK_ID}; expected array 0-23." >&2; exit 2 ;;
esac

case "${POLICY_IDX}" in
  0) POLICY="xvla"; RUN_SCRIPT="scripts/calvin/run_xvla_calvin_smoke.sh" ;;
  1) POLICY="gr1"; RUN_SCRIPT="scripts/calvin/run_gr1_calvin_smoke.sh" ;;
  2) POLICY="roboflamingo"; RUN_SCRIPT="scripts/calvin/run_roboflamingo_calvin_smoke.sh" ;;
  *) echo "Unexpected POLICY_IDX=${POLICY_IDX} from task ${TASK_ID}" >&2; exit 2 ;;
esac

EVAL_START=$((CHUNK_ID * CHUNK_SIZE))
EVAL_END=$(((CHUNK_ID + 1) * CHUNK_SIZE))
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-manual}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-manual}"
RUN_TAG="${POLICY}_abc_d_${RUN_KIND}_${CONDITION_TAG}_${NUM_SEQUENCES}seq_chunk${CHUNK_ID}_${EVAL_START}_${EVAL_END}_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}"
RESULTS_DIR="${LAYER2_RESULTS_ROOT}/${CONDITION_TAG}/${POLICY}/${RUN_TAG}"

manifest_info="$(
  PROJECT_ROOT="${PROJECT_ROOT}" MANIFEST_LIST="${MANIFEST_LIST}" CONDITION_TAG="${CONDITION_TAG}" "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

EXPECTED = {
    "layer2_fresh_seed2026052601": {
        "sha256": "f2a339b2ec42320cac1b001aa6763d477015f26f55089528914f9ecb5a137e93",
        "seed": 2026052601,
        "filename": "abc_d_layer2_fresh_1000seq_seed2026052601.json",
    },
    "layer2_fresh_seed2026052602": {
        "sha256": "bf2b9a19c73b8a54c5c2aa5f85ef26210df508c41e9a37385da16b5da1f73ee6",
        "seed": 2026052602,
        "filename": "abc_d_layer2_fresh_1000seq_seed2026052602.json",
    },
}
SHA_FIELDS = ("sha256", "manifest_sha256", "sequence_manifest_sha256")
TAG_FIELDS = ("condition_tag", "condition", "tag", "name")
CURATED_PACKAGE_REL = "paper_artifacts/calvin_creeping_overfitting/layer2_fresh_1000_20260526"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entries(payload):
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield dict(item)
        return
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest list must be a dict or list, got {type(payload).__name__}")
    for key in ("manifests", "manifest_list", "layer2_fresh_manifests"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield dict(item)
            return
        if isinstance(value, dict):
            for tag, item in value.items():
                if isinstance(item, dict):
                    row = dict(item)
                else:
                    row = {"manifest_path": item}
                row.setdefault("condition_tag", tag)
                yield row
            return
    for tag, item in payload.items():
        if str(tag).startswith("layer2_fresh_seed"):
            if isinstance(item, dict):
                row = dict(item)
            else:
                row = {"manifest_path": item}
            row.setdefault("condition_tag", tag)
            yield row


def first(row, fields):
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def require_under(path: Path, parent: Path, context: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(f"{context}: selected manifest is outside curated package: {path}") from exc


project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
list_path = Path(os.environ["MANIFEST_LIST"]).resolve()
curated_dir = (project_root / CURATED_PACKAGE_REL).resolve()
condition_tag = os.environ["CONDITION_TAG"]
payload = json.loads(list_path.read_text())
selected = None
for row in entries(payload):
    tag = first(row, TAG_FIELDS)
    seed = row.get("sample_seed", row.get("seed"))
    if tag == condition_tag or str(seed) == str(EXPECTED[condition_tag]["seed"]):
        selected = row
        break
if selected is None:
    raise RuntimeError(f"{list_path} does not contain required condition tag {condition_tag}")

manifest_path_raw = selected.get("curated_repo_relative_path")
if not manifest_path_raw:
    raise RuntimeError(f"{condition_tag}: manifest list entry lacks curated_repo_relative_path; source_path/source_absolute_path are provenance-only")
manifest_path_rel = Path(str(manifest_path_raw))
if manifest_path_rel.is_absolute():
    raise RuntimeError(f"{condition_tag}: curated_repo_relative_path must be repo-relative, got {manifest_path_raw}")
manifest_path = (project_root / manifest_path_rel).resolve()
require_under(manifest_path, curated_dir, condition_tag)
if manifest_path.name != EXPECTED[condition_tag]["filename"]:
    raise RuntimeError(f"{condition_tag}: selected manifest filename {manifest_path.name} != expected {EXPECTED[condition_tag]['filename']}")
if not manifest_path.is_file():
    raise RuntimeError(f"{condition_tag}: manifest file from Worker A list does not exist: {manifest_path}")

listed_sha = first(selected, SHA_FIELDS)
if not listed_sha:
    raise RuntimeError(f"{condition_tag}: manifest list entry lacks sha256")
expected_sha = EXPECTED[condition_tag]["sha256"]
if str(listed_sha) != expected_sha:
    raise RuntimeError(f"{condition_tag}: listed sha256 {listed_sha} != expected {expected_sha}")
actual_sha = sha256(manifest_path)
if actual_sha != expected_sha:
    raise RuntimeError(f"{condition_tag}: actual sha256 {actual_sha} != expected {expected_sha}: {manifest_path}")

manifest = json.loads(manifest_path.read_text())
if manifest.get("sample_kind") != "layer2_fresh_stratified":
    raise RuntimeError(f"{condition_tag}: sample_kind {manifest.get('sample_kind')!r} is not layer2_fresh_stratified")
if int(manifest.get("num_sequences", -1)) != 1000:
    raise RuntimeError(f"{condition_tag}: num_sequences {manifest.get('num_sequences')} is not 1000")
if int(manifest.get("generation", {}).get("sample_seed", -1)) != EXPECTED[condition_tag]["seed"]:
    raise RuntimeError(f"{condition_tag}: manifest sample seed mismatch")
if manifest.get("calvin_reset_bank_used") is not False:
    raise RuntimeError(f"{condition_tag}: manifest indicates calvin_reset_bank_used={manifest.get('calvin_reset_bank_used')!r}")
rows = manifest.get("sequences")
if not isinstance(rows, list) or len(rows) != 1000:
    raise RuntimeError(f"{condition_tag}: manifest rows are not exactly 1000")
indices = sorted(int(row.get("global_index", row.get("index", -1))) for row in rows)
if indices != list(range(1000)):
    raise RuntimeError(f"{condition_tag}: manifest global_index values are not exactly 0..999")
print(str(manifest_path), expected_sha, "layer2_fresh_stratified", EXPECTED[condition_tag]["seed"])
PY
)"
read -r SEQUENCE_MANIFEST MANIFEST_SHA256 MANIFEST_SAMPLE_KIND MANIFEST_SEED <<< "${manifest_info}"

export NUM_SEQUENCES
export EVAL_START
export EVAL_END
export EVAL_COUNT="${CHUNK_SIZE}"
export CHUNK_ID
export CHUNK_SIZE
export GLOBAL_INDEX_START="${EVAL_START}"
export GLOBAL_INDEX_END_EXCLUSIVE="${EVAL_END}"
export CALVIN_GLOBAL_INDEX_START="${EVAL_START}"
export CALVIN_GLOBAL_INDEX_END_EXCLUSIVE="${EVAL_END}"
export CALVIN_CHUNK_INDEX="${CHUNK_ID}"
export CALVIN_CHUNK_START="${EVAL_START}"
export CALVIN_CHUNK_COUNT="${CHUNK_SIZE}"
export CALVIN_SEQUENCE_MANIFEST="${SEQUENCE_MANIFEST}"
export CALVIN_SEQUENCE_MANIFEST_SHA256="${MANIFEST_SHA256}"
export CALVIN_SEQUENCE_SAMPLE_KIND="${MANIFEST_SAMPLE_KIND}"
export CALVIN_SEQUENCE_MODE="manifest_direct"
export CALVIN_LAYER2_FRESH="1"
export CALVIN_LAYER2_CONDITION_TAG="${CONDITION_TAG}"
export CALVIN_CONDITION_TAG="${CONDITION_TAG}"
export CALVIN_SAMPLE_KIND="${MANIFEST_SAMPLE_KIND}"
export XVLA_EP_LEN=360
export REQUIRE_XVLA_EP_LEN=360
export RUN_KIND
export RUN_TAG
export RESULTS_DIR
export POLICY

mkdir -p "${RESULTS_DIR}"
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULTS_DIR"])
payload = {
    "launcher": "scripts/slurm/calvin_layer2_fresh_1000_array.sbatch",
    "run_kind": os.environ["RUN_KIND"],
    "run_tag": os.environ["RUN_TAG"],
    "policy": os.environ["POLICY"],
    "condition_tag": os.environ["CALVIN_LAYER2_CONDITION_TAG"],
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "manual"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "manual"),
    "num_sequences": int(os.environ["NUM_SEQUENCES"]),
    "chunk_id": int(os.environ["CHUNK_ID"]),
    "chunk_size": int(os.environ["CHUNK_SIZE"]),
    "chunk_count": int(os.environ["EVAL_COUNT"]),
    "eval_start": int(os.environ["EVAL_START"]),
    "eval_end": int(os.environ["EVAL_END"]),
    "global_index_start": int(os.environ["GLOBAL_INDEX_START"]),
    "global_index_end_exclusive": int(os.environ["GLOBAL_INDEX_END_EXCLUSIVE"]),
    "calvin_sequence_manifest": os.environ["CALVIN_SEQUENCE_MANIFEST"],
    "calvin_sequence_manifest_sha256": os.environ["CALVIN_SEQUENCE_MANIFEST_SHA256"],
    "calvin_sequence_sample_kind": os.environ["CALVIN_SEQUENCE_SAMPLE_KIND"],
    "calvin_sequence_mode": os.environ["CALVIN_SEQUENCE_MODE"],
    "calvin_layer2_fresh": os.environ["CALVIN_LAYER2_FRESH"],
    "xvla_ep_len": int(os.environ["XVLA_EP_LEN"]),
    "forbidden_reset_envs": {
        "CALVIN_RESET_BANK": os.environ.get("CALVIN_RESET_BANK", ""),
        "RESET_BANK": os.environ.get("RESET_BANK", ""),
        "CALVIN_RESET_PROTOCOL": os.environ.get("CALVIN_RESET_PROTOCOL", ""),
    },
}
tmp = result_dir / "layer2_chunk_metadata.json.tmp"
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(result_dir / "layer2_chunk_metadata.json")
PY

echo "policy=${POLICY}"
echo "condition_tag=${CONDITION_TAG}"
echo "run_kind=${RUN_KIND}"
echo "chunk_id=${CHUNK_ID}"
echo "eval_start=${EVAL_START}"
echo "eval_end=${EVAL_END}"
echo "run_tag=${RUN_TAG}"
echo "results_dir=${RESULTS_DIR}"
echo "calvin_sequence_manifest=${CALVIN_SEQUENCE_MANIFEST}"
echo "calvin_sequence_manifest_sha256=${CALVIN_SEQUENCE_MANIFEST_SHA256}"
echo "calvin_sequence_sample_kind=${CALVIN_SEQUENCE_SAMPLE_KIND}"
echo "xvla_ep_len=${XVLA_EP_LEN}"
echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"

bash "${RUN_SCRIPT}"
