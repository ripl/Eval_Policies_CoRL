#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-protocol_abcde_full_288_20260525T080007Z}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/simplerenv/protocol_abcde_full_288/${SOURCE_RUN_ID}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPORT_ROOT="${SUPPORT_ROOT:-${SCRIPT_DIR}}"
REPLACEMENT_ID="${REPLACEMENT_ID:-dexbotic_replacement_r2_$(date -u +%Y%m%dT%H%M%SZ)}"
REPLACEMENT_ROOT="${REPLACEMENT_ROOT:-${RUN_ROOT}/dexbotic_replacements/${REPLACEMENT_ID}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs/slurm/${REPLACEMENT_ID}}"
MANIFEST="${MANIFEST:-${RUN_ROOT}/bundle/${REPLACEMENT_ID}_submission.tsv}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${RUN_ROOT}/bundle/${REPLACEMENT_ID}_preflight_report.json}"
ASSET_PREFLIGHT_REPORT="${ASSET_PREFLIGHT_REPORT:-${RUN_ROOT}/bundle/${REPLACEMENT_ID}_asset_preflight/dexbotic_runtime_asset_preflight.json}"
PROTOCOL_SHA256="1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"
DEXBOTIC_NORM_STATS_SOURCE="${DEXBOTIC_NORM_STATS_SOURCE:-${PROJECT_ROOT}/results/simplerenv/strict_calibrations_20260523/dexbotic_official_workspace_strict_20260523T015147Z/run/norm_stats.json}"
DEXBOTIC_NORM_STATS_SHA256="${DEXBOTIC_NORM_STATS_SHA256:-ddbeb68786543c68f8fa198c33fd9a265025d35c5f7ebba144d6ec9c655693e6}"
DEXBOTIC_MS2_REAL2SIM_ASSET_DIR="${DEXBOTIC_MS2_REAL2SIM_ASSET_DIR:-/workspace/simpler/ManiSkill2_real2sim/data}"
DRY_RUN=0
PREFLIGHT_ONLY=0
ALLOW_SUBMIT=0

conditions=(
  protocol_C2_blue_on_red
  protocol_C3_red_on_blue
  protocol_D
  protocol_E
)

declare -A failed_r1_jobs=(
  [protocol_C2_blue_on_red]=2076515
  [protocol_C3_red_on_blue]=2076516
  [protocol_D]=2076517
  [protocol_E]=2076518
)

usage() {
  cat <<'EOF'
Usage: submit_dexbotic_r2_replacements.sh [--dry-run] [--preflight-only] [--allow-submit]

Submits exactly four Dexbotic r2 replacement jobs for Protocol C2/C3/D/E.
No Slurm arrays are used.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --allow-submit) ALLOW_SUBMIT=1; shift ;;
    --replacement-id)
      REPLACEMENT_ID="$2"
      REPLACEMENT_ROOT="${RUN_ROOT}/dexbotic_replacements/${REPLACEMENT_ID}"
      LOG_DIR="${RUN_ROOT}/logs/slurm/${REPLACEMENT_ID}"
      MANIFEST="${RUN_ROOT}/bundle/${REPLACEMENT_ID}_submission.tsv"
      PREFLIGHT_REPORT="${RUN_ROOT}/bundle/${REPLACEMENT_ID}_preflight_report.json"
      ASSET_PREFLIGHT_REPORT="${RUN_ROOT}/bundle/${REPLACEMENT_ID}_asset_preflight/dexbotic_runtime_asset_preflight.json"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 2
  fi
}

require_no_array() {
  if grep -Eq '^[[:space:]]*#SBATCH[[:space:]]+(-a|--array)([=[:space:]]|$)' "$1"; then
    echo "Slurm array directive forbidden in $1" >&2
    exit 2
  fi
}

condition_short() {
  local short="${1#protocol_}"
  printf '%s' "${short//_/-}"
}

job_name_for() {
  printf 'se_abcde_dexbotic_%s_r2' "$(condition_short "$1")"
}

r1_result_dir_for() {
  printf '%s/dexbotic_replacements/dexbotic_replacement_20260525T104356Z/%s' "${RUN_ROOT}" "$1"
}

r2_result_dir_for() {
  printf '%s/%s' "${REPLACEMENT_ROOT}" "$1"
}

r1_stdout_for() {
  local condition="$1"
  local short
  short="$(condition_short "${condition}")"
  printf '%s/logs/slurm/dexbotic_replacement_20260525T104356Z/se_abcde_dexbotic_%s_r1-%s.out' "${RUN_ROOT}" "${short}" "${failed_r1_jobs[${condition}]}"
}

r1_stderr_for() {
  local condition="$1"
  local short
  short="$(condition_short "${condition}")"
  printf '%s/logs/slurm/dexbotic_replacement_20260525T104356Z/se_abcde_dexbotic_%s_r1-%s.err' "${RUN_ROOT}" "${short}" "${failed_r1_jobs[${condition}]}"
}

export_vars_for() {
  local condition="$1"
  local output_root="$2"
  printf 'ALL,PROJECT_ROOT=%s,SUPPORT_ROOT=%s,RUN_ID=%s,RUN_ROOT=%s,POLICY=dexbotic,CONDITION=%s,OUTPUT_ROOT=%s,ASSET_MANIFEST=%s,SIMPLERENV_PROTOCOL_CONDITION=%s,SIMPLERENV_PROTOCOL_CONFIG=%s,SIMPLERENV_PROTOCOL_SHA256=%s,DEXBOTIC_NORM_STATS_SOURCE=%s,DEXBOTIC_NORM_STATS_SHA256=%s,DEXBOTIC_MS2_REAL2SIM_ASSET_DIR=%s,PYTHONDONTWRITEBYTECODE=1' \
    "${PROJECT_ROOT}" \
    "${SUPPORT_ROOT}" \
    "${SOURCE_RUN_ID}" \
    "${RUN_ROOT}" \
    "${condition}" \
    "${output_root}" \
    "${RUN_ROOT}/bundle/runtime_asset_manifest.json" \
    "${condition}" \
    "${PROJECT_ROOT}/configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json" \
    "${PROTOCOL_SHA256}" \
    "${DEXBOTIC_NORM_STATS_SOURCE}" \
    "${DEXBOTIC_NORM_STATS_SHA256}" \
    "${DEXBOTIC_MS2_REAL2SIM_ASSET_DIR}"
}

write_preflight_report() {
  mkdir -p "$(dirname "${PREFLIGHT_REPORT}")"
  PREFLIGHT_REPORT="${PREFLIGHT_REPORT}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  SUPPORT_ROOT="${SUPPORT_ROOT}" \
  REPLACEMENT_ID="${REPLACEMENT_ID}" \
  REPLACEMENT_ROOT="${REPLACEMENT_ROOT}" \
  LOG_DIR="${LOG_DIR}" \
  MANIFEST="${MANIFEST}" \
  ASSET_PREFLIGHT_REPORT="${ASSET_PREFLIGHT_REPORT}" \
  DEXBOTIC_MS2_REAL2SIM_ASSET_DIR="${DEXBOTIC_MS2_REAL2SIM_ASSET_DIR}" \
  DEXBOTIC_NORM_STATS_SOURCE="${DEXBOTIC_NORM_STATS_SOURCE}" \
  DEXBOTIC_NORM_STATS_SHA256="${DEXBOTIC_NORM_STATS_SHA256}" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

conditions = [
    "protocol_C2_blue_on_red",
    "protocol_C3_red_on_blue",
    "protocol_D",
    "protocol_E",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

support = Path(os.environ["SUPPORT_ROOT"])
payload = {
    "status": "passed",
    "policy": "dexbotic",
    "source_run_root": os.environ["RUN_ROOT"],
    "replacement_id": os.environ["REPLACEMENT_ID"],
    "replacement_root": os.environ["REPLACEMENT_ROOT"],
    "log_dir": os.environ["LOG_DIR"],
    "submission_manifest": os.environ["MANIFEST"],
    "asset_preflight_report": os.environ["ASSET_PREFLIGHT_REPORT"],
    "conditions": conditions,
    "job_count": 4,
    "no_arrays": True,
    "episode_ids": [0, 287],
    "episodes_per_job": 288,
    "horizon": 60,
    "protocol_sha256": "1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b",
    "fix": "Dexbotic benchmark container sets MS2_REAL2SIM_ASSET_DIR to the staged /workspace/simpler/ManiSkill2_real2sim/data asset tree.",
    "ms2_real2sim_asset_dir": os.environ["DEXBOTIC_MS2_REAL2SIM_ASSET_DIR"],
    "support_files": {
        str(path.relative_to(support)): sha256(path)
        for path in sorted(support.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    },
    "norm_stats_source": os.environ["DEXBOTIC_NORM_STATS_SOURCE"],
    "norm_stats_sha256": os.environ["DEXBOTIC_NORM_STATS_SHA256"],
}
Path(os.environ["PREFLIGHT_REPORT"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

check_asset_preflight_if_submitting() {
  if [[ "${ALLOW_SUBMIT}" != "1" ]]; then
    return
  fi
  require_file "${ASSET_PREFLIGHT_REPORT}"
  python3 - "${ASSET_PREFLIGHT_REPORT}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("status") != "passed":
    raise SystemExit(f"asset preflight did not pass: {payload.get('status')} {payload.get('errors')}")
missing = payload.get("condition_episode0_missing_model_ids") or {}
bad = {k: v for k, v in missing.items() if v}
if bad:
    raise SystemExit(f"asset preflight missing condition model IDs: {bad}")
print("asset_preflight_status=passed")
PY
}

check_static() {
  require_file "${SUPPORT_ROOT}/protocol_abcde_common.py"
  require_file "${SUPPORT_ROOT}/run_dexbotic_condition.sh"
  require_file "${SUPPORT_ROOT}/run_dexbotic_condition.sbatch"
  require_file "${SUPPORT_ROOT}/make_dexbotic_abcde_config.py"
  require_file "${SUPPORT_ROOT}/seeded_runpy_with_protocol.py"
  require_file "${SUPPORT_ROOT}/bin/launch_memvla_server_seeded.py"
  require_file "${SUPPORT_ROOT}/check_dexbotic_runtime_assets.py"
  require_file "${SUPPORT_ROOT}/run_dexbotic_asset_preflight.sbatch"
  require_no_array "${SUPPORT_ROOT}/run_dexbotic_condition.sbatch"
  require_no_array "${SUPPORT_ROOT}/run_dexbotic_asset_preflight.sbatch"
  bash -n "${SUPPORT_ROOT}/run_dexbotic_condition.sh"
  bash -n "${SUPPORT_ROOT}/run_dexbotic_condition.sbatch"
  bash -n "${SUPPORT_ROOT}/run_dexbotic_asset_preflight.sbatch"
  bash -n "${SUPPORT_ROOT}/submit_dexbotic_r2_replacements.sh"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${SUPPORT_ROOT}:${SUPPORT_ROOT}/bin:${PYTHONPATH:-}" python3 -m py_compile \
    "${SUPPORT_ROOT}/protocol_abcde_common.py" \
    "${SUPPORT_ROOT}/make_dexbotic_abcde_config.py" \
    "${SUPPORT_ROOT}/seeded_runpy_with_protocol.py" \
    "${SUPPORT_ROOT}/bin/launch_memvla_server_seeded.py" \
    "${SUPPORT_ROOT}/check_dexbotic_runtime_assets.py"
  PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 - <<'PY'
from protocol_abcde_common import EPISODE_IDS, HORIZON, load_protocol_config
load_protocol_config()
assert EPISODE_IDS == tuple(range(288)), EPISODE_IDS[:3]
assert HORIZON == 60, HORIZON
PY
  if [[ ! -s "${DEXBOTIC_NORM_STATS_SOURCE}" ]]; then
    echo "missing norm_stats source: ${DEXBOTIC_NORM_STATS_SOURCE}" >&2
    exit 2
  fi
  actual_norm_sha="$(sha256sum "${DEXBOTIC_NORM_STATS_SOURCE}" | awk '{print $1}')"
  if [[ "${actual_norm_sha}" != "${DEXBOTIC_NORM_STATS_SHA256}" ]]; then
    echo "norm_stats SHA mismatch: ${actual_norm_sha} != ${DEXBOTIC_NORM_STATS_SHA256}" >&2
    exit 2
  fi
  require_file "${RUN_ROOT}/bundle/runtime_asset_manifest.json"
  require_file "${RUN_ROOT}/bundle/launch_preflight_report.json"
  for condition in "${conditions[@]}"; do
    require_file "$(r1_stdout_for "${condition}")"
    require_file "$(r1_stderr_for "${condition}")"
    if [[ ! -d "$(r1_result_dir_for "${condition}")" ]]; then
      echo "missing r1 failed result dir: $(r1_result_dir_for "${condition}")" >&2
      exit 2
    fi
    if [[ -e "$(r2_result_dir_for "${condition}")" ]]; then
      echo "refusing to reuse r2 result dir: $(r2_result_dir_for "${condition}")" >&2
      exit 2
    fi
  done
  check_asset_preflight_if_submitting
  write_preflight_report
}

dry_run_commands() {
  local count=0
  for condition in "${conditions[@]}"; do
    local output_root job_name export_vars
    output_root="$(r2_result_dir_for "${condition}")"
    job_name="$(job_name_for "${condition}")"
    export_vars="$(export_vars_for "${condition}" "${output_root}")"
    printf 'sbatch --parsable --no-requeue -J %q --output=%q --error=%q --export=%q %q\n' \
      "${job_name}" \
      "${LOG_DIR}/${job_name}-%j.out" \
      "${LOG_DIR}/${job_name}-%j.err" \
      "${export_vars}" \
      "${SUPPORT_ROOT}/run_dexbotic_condition.sbatch"
    count=$((count + 1))
  done
  if [[ "${count}" != "4" ]]; then
    echo "ERROR: dry-run command count ${count}, expected 4" >&2
    exit 1
  fi
  echo "dry_run_count=4"
  echo "preflight_report=${PREFLIGHT_REPORT}"
}

submit_one() {
  local condition="$1"
  local output_root job_name export_vars
  output_root="$(r2_result_dir_for "${condition}")"
  job_name="$(job_name_for "${condition}")"
  export_vars="$(export_vars_for "${condition}" "${output_root}")"
  sbatch --parsable --no-requeue -J "${job_name}" \
    --output="${LOG_DIR}/${job_name}-%j.out" \
    --error="${LOG_DIR}/${job_name}-%j.err" \
    --export="${export_vars}" \
    "${SUPPORT_ROOT}/run_dexbotic_condition.sbatch"
}

check_static

if [[ "${DRY_RUN}" == "1" ]]; then
  dry_run_commands
  exit 0
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "preflight_status=passed"
  echo "preflight_report=${PREFLIGHT_REPORT}"
  exit 0
fi

if [[ "${ALLOW_SUBMIT}" != "1" ]]; then
  echo "refusing to submit without --allow-submit" >&2
  exit 2
fi
if [[ -e "${MANIFEST}" ]]; then
  echo "refusing to overwrite r2 replacement manifest: ${MANIFEST}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "$(dirname "${MANIFEST}")" "${REPLACEMENT_ROOT}"
tmp="${MANIFEST}.tmp"
printf 'replacement_job_id\tfailed_r1_job_id\tpolicy\tcondition\tjob_name\tr1_result_dir\tr2_result_dir\tr1_stdout_log\tr1_stderr_log\tsbatch_script\tstdout_log\tstderr_log\tprotocol_sha256\tsubmitted_at_utc\tsource_run_id\treplacement_id\tsupport_root\tasset_preflight_report\tfix_summary\n' > "${tmp}"
count=0
for condition in "${conditions[@]}"; do
  job_name="$(job_name_for "${condition}")"
  job_id="$(submit_one "${condition}")"
  job_num="${job_id%%;*}"
  printf '%s\t%s\tdexbotic\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" \
    "${failed_r1_jobs[${condition}]}" \
    "${condition}" \
    "${job_name}" \
    "$(r1_result_dir_for "${condition}")" \
    "$(r2_result_dir_for "${condition}")" \
    "$(r1_stdout_for "${condition}")" \
    "$(r1_stderr_for "${condition}")" \
    "${SUPPORT_ROOT}/run_dexbotic_condition.sbatch" \
    "${LOG_DIR}/${job_name}-${job_num}.out" \
    "${LOG_DIR}/${job_name}-${job_num}.err" \
    "${PROTOCOL_SHA256}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${SOURCE_RUN_ID}" \
    "${REPLACEMENT_ID}" \
    "${SUPPORT_ROOT}" \
    "${ASSET_PREFLIGHT_REPORT}" \
    "set MS2_REAL2SIM_ASSET_DIR=/workspace/simpler/ManiSkill2_real2sim/data for Dexbotic benchmark runtime" >> "${tmp}"
  count=$((count + 1))
done
if [[ "${count}" != "4" ]]; then
  echo "ERROR: submitted ${count} jobs, expected 4" >&2
  exit 1
fi
mv "${tmp}" "${MANIFEST}"
echo "r2_replacement_manifest=${MANIFEST}"
