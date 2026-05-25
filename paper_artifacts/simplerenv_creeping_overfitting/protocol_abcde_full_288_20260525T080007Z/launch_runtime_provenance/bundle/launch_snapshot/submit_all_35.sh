#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${PROJECT_ROOT}/scratch/protocol_abcde_full_288_20260525}"
RUN_ID="${RUN_ID:-protocol_abcde_full_288_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/simplerenv/protocol_abcde_full_288/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs/slurm}"
SUBMISSION_TSV="${SUBMISSION_TSV:-${RUN_ROOT}/submission.tsv}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${RUN_ROOT}/bundle/launch_snapshot}"
DRY_RUN=0
PREFLIGHT_ONLY=0
ALLOW_SUBMIT=0

usage() {
  cat <<'EOF'
Usage: bash submit_all_35.sh [--dry-run] [--preflight-only] [--allow-submit] [--run-id RUN_ID]

Dry-run prints exactly 35 sbatch commands, one for each policy x condition.
Non-dry submission is gated by --allow-submit and submits no Slurm arrays.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --allow-submit) ALLOW_SUBMIT=1; shift ;;
    --run-id)
      RUN_ID="$2"
      RUN_ROOT="${PROJECT_ROOT}/results/simplerenv/protocol_abcde_full_288/${RUN_ID}"
      LOG_DIR="${RUN_ROOT}/logs/slurm"
      SUBMISSION_TSV="${RUN_ROOT}/submission.tsv"
      SNAPSHOT_DIR="${RUN_ROOT}/bundle/launch_snapshot"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

policies=(cogact spatialvla internvla_m1 xvla dexbotic)
conditions=(protocol_A protocol_B protocol_C1_yellow_on_green protocol_C2_blue_on_red protocol_C3_red_on_blue protocol_D protocol_E)

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

check_static() {
  require_file "${SUPPORT_ROOT}/protocol_abcde_common.py"
  require_file "${SUPPORT_ROOT}/stage_protocol_assets.py"
  require_file "${SUPPORT_ROOT}/preflight_launch_shape.py"
  require_file "${SUPPORT_ROOT}/run_cogact_spatialvla_condition.py"
  require_file "${SUPPORT_ROOT}/run_cogact_spatialvla_condition.sbatch"
  require_file "${SUPPORT_ROOT}/run_internvla_condition.py"
  require_file "${SUPPORT_ROOT}/run_internvla_condition.sbatch"
  require_file "${SUPPORT_ROOT}/run_seeded_simpler_episode_with_protocol.py"
  require_file "${SUPPORT_ROOT}/run_xvla_condition.sh"
  require_file "${SUPPORT_ROOT}/run_xvla_condition.sbatch"
  require_file "${SUPPORT_ROOT}/run_dexbotic_condition.sh"
  require_file "${SUPPORT_ROOT}/run_dexbotic_condition.sbatch"
  require_file "${SUPPORT_ROOT}/make_dexbotic_abcde_config.py"
  require_file "${SUPPORT_ROOT}/seeded_runpy_with_protocol.py"
  require_no_array "${SUPPORT_ROOT}/run_cogact_spatialvla_condition.sbatch"
  require_no_array "${SUPPORT_ROOT}/run_internvla_condition.sbatch"
  require_no_array "${SUPPORT_ROOT}/run_xvla_condition.sbatch"
  require_no_array "${SUPPORT_ROOT}/run_dexbotic_condition.sbatch"
  PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 "${SUPPORT_ROOT}/stage_protocol_assets.py" --source-only >/dev/null
  PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 -m py_compile \
    "${SUPPORT_ROOT}/protocol_abcde_common.py" \
    "${SUPPORT_ROOT}/stage_protocol_assets.py" \
    "${SUPPORT_ROOT}/preflight_launch_shape.py" \
    "${SUPPORT_ROOT}/run_cogact_spatialvla_condition.py" \
    "${SUPPORT_ROOT}/run_internvla_condition.py" \
    "${SUPPORT_ROOT}/run_seeded_simpler_episode_with_protocol.py" \
    "${SUPPORT_ROOT}/make_dexbotic_abcde_config.py" \
    "${SUPPORT_ROOT}/seeded_runpy_with_protocol.py"
  for shell_file in \
    "${SUPPORT_ROOT}/submit_all_35.sh" \
    "${SUPPORT_ROOT}/run_xvla_condition.sh" \
    "${SUPPORT_ROOT}/run_dexbotic_condition.sh" \
    "${SUPPORT_ROOT}/run_cogact_spatialvla_condition.sbatch" \
    "${SUPPORT_ROOT}/run_internvla_condition.sbatch" \
    "${SUPPORT_ROOT}/run_xvla_condition.sbatch" \
    "${SUPPORT_ROOT}/run_dexbotic_condition.sbatch"; do
    bash -n "${shell_file}"
  done
}

output_root_for() {
  local policy="$1"
  local condition="$2"
  printf '%s/%s/%s' "${RUN_ROOT}" "${policy}" "${condition}"
}

sbatch_script_for() {
  local policy="$1"
  case "${policy}" in
    cogact|spatialvla) printf '%s/run_cogact_spatialvla_condition.sbatch' "${SNAPSHOT_DIR}" ;;
    internvla_m1) printf '%s/run_internvla_condition.sbatch' "${SNAPSHOT_DIR}" ;;
    xvla) printf '%s/run_xvla_condition.sbatch' "${SNAPSHOT_DIR}" ;;
    dexbotic) printf '%s/run_dexbotic_condition.sbatch' "${SNAPSHOT_DIR}" ;;
    *) echo "unsupported policy ${policy}" >&2; return 2 ;;
  esac
}

job_name_for() {
  local policy="$1"
  local condition="$2"
  local short="${condition#protocol_}"
  short="${short//_/-}"
  printf 'se_abcde_%s_%s' "${policy}" "${short}"
}

export_vars_for() {
  local policy="$1"
  local condition="$2"
  local output_root="$3"
  printf 'ALL,PROJECT_ROOT=%s,SUPPORT_ROOT=%s,RUN_ID=%s,RUN_ROOT=%s,POLICY=%s,CONDITION=%s,OUTPUT_ROOT=%s,ASSET_MANIFEST=%s,SIMPLERENV_PROTOCOL_CONDITION=%s,SIMPLERENV_PROTOCOL_CONFIG=%s,SIMPLERENV_PROTOCOL_SHA256=%s,PYTHONDONTWRITEBYTECODE=1' \
    "${PROJECT_ROOT}" \
    "${SNAPSHOT_DIR}" \
    "${RUN_ID}" \
    "${RUN_ROOT}" \
    "${policy}" \
    "${condition}" \
    "${output_root}" \
    "${RUN_ROOT}/bundle/runtime_asset_manifest.json" \
    "${condition}" \
    "${PROJECT_ROOT}/configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json" \
    "1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"
}

dry_run_commands() {
  local count=0
  for policy in "${policies[@]}"; do
    for condition in "${conditions[@]}"; do
      local output_root job_name script export_vars
      output_root="$(output_root_for "${policy}" "${condition}")"
      job_name="$(job_name_for "${policy}" "${condition}")"
      script="$(sbatch_script_for "${policy}")"
      export_vars="$(export_vars_for "${policy}" "${condition}" "${output_root}")"
      printf 'sbatch --parsable --no-requeue -J %q --output=%q --error=%q --export=%q %q\n' \
        "${job_name}" \
        "${LOG_DIR}/${job_name}-%j.out" \
        "${LOG_DIR}/${job_name}-%j.err" \
        "${export_vars}" \
        "${script}"
      count=$((count + 1))
    done
  done
  if [[ "${count}" != "35" ]]; then
    echo "ERROR: dry-run command count ${count}, expected 35" >&2
    exit 1
  fi
  echo "dry_run_count=35"
}

create_snapshot() {
  if [[ -e "${SNAPSHOT_DIR}" ]]; then
    echo "refusing to reuse launch snapshot: ${SNAPSHOT_DIR}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${SNAPSHOT_DIR}")"
  rsync -a --exclude __pycache__ --exclude '*.pyc' "${SUPPORT_ROOT}/" "${SNAPSHOT_DIR}/"
  find "${SNAPSHOT_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/bundle/submission_support_files.sha256"
  chmod -R a-w "${SNAPSHOT_DIR}"
}

submit_one() {
  local policy="$1"
  local condition="$2"
  local output_root job_name script export_vars
  output_root="$(output_root_for "${policy}" "${condition}")"
  job_name="$(job_name_for "${policy}" "${condition}")"
  script="$(sbatch_script_for "${policy}")"
  export_vars="$(export_vars_for "${policy}" "${condition}" "${output_root}")"
  sbatch --parsable --no-requeue -J "${job_name}" --output="${LOG_DIR}/${job_name}-%j.out" --error="${LOG_DIR}/${job_name}-%j.err" --export="${export_vars}" "${script}"
}

check_static

if [[ "${DRY_RUN}" == "1" ]]; then
  mkdir -p "${RUN_ROOT}/bundle"
  ASSET_MANIFEST="${RUN_ROOT}/bundle/runtime_asset_manifest.json"
  PREFLIGHT_REPORT="${RUN_ROOT}/bundle/launch_preflight_report.json"
  PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 "${SUPPORT_ROOT}/stage_protocol_assets.py" \
    --allow-missing-targets \
    --manifest "${ASSET_MANIFEST}" >/dev/null
  PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 "${SUPPORT_ROOT}/preflight_launch_shape.py" \
    --support-root "${SUPPORT_ROOT}" \
    --asset-manifest "${ASSET_MANIFEST}" \
    --report "${PREFLIGHT_REPORT}" >/dev/null
  dry_run_commands
  echo "asset_manifest=${ASSET_MANIFEST}"
  echo "preflight_report=${PREFLIGHT_REPORT}"
  exit 0
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  mkdir -p "${RUN_ROOT}/bundle"
  ASSET_MANIFEST="${RUN_ROOT}/bundle/runtime_asset_manifest.json"
  PREFLIGHT_REPORT="${RUN_ROOT}/bundle/launch_preflight_report.json"
  PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 "${SUPPORT_ROOT}/stage_protocol_assets.py" \
    --allow-missing-targets \
    --manifest "${ASSET_MANIFEST}" >/dev/null
  PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 "${SUPPORT_ROOT}/preflight_launch_shape.py" \
    --support-root "${SUPPORT_ROOT}" \
    --asset-manifest "${ASSET_MANIFEST}" \
    --report "${PREFLIGHT_REPORT}" >/dev/null
  echo "preflight_status=passed"
  echo "asset_manifest=${ASSET_MANIFEST}"
  echo "preflight_report=${PREFLIGHT_REPORT}"
  exit 0
fi

if [[ "${ALLOW_SUBMIT}" != "1" ]]; then
  echo "refusing to submit without --allow-submit; wait for Worker A and critic gates" >&2
  exit 2
fi
if [[ -e "${SUBMISSION_TSV}" ]]; then
  echo "refusing to overwrite submission TSV: ${SUBMISSION_TSV}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${RUN_ROOT}/bundle"
ASSET_MANIFEST="${RUN_ROOT}/bundle/runtime_asset_manifest.json"
PREFLIGHT_REPORT="${RUN_ROOT}/bundle/launch_preflight_report.json"
PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 "${SUPPORT_ROOT}/stage_protocol_assets.py" --write --manifest "${ASSET_MANIFEST}"
PYTHONPATH="${SUPPORT_ROOT}:${PYTHONPATH:-}" python3 "${SUPPORT_ROOT}/preflight_launch_shape.py" \
  --support-root "${SUPPORT_ROOT}" \
  --asset-manifest "${ASSET_MANIFEST}" \
  --report "${PREFLIGHT_REPORT}" >/dev/null
create_snapshot

tmp="${SUBMISSION_TSV}.tmp"
printf 'job_id\tpolicy\tcondition\tjob_name\tresult_dir\tsbatch_script\tstdout_log\tstderr_log\tprotocol_sha256\tsubmitted_at_utc\trun_id\n' > "${tmp}"
count=0
for policy in "${policies[@]}"; do
  for condition in "${conditions[@]}"; do
    job_name="$(job_name_for "${policy}" "${condition}")"
    job_id="$(submit_one "${policy}" "${condition}")"
    job_num="${job_id%%;*}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${job_id}" \
      "${policy}" \
      "${condition}" \
      "${job_name}" \
      "$(output_root_for "${policy}" "${condition}")" \
      "$(sbatch_script_for "${policy}")" \
      "${LOG_DIR}/${job_name}-${job_num}.out" \
      "${LOG_DIR}/${job_name}-${job_num}.err" \
      "1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "${RUN_ID}" >> "${tmp}"
    count=$((count + 1))
  done
done
if [[ "${count}" != "35" ]]; then
  echo "ERROR: submitted ${count} jobs, expected 35" >&2
  exit 1
fi
mv "${tmp}" "${SUBMISSION_TSV}"
echo "submission_tsv=${SUBMISSION_TSV}"
