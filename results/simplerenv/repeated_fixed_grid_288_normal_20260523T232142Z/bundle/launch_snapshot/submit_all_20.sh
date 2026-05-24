#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
V2_DIR="${V2_DIR:-${PROJECT_ROOT}/scratch/repeated_fixed_grid_calibration_288_20260523_v2}"
RUN_ID="${RUN_ID:-repeated_fixed_grid_calibration_288_20260523_v2}"
RUN_ROOT="${RUN_ROOT:-${V2_DIR}/runs/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs/slurm}"
SUBMISSION_CSV="${SUBMISSION_CSV:-${RUN_ROOT}/submission_jobs.csv}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${RUN_ROOT}/bundle/launch_snapshot}"
SPATIALVLA_REPO="${SPATIALVLA_REPO:-${V2_DIR}/sources/simplerenv_openvla_ccfe380}"
LAUNCH_DIR="${V2_DIR}"
DRY_RUN=0
PREFLIGHT_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash submit_all_20.sh [--dry-run] [--preflight-only] [--run-id RUN_ID] [--submission-csv PATH]

Submits exactly 20 independent sbatch jobs, one per policy/task pair, after
all worker scripts and source/patch guards pass. Actual jobs execute from an
immutable per-run launch snapshot, and this script never uses Slurm arrays.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --run-id) RUN_ID="$2"; RUN_ROOT="${V2_DIR}/runs/${RUN_ID}"; LOG_DIR="${RUN_ROOT}/logs/slurm"; SUBMISSION_CSV="${RUN_ROOT}/submission_jobs.csv"; SNAPSHOT_DIR="${RUN_ROOT}/bundle/launch_snapshot"; shift 2 ;;
    --submission-csv) SUBMISSION_CSV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing required file: $1" >&2
    exit 1
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    echo "ERROR: required file is not executable: $1" >&2
    exit 1
  fi
}

require_no_slurm_array() {
  local script="$1"
  if grep -Eq '^[[:space:]]*#SBATCH[[:space:]]+(-a|--array)([=[:space:]]|$)' "${script}"; then
    echo "ERROR: Slurm array directive found in ${script}" >&2
    exit 1
  fi
}

check_static_inputs() {
  local root="${1:-${V2_DIR}}"
  require_file "${root}/README.md"
  require_executable "${root}/prepare_spatialvla_source.sh"
  require_executable "${root}/run_one_policy_task.py"
  require_file "${root}/run_one_policy_task.sbatch"
  require_executable "${root}/validate_job_csv.py"
  require_executable "${root}/internvla_m1_task_driver.py"
  require_file "${root}/internvla_m1_task.sbatch"
  require_executable "${root}/run_seeded_policy_server.py"
  require_executable "${root}/run_seeded_simpler_episode.py"
  require_executable "${root}/validate_internvla_task_results.py"
  require_executable "${root}/xvla_worker_a_support/run_one_task.sh"
  require_file "${root}/xvla_worker_a_support/launch/run_xvla_task.sbatch"
  require_executable "${root}/xvla_worker_a_support/bin/preflight.py"
  require_executable "${root}/dexbotic_support/run_one_task.sh"
  require_file "${root}/dexbotic_support/launch/run_repeated_fixed_grid_task.sbatch"
  require_executable "${root}/dexbotic_support/bin/preflight.py"
  require_executable "${root}/final_validate_and_summarize.py"
  require_no_slurm_array "${root}/run_one_policy_task.sbatch"
  require_no_slurm_array "${root}/internvla_m1_task.sbatch"
  require_no_slurm_array "${root}/xvla_worker_a_support/launch/run_xvla_task.sbatch"
  require_no_slurm_array "${root}/dexbotic_support/launch/run_repeated_fixed_grid_task.sbatch"
}

run_preflights() {
  bash "${V2_DIR}/prepare_spatialvla_source.sh" --verify-only
  for task in stack carrot spoon eggplant; do
    python3 "${V2_DIR}/run_one_policy_task.py" --policy cogact --task "${task}" --run-id "${RUN_ID}" --output-root "${RUN_ROOT}/cogact/${task}" --preflight-only
    SPATIALVLA_REPO="${SPATIALVLA_REPO}" python3 "${V2_DIR}/run_one_policy_task.py" --policy spatialvla --task "${task}" --run-id "${RUN_ID}" --output-root "${RUN_ROOT}/spatialvla/${task}" --preflight-only
    python3 "${V2_DIR}/internvla_m1_task_driver.py" --task "${task}" --output-root "${RUN_ROOT}/internvla_m1" --preflight
    RUN_ROOT="${RUN_ROOT}" bash "${V2_DIR}/xvla_worker_a_support/run_one_task.sh" --task "${task}" --dry-run >/dev/null
    RUN_ROOT="${RUN_ROOT}" bash "${V2_DIR}/dexbotic_support/run_one_task.sh" --policy dexbotic --task "${task}" --dry-run >/dev/null
  done
}

create_launch_snapshot() {
  if [[ -e "${SNAPSHOT_DIR}" ]]; then
    echo "ERROR: refusing to reuse existing launch snapshot: ${SNAPSHOT_DIR}" >&2
    exit 1
  fi
  require_executable "$(command -v rsync)"
  mkdir -p "$(dirname "${SNAPSHOT_DIR}")" "${RUN_ROOT}/bundle"
  rsync -a \
    --exclude sources \
    --exclude logs \
    --exclude runs \
    --exclude preflight \
    --exclude slurm \
    --exclude __pycache__ \
    --exclude '*.pyc' \
    --exclude 'submission_jobs.csv*' \
    "${V2_DIR}/" "${SNAPSHOT_DIR}/"
  find "${SNAPSHOT_DIR}" -type f \
    \( -path '*/__pycache__/*' -o -name '*.pyc' \) -prune -o \
    -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${RUN_ROOT}/bundle/submission_support_files.sha256"
  {
    echo "run_id=${RUN_ID}"
    echo "run_root=${RUN_ROOT}"
    echo "snapshot_dir=${SNAPSHOT_DIR}"
    echo "source_v2_dir=${V2_DIR}"
    echo "spatialvla_repo=${SPATIALVLA_REPO}"
    echo "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${RUN_ROOT}/bundle/submission_snapshot_metadata.txt"
  chmod -R a-w "${SNAPSHOT_DIR}"
  LAUNCH_DIR="${SNAPSHOT_DIR}"
}

sbatch_command_for() {
  local policy="$1"
  local task="$2"
  case "${policy}" in
    cogact|spatialvla)
      local result_dir="${RUN_ROOT}/${policy}/${task}"
      local export_vars="ALL,RUN_ID=${RUN_ID},RUN_ROOT=${RUN_ROOT},SCRATCH_DIR=${LAUNCH_DIR},POLICY=${policy},TASK=${task},RESULTS_ROOT=${result_dir},SPATIALVLA_REPO=${SPATIALVLA_REPO},PYTHONDONTWRITEBYTECODE=1"
      printf 'sbatch --parsable --no-requeue -J rep288_%s_%s --output=%q --error=%q --export=%q %q' "${policy}" "${task}" "${LOG_DIR}/rep288_${policy}_${task}-%j.out" "${LOG_DIR}/rep288_${policy}_${task}-%j.err" "${export_vars}" "${LAUNCH_DIR}/run_one_policy_task.sbatch"
      ;;
    internvla_m1)
      local export_vars="ALL,RUN_ID=${RUN_ID},RUN_ROOT=${RUN_ROOT},SCRATCH=${LAUNCH_DIR},OUTPUT_ROOT=${RUN_ROOT}/internvla_m1,PYTHONDONTWRITEBYTECODE=1"
      printf 'sbatch --parsable --no-requeue -J rep288_%s_%s --output=%q --error=%q --export=%q %q %q' "${policy}" "${task}" "${LOG_DIR}/rep288_${policy}_${task}-%j.out" "${LOG_DIR}/rep288_${policy}_${task}-%j.err" "${export_vars}" "${LAUNCH_DIR}/internvla_m1_task.sbatch" "${task}"
      ;;
    xvla)
      local export_vars="ALL,RUN_ID=${RUN_ID},RUN_ROOT=${RUN_ROOT},SUPPORT_ROOT=${LAUNCH_DIR}/xvla_worker_a_support,RESULT_BASE=${RUN_ROOT}/${policy},RUN_TAG=${task},PYTHONDONTWRITEBYTECODE=1"
      printf 'sbatch --parsable --no-requeue -J rep288_%s_%s --output=%q --error=%q --export=%q %q %q' "${policy}" "${task}" "${LOG_DIR}/rep288_${policy}_${task}-%j.out" "${LOG_DIR}/rep288_${policy}_${task}-%j.err" "${export_vars}" "${LAUNCH_DIR}/xvla_worker_a_support/launch/run_xvla_task.sbatch" "${task}"
      ;;
    dexbotic)
      local export_vars="ALL,RUN_ID=${RUN_ID},RUN_ROOT=${RUN_ROOT},SUPPORT_ROOT=${LAUNCH_DIR}/dexbotic_support,RESULT_BASE=${RUN_ROOT}/${policy},RUN_TAG=${task},PYTHONDONTWRITEBYTECODE=1"
      printf 'sbatch --parsable --no-requeue -J rep288_%s_%s --output=%q --error=%q --export=%q %q %q' "${policy}" "${task}" "${LOG_DIR}/rep288_${policy}_${task}-%j.out" "${LOG_DIR}/rep288_${policy}_${task}-%j.err" "${export_vars}" "${LAUNCH_DIR}/dexbotic_support/launch/run_repeated_fixed_grid_task.sbatch" "${task}"
      ;;
    *) echo "unsupported policy ${policy}" >&2; return 2 ;;
  esac
}

submit_one() {
  local policy="$1"
  local task="$2"
  case "${policy}" in
    cogact|spatialvla)
      local result_dir="${RUN_ROOT}/${policy}/${task}"
      local export_vars="ALL,RUN_ID=${RUN_ID},RUN_ROOT=${RUN_ROOT},SCRATCH_DIR=${LAUNCH_DIR},POLICY=${policy},TASK=${task},RESULTS_ROOT=${result_dir},SPATIALVLA_REPO=${SPATIALVLA_REPO},PYTHONDONTWRITEBYTECODE=1"
      sbatch --parsable --no-requeue -J "rep288_${policy}_${task}" --output="${LOG_DIR}/rep288_${policy}_${task}-%j.out" --error="${LOG_DIR}/rep288_${policy}_${task}-%j.err" --export="${export_vars}" "${LAUNCH_DIR}/run_one_policy_task.sbatch"
      ;;
    internvla_m1)
      local export_vars="ALL,RUN_ID=${RUN_ID},RUN_ROOT=${RUN_ROOT},SCRATCH=${LAUNCH_DIR},OUTPUT_ROOT=${RUN_ROOT}/internvla_m1,PYTHONDONTWRITEBYTECODE=1"
      sbatch --parsable --no-requeue -J "rep288_${policy}_${task}" --output="${LOG_DIR}/rep288_${policy}_${task}-%j.out" --error="${LOG_DIR}/rep288_${policy}_${task}-%j.err" --export="${export_vars}" "${LAUNCH_DIR}/internvla_m1_task.sbatch" "${task}"
      ;;
    xvla)
      local export_vars="ALL,RUN_ID=${RUN_ID},RUN_ROOT=${RUN_ROOT},SUPPORT_ROOT=${LAUNCH_DIR}/xvla_worker_a_support,RESULT_BASE=${RUN_ROOT}/${policy},RUN_TAG=${task},PYTHONDONTWRITEBYTECODE=1"
      sbatch --parsable --no-requeue -J "rep288_${policy}_${task}" --output="${LOG_DIR}/rep288_${policy}_${task}-%j.out" --error="${LOG_DIR}/rep288_${policy}_${task}-%j.err" --export="${export_vars}" "${LAUNCH_DIR}/xvla_worker_a_support/launch/run_xvla_task.sbatch" "${task}"
      ;;
    dexbotic)
      local export_vars="ALL,RUN_ID=${RUN_ID},RUN_ROOT=${RUN_ROOT},SUPPORT_ROOT=${LAUNCH_DIR}/dexbotic_support,RESULT_BASE=${RUN_ROOT}/${policy},RUN_TAG=${task},PYTHONDONTWRITEBYTECODE=1"
      sbatch --parsable --no-requeue -J "rep288_${policy}_${task}" --output="${LOG_DIR}/rep288_${policy}_${task}-%j.out" --error="${LOG_DIR}/rep288_${policy}_${task}-%j.err" --export="${export_vars}" "${LAUNCH_DIR}/dexbotic_support/launch/run_repeated_fixed_grid_task.sbatch" "${task}"
      ;;
    *) echo "unsupported policy ${policy}" >&2; return 2 ;;
  esac
}

result_dir_for() {
  local policy="$1"
  local task="$2"
  if [[ "${policy}" == "internvla_m1" ]]; then
    printf '%s/internvla_m1/%s' "${RUN_ROOT}" "${task}"
  else
    printf '%s/%s/%s' "${RUN_ROOT}" "${policy}" "${task}"
  fi
}

check_static_inputs "${V2_DIR}"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}"
run_preflights

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "preflight_status=passed"
  exit 0
fi

policies=(cogact spatialvla internvla_m1 xvla dexbotic)
tasks=(stack carrot spoon eggplant)

if [[ "${DRY_RUN}" == "1" ]]; then
  count=0
  for policy in "${policies[@]}"; do
    for task in "${tasks[@]}"; do
      sbatch_command_for "${policy}" "${task}"
      printf '\n'
      count=$((count + 1))
    done
  done
  if [[ "${count}" != "20" ]]; then
    echo "ERROR: dry-run command count is ${count}, expected 20" >&2
    exit 1
  fi
  echo "dry_run_count=20"
  exit 0
fi

if [[ -e "${SUBMISSION_CSV}" ]]; then
  echo "ERROR: refusing to overwrite existing submission CSV: ${SUBMISSION_CSV}" >&2
  exit 1
fi
create_launch_snapshot
check_static_inputs "${LAUNCH_DIR}"

tmp_csv="${SUBMISSION_CSV}.tmp"
printf 'job_id,policy,task,sbatch_script,result_dir,submitted_at_utc,run_id\n' > "${tmp_csv}"
count=0
for policy in "${policies[@]}"; do
  for task in "${tasks[@]}"; do
    job_id="$(submit_one "${policy}" "${task}")"
    result_dir="$(result_dir_for "${policy}" "${task}")"
    case "${policy}" in
      cogact|spatialvla) sbatch_script="${LAUNCH_DIR}/run_one_policy_task.sbatch" ;;
      internvla_m1) sbatch_script="${LAUNCH_DIR}/internvla_m1_task.sbatch" ;;
      xvla) sbatch_script="${LAUNCH_DIR}/xvla_worker_a_support/launch/run_xvla_task.sbatch" ;;
      dexbotic) sbatch_script="${LAUNCH_DIR}/dexbotic_support/launch/run_repeated_fixed_grid_task.sbatch" ;;
    esac
    printf '%s,%s,%s,%s,%s,%s,%s\n' "${job_id}" "${policy}" "${task}" "${sbatch_script}" "${result_dir}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RUN_ID}" >> "${tmp_csv}"
    count=$((count + 1))
  done
done

if [[ "${count}" != "20" ]]; then
  echo "ERROR: submitted ${count} jobs, expected 20" >&2
  exit 1
fi
mv "${tmp_csv}" "${SUBMISSION_CSV}"
echo "submission_csv=${SUBMISSION_CSV}"
