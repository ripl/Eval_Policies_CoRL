#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
SBATCH_SCRIPT="${PROJECT_ROOT}/scripts/slurm/libero_official_calibration_policy_suite.sbatch"
PREP_SCRIPT="${PROJECT_ROOT}/scripts/libero/prepare_libero_official_calibration_artifact.py"
RUN_GROUP="official_calibration_$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_FAMILY="official_calibration"
EPISODES_PER_TASK=50
NICE=10000
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/slurm/submit_libero_official_calibration_12jobs.sh [--submit] [--nice N] [--run-group NAME]

Defaults to dry-run. Add --submit to call sbatch.

This submits exactly 12 non-array jobs: 3 policies x 4 suites. Each job loops
serially over the 10 LIBERO task IDs and evaluates 50 official init states per
task.

Options:
  --submit          Actually submit the 12 policy-suite Slurm jobs.
  --nice N          Slurm nice value; default 10000 lowers priority.
  --run-group NAME  Override the result run group.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --nice) NICE="${2:?missing value for --nice}"; shift 2 ;;
    --run-group) RUN_GROUP="${2:?missing value for --run-group}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if ! [[ "${NICE}" =~ ^[0-9]+$ ]]; then
  echo "--nice must be a non-negative integer, got ${NICE}" >&2
  exit 1
fi

POLICIES=(spatial_forcing simvla pi05_lerobot)
SUITES=(libero_spatial libero_object libero_goal libero_10)

cd "${PROJECT_ROOT}"
mkdir -p logs/slurm

for path in "${SBATCH_SCRIPT}" "${PREP_SCRIPT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required script: ${path}" >&2
    exit 1
  fi
done

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RUN_GROUP=${RUN_GROUP}"
echo "RESULT_FAMILY=${RESULT_FAMILY}"
echo "EPISODES_PER_TASK=${EPISODES_PER_TASK}"
echo "NICE=${NICE}"
echo "SUBMIT=${SUBMIT}"

if [[ "${SUBMIT}" == "1" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    export PATH="/share/data/ripl/tianchong/conda/bin:${PATH}"
  fi
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate /share/data/ripl/tianchong/conda/envs/libero_SF
  python "${PREP_SCRIPT}"
else
  echo "dry-run: not preparing artifact and not submitting jobs"
fi

jobids=()
for policy in "${POLICIES[@]}"; do
  for suite in "${SUITES[@]}"; do
    extra_sbatch=(-c 4 --mem=80G)
    if [[ "${policy}" == "pi05_lerobot" ]]; then
      extra_sbatch=(-c 1 --mem=50G)
    fi
    job_name="liboff_${policy}_${suite}"
    export_arg="ALL,POLICY=${policy},TASK_SUITE=${suite},RUN_GROUP=${RUN_GROUP},RESULT_FAMILY=${RESULT_FAMILY},EPISODES_PER_TASK=${EPISODES_PER_TASK}"
    cmd=(sbatch --parsable --nice="${NICE}" --job-name="${job_name}" "${extra_sbatch[@]}" --export="${export_arg}" "${SBATCH_SCRIPT}")
    printf 'JOB policy=%s suite=%s\n' "${policy}" "${suite}"
    printf '  %q' "${cmd[@]}"
    printf '\n'
    if [[ "${SUBMIT}" == "1" ]]; then
      jobid="$("${cmd[@]}")"
      jobids+=("${jobid}")
      echo "  submitted ${jobid}"
    fi
  done
done

if [[ "${SUBMIT}" == "1" ]]; then
  manifest_dir="${PROJECT_ROOT}/results/libero/${RESULT_FAMILY}/${RUN_GROUP}"
  mkdir -p "${manifest_dir}"
  {
    echo "run_group=${RUN_GROUP}"
    echo "result_family=${RESULT_FAMILY}"
    echo "episodes_per_task=${EPISODES_PER_TASK}"
    echo "nice=${NICE}"
    echo "submitted_utc=$(date -u --iso-8601=seconds)"
    echo "submission_shape=12 non-array jobs, one policy-suite cell per job"
    printf 'jobids=%s\n' "${jobids[*]}"
  } > "${manifest_dir}/SUBMITTED_JOBS.txt"
  echo "submitted ${#jobids[@]} jobs"
  echo "manifest=${manifest_dir}/SUBMITTED_JOBS.txt"
fi
