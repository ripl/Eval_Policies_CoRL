#!/usr/bin/env bash
set -Eeuo pipefail

SUPPORT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
POLICY="dexbotic"
TASK=""
DRY_RUN=0
RUN_TAG="${RUN_TAG:-}"

usage() {
  cat <<'EOF'
Usage: bash run_one_task.sh [--policy dexbotic] --task {stack,carrot,spoon,eggplant} [--dry-run] [--run-tag TAG]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICY="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${TASK}" ]]; then
  usage >&2
  exit 2
fi
if [[ "${POLICY}" != "dexbotic" ]]; then
  echo "unsupported policy in this support tree: ${POLICY}" >&2
  exit 2
fi

python3 "${SUPPORT_ROOT}/bin/preflight.py" --policy "${POLICY}" --task "${TASK}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run=1"
  echo "normal_command=bash ${SUPPORT_ROOT}/run_one_task.sh --policy ${POLICY} --task ${TASK}"
  echo "sbatch_command=sbatch -J repcal_${POLICY}_${TASK} ${SUPPORT_ROOT}/launch/run_repeated_fixed_grid_task.sbatch ${TASK}"
  exit 0
fi

export RUN_TAG
exec bash "${SUPPORT_ROOT}/launch/run_dexbotic_task.sh" "${TASK}"
