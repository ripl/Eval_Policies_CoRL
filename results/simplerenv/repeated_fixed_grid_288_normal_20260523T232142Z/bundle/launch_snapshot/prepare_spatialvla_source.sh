#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/data/ripl/tianchong/projects/Eval_Policies_CoRL}"
V2_DIR="${V2_DIR:-${PROJECT_ROOT}/scratch/repeated_fixed_grid_calibration_288_20260523_v2}"
SOURCE_REPO="${SOURCE_REPO:-${PROJECT_ROOT}/third_party/simplerenv_openvla}"
SOURCE_SUBMODULE="${SOURCE_SUBMODULE:-${SOURCE_REPO}/ManiSkill2_real2sim}"
DEST_REPO="${DEST_REPO:-${V2_DIR}/sources/simplerenv_openvla_ccfe380}"
EXPECTED_COMMIT="ccfe3809766839a2fcfb7a3d3c9abff585189188"
EXPECTED_SUBMODULE_COMMIT="cd45dd27dc6bb26d048cb6570cdab4e3f935cc37"

verify_clean() {
  local repo="$1"
  local expected="$2"
  local label="$3"
  local head
  head="$(git -C "${repo}" rev-parse HEAD)"
  if [[ "${head}" != "${expected}" ]]; then
    echo "ERROR: ${label} commit mismatch: ${head} != ${expected}" >&2
    return 1
  fi
  local status
  status="$(git -C "${repo}" status --short)"
  if [[ -n "${status}" ]]; then
    echo "ERROR: ${label} is dirty:" >&2
    printf '%s\n' "${status}" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--verify-only" ]]; then
  if [[ ! -e "${DEST_REPO}/.git" ]]; then
    echo "ERROR: missing clean SpatialVLA source: ${DEST_REPO}" >&2
    exit 1
  fi
  verify_clean "${DEST_REPO}" "${EXPECTED_COMMIT}" "SpatialVLA source"
  verify_clean "${DEST_REPO}/ManiSkill2_real2sim" "${EXPECTED_SUBMODULE_COMMIT}" "SpatialVLA ManiSkill2_real2sim"
  echo "spatialvla_source_status=passed"
  exit 0
fi

if [[ -e "${DEST_REPO}" ]]; then
  "${BASH_SOURCE[0]}" --verify-only
  exit 0
fi

git -C "${SOURCE_REPO}" cat-file -e "${EXPECTED_COMMIT}^{commit}"
git -C "${SOURCE_SUBMODULE}" cat-file -e "${EXPECTED_SUBMODULE_COMMIT}^{commit}"
mkdir -p "$(dirname "${DEST_REPO}")"
git clone --shared "${SOURCE_REPO}" "${DEST_REPO}"
git -C "${DEST_REPO}" checkout --detach "${EXPECTED_COMMIT}"
git clone --shared "${SOURCE_SUBMODULE}" "${DEST_REPO}/ManiSkill2_real2sim"
git -C "${DEST_REPO}/ManiSkill2_real2sim" checkout --detach "${EXPECTED_SUBMODULE_COMMIT}"
"${BASH_SOURCE[0]}" --verify-only
