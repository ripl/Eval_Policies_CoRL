#!/usr/bin/env python3
"""Validate and summarize a full SimplerEnv Protocol A-E rollout bundle.

Expected run shape:
  run_root/
    submission_jobs.csv or submission_jobs.tsv
    <policy>/<condition>/per_episode_results.csv
    metadata/ or policy_runtime_metadata.json with policy/runtime/checkpoint evidence

The script writes only lightweight CSV/JSON summaries under final_summary by
default. Raw videos, observations, caches, and weights must stay outside the
curated paper bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_PROTOCOL_NAME = "simplerenv_protocol_abcde_stack_v1"
EXPECTED_PROTOCOL_SHA256 = "1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"
DEFAULT_PROTOCOL_CONFIG = Path("configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json")
DEFAULT_ASSET_MANIFEST = Path(
    "/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/artifacts/simplerenv/"
    "protocol_abcde_asset_preflight_20260525T072500Z/runtime_asset_manifest.json"
)

POLICIES = {
    "cogact": "CogACT-Base",
    "spatialvla": "SpatialVLA",
    "internvla_m1": "InternVLA-M1",
    "xvla": "X-VLA-WidowX",
    "dexbotic": "Dexbotic / DB-MemVLA",
}
POLICY_ALIASES = {
    "cogact-base": "cogact",
    "cogact_base": "cogact",
    "spatialvla_bridge": "spatialvla",
    "internvla": "internvla_m1",
    "internvla-m1": "internvla_m1",
    "internvla_m1": "internvla_m1",
    "x-vla-widowx": "xvla",
    "xvla_widowx": "xvla",
    "xvla": "xvla",
    "db-memvla": "dexbotic",
    "dexbotic_db-memvla": "dexbotic",
    "dexbotic_db_memvla": "dexbotic",
    "dexbotic": "dexbotic",
}
CONDITIONS = [
    "protocol_A",
    "protocol_B",
    "protocol_C1_yellow_on_green",
    "protocol_C2_blue_on_red",
    "protocol_C3_red_on_blue",
    "protocol_D",
    "protocol_E",
]
EXPECTED_ROWS_PER_JOB = 288
EXPECTED_TOTAL_ROWS = len(POLICIES) * len(CONDITIONS) * EXPECTED_ROWS_PER_JOB
STANDARD_HORIZON = 60
ACCEPTED_RUNTIME_ASSETS = {
    "blue": "render_candidate_blue_hybrid_v4",
    "red": "render_candidate_red_corrected_v6e",
    "white": "render_candidate_white_offwhite_hybrid_v4",
}
HEX64_RE = re.compile(r"\b[0-9a-fA-F]{64}\b")
BANNED_BUNDLE_SUFFIXES = {
    ".avi",
    ".gif",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".png",
    ".webm",
}
EPISODE_COLUMNS = [
    "policy",
    "policy_label",
    "condition",
    "episode_id",
    "success",
    "steps",
    "error",
    "timeout",
    "horizon",
    "job_id",
    "result_dir",
    "artifact_path",
    "protocol_sha256",
    "protocol_sha256_source",
    "instruction",
    "source_color",
    "target_color",
    "source_model_id",
    "target_model_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--submission-jobs", type=Path, default=None)
    parser.add_argument(
        "--replacement-submission",
        type=Path,
        action="append",
        default=None,
        help="Replacement submission TSV to overlay onto the original 35-row submission table. Defaults to bundle/*replacement*submission.tsv.",
    )
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL_CONFIG)
    parser.add_argument("--asset-manifest", type=Path, default=None)
    parser.add_argument("--expected-protocol-sha256", default=EXPECTED_PROTOCOL_SHA256)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-error-rows", action="store_true")
    parser.add_argument("--skip-artifact-existence", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(errors="replace").splitlines()
    while text and "\t" in text[0] and text[0].split("\t", 1)[0] in {"stamp", "invalid_previous_jobs"}:
        text.pop(0)
    if not text:
        return [], []
    sample = "\n".join(text[:5])
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    reader = csv.DictReader(text, dialect=dialect)
    return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def norm_policy(value: str) -> str:
    token = value.strip()
    lowered = token.lower().replace(" ", "_").replace("/", "_")
    lowered = re.sub(r"_+", "_", lowered)
    lowered = lowered.strip("_")
    if lowered in POLICIES:
        return lowered
    return POLICY_ALIASES.get(lowered.replace("_", "-"), POLICY_ALIASES.get(lowered, lowered))


def first_nonblank(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "timeout"}


def bool_to_01(value: Any) -> str:
    text = str(value).strip()
    if text in {"0", "1"}:
        return text
    if text.lower() in {"true", "yes", "success"}:
        return "1"
    if text.lower() in {"false", "no", "fail", "failure"}:
        return "0"
    return text


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def load_protocol_config(path: Path, expected_sha: str, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing Protocol A-E config: {path}")
        return {}
    actual = sha256_file(path)
    if actual != expected_sha:
        errors.append(f"Protocol A-E config SHA mismatch: expected {expected_sha}, got {actual}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.is_file():
        declared = sidecar.read_text(errors="replace").split()[0]
        if declared != expected_sha:
            errors.append(f"Protocol A-E sidecar SHA mismatch: expected {expected_sha}, got {declared}")
    else:
        errors.append(f"missing Protocol A-E config sidecar: {sidecar}")
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"failed to parse Protocol A-E config {path}: {exc}")
        return {}
    if data.get("name") != EXPECTED_PROTOCOL_NAME:
        errors.append(f"unexpected protocol config name: {data.get('name')!r}")
    conditions = data.get("conditions")
    if not isinstance(conditions, dict):
        errors.append("Protocol config missing conditions object")
        return data
    if sorted(conditions) != sorted(CONDITIONS):
        errors.append(f"Protocol config conditions mismatch: {sorted(conditions)} != {CONDITIONS}")
    for condition in CONDITIONS:
        episodes = (conditions.get(condition) or {}).get("episodes")
        if not isinstance(episodes, list) or len(episodes) != EXPECTED_ROWS_PER_JOB:
            errors.append(f"{condition}: expected {EXPECTED_ROWS_PER_JOB} config episodes, found {0 if episodes is None else len(episodes)}")
            continue
        ids = [ep.get("episode_id") for ep in episodes if isinstance(ep, dict)]
        if ids != list(range(EXPECTED_ROWS_PER_JOB)):
            errors.append(f"{condition}: config episode_id sequence is not exactly 0..287")
    return data


def _record_asset_hash(
    *,
    asset_hashes: dict[str, set[str]],
    asset_id: str,
    rel: str,
    digest: str,
    root_text: str,
    errors: list[str],
) -> None:
    expected_ids = set(ACCEPTED_RUNTIME_ASSETS.values())
    if asset_id not in expected_ids:
        errors.append(f"asset manifest has unexpected asset_id {asset_id!r}")
        return
    if not HEX64_RE.fullmatch(digest):
        errors.append(f"asset manifest invalid sha256 for {asset_id}/{rel}: {digest!r}")
        return
    asset_hashes[asset_id].add(digest)
    if root_text and rel:
        file_path = Path(root_text) / rel
        if not file_path.is_file():
            errors.append(f"asset manifest runtime asset file missing: {file_path}")
        elif sha256_file(file_path) != digest:
            errors.append(f"asset manifest runtime asset hash mismatch: {file_path}")


def load_asset_manifest(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing Protocol A-E runtime asset manifest: {path}")
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"failed to parse Protocol A-E runtime asset manifest {path}: {exc}")
        return {}
    manifest_sha256 = sha256_file(path)
    accepted = set(data.get("accepted_asset_ids") or data.get("asset_ids") or [])
    expected_ids = set(ACCEPTED_RUNTIME_ASSETS.values())
    if accepted != expected_ids:
        errors.append(f"asset manifest asset id set mismatch: {sorted(accepted)} != {sorted(expected_ids)}")
    asset_hashes: dict[str, set[str]] = {asset_id: set() for asset_id in expected_ids}
    policy_asset_hashes: dict[str, dict[str, set[str]]] = {
        policy: {asset_id: set() for asset_id in expected_ids} for policy in POLICIES
    }
    policy_paths: dict[str, list[str]] = defaultdict(list)

    runtime_paths = data.get("runtime_paths")
    policy_targets = data.get("policy_targets")
    if isinstance(runtime_paths, list) and runtime_paths:
        schema = "runtime_paths"
        policy_targets = []
    elif isinstance(policy_targets, list) and policy_targets:
        schema = "policy_targets"
        runtime_paths = []
    else:
        errors.append(f"asset manifest has neither runtime_paths nor policy_targets: {path}")
        runtime_paths = []
        policy_targets = []
        schema = "unknown"

    for idx, runtime in enumerate(runtime_paths):
        if not isinstance(runtime, dict):
            errors.append(f"asset manifest runtime_paths[{idx}] is not an object")
            continue
        policy = norm_policy(str(runtime.get("policy", "")))
        root_text = str(runtime.get("maniskill2_real2sim", "")).strip()
        if policy in POLICIES and root_text:
            policy_paths[policy].append(root_text)
        info_json = Path(str(runtime.get("info_json", "")))
        info_sha = str(runtime.get("info_json_sha256", "")).strip()
        if info_json.is_file() and info_sha:
            actual = sha256_file(info_json)
            if actual != info_sha:
                errors.append(f"asset manifest info_json hash mismatch for {info_json}: {actual} != {info_sha}")
        elif not info_json.is_file():
            errors.append(f"asset manifest info_json missing: {info_json}")
        for asset_file in runtime.get("asset_files") or []:
            asset_id = str(asset_file.get("asset_id", "")).strip()
            rel = str(asset_file.get("relative_path", "")).strip()
            digest = str(asset_file.get("sha256", "")).strip()
            _record_asset_hash(asset_hashes=asset_hashes, asset_id=asset_id, rel=rel, digest=digest, root_text=root_text, errors=errors)
            if policy in POLICIES and asset_id in expected_ids and HEX64_RE.fullmatch(digest):
                policy_asset_hashes[policy][asset_id].add(digest)

    for idx, target in enumerate(policy_targets or []):
        if not isinstance(target, dict):
            errors.append(f"asset manifest policy_targets[{idx}] is not an object")
            continue
        policy = norm_policy(str(target.get("policy", "")))
        target_custom_dir = str(target.get("target_custom_dir", "")).strip()
        if policy in POLICIES and target_custom_dir:
            policy_paths[policy].append(target_custom_dir)
        if target.get("status") not in {"passed", "verified", "written", None}:
            errors.append(f"asset manifest target for {policy or target.get('policy')} did not pass: {target.get('status')}")
        for error in target.get("errors") or []:
            errors.append(f"asset manifest target error for {policy or target.get('policy')}: {error}")
        target_info = Path(str(target.get("target_info", "")))
        target_info_entry_seen = False
        for asset_id, asset_record in (target.get("assets") or {}).items():
            if not isinstance(asset_record, dict):
                errors.append(f"asset manifest target {policy}/{asset_id} record is not an object")
                continue
            if not asset_record.get("target_exists", False):
                errors.append(f"asset manifest target {policy}/{asset_id} target_exists is false")
            if asset_record.get("target_info_entry"):
                target_info_entry_seen = True
            for rel, digest in (asset_record.get("target_file_sha256") or {}).items():
                root_text = str(Path(target_custom_dir) / "models" / asset_id) if target_custom_dir else ""
                _record_asset_hash(asset_hashes=asset_hashes, asset_id=str(asset_id), rel=str(rel), digest=str(digest), root_text=root_text, errors=errors)
                if policy in POLICIES and str(asset_id) in expected_ids and HEX64_RE.fullmatch(str(digest)):
                    policy_asset_hashes[policy][str(asset_id)].add(str(digest))
        if target_info and str(target_info) != "." and not target_info.is_file():
            errors.append(f"asset manifest target info JSON missing: {target_info}")
        if not target_info_entry_seen:
            errors.append(f"asset manifest target for {policy or target.get('policy')} lacks target_info_entry records")

    for asset_id, hashes in asset_hashes.items():
        if not hashes:
            errors.append(f"asset manifest lacks hashes for accepted asset {asset_id}")
    for policy in POLICIES:
        for color, asset_id in ACCEPTED_RUNTIME_ASSETS.items():
            if not policy_asset_hashes[policy][asset_id]:
                errors.append(f"asset manifest lacks {color} asset hashes for policy runtime path {policy}: {asset_id}")
    return {
        "path": str(path),
        "sha256": manifest_sha256,
        "schema": schema,
        "accepted_asset_ids": sorted(expected_ids),
        "asset_hashes": {asset_id: sorted(hashes) for asset_id, hashes in asset_hashes.items()},
        "policy_asset_hashes": {
            policy: {asset_id: sorted(hashes) for asset_id, hashes in by_asset.items()}
            for policy, by_asset in policy_asset_hashes.items()
        },
        "policy_runtime_paths": {policy: sorted(paths) for policy, paths in policy_paths.items()},
    }


def test_set_rows(config: dict[str, Any], expected_sha: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        episodes = ((config.get("conditions") or {}).get(condition) or {}).get("episodes") or []
        for ep in episodes:
            source = ep.get("source") or {}
            target = ep.get("target") or {}
            rows.append(
                {
                    "condition": condition,
                    "episode_id": ep.get("episode_id", ""),
                    "protocol_sha256": expected_sha,
                    "instruction": ep.get("instruction", ""),
                    "source_color": source.get("color", ""),
                    "target_color": target.get("color", ""),
                    "source_model_id": source.get("model_id", ""),
                    "target_model_id": target.get("model_id", ""),
                }
            )
    return rows


def find_submission_jobs(run_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    for name in ["submission_jobs.csv", "submission_jobs.tsv", "submission.csv", "submission.tsv"]:
        path = run_root / name
        if path.is_file():
            return path
    return None


def find_replacement_submissions(run_root: Path, explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return [p.resolve() for p in explicit]
    bundle = run_root / "bundle"
    if not bundle.is_dir():
        return []
    return sorted(bundle.glob("*replacement*submission.tsv"))


def resolve_asset_manifest_path(run_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    run_manifest = run_root / "bundle" / "runtime_asset_manifest.json"
    if run_manifest.is_file():
        return run_manifest.resolve()
    return DEFAULT_ASSET_MANIFEST.resolve()


def derived_log_paths(run_root: Path, row: dict[str, str]) -> tuple[str, str]:
    job_id = first_nonblank(row, ["job_id", "jobid", "slurm_job_id"])
    job_name = first_nonblank(row, ["job_name", "name"])
    if not job_id or not job_name:
        return "", ""
    return (
        str(run_root / "logs" / "slurm" / f"{job_name}-{job_id}.out"),
        str(run_root / "logs" / "slurm" / f"{job_name}-{job_id}.err"),
    )


def parse_submission(path: Path | None, run_root: Path, errors: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.is_file():
        errors.append(f"missing submission job table under {run_root}")
        return {}
    _header, rows = read_table(path)
    expected_pairs = {(p, c) for p in POLICIES for c in CONDITIONS}
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    seen_jobs: set[str] = set()
    for idx, row in enumerate(rows, start=2):
        policy = norm_policy(first_nonblank(row, ["policy", "policy_name", "policy_key"]))
        condition = first_nonblank(row, ["condition", "protocol_condition"])
        if not policy and "run_tag" in row:
            # Smoke-only tables omit policy; full paper runs must not.
            policy = norm_policy(first_nonblank(row, ["run_policy"]))
        job_id = first_nonblank(row, ["job_id", "jobid", "slurm_job_id"])
        pair = (policy, condition)
        if pair in by_pair:
            errors.append(f"{path}: duplicate policy/condition pair on line {idx}: {pair}")
        by_pair[pair] = row
        if pair not in expected_pairs:
            errors.append(f"{path}: unexpected policy/condition on line {idx}: {pair}")
        if not job_id:
            errors.append(f"{path}: blank Slurm job_id on line {idx}")
        elif job_id in seen_jobs:
            errors.append(f"{path}: duplicate Slurm job_id {job_id} on line {idx}")
        seen_jobs.add(job_id)
        result_dir_text = first_nonblank(row, ["result_dir", "results_dir", "output_dir"])
        if not result_dir_text and policy and condition:
            result_dir_text = str(run_root / policy / condition)
            row["result_dir"] = result_dir_text
        if result_dir_text and not resolve_path(result_dir_text, run_root).is_dir():
            errors.append(f"{path}: result_dir does not exist on line {idx}: {result_dir_text}")
        command = first_nonblank(row, ["command", "resolved_command"])
        command_file = first_nonblank(row, ["command_file", "sbatch_script", "script_path"])
        if not command and not command_file:
            errors.append(f"{path}: line {idx} lacks exact command or command_file/sbatch_script")
        if command_file and not resolve_path(command_file, run_root).is_file():
            errors.append(f"{path}: command_file/sbatch_script does not exist on line {idx}: {command_file}")
        logs = [
            first_nonblank(row, ["stdout_log", "log_out", "slurm_log", "log_path"]),
            first_nonblank(row, ["stderr_log", "log_err"]),
        ]
        if not any(logs):
            derived_out, derived_err = derived_log_paths(run_root, row)
            if derived_out and derived_err:
                row["stdout_log"] = derived_out
                row["stderr_log"] = derived_err
                logs = [derived_out, derived_err]
        if not any(logs):
            errors.append(f"{path}: line {idx} lacks Slurm stdout/stderr log path and cannot derive logs from job_name/job_id")
        for log in [x for x in logs if x]:
            if not resolve_path(log, run_root).is_file():
                errors.append(f"{path}: Slurm log path does not exist on line {idx}: {log}")
    missing = sorted(expected_pairs - set(by_pair))
    extra = sorted(set(by_pair) - expected_pairs)
    if len(rows) != len(expected_pairs):
        errors.append(f"{path}: expected {len(expected_pairs)} submission rows, found {len(rows)}")
    if missing:
        errors.append(f"{path}: missing policy/condition pairs: {missing}")
    if extra:
        errors.append(f"{path}: extra policy/condition pairs: {extra}")
    return by_pair


def _validate_existing_path(path_text: str, run_root: Path, errors: list[str], context: str, *, is_dir: bool = False) -> None:
    if not path_text:
        errors.append(f"{context}: missing path")
        return
    path = resolve_path(path_text, run_root)
    if is_dir:
        if not path.is_dir():
            errors.append(f"{context}: directory does not exist: {path_text}")
    elif not path.is_file():
        errors.append(f"{context}: file does not exist: {path_text}")


def validate_replacement_preflight(path: Path, run_root: Path, rows: list[dict[str, str]], errors: list[str]) -> None:
    stem = path.name.removesuffix("_submission.tsv")
    candidates = [
        path.with_name(f"{stem}_preflight_report.json"),
        run_root / "bundle" / f"{stem}_preflight_report.json",
    ]
    preflight = next((p for p in candidates if p.is_file()), None)
    if preflight is None:
        errors.append(f"{path}: missing replacement preflight report")
        return
    try:
        payload = json.loads(preflight.read_text(errors="replace"))
    except Exception as exc:
        errors.append(f"{path}: failed to parse replacement preflight report {preflight}: {exc}")
        return
    if payload.get("status") != "passed":
        errors.append(f"{path}: replacement preflight did not pass: {payload.get('status')}")
    if payload.get("policy") != "dexbotic":
        errors.append(f"{path}: replacement preflight policy {payload.get('policy')!r} != 'dexbotic'")
    if payload.get("job_count") != len(rows):
        errors.append(f"{path}: replacement preflight job_count {payload.get('job_count')} != TSV rows {len(rows)}")
    if payload.get("episodes_per_job") != EXPECTED_ROWS_PER_JOB:
        errors.append(f"{path}: replacement preflight episodes_per_job {payload.get('episodes_per_job')} != {EXPECTED_ROWS_PER_JOB}")
    if payload.get("horizon") != STANDARD_HORIZON:
        errors.append(f"{path}: replacement preflight horizon {payload.get('horizon')} != {STANDARD_HORIZON}")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        errors.append(f"{path}: replacement preflight protocol SHA {payload.get('protocol_sha256')} != {EXPECTED_PROTOCOL_SHA256}")
    row_conditions = [first_nonblank(row, ["condition", "protocol_condition"]) for row in rows]
    preflight_conditions = payload.get("conditions")
    if not rows:
        errors.append(f"{path}: replacement TSV has no rows")
    if not isinstance(preflight_conditions, list):
        errors.append(f"{path}: replacement preflight conditions is not a list: {preflight_conditions!r}")
    else:
        bad_conditions = sorted(set(preflight_conditions) - set(CONDITIONS))
        if bad_conditions:
            errors.append(f"{path}: replacement preflight has non-Protocol A-E conditions: {bad_conditions}")
        if set(preflight_conditions) != set(row_conditions):
            errors.append(f"{path}: replacement preflight conditions {preflight_conditions} do not match TSV conditions {row_conditions}")
    submission_manifest = str(payload.get("submission_manifest", "")).strip()
    if submission_manifest and resolve_path(submission_manifest, run_root) != path.resolve():
        errors.append(f"{path}: replacement preflight submission_manifest mismatch: {submission_manifest}")
    norm_stats_sha = str(payload.get("norm_stats_sha256", "")).strip()
    if norm_stats_sha and not HEX64_RE.fullmatch(norm_stats_sha):
        errors.append(f"{path}: replacement preflight norm_stats_sha256 is not a sha256: {norm_stats_sha!r}")
    support_root = Path(str(rows[0].get("support_root", "")).strip()) if rows and rows[0].get("support_root") else None
    support_files = payload.get("support_files") or {}
    if not isinstance(support_files, dict) or not support_files:
        errors.append(f"{path}: replacement preflight lacks support_files hashes")
        return
    if support_root is None:
        errors.append(f"{path}: replacement TSV lacks support_root for support hash validation")
        return
    for rel, digest in support_files.items():
        digest = str(digest).strip()
        if not HEX64_RE.fullmatch(digest):
            errors.append(f"{path}: replacement support hash for {rel} is invalid: {digest!r}")
            continue
        support_file = support_root / str(rel)
        if not support_file.is_file():
            errors.append(f"{path}: replacement support file missing: {support_file}")
        elif sha256_file(support_file) != digest:
            errors.append(f"{path}: replacement support file hash drift: {support_file}")


def apply_replacement_submissions(
    *,
    run_root: Path,
    original_by_pair: dict[tuple[str, str], dict[str, str]],
    replacement_paths: list[Path],
    errors: list[str],
) -> tuple[dict[tuple[str, str], dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    active_by_pair = dict(original_by_pair)
    replacement_records: list[dict[str, Any]] = []
    chains: dict[tuple[str, str], list[dict[str, str]]] = {}
    for pair, row in original_by_pair.items():
        chains[pair] = [
            {
                "job_id": first_nonblank(row, ["job_id", "jobid", "slurm_job_id"]),
                "result_dir": first_nonblank(row, ["result_dir", "results_dir", "output_dir"]),
                "stdout_log": first_nonblank(row, ["stdout_log", "log_out", "slurm_log", "log_path"]),
                "stderr_log": first_nonblank(row, ["stderr_log", "log_err"]),
                "source_tsv": "original_submission",
                "fix_summary": "",
            }
        ]
    if not replacement_paths:
        return active_by_pair, replacement_records, []
    expected_replacement_pairs = {("dexbotic", condition) for condition in CONDITIONS}
    seen_replacement_jobs: set[str] = set()
    for path in replacement_paths:
        if not path.is_file():
            errors.append(f"missing replacement submission TSV: {path}")
            continue
        _header, rows = read_table(path)
        validate_replacement_preflight(path, run_root, rows, errors)
        seen_pairs_in_file: set[tuple[str, str]] = set()
        for idx, row in enumerate(rows, start=2):
            policy = norm_policy(first_nonblank(row, ["policy", "policy_name", "policy_key"]))
            condition = first_nonblank(row, ["condition", "protocol_condition"])
            pair = (policy, condition)
            if pair not in expected_replacement_pairs:
                errors.append(f"{path}: line {idx} replacement pair {pair} is not a Dexbotic Protocol A-E cell")
                continue
            if pair in seen_pairs_in_file:
                errors.append(f"{path}: duplicate replacement pair within this TSV: {pair}")
            seen_pairs_in_file.add(pair)
            if pair not in original_by_pair:
                errors.append(f"{path}: line {idx} replacement has no original submission row for {pair}")
                continue
            current = active_by_pair.get(pair, {})
            replacement_job_id = first_nonblank(row, ["replacement_job_id", "job_id", "jobid", "slurm_job_id"])
            failed_job_id = first_nonblank(row, ["failed_job_id", "failed_r1_job_id", "original_job_id", "superseded_job_id"])
            current_job_id = first_nonblank(current, ["job_id", "jobid", "slurm_job_id", "replacement_job_id"])
            if not replacement_job_id:
                errors.append(f"{path}: line {idx} missing replacement job id")
            elif replacement_job_id in seen_replacement_jobs:
                errors.append(f"{path}: duplicate replacement job id {replacement_job_id}")
            elif replacement_job_id in [entry.get("job_id", "") for entry in chains.get(pair, [])]:
                errors.append(f"{path}: line {idx} replacement job id {replacement_job_id} already appears in supersession chain for {pair}")
            seen_replacement_jobs.add(replacement_job_id)
            if not failed_job_id:
                errors.append(f"{path}: line {idx} missing failed/superseded job id")
            elif current_job_id and failed_job_id != current_job_id:
                errors.append(f"{path}: line {idx} failed_job_id {failed_job_id} != current active job_id {current_job_id} for {pair}")
            if replacement_job_id and failed_job_id and replacement_job_id == failed_job_id:
                errors.append(f"{path}: line {idx} replacement job id equals failed job id {replacement_job_id}")

            replacement_result_dir = first_nonblank(row, ["replacement_result_dir", "r2_result_dir", "result_dir", "results_dir", "output_dir"])
            failed_result_dir = first_nonblank(row, ["failed_result_dir", "r1_result_dir", "original_result_dir", "superseded_result_dir"])
            if current and failed_result_dir:
                current_result_dir = first_nonblank(current, ["result_dir", "results_dir", "output_dir", "replacement_result_dir"])
                if current_result_dir and resolve_path(failed_result_dir, run_root) != resolve_path(current_result_dir, run_root):
                    errors.append(f"{path}: line {idx} failed_result_dir does not match current active result_dir for {pair}")
            _validate_existing_path(replacement_result_dir, run_root, errors, f"{path}: line {idx} replacement_result_dir", is_dir=True)
            _validate_existing_path(failed_result_dir, run_root, errors, f"{path}: line {idx} failed_result_dir", is_dir=True)
            _validate_existing_path(first_nonblank(row, ["sbatch_script", "command_file", "script_path"]), run_root, errors, f"{path}: line {idx} replacement sbatch_script")
            _validate_existing_path(first_nonblank(row, ["stdout_log", "log_out", "slurm_log", "log_path"]), run_root, errors, f"{path}: line {idx} replacement stdout_log")
            _validate_existing_path(first_nonblank(row, ["stderr_log", "log_err"]), run_root, errors, f"{path}: line {idx} replacement stderr_log")
            failed_stdout_log = first_nonblank(row, ["failed_stdout_log", "r1_stdout_log"])
            failed_stderr_log = first_nonblank(row, ["failed_stderr_log", "r1_stderr_log"])
            asset_preflight_report = first_nonblank(row, ["asset_preflight_report"])
            _validate_existing_path(failed_stdout_log, run_root, errors, f"{path}: line {idx} failed_stdout_log")
            _validate_existing_path(failed_stderr_log, run_root, errors, f"{path}: line {idx} failed_stderr_log")
            if asset_preflight_report:
                _validate_existing_path(asset_preflight_report, run_root, errors, f"{path}: line {idx} asset_preflight_report")

            replacement_sha = first_nonblank(row, ["protocol_sha256", "runtime_protocol_sha256", "config_sha256", "simplerenv_protocol_sha256"])
            if replacement_sha != EXPECTED_PROTOCOL_SHA256:
                errors.append(f"{path}: line {idx} replacement protocol SHA {replacement_sha!r} != {EXPECTED_PROTOCOL_SHA256}")
            active_row = dict(row)
            active_row["job_id"] = replacement_job_id
            active_row["result_dir"] = replacement_result_dir
            active_row["replacement_source_tsv"] = str(path)
            active_row["supersedes_job_id"] = failed_job_id
            active_row["supersedes_result_dir"] = failed_result_dir
            chain = chains.setdefault(pair, [])
            chain.append(
                {
                    "job_id": replacement_job_id,
                    "result_dir": replacement_result_dir,
                    "stdout_log": first_nonblank(row, ["stdout_log", "log_out", "slurm_log", "log_path"]),
                    "stderr_log": first_nonblank(row, ["stderr_log", "log_err"]),
                    "source_tsv": str(path),
                    "fix_summary": first_nonblank(row, ["fix_summary"]),
                }
            )
            active_row["supersession_chain_job_ids"] = json.dumps([entry.get("job_id", "") for entry in chain])
            active_row["supersession_chain_result_dirs"] = json.dumps([entry.get("result_dir", "") for entry in chain])
            active_row["supersession_chain_tsvs"] = json.dumps([entry.get("source_tsv", "") for entry in chain])
            active_by_pair[pair] = active_row
            replacement_records.append(
                {
                    "replacement_order": len(replacement_records) + 1,
                    "policy": policy,
                    "condition": condition,
                    "failed_job_id": failed_job_id,
                    "failed_result_dir": failed_result_dir,
                    "failed_stdout_log": failed_stdout_log,
                    "failed_stderr_log": failed_stderr_log,
                    "replacement_job_id": replacement_job_id,
                    "replacement_result_dir": replacement_result_dir,
                    "replacement_stdout_log": first_nonblank(row, ["stdout_log", "log_out", "slurm_log", "log_path"]),
                    "replacement_stderr_log": first_nonblank(row, ["stderr_log", "log_err"]),
                    "replacement_sbatch_script": first_nonblank(row, ["sbatch_script", "command_file", "script_path"]),
                    "replacement_id": first_nonblank(row, ["replacement_id"]),
                    "replacement_source_tsv": str(path),
                    "asset_preflight_report": asset_preflight_report,
                    "previous_active_job_id": current_job_id,
                    "previous_active_result_dir": first_nonblank(current, ["result_dir", "results_dir", "output_dir", "replacement_result_dir"]),
                    "support_root": first_nonblank(row, ["support_root"]),
                    "fix_summary": first_nonblank(row, ["fix_summary"]),
                    "protocol_sha256": replacement_sha,
                    "chain_job_ids": json.dumps([entry.get("job_id", "") for entry in chain]),
                    "chain_result_dirs": json.dumps([entry.get("result_dir", "") for entry in chain]),
                    "chain_tsvs": json.dumps([entry.get("source_tsv", "") for entry in chain]),
                }
            )
    replacement_chains = [
        {
            "policy": pair[0],
            "condition": pair[1],
            "chain_job_ids": [entry.get("job_id", "") for entry in chain],
            "chain_result_dirs": [entry.get("result_dir", "") for entry in chain],
            "chain_tsvs": [entry.get("source_tsv", "") for entry in chain],
            "active_job_id": chain[-1].get("job_id", "") if chain else "",
            "active_result_dir": chain[-1].get("result_dir", "") if chain else "",
        }
        for pair, chain in sorted(chains.items())
        if len(chain) > 1
    ]
    return active_by_pair, replacement_records, replacement_chains


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_job_rows(result_dir: Path, policy: str, condition: str, job_id: str, errors: list[str]) -> list[dict[str, Any]]:
    csv_path = result_dir / "per_episode_results.csv"
    raw_xvla_path = result_dir / "videos" / "widowx_results.txt"
    rows: list[dict[str, Any]] = []
    if csv_path.is_file():
        _header, raw_rows = read_table(csv_path)
        for row in raw_rows:
            rows.append(
                {
                    "policy": norm_policy(first_nonblank(row, ["policy"]) or policy),
                    "condition": first_nonblank(row, ["condition", "protocol_condition"]) or condition,
                    "episode_id": first_nonblank(row, ["episode_id", "official_episode_id", "proc_id"]),
                    "success": bool_to_01(first_nonblank(row, ["success", "done"])),
                    "steps": first_nonblank(row, ["steps", "step_count"]),
                    "error": first_nonblank(row, ["error", "exception"]),
                    "timeout": first_nonblank(row, ["timeout"]),
                    "horizon": first_nonblank(row, ["horizon", "max_steps"]) or str(STANDARD_HORIZON),
                    "job_id": first_nonblank(row, ["job_id", "slurm_job_id"]) or job_id,
                    "artifact_path": first_nonblank(row, ["artifact_path", "video_path", "output"]),
                    "protocol_sha256": first_nonblank(row, ["protocol_sha256"]),
                    "protocol_sha256_source": "row:protocol_sha256" if first_nonblank(row, ["protocol_sha256"]) else "",
                }
            )
        return rows
    if policy == "xvla" and raw_xvla_path.is_file():
        try:
            raw_rows = load_json_lines(raw_xvla_path)
        except Exception as exc:
            errors.append(f"failed to parse X-VLA raw rows {raw_xvla_path}: {exc}")
            return []
        for row in raw_rows:
            rows.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "episode_id": str(row.get("proc_id", "")),
                    "success": "1" if bool(row.get("done")) else "0",
                    "steps": str(row.get("steps", "")),
                    "error": str(row.get("error", "") or ""),
                    "timeout": "",
                    "horizon": str(STANDARD_HORIZON),
                    "job_id": job_id,
                    "artifact_path": str(row.get("output", "") or ""),
                    "protocol_sha256": "",
                    "protocol_sha256_source": "",
                }
            )
        return rows
    errors.append(f"missing per-episode rows for {policy}/{condition}: expected {csv_path}")
    return []


def metadata_candidates(run_root: Path, result_dir: Path, policy: str) -> list[Path]:
    candidates = [
        run_root / "bundle" / "submission_support_files.sha256",
        run_root / "bundle" / "runtime_asset_manifest.json",
        run_root / "bundle" / "launch_preflight_report.json",
        run_root / "bundle" / "code_config_snapshot.sha256",
        run_root / "bundle" / "code_config_snapshot_hashes.json",
        run_root / "bundle" / "launch_snapshot_hashes.sha256",
        run_root / "metadata" / "code_config_snapshot_hashes.json",
        run_root / "metadata" / "accepted_asset_hashes.json",
        run_root / "metadata" / "runtime_assets.json",
        result_dir / "runtime_metadata.json",
        result_dir / "run_metadata.json",
        result_dir / "metadata.json",
        result_dir / "code_config_snapshot_hashes.json",
        result_dir / "accepted_asset_hashes.json",
        result_dir / "runtime_assets.json",
        result_dir / "preflight.json",
        result_dir / "preflight_report.json",
        result_dir / "manifest.json",
        result_dir / "summary.json",
        result_dir / "server" / "info.json",
        run_root / "metadata" / f"{policy}.json",
        run_root / "policy_metadata" / f"{policy}.json",
        run_root / "policy_runtime_metadata.json",
        run_root / "checkpoint_hashes.json",
    ]
    return [p for p in candidates if p.is_file()]


def job_metadata_candidates(result_dir: Path) -> list[Path]:
    candidates = [
        result_dir / "runtime_metadata.json",
        result_dir / "run_metadata.json",
        result_dir / "metadata.json",
        result_dir / "preflight.json",
        result_dir / "preflight_report.json",
        result_dir / "manifest.json",
        result_dir / "summary.json",
        result_dir / "server" / "info.json",
    ]
    return [p for p in candidates if p.is_file()]


def flatten_keys(payload: Any, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_keys(value, child))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            flat.update(flatten_keys(value, f"{prefix}.{idx}"))
    else:
        flat[prefix] = str(payload)
    return flat


def add_metadata_file(path: Path, flat: dict[str, str], errors: list[str], policy: str) -> bool:
    text = path.read_text(errors="replace")
    if path.suffix == ".json":
        try:
            payload = json.loads(text)
        except Exception as exc:
            errors.append(f"{policy}: failed to parse metadata {path}: {exc}")
            return False
        for key, value in flatten_keys(payload).items():
            flat[f"{path.name}.{key}"] = value
        return True
    flat[path.name] = text
    return True


def has_snapshot_hash(keys: dict[str, str], *, kind: str) -> bool:
    for key, value in keys.items():
        blob = f"{key} {value}".lower()
        if kind not in blob:
            continue
        if "snapshot" not in blob and "support_files" not in blob and "launch" not in blob:
            continue
        if ("sha" in blob or "hash" in blob or "digest" in blob) and HEX64_RE.search(value):
            return True
    return False


def has_checkpoint_or_stat_hash(keys: dict[str, str]) -> bool:
    for key, value in keys.items():
        blob = f"{key} {value}".lower()
        if "protocol" in key.lower():
            continue
        if not any(token in blob for token in ["checkpoint", "weight", "stat", "normalization", "dataset_statistics", "processor", "tokenizer", "config"]):
            continue
        if ("sha" in blob or "hash" in blob or "digest" in blob) and HEX64_RE.search(value):
            return True
    return False


def validate_runtime_asset_hashes(policy: str, keys: dict[str, str], asset_manifest: dict[str, Any], errors: list[str]) -> None:
    expected_hashes_by_asset = asset_manifest.get("asset_hashes") or {}
    expected_paths = asset_manifest.get("policy_runtime_paths", {}).get(policy, [])
    for color, asset_id in ACCEPTED_RUNTIME_ASSETS.items():
        expected_hashes = set(expected_hashes_by_asset.get(asset_id, []))
        if not expected_hashes:
            errors.append(f"{policy}: validator asset manifest lacks expected hashes for {color} asset {asset_id}")
    if not expected_paths:
        errors.append(f"{policy}: asset manifest lacks a runtime target path for this policy")


def flatten_metadata_paths(paths: list[Path], errors: list[str], context: str) -> dict[str, str]:
    flat: dict[str, str] = {}
    for path in paths:
        text = path.read_text(errors="replace")
        if path.suffix == ".json":
            try:
                payload = json.loads(text)
            except Exception as exc:
                errors.append(f"{context}: failed to parse metadata {path}: {exc}")
                continue
            for key, value in flatten_keys(payload).items():
                flat[f"{path.name}.{key}"] = value
        else:
            flat[path.name] = text
    return flat


def extract_protocol_sha_from_job_metadata(paths: list[Path], expected_sha: str, context: str, errors: list[str]) -> tuple[str, str]:
    flat = flatten_metadata_paths(paths, errors, context)
    found: list[tuple[str, str]] = []
    for key, value in flat.items():
        key_l = key.lower()
        value_s = str(value).strip()
        if not value_s:
            continue
        if ("protocol" in key_l and "sha" in key_l) or key_l.endswith("simplerenv_protocol_sha256"):
            match = HEX64_RE.search(value_s)
            if match:
                found.append((match.group(0), key))
        elif "simplerenv_protocol_sha256" in value_s.lower():
            for match in HEX64_RE.finditer(value_s):
                found.append((match.group(0), key))
    if not found:
        return "", ""
    bad = [(sha, source) for sha, source in found if sha != expected_sha]
    if bad:
        errors.append(f"{context}: runtime metadata has non-final Protocol A-E SHA values: {bad[:5]} expected={expected_sha}")
    good_sources = [source for sha, source in found if sha == expected_sha]
    if good_sources:
        return expected_sha, "job_metadata:" + ";".join(good_sources[:3])
    return found[0][0], "job_metadata:" + found[0][1]


def validate_job_metadata_contract(
    *,
    policy: str,
    condition: str,
    result_dir: Path,
    paths: list[Path],
    expected_sha: str,
    asset_manifest: dict[str, Any],
    errors: list[str],
) -> None:
    runtime_path = result_dir / "runtime_metadata.json"
    if not runtime_path.is_file():
        errors.append(f"{policy}/{condition}: missing runtime metadata: {runtime_path}")
    flat = flatten_metadata_paths(paths, errors, f"{policy}/{condition}")
    lowered = {key.lower(): str(value).strip() for key, value in flat.items() if str(value).strip()}
    if not lowered:
        return
    if not any(key.endswith(".policy") and norm_policy(value) == policy for key, value in lowered.items()):
        errors.append(f"{policy}/{condition}: runtime metadata does not capture matching policy")
    if not any(key.endswith(".condition") and value == condition for key, value in lowered.items()):
        errors.append(f"{policy}/{condition}: runtime metadata does not capture matching condition")
    horizon_values = [value for key, value in lowered.items() if key.endswith(".horizon")]
    if not horizon_values:
        errors.append(f"{policy}/{condition}: runtime metadata does not capture horizon")
    for value in horizon_values:
        try:
            horizon = int(value)
        except Exception:
            errors.append(f"{policy}/{condition}: runtime metadata horizon is not an integer: {value!r}")
            continue
        if horizon != STANDARD_HORIZON:
            errors.append(f"{policy}/{condition}: runtime metadata horizon {horizon} != {STANDARD_HORIZON}")
    blob = "\n".join(f"{key}: {value}" for key, value in lowered.items())
    if expected_sha not in blob:
        errors.append(f"{policy}/{condition}: runtime metadata does not capture final Protocol A-E SHA {expected_sha}")
    manifest_path = str(asset_manifest.get("path", ""))
    manifest_sha = str(asset_manifest.get("sha256", ""))
    if manifest_path not in blob and (not manifest_sha or manifest_sha not in blob):
        errors.append(f"{policy}/{condition}: runtime metadata does not reference required asset manifest path or SHA")


def validate_policy_metadata(policy: str, paths: list[Path], expected_sha: str, asset_manifest: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    payloads = []
    flat: dict[str, str] = {}
    for path in paths:
        if add_metadata_file(path, flat, errors, policy):
            payloads.append(str(path))
    keys = {k.lower(): v for k, v in flat.items() if str(v).strip()}
    if not payloads:
        errors.append(f"{policy}: missing policy/job runtime metadata")
    for key, value in keys.items():
        if "protocol" in key and "sha" in key and value.strip() != expected_sha:
            errors.append(f"{policy}: mixed Protocol A-E SHA in metadata {key}: {value.strip()} != {expected_sha}")
    validate_runtime_asset_hashes(policy, keys, asset_manifest, errors)
    return {"policy": policy, "metadata_files": payloads}


def evidence_references_asset_manifest(
    submission_rows: dict[tuple[str, str], dict[str, str]],
    metadata_paths_by_policy: dict[str, set[Path]],
    asset_manifest: dict[str, Any],
) -> bool:
    manifest_path = str(asset_manifest.get("path", ""))
    manifest_sha = str(asset_manifest.get("sha256", ""))
    if not manifest_path:
        return False
    chunks = []
    for row in submission_rows.values():
        chunks.extend(str(value) for value in row.values())
    for paths in metadata_paths_by_policy.values():
        for path in paths:
            try:
                chunks.append(path.read_text(errors="replace"))
            except Exception:
                pass
    blob = "\n".join(chunks)
    return manifest_path in blob or (bool(manifest_sha) and manifest_sha in blob)


def enrich_with_config(row: dict[str, Any], config_index: dict[tuple[str, int], dict[str, str]], expected_sha: str) -> dict[str, Any]:
    condition = str(row["condition"])
    episode_id = int(row["episode_id"])
    cfg = config_index.get((condition, episode_id), {})
    return {
        "policy": row["policy"],
        "policy_label": POLICIES.get(row["policy"], row["policy"]),
        "condition": condition,
        "episode_id": episode_id,
        "success": row["success"],
        "steps": row["steps"],
        "error": row["error"],
        "timeout": row["timeout"],
        "horizon": row["horizon"],
        "job_id": row["job_id"],
        "result_dir": row["result_dir"],
        "artifact_path": row["artifact_path"],
        "protocol_sha256": row.get("protocol_sha256"),
        "protocol_sha256_source": row.get("protocol_sha256_source", ""),
        "instruction": cfg.get("instruction", ""),
        "source_color": cfg.get("source_color", ""),
        "target_color": cfg.get("target_color", ""),
        "source_model_id": cfg.get("source_model_id", ""),
        "target_model_id": cfg.get("target_model_id", ""),
    }


def validate_job_rows(
    rows: list[dict[str, Any]],
    policy: str,
    condition: str,
    job_id: str,
    result_dir: Path,
    expected_sha: str,
    allow_error_rows: bool,
    skip_artifact_existence: bool,
    job_protocol_sha: str,
    job_protocol_sha_source: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    expected_ids = set(range(EXPECTED_ROWS_PER_JOB))
    seen: set[int] = set()
    duplicate_ids: list[int] = []
    validated: list[dict[str, Any]] = []
    if len(rows) != EXPECTED_ROWS_PER_JOB:
        errors.append(f"{policy}/{condition}: expected {EXPECTED_ROWS_PER_JOB} rows, found {len(rows)}")
    for idx, row in enumerate(rows, start=2):
        ctx = f"{policy}/{condition}: row {idx}"
        if row.get("policy") != policy:
            errors.append(f"{ctx}: policy {row.get('policy')!r} != {policy!r}")
        if row.get("condition") != condition:
            errors.append(f"{ctx}: condition {row.get('condition')!r} != {condition!r}")
        if str(row.get("job_id", "")).split(";", 1)[0] != str(job_id).split(";", 1)[0]:
            errors.append(f"{ctx}: job_id {row.get('job_id')!r} does not match submission {job_id!r}")
        try:
            episode_id = int(row.get("episode_id", ""))
        except Exception:
            errors.append(f"{ctx}: episode_id is not an integer: {row.get('episode_id')!r}")
            continue
        if episode_id in seen:
            duplicate_ids.append(episode_id)
        seen.add(episode_id)
        if episode_id not in expected_ids:
            errors.append(f"{ctx}: episode_id outside 0..287: {episode_id}")
        success = str(row.get("success", "")).strip()
        if success not in {"0", "1"}:
            errors.append(f"{ctx}: success must be 0/1, found {success!r}")
        try:
            steps = int(str(row.get("steps", "")).strip())
            if steps < 0 or steps > STANDARD_HORIZON:
                errors.append(f"{ctx}: steps {steps} outside 0..{STANDARD_HORIZON}")
        except Exception:
            errors.append(f"{ctx}: steps must be an integer, found {row.get('steps')!r}")
        try:
            horizon = int(str(row.get("horizon", "")).strip())
            if horizon != STANDARD_HORIZON:
                errors.append(f"{ctx}: horizon {horizon} != {STANDARD_HORIZON}")
        except Exception:
            errors.append(f"{ctx}: horizon must be an integer, found {row.get('horizon')!r}")
        error_text = str(row.get("error", "")).strip()
        timeout = truthy(row.get("timeout", ""))
        if (error_text or timeout) and not allow_error_rows:
            errors.append(f"{ctx}: error/timeout row is not allowed: error={error_text!r} timeout={row.get('timeout')!r}")
        protocol_sha = str(row.get("protocol_sha256", "")).strip()
        if not protocol_sha:
            if job_protocol_sha == expected_sha:
                row["protocol_sha256"] = job_protocol_sha
                row["protocol_sha256_source"] = job_protocol_sha_source
            else:
                errors.append(
                    f"{ctx}: missing runtime-captured protocol_sha256 in row and no final-SHA job metadata/job record evidence; "
                    "refusing to infer final config identity posthoc"
                )
        elif protocol_sha != expected_sha:
            errors.append(f"{ctx}: protocol_sha256 {protocol_sha} != {expected_sha}")
        else:
            row["protocol_sha256_source"] = row.get("protocol_sha256_source") or "row:protocol_sha256"
        artifact_text = str(row.get("artifact_path", "")).strip()
        if artifact_text and not skip_artifact_existence:
            artifact = resolve_path(artifact_text, result_dir)
            if not artifact.is_file():
                errors.append(f"{ctx}: artifact_path does not exist: {artifact}")
        row["result_dir"] = str(result_dir)
        validated.append(row)
    missing = sorted(expected_ids - seen)
    extras = sorted(seen - expected_ids)
    if missing:
        errors.append(f"{policy}/{condition}: missing episode IDs first20={missing[:20]}")
    if extras:
        errors.append(f"{policy}/{condition}: extra episode IDs first20={extras[:20]}")
    if duplicate_ids:
        errors.append(f"{policy}/{condition}: duplicate episode IDs first20={sorted(set(duplicate_ids))[:20]}")
    return validated


def summary_row(group: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(1 for r in rows if str(r.get("success")) == "1")
    error_rows = sum(1 for r in rows if str(r.get("error", "")).strip())
    timeout_rows = sum(1 for r in rows if truthy(r.get("timeout", "")))
    rate = successes / total if total else float("nan")
    return {
        **group,
        "successes": successes,
        "total": total,
        "success_rate": "" if math.isnan(rate) else f"{rate:.8f}",
        "error_rows": error_rows,
        "timeout_rows": timeout_rows,
    }


def current_repo_state(project_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "project_root": str(project_root),
        "head_sha": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": run_git(["status", "--short"]),
        "submodule_status": run_git(["submodule", "status", "--recursive"]),
    }


def tracked_large_files(project_root: Path, run_root: Path, output_dir: Path) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "--", str(run_root), str(output_dir)],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    bad = []
    for line in output.splitlines():
        if Path(line).suffix.lower() in BANNED_BUNDLE_SUFFIXES:
            bad.append(line)
    return bad


def output_large_files(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return [str(p) for p in output_dir.rglob("*") if p.is_file() and p.suffix.lower() in BANNED_BUNDLE_SUFFIXES]


def validate_run_snapshot(run_root: Path, expected_sha: str, asset_manifest: dict[str, Any], errors: list[str]) -> None:
    support_hashes = run_root / "bundle" / "submission_support_files.sha256"
    if not support_hashes.is_file():
        errors.append(f"missing code/support snapshot hash manifest: {support_hashes}")
    elif not support_hashes.read_text(errors="replace").strip():
        errors.append(f"empty code/support snapshot hash manifest: {support_hashes}")

    preflight = run_root / "bundle" / "launch_preflight_report.json"
    if not preflight.is_file():
        errors.append(f"missing launch preflight report: {preflight}")
        return
    try:
        payload = json.loads(preflight.read_text(errors="replace"))
    except Exception as exc:
        errors.append(f"failed to parse launch preflight report {preflight}: {exc}")
        return
    if payload.get("status") != "passed":
        errors.append(f"launch preflight report did not pass: {preflight}")
    if payload.get("job_count") != len(POLICIES) * len(CONDITIONS):
        errors.append(f"launch preflight job_count {payload.get('job_count')} != 35")
    if payload.get("policies") != list(POLICIES):
        errors.append(f"launch preflight policies mismatch: {payload.get('policies')}")
    if payload.get("conditions") != CONDITIONS:
        errors.append(f"launch preflight conditions mismatch: {payload.get('conditions')}")
    if payload.get("horizon") != STANDARD_HORIZON:
        errors.append(f"launch preflight horizon {payload.get('horizon')} != {STANDARD_HORIZON}")
    if payload.get("protocol_sha256") != expected_sha:
        errors.append(f"launch preflight protocol SHA {payload.get('protocol_sha256')} != {expected_sha}")
    manifest_path = str(asset_manifest.get("path", ""))
    manifest_sha = str(asset_manifest.get("sha256", ""))
    preflight_blob = json.dumps(payload, sort_keys=True)
    if manifest_path not in preflight_blob and (not manifest_sha or manifest_sha not in preflight_blob):
        errors.append(f"launch preflight report does not reference required asset manifest path or SHA")


def validate_per_job_report(result_dir: Path, policy: str, condition: str, errors: list[str]) -> None:
    path = result_dir / "validation_report.json"
    if not path.is_file():
        errors.append(f"{policy}/{condition}: missing per-job validation report: {path}")
        return
    try:
        payload = json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        errors.append(f"{policy}/{condition}: failed to parse per-job validation report {path}: {exc}")
        return
    status = payload.get("validation_status", payload.get("status"))
    if status != "passed":
        errors.append(f"{policy}/{condition}: per-job validation report did not pass: {status}")
    expected_rows = payload.get("expected_rows")
    actual_rows = payload.get("actual_rows")
    if expected_rows not in {None, EXPECTED_ROWS_PER_JOB}:
        errors.append(f"{policy}/{condition}: per-job expected_rows {expected_rows} != {EXPECTED_ROWS_PER_JOB}")
    if actual_rows not in {None, EXPECTED_ROWS_PER_JOB}:
        errors.append(f"{policy}/{condition}: per-job actual_rows {actual_rows} != {EXPECTED_ROWS_PER_JOB}")
    for key in ("error_rows", "timeout_rows"):
        value = payload.get(key)
        if value not in {None, 0}:
            errors.append(f"{policy}/{condition}: per-job {key}={value}")
    integer_step_rows = payload.get("integer_step_rows")
    non_integer_step_rows = payload.get("non_integer_step_rows")
    if integer_step_rows != EXPECTED_ROWS_PER_JOB:
        errors.append(f"{policy}/{condition}: per-job integer_step_rows {integer_step_rows} != {EXPECTED_ROWS_PER_JOB}")
    if non_integer_step_rows != 0:
        errors.append(f"{policy}/{condition}: per-job non_integer_step_rows {non_integer_step_rows} != 0")


def write_contract(path: Path, asset_manifest: dict[str, Any]) -> None:
    contract = {
        "run_shape": {"policies": list(POLICIES), "conditions": CONDITIONS, "rows_per_policy_condition": EXPECTED_ROWS_PER_JOB, "total_rows": EXPECTED_TOTAL_ROWS},
        "required_job_record_fields": [
            "policy",
            "condition",
            "job_id",
            "result_dir",
            "job_name plus job_id, or explicit stdout/stderr log paths",
            "command or command_file/sbatch_script",
            "protocol_sha256 if not captured per row/job metadata",
        ],
        "required_per_episode_identity": ["policy", "condition", "episode_id 0..287", "success 0/1", "steps", "horizon=60", "job_id", "runtime-captured protocol_sha256 per row or per job"],
        "required_metadata": [
            "launch snapshot hash manifest in bundle/submission_support_files.sha256",
            "frozen config SHA and sidecar",
            "launch preflight report with 35 policy x condition jobs",
            "frozen config SHA on every row",
            "per-job runtime_metadata.json with policy, condition, horizon, final protocol SHA, and asset manifest reference",
            "runtime asset manifest with target runtime paths and hashes for accepted blue/red/white assets",
            "per-job validation_report.json with integer_step_rows=288 and non_integer_step_rows=0",
        ],
        "accepted_runtime_assets": ACCEPTED_RUNTIME_ASSETS,
        "required_runtime_asset_manifest": asset_manifest.get("path", ""),
        "required_runtime_asset_manifest_sha256": asset_manifest.get("sha256", ""),
        "curated_outputs": [
            "validation_report.json",
            "per_episode_results_all.csv",
            "per_policy_condition_summary.csv",
            "per_condition_summary.csv",
            "per_policy_summary.csv",
            "job_records.csv",
            "replacement_job_records.csv",
            "test_set_manifest.csv",
        ],
        "excluded_from_curated_bundle": sorted(BANNED_BUNDLE_SUFFIXES),
    }
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    run_root = args.run_root.resolve()
    protocol_config = args.protocol_config.resolve()
    asset_manifest_path = resolve_asset_manifest_path(run_root, args.asset_manifest)
    output_dir = (args.output_dir or (run_root / "final_summary")).resolve()
    errors: list[str] = []

    config = load_protocol_config(protocol_config, args.expected_protocol_sha256, errors)
    asset_manifest = load_asset_manifest(asset_manifest_path, errors)
    validate_run_snapshot(run_root, args.expected_protocol_sha256, asset_manifest, errors)
    test_rows = test_set_rows(config, args.expected_protocol_sha256)
    config_index = {(str(r["condition"]), int(r["episode_id"])): r for r in test_rows if str(r.get("episode_id", "")).isdigit()}

    submission_path = find_submission_jobs(run_root, args.submission_jobs)
    original_by_pair = parse_submission(submission_path, run_root, errors)
    replacement_paths = find_replacement_submissions(run_root, args.replacement_submission)
    by_pair, replacement_records, replacement_chains = apply_replacement_submissions(
        run_root=run_root,
        original_by_pair=original_by_pair,
        replacement_paths=replacement_paths,
        errors=errors,
    )
    expected_pairs = {(p, c) for p in POLICIES for c in CONDITIONS}
    active_pairs = set(by_pair)
    if len(by_pair) != len(expected_pairs):
        errors.append(f"active submission matrix expected {len(expected_pairs)} cells, found {len(by_pair)}")
    if active_pairs != expected_pairs:
        errors.append(f"active submission matrix mismatch missing={sorted(expected_pairs - active_pairs)} extra={sorted(active_pairs - expected_pairs)}")

    all_rows: list[dict[str, Any]] = []
    by_policy_condition: dict[tuple[str, str], list[dict[str, Any]]] = {}
    metadata_files_by_policy: dict[str, set[Path]] = defaultdict(set)

    for policy in POLICIES:
        for condition in CONDITIONS:
            sub = by_pair.get((policy, condition), {})
            job_id = first_nonblank(sub, ["job_id", "jobid", "slurm_job_id"])
            result_dir_text = first_nonblank(sub, ["result_dir", "results_dir", "output_dir"]) or str(run_root / policy / condition)
            result_dir = resolve_path(result_dir_text, run_root)
            job_metadata_paths = job_metadata_candidates(result_dir)
            metadata_files_by_policy[policy].update(metadata_candidates(run_root, result_dir, policy))
            validate_job_metadata_contract(
                policy=policy,
                condition=condition,
                result_dir=result_dir,
                paths=job_metadata_paths,
                expected_sha=args.expected_protocol_sha256,
                asset_manifest=asset_manifest,
                errors=errors,
            )
            validate_per_job_report(result_dir, policy, condition, errors)
            job_record_sha = first_nonblank(sub, ["protocol_sha256", "runtime_protocol_sha256", "config_sha256", "simplerenv_protocol_sha256"])
            if job_record_sha and job_record_sha != args.expected_protocol_sha256:
                errors.append(f"{policy}/{condition}: submission job record protocol SHA {job_record_sha} != {args.expected_protocol_sha256}")
            metadata_sha, metadata_sha_source = extract_protocol_sha_from_job_metadata(
                job_metadata_paths,
                args.expected_protocol_sha256,
                f"{policy}/{condition}",
                errors,
            )
            if job_record_sha == args.expected_protocol_sha256:
                job_protocol_sha = job_record_sha
                job_protocol_sha_source = "submission_job_record"
            else:
                job_protocol_sha = metadata_sha
                job_protocol_sha_source = metadata_sha_source
            rows = read_job_rows(result_dir, policy, condition, job_id, errors)
            rows = validate_job_rows(
                rows,
                policy,
                condition,
                job_id,
                result_dir,
                args.expected_protocol_sha256,
                args.allow_error_rows,
                args.skip_artifact_existence,
                job_protocol_sha,
                job_protocol_sha_source,
                errors,
            )
            enriched = [enrich_with_config(r, config_index, args.expected_protocol_sha256) for r in rows if str(r.get("episode_id", "")).isdigit()]
            by_policy_condition[(policy, condition)] = enriched
            all_rows.extend(enriched)

    metadata_summary = [
        validate_policy_metadata(policy, sorted(metadata_files_by_policy.get(policy, set())), args.expected_protocol_sha256, asset_manifest, errors)
        for policy in POLICIES
    ]
    if not evidence_references_asset_manifest(by_pair, metadata_files_by_policy, asset_manifest):
        errors.append(
            "run evidence does not reference the required Protocol A-E runtime asset manifest path or SHA: "
            f"{asset_manifest.get('path', str(asset_manifest_path))}"
        )

    per_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        per_policy[row["policy"]].append(row)
        per_condition[row["condition"]].append(row)

    for policy in POLICIES:
        if len(per_policy[policy]) != len(CONDITIONS) * EXPECTED_ROWS_PER_JOB:
            errors.append(f"{policy}: expected {len(CONDITIONS) * EXPECTED_ROWS_PER_JOB} rows, found {len(per_policy[policy])}")
    for condition in CONDITIONS:
        if len(per_condition[condition]) != len(POLICIES) * EXPECTED_ROWS_PER_JOB:
            errors.append(f"{condition}: expected {len(POLICIES) * EXPECTED_ROWS_PER_JOB} rows, found {len(per_condition[condition])}")
    if len(all_rows) != EXPECTED_TOTAL_ROWS:
        errors.append(f"full run: expected {EXPECTED_TOTAL_ROWS} rows, found {len(all_rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_contract(output_dir / "artifact_bundle_contract.json", asset_manifest)
    write_csv(output_dir / "test_set_manifest.csv", ["condition", "episode_id", "protocol_sha256", "instruction", "source_color", "target_color", "source_model_id", "target_model_id"], test_rows)

    job_records = []
    for (policy, condition), row in sorted(by_pair.items()):
        if policy in POLICIES and condition in CONDITIONS:
            job_records.append(
                {
                    "policy": policy,
                    "policy_label": POLICIES[policy],
                    "condition": condition,
                    "job_id": first_nonblank(row, ["job_id", "jobid", "slurm_job_id"]),
                    "result_dir": first_nonblank(row, ["result_dir", "results_dir", "output_dir"]),
                    "command": first_nonblank(row, ["command", "resolved_command"]),
                    "command_file": first_nonblank(row, ["command_file", "sbatch_script", "script_path"]),
                    "stdout_log": first_nonblank(row, ["stdout_log", "log_out", "slurm_log", "log_path"]),
                    "stderr_log": first_nonblank(row, ["stderr_log", "log_err"]),
                    "protocol_sha256": first_nonblank(row, ["protocol_sha256", "runtime_protocol_sha256", "config_sha256", "simplerenv_protocol_sha256"]),
                    "replacement_source_tsv": first_nonblank(row, ["replacement_source_tsv"]),
                    "supersedes_job_id": first_nonblank(row, ["supersedes_job_id"]),
                    "supersedes_result_dir": first_nonblank(row, ["supersedes_result_dir"]),
                    "supersession_chain_job_ids": first_nonblank(row, ["supersession_chain_job_ids"]),
                    "supersession_chain_result_dirs": first_nonblank(row, ["supersession_chain_result_dirs"]),
                    "supersession_chain_tsvs": first_nonblank(row, ["supersession_chain_tsvs"]),
                }
            )
    write_csv(
        output_dir / "job_records.csv",
        [
            "policy",
            "policy_label",
            "condition",
            "job_id",
            "result_dir",
            "command",
            "command_file",
            "stdout_log",
            "stderr_log",
            "protocol_sha256",
            "replacement_source_tsv",
            "supersedes_job_id",
            "supersedes_result_dir",
            "supersession_chain_job_ids",
            "supersession_chain_result_dirs",
            "supersession_chain_tsvs",
        ],
        job_records,
    )
    write_csv(
        output_dir / "replacement_job_records.csv",
        [
            "policy",
            "condition",
            "failed_job_id",
            "failed_result_dir",
            "failed_stdout_log",
            "failed_stderr_log",
            "replacement_job_id",
            "replacement_result_dir",
            "replacement_stdout_log",
            "replacement_stderr_log",
            "replacement_sbatch_script",
            "replacement_id",
            "replacement_source_tsv",
            "asset_preflight_report",
            "previous_active_job_id",
            "previous_active_result_dir",
            "support_root",
            "fix_summary",
            "protocol_sha256",
            "replacement_order",
            "chain_job_ids",
            "chain_result_dirs",
            "chain_tsvs",
        ],
        replacement_records,
    )

    tracked_bad = tracked_large_files(project_root, run_root, output_dir)
    output_bad = output_large_files(output_dir)
    if tracked_bad:
        errors.append(f"tracked large video/observation artifacts are in git: {tracked_bad[:20]}")
    if output_bad:
        errors.append(f"curated output dir contains large video/observation artifacts: {output_bad[:20]}")

    report = {
        "validation_status": "failed" if errors else "passed",
        "run_root": str(run_root),
        "submission_jobs": str(submission_path) if submission_path else None,
        "replacement_submissions": [str(path) for path in replacement_paths],
        "output_dir": str(output_dir),
        "protocol_config": str(protocol_config),
        "protocol_sha256": args.expected_protocol_sha256,
        "asset_manifest": asset_manifest,
        "expected_total_rows": EXPECTED_TOTAL_ROWS,
        "actual_total_rows": len(all_rows),
        "row_counts": {
            "per_policy": {policy: len(per_policy[policy]) for policy in POLICIES},
            "per_condition": {condition: len(per_condition[condition]) for condition in CONDITIONS},
            "per_policy_condition": {f"{policy}/{condition}": len(by_policy_condition[(policy, condition)]) for policy in POLICIES for condition in CONDITIONS},
        },
        "metadata_summary": metadata_summary,
        "replacement_summary": {
            "active_replacement_rows": len(replacement_chains),
            "total_replacement_rows": len(replacement_records),
            "chains": replacement_chains,
            "records": replacement_records,
        },
        "current_repo_state": current_repo_state(project_root),
        "large_artifact_git_check": {"tracked_banned_files": tracked_bad, "output_banned_files": output_bad},
        "errors": errors,
    }
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    write_csv(output_dir / "per_episode_results_all.csv", EPISODE_COLUMNS, all_rows)
    write_csv(
        output_dir / "per_policy_condition_summary.csv",
        ["policy", "policy_label", "condition", "successes", "total", "success_rate", "error_rows", "timeout_rows"],
        [
            summary_row({"policy": p, "policy_label": POLICIES[p], "condition": c}, by_policy_condition[(p, c)])
            for p in POLICIES
            for c in CONDITIONS
        ],
    )
    write_csv(
        output_dir / "per_condition_summary.csv",
        ["condition", "successes", "total", "success_rate", "error_rows", "timeout_rows"],
        [summary_row({"condition": c}, per_condition[c]) for c in CONDITIONS],
    )
    write_csv(
        output_dir / "per_policy_summary.csv",
        ["policy", "policy_label", "successes", "total", "success_rate", "error_rows", "timeout_rows"],
        [summary_row({"policy": p, "policy_label": POLICIES[p]}, per_policy[p]) for p in POLICIES],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
