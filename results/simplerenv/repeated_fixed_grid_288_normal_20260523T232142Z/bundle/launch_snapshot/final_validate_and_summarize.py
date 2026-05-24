#!/usr/bin/env python3
"""Final validation and summaries for the v2 repeated fixed-grid calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

V2_DIR = Path("/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/scratch/repeated_fixed_grid_calibration_288_20260523_v2")
DEFAULT_RUN_ID = "repeated_fixed_grid_calibration_288_20260523_v2"

SUBMISSION_COLUMNS = ["job_id", "policy", "task", "sbatch_script", "result_dir", "submitted_at_utc", "run_id"]
RESULT_COLUMNS = [
    "policy",
    "task",
    "official_episode_id",
    "repeat_id",
    "seed",
    "success",
    "steps",
    "error",
    "timeout",
    "horizon",
    "job_id",
    "video_path",
]
TASKS = OrderedDict([("stack", 60), ("carrot", 60), ("spoon", 60), ("eggplant", 120)])
POLICIES = OrderedDict(
    [
        ("cogact", {"label": "CogACT-Base", "result_policy": "cogact"}),
        ("spatialvla", {"label": "SpatialVLA", "result_policy": "spatialvla"}),
        ("internvla_m1", {"label": "InternVLA-M1", "result_policy": "InternVLA-M1"}),
        ("xvla", {"label": "X-VLA-WidowX", "result_policy": "xvla"}),
        ("dexbotic", {"label": "Dexbotic / DB-MemVLA", "result_policy": "dexbotic"}),
    ]
)

COGACT_SPATIAL_BASE_SEED = 202605230
COGACT_SPATIAL_POLICY_OFFSETS = {"cogact": 100_000, "spatialvla": 200_000}
TASK_OFFSETS_1K = {"stack": 1_000, "carrot": 2_000, "spoon": 3_000, "eggplant": 4_000}
INTERNVLA_BASE_SEED = 20260523
XVLA_DEXBOTIC_BASE_SEED = 20260523
XVLA_DEXBOTIC_POLICY_INDEX = {"xvla": 0, "dexbotic": 1}
TASK_INDEX = {"stack": 0, "carrot": 1, "spoon": 2, "eggplant": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--submission-jobs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-error-rows", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_result_dir(run_root: Path, policy: str, task: str) -> Path:
    return run_root / policy / task


def expected_seed(policy: str, task: str, repeat_id: int, episode_id: int) -> int:
    if policy in COGACT_SPATIAL_POLICY_OFFSETS:
        return COGACT_SPATIAL_BASE_SEED + COGACT_SPATIAL_POLICY_OFFSETS[policy] + TASK_OFFSETS_1K[task] + repeat_id * 24 + episode_id
    if policy == "internvla_m1":
        return INTERNVLA_BASE_SEED + TASK_INDEX[task] * 10_000 + repeat_id * 24 + episode_id
    if policy in XVLA_DEXBOTIC_POLICY_INDEX:
        return XVLA_DEXBOTIC_BASE_SEED + XVLA_DEXBOTIC_POLICY_INDEX[policy] * 1_000_000 + TASK_INDEX[task] * 10_000 + repeat_id * 100 + episode_id
    raise KeyError(policy)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def truthy_timeout(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "timeout"}


def resolve_artifact(path_text: str, csv_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = csv_dir / path
    return path


def path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def current_snapshot_hashes(snapshot_dir: Path, manifest_root: Path) -> list[str]:
    lines = []
    for path in sorted(p for p in snapshot_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
        rel_path = path.resolve().relative_to(manifest_root.resolve())
        lines.append(f"{sha256_file(path)}  {rel_path.as_posix()}")
    return lines


def validate_run_bundle(run_root: Path, errors: list[str]) -> None:
    bundle_dir = run_root / "bundle"
    metadata_path = bundle_dir / "submission_snapshot_metadata.txt"
    hash_path = bundle_dir / "submission_support_files.sha256"
    if not metadata_path.is_file():
        errors.append(f"missing launch snapshot metadata: {metadata_path}")
        return
    if not hash_path.is_file():
        errors.append(f"missing launch snapshot hash manifest: {hash_path}")
        return
    metadata = {}
    for line in metadata_path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    snapshot_dir = Path(metadata.get("snapshot_dir", ""))
    if not snapshot_dir.is_dir():
        errors.append(f"launch snapshot dir missing: {snapshot_dir}")
        return
    expected = [line.strip() for line in hash_path.read_text(errors="replace").splitlines() if line.strip()]
    expected_paths = []
    for line in expected:
        try:
            _digest, path_text = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed launch snapshot hash manifest line: {line}")
            continue
        expected_paths.append(Path(path_text))
    manifest_root = None
    for candidate in [snapshot_dir.resolve(), *snapshot_dir.resolve().parents]:
        if expected_paths and all((candidate / path).is_file() for path in expected_paths):
            manifest_root = candidate
            break
    if manifest_root is None:
        errors.append(f"could not resolve launch snapshot hash manifest paths relative to {snapshot_dir}")
        return
    current = current_snapshot_hashes(snapshot_dir, manifest_root)
    if current != expected:
        errors.append(f"launch snapshot hash manifest mismatch: {hash_path}")


def one_sided_wilson_bounds(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.6448536269514722
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def format_float(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.8f}"


def validate_submission(path: Path, run_root: Path, run_id: str, errors: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        errors.append(f"missing submission_jobs.csv: {path}")
        return {}
    header, rows = read_csv(path)
    if header != SUBMISSION_COLUMNS:
        errors.append(f"submission_jobs.csv header mismatch: {header} != {SUBMISSION_COLUMNS}")
    if len(rows) != 20:
        errors.append(f"submission_jobs.csv must have exactly 20 rows, found {len(rows)}")
    expected_pairs = {(policy, task) for policy in POLICIES for task in TASKS}
    seen_pairs: set[tuple[str, str]] = set()
    seen_jobs: set[str] = set()
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for idx, row in enumerate(rows, start=2):
        policy = row.get("policy", "")
        task = row.get("task", "")
        pair = (policy, task)
        if pair in seen_pairs:
            errors.append(f"submission_jobs.csv line {idx}: duplicate policy/task pair {pair}")
        seen_pairs.add(pair)
        by_pair[pair] = row
        if pair not in expected_pairs:
            errors.append(f"submission_jobs.csv line {idx}: unexpected policy/task pair {pair}")
        job_id = row.get("job_id", "")
        if not job_id:
            errors.append(f"submission_jobs.csv line {idx}: blank job_id")
        if job_id in seen_jobs:
            errors.append(f"submission_jobs.csv line {idx}: duplicate job_id {job_id}")
        seen_jobs.add(job_id)
        if row.get("run_id") != run_id:
            errors.append(f"submission_jobs.csv line {idx}: run_id {row.get('run_id')} != {run_id}")
        expected_dir = expected_result_dir(run_root, policy, task) if policy in POLICIES and task in TASKS else None
        if expected_dir is not None and Path(row.get("result_dir", "")) != expected_dir:
            errors.append(f"submission_jobs.csv line {idx}: result_dir {row.get('result_dir')} != {expected_dir}")
        sbatch_script = Path(row.get("sbatch_script", ""))
        if not sbatch_script.is_file():
            errors.append(f"submission_jobs.csv line {idx}: missing sbatch_script {sbatch_script}")
        elif "--array" in sbatch_script.read_text(errors="replace"):
            errors.append(f"submission_jobs.csv line {idx}: sbatch_script contains --array: {sbatch_script}")
        if expected_dir is not None:
            snapshot_root = run_root / "bundle" / "launch_snapshot"
            if not path_under(sbatch_script, snapshot_root):
                errors.append(f"submission_jobs.csv line {idx}: sbatch_script is not under launch snapshot: {sbatch_script}")
    missing = sorted(expected_pairs - seen_pairs)
    extra = sorted(seen_pairs - expected_pairs)
    if missing:
        errors.append(f"submission_jobs.csv missing policy/task pairs: {missing}")
    if extra:
        errors.append(f"submission_jobs.csv has extra policy/task pairs: {extra}")
    return by_pair


def validate_result_csv(
    csv_path: Path,
    policy: str,
    task: str,
    expected_job_id: str,
    allow_error_rows: bool,
    artifact_paths: dict[str, tuple[str, str, int, int]],
    errors: list[str],
) -> list[dict[str, str]]:
    if not csv_path.is_file():
        errors.append(f"missing per_episode_results.csv for {policy}/{task}: {csv_path}")
        return []
    header, rows = read_csv(csv_path)
    if header != RESULT_COLUMNS:
        errors.append(f"{csv_path}: header mismatch: {header} != {RESULT_COLUMNS}")
    if len(rows) != 288:
        errors.append(f"{csv_path}: expected 288 rows, found {len(rows)}")
    expected_keys = {(repeat_id, episode_id) for repeat_id in range(12) for episode_id in range(24)}
    seen_keys: set[tuple[int, int]] = set()
    duplicate_keys: list[tuple[int, int]] = []
    expected_result_policy = POLICIES[policy]["result_policy"]
    job_id_expected_prefix = expected_job_id.split(";", 1)[0]

    for idx, row in enumerate(rows, start=2):
        row_context = f"{csv_path}: line {idx}"
        if row.get("policy") != expected_result_policy:
            errors.append(f"{row_context}: policy {row.get('policy')} != {expected_result_policy}")
        if row.get("task") != task:
            errors.append(f"{row_context}: task {row.get('task')} != {task}")
        if row.get("job_id", "").split(";", 1)[0] != job_id_expected_prefix:
            errors.append(f"{row_context}: job_id {row.get('job_id')} does not match submission job_id {expected_job_id}")
        try:
            repeat_id = int(row["repeat_id"])
            episode_id = int(row["official_episode_id"])
            seed = int(row["seed"])
            horizon = int(row["horizon"])
        except Exception as exc:
            errors.append(f"{row_context}: failed to parse integer key/seed/horizon: {exc}")
            continue
        key = (repeat_id, episode_id)
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)
        if key not in expected_keys:
            errors.append(f"{row_context}: key outside expected repeat/episode grid: {key}")
        expected = expected_seed(policy, task, repeat_id, episode_id)
        if seed != expected:
            errors.append(f"{row_context}: seed {seed} != {expected}")
        if horizon != TASKS[task]:
            errors.append(f"{row_context}: horizon {horizon} != {TASKS[task]}")
        error_text = row.get("error", "").strip()
        timeout = truthy_timeout(row.get("timeout", ""))
        if (error_text or timeout) and not allow_error_rows:
            errors.append(f"{row_context}: error/timeout row is not allowed: error={error_text!r} timeout={row.get('timeout')!r}")
        if not error_text and not timeout:
            if row.get("success") not in {"0", "1"}:
                errors.append(f"{row_context}: success must be 0/1 for non-error rows, found {row.get('success')!r}")
            try:
                steps = int(row.get("steps", ""))
                if steps < 0 or steps > TASKS[task]:
                    errors.append(f"{row_context}: steps {steps} outside 0..{TASKS[task]}")
            except Exception as exc:
                errors.append(f"{row_context}: steps must be integer for non-error rows: {exc}")
            video_path = row.get("video_path", "").strip()
            if not video_path:
                errors.append(f"{row_context}: blank video_path for non-error row")
            else:
                video = resolve_artifact(video_path, csv_path.parent)
                if not video.is_file():
                    errors.append(f"{row_context}: video_path does not exist: {video}")
                if not path_under(video, csv_path.parent):
                    errors.append(f"{row_context}: video_path is outside result dir: {video}")
                artifact_key = str(video)
                previous = artifact_paths.get(artifact_key)
                if previous is not None:
                    errors.append(f"{row_context}: duplicate video_path also used by {previous}: {artifact_key}")
                artifact_paths[artifact_key] = (policy, task, repeat_id, episode_id)
    missing = sorted(expected_keys - seen_keys)
    extra = sorted(seen_keys - expected_keys)
    if missing:
        errors.append(f"{csv_path}: missing keys first20={missing[:20]}")
    if extra:
        errors.append(f"{csv_path}: extra keys first20={extra[:20]}")
    if duplicate_keys:
        errors.append(f"{csv_path}: duplicate keys first20={sorted(set(duplicate_keys))[:20]}")
    return rows


def validate_side_artifacts(result_dir: Path, policy: str, task: str, errors: list[str]) -> None:
    manifest = result_dir / "manifest.csv"
    if not manifest.is_file():
        errors.append(f"missing manifest.csv for {policy}/{task}: {manifest}")

    if policy == "internvla_m1":
        validation = result_dir / "validation_summary.json"
    else:
        validation = result_dir / "validation_report.json"
    if not validation.is_file():
        errors.append(f"missing per-job validation report for {policy}/{task}: {validation}")
    else:
        try:
            payload = json.loads(validation.read_text(errors="replace"))
            status = payload.get("validation_status", payload.get("status"))
            if status != "passed":
                errors.append(f"per-job validation did not pass for {policy}/{task}: {validation}")
        except Exception as exc:
            errors.append(f"failed to parse per-job validation report for {policy}/{task}: {validation}: {exc}")

    if policy in {"cogact", "spatialvla"}:
        runtime = result_dir / "runtime_metadata.json"
        if not runtime.is_file():
            errors.append(f"missing runtime metadata for {policy}/{task}: {runtime}")
            return
        try:
            payload = json.loads(runtime.read_text(errors="replace"))
        except Exception as exc:
            errors.append(f"failed to parse runtime metadata for {policy}/{task}: {runtime}: {exc}")
            return
        for key in ["stochastic_action_sampling", "policy_state_reset", "rollout_seed_control"]:
            if not str(payload.get(key, "")).strip():
                errors.append(f"runtime metadata for {policy}/{task} missing {key}: {runtime}")
    else:
        note_names = ["seed_caveat.txt", "seed_control_caveat.txt"]
        if not any((result_dir / name).is_file() for name in note_names):
            errors.append(f"missing stochastic/reset note for {policy}/{task}: expected one of {note_names} under {result_dir}")


def summary_row(policy: str, task: str, rows: list[dict[str, str]]) -> dict[str, object]:
    total = len(rows)
    successes = sum(1 for row in rows if row.get("success") == "1")
    error_rows = sum(1 for row in rows if row.get("error", "").strip())
    timeout_rows = sum(1 for row in rows if truthy_timeout(row.get("timeout", "")))
    lower, upper = one_sided_wilson_bounds(successes, total)
    rate = successes / total if total else float("nan")
    return {
        "policy": policy,
        "policy_label": POLICIES[policy]["label"],
        "task": task,
        "successes": successes,
        "total": total,
        "success_rate": format_float(rate),
        "ci95_one_sided_lower": format_float(lower),
        "ci95_one_sided_upper": format_float(upper),
        "ci95_one_sided_lower_width": format_float(rate - lower if total else float("nan")),
        "ci95_one_sided_upper_width": format_float(upper - rate if total else float("nan")),
        "error_rows": error_rows,
        "timeout_rows": timeout_rows,
    }


def main() -> int:
    args = parse_args()
    run_root = args.run_root or (V2_DIR / "runs" / args.run_id)
    submission_jobs = args.submission_jobs or (run_root / "submission_jobs.csv")
    output_dir = args.output_dir or (run_root / "final_summary")
    errors: list[str] = []
    validate_run_bundle(run_root, errors)
    by_pair = validate_submission(submission_jobs, run_root, args.run_id, errors)
    all_rows: list[dict[str, str]] = []
    by_policy: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_policy_task: dict[tuple[str, str], list[dict[str, str]]] = {}
    artifact_paths: dict[str, tuple[str, str, int, int]] = {}

    for policy in POLICIES:
        for task in TASKS:
            row = by_pair.get((policy, task), {})
            result_dir = Path(row.get("result_dir", str(expected_result_dir(run_root, policy, task))))
            rows = validate_result_csv(
                result_dir / "per_episode_results.csv",
                policy,
                task,
                row.get("job_id", ""),
                args.allow_error_rows,
                artifact_paths,
                errors,
            )
            validate_side_artifacts(result_dir, policy, task, errors)
            by_policy_task[(policy, task)] = rows
            by_policy[policy].extend(rows)
            all_rows.extend(rows)

    for policy, rows in by_policy.items():
        if len(rows) != 1152:
            errors.append(f"{policy}: expected 1152 rows, found {len(rows)}")
    if len(all_rows) != 5760:
        errors.append(f"full run: expected 5760 rows, found {len(all_rows)}")

    report = {
        "validation_status": "failed" if errors else "passed",
        "run_id": args.run_id,
        "run_root": str(run_root),
        "submission_jobs": str(submission_jobs),
        "output_dir": str(output_dir),
        "allow_error_rows": args.allow_error_rows,
        "row_counts": {
            "full": len(all_rows),
            "per_policy": {policy: len(rows) for policy, rows in by_policy.items()},
            "per_task_job": {f"{policy}/{task}": len(rows) for (policy, task), rows in by_policy_task.items()},
        },
        "artifact_path_count": len(artifact_paths),
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    write_csv(output_dir / "per_episode_results_all.csv", RESULT_COLUMNS, all_rows)
    task_summary_rows = [summary_row(policy, task, by_policy_task[(policy, task)]) for policy in POLICIES for task in TASKS]
    write_csv(
        output_dir / "per_task_summary.csv",
        [
            "policy",
            "policy_label",
            "task",
            "successes",
            "total",
            "success_rate",
            "ci95_one_sided_lower",
            "ci95_one_sided_upper",
            "ci95_one_sided_lower_width",
            "ci95_one_sided_upper_width",
            "error_rows",
            "timeout_rows",
        ],
        task_summary_rows,
    )
    policy_summary_rows = []
    for policy in POLICIES:
        row = summary_row(policy, "all", by_policy[policy])
        row["task"] = "all"
        policy_summary_rows.append(row)
    write_csv(
        output_dir / "per_policy_summary.csv",
        [
            "policy",
            "policy_label",
            "task",
            "successes",
            "total",
            "success_rate",
            "ci95_one_sided_lower",
            "ci95_one_sided_upper",
            "ci95_one_sided_lower_width",
            "ci95_one_sided_upper_width",
            "error_rows",
            "timeout_rows",
        ],
        policy_summary_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
