#!/usr/bin/env python3
"""Validate one InternVLA-M1 repeated fixed-grid task result table."""

import argparse
import csv
import json
from pathlib import Path
import sys

POLICY = "InternVLA-M1"
BASE_SEED = 20260523
TASK_ORDER = ["stack", "carrot", "spoon", "eggplant"]
HORIZONS = {"stack": 60, "carrot": 60, "spoon": 60, "eggplant": 120}
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


def seed_for(task, repeat_id, official_episode_id):
    return BASE_SEED + TASK_ORDER.index(task) * 10000 + repeat_id * 24 + official_episode_id


def read_rows(path, errors):
    if not path.exists():
        errors.append("missing per_episode_results.csv: {}".format(path))
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != RESULT_COLUMNS:
            errors.append("header mismatch: {} != {}".format(reader.fieldnames, RESULT_COLUMNS))
            return []
        return list(reader)


def load_manifest_keys(path, task, errors):
    if not path.exists():
        errors.append("missing manifest.csv: {}".format(path))
        return set()
    keys = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("policy") != POLICY or row.get("task") != task:
                errors.append("manifest row has wrong policy/task: {}".format(row))
                continue
            keys.add((int(row["repeat_id"]), int(row["official_episode_id"])))
    return keys


def validate(args):
    task_root = Path(args.task_root)
    errors = []
    warnings = []
    result_path = task_root / "per_episode_results.csv"
    manifest_path = task_root / "manifest.csv"
    rows = read_rows(result_path, errors)
    manifest_keys = load_manifest_keys(manifest_path, args.task, errors)

    expected_keys = {(repeat_id, episode_id) for repeat_id in range(12) for episode_id in range(24)}
    if manifest_keys and manifest_keys != expected_keys:
        errors.append("manifest keys mismatch: missing={} extra={}".format(
            sorted(expected_keys - manifest_keys)[:20], sorted(manifest_keys - expected_keys)[:20]
        ))

    keys = []
    duplicate_keys = []
    seen = set()
    bad_rows = []
    error_rows = []
    timeout_rows = []
    successes = 0
    failures = 0

    for idx, row in enumerate(rows, start=2):
        try:
            repeat_id = int(row["repeat_id"])
            episode_id = int(row["official_episode_id"])
            seed = int(row["seed"])
            horizon = int(row["horizon"])
        except Exception as exc:
            bad_rows.append("line {} has non-integer id/seed/horizon: {}".format(idx, exc))
            continue
        key = (repeat_id, episode_id)
        keys.append(key)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)

        if row["policy"] != POLICY:
            bad_rows.append("line {} wrong policy {}".format(idx, row["policy"]))
        if row["task"] != args.task:
            bad_rows.append("line {} wrong task {}".format(idx, row["task"]))
        if key not in expected_keys:
            bad_rows.append("line {} key outside expected range {}".format(idx, key))
        if seed != seed_for(args.task, repeat_id, episode_id):
            bad_rows.append("line {} seed mismatch {} != {}".format(idx, seed, seed_for(args.task, repeat_id, episode_id)))
        if horizon != HORIZONS[args.task]:
            bad_rows.append("line {} horizon mismatch {} != {}".format(idx, horizon, HORIZONS[args.task]))
        if not row["job_id"]:
            bad_rows.append("line {} has blank job_id".format(idx))
        if row["timeout"] not in ("0", "1"):
            bad_rows.append("line {} timeout must be 0 or 1".format(idx))
        if row["timeout"] == "1":
            timeout_rows.append(idx)
        if row["error"]:
            error_rows.append(idx)
        if not row["error"] and row["timeout"] == "0":
            if row["success"] not in ("0", "1"):
                bad_rows.append("line {} success must be 0 or 1 for non-error row".format(idx))
            else:
                successes += int(row["success"])
                failures += 1 - int(row["success"])
            try:
                steps = int(row["steps"])
                if steps < 1 or steps > HORIZONS[args.task]:
                    bad_rows.append("line {} steps {} outside 1..{}".format(idx, steps, HORIZONS[args.task]))
            except Exception:
                bad_rows.append("line {} steps must be integer for non-error row".format(idx))
            if not row["video_path"]:
                bad_rows.append("line {} missing video_path".format(idx))
            elif not Path(row["video_path"]).exists():
                bad_rows.append("line {} video_path does not exist: {}".format(idx, row["video_path"]))

    missing = sorted(expected_keys - set(keys))
    extra = sorted(set(keys) - expected_keys)
    if len(rows) != 288:
        errors.append("expected 288 data rows, found {}".format(len(rows)))
    if len(set(keys)) != 288:
        errors.append("expected 288 unique (repeat_id, official_episode_id) keys, found {}".format(len(set(keys))))
    if duplicate_keys:
        errors.append("duplicate keys: {}".format(sorted(set(duplicate_keys))[:20]))
    if missing:
        errors.append("missing keys: {}".format(missing[:20]))
    if extra:
        errors.append("extra keys: {}".format(extra[:20]))
    if bad_rows:
        errors.extend(bad_rows[:50])
        if len(bad_rows) > 50:
            errors.append("additional bad rows omitted: {}".format(len(bad_rows) - 50))
    if (error_rows or timeout_rows) and not args.allow_error_rows:
        errors.append("error/timeout rows present: error_lines={} timeout_lines={}".format(error_rows[:20], timeout_rows[:20]))
    elif error_rows or timeout_rows:
        warnings.append("error/timeout rows allowed by flag: error_lines={} timeout_lines={}".format(error_rows[:20], timeout_rows[:20]))

    summary = {
        "policy": POLICY,
        "task": args.task,
        "task_root": str(task_root),
        "result_path": str(result_path),
        "manifest_path": str(manifest_path),
        "expected_rows": 288,
        "row_count": len(rows),
        "unique_key_count": len(set(keys)),
        "successes": successes,
        "failures": failures,
        "error_row_count": len(error_rows),
        "timeout_row_count": len(timeout_rows),
        "missing_key_count": len(missing),
        "duplicate_key_count": len(set(duplicate_keys)),
        "validation_status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
    }
    out = task_root / "validation_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--task", required=True, choices=TASK_ORDER)
    parser.add_argument("--allow-error-rows", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    return validate(parse_args(argv or sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
