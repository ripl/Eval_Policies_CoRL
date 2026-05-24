#!/usr/bin/env python3
"""Validate one repeated fixed-grid per-episode CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CSV_COLUMNS = [
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
POLICIES = {"cogact", "spatialvla"}
TASKS = {"stack": 60, "carrot": 60, "spoon": 60, "eggplant": 120}
EXPECTED_KEYS = {(repeat_id, episode_id) for repeat_id in range(12) for episode_id in range(24)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-error-rows", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    rows = []
    if not args.csv.is_file():
        errors.append(f"missing CSV: {args.csv}")
    else:
        with args.csv.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CSV_COLUMNS:
                errors.append(f"columns mismatch: expected {CSV_COLUMNS}, found {reader.fieldnames}")
            rows = list(reader)

    keys: list[tuple[int, int]] = []
    duplicate_keys: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    bad_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=2):
        if row.get("policy") != args.policy or row.get("task") != args.task:
            errors.append(f"line {index}: wrong job key policy={row.get('policy')} task={row.get('task')}")
            bad_rows.append(row)
            continue
        try:
            repeat_id = int(row["repeat_id"])
            episode_id = int(row["official_episode_id"])
            horizon = int(row["horizon"])
        except Exception as exc:
            errors.append(f"line {index}: non-integer key/horizon: {exc}")
            bad_rows.append(row)
            continue
        key = (repeat_id, episode_id)
        keys.append(key)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
        if repeat_id not in range(12):
            errors.append(f"line {index}: repeat_id out of range: {repeat_id}")
        if episode_id not in range(24):
            errors.append(f"line {index}: official_episode_id out of range: {episode_id}")
        if horizon != TASKS[args.task]:
            errors.append(f"line {index}: horizon mismatch: expected {TASKS[args.task]}, found {horizon}")
        if row.get("success") not in {"0", "1", ""}:
            errors.append(f"line {index}: success must be 0/1/blank, found {row.get('success')!r}")
        if row.get("timeout") not in {"0", "1", ""}:
            errors.append(f"line {index}: timeout must be 0/1/blank, found {row.get('timeout')!r}")
        error_text = row.get("error", "").strip()
        timeout_text = row.get("timeout", "").strip()
        if (error_text or timeout_text == "1") and not args.allow_error_rows:
            errors.append(f"line {index}: error/timeout row is not allowed: error={error_text!r} timeout={timeout_text!r}")
        if not error_text and timeout_text != "1":
            if row.get("success") not in {"0", "1"}:
                errors.append(f"line {index}: success must be 0/1 for non-error rows")
            try:
                steps = int(row.get("steps", ""))
                if steps < 0 or steps > TASKS[args.task]:
                    errors.append(f"line {index}: steps out of range: {steps}")
            except Exception as exc:
                errors.append(f"line {index}: steps must be integer for non-error rows: {exc}")
            video_path = row.get("video_path", "").strip()
            if not video_path:
                errors.append(f"line {index}: missing video_path for non-error row")
            elif not Path(video_path).is_file():
                errors.append(f"line {index}: video_path does not exist: {video_path}")

    missing = sorted(EXPECTED_KEYS - seen)
    extras = sorted(seen - EXPECTED_KEYS)
    if len(rows) != 288:
        errors.append(f"row count mismatch: expected 288, found {len(rows)}")
    if len(seen) != 288:
        errors.append(f"unique key count mismatch: expected 288, found {len(seen)}")
    if duplicate_keys:
        errors.append(f"duplicate keys: {duplicate_keys[:20]}{' ...' if len(duplicate_keys) > 20 else ''}")
    if missing:
        errors.append(f"missing keys: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if extras:
        errors.append(f"extra keys: {extras[:20]}{' ...' if len(extras) > 20 else ''}")

    successes = sum(1 for row in rows if row.get("success") == "1")
    error_rows = sum(1 for row in rows if row.get("error"))
    timeout_rows = sum(1 for row in rows if row.get("timeout") == "1")
    report = {
        "policy": args.policy,
        "task": args.task,
        "csv": str(args.csv),
        "status": "failed" if errors else "passed",
        "row_count": len(rows),
        "unique_key_count": len(seen),
        "successes": successes,
        "error_rows": error_rows,
        "timeout_rows": timeout_rows,
        "missing_count": len(missing),
        "duplicate_count": len(duplicate_keys),
        "errors": errors,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
