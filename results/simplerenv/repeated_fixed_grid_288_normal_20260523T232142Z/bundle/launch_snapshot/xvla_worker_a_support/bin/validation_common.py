#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from protocol import OFFICIAL_EPISODES, POLICY, REPEATS, RESULT_COLUMNS, expected_keys, horizon_for, seed_for


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def validate_manifest(task: str, manifest_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    errors: list[str] = []
    header, rows = read_csv(manifest_path)
    expected_header = ["policy", "task", "official_episode_id", "repeat_id", "seed", "horizon"]
    if header != expected_header:
        errors.append(f"manifest header mismatch: {header}")
    if len(rows) != len(REPEATS) * len(OFFICIAL_EPISODES):
        errors.append(f"manifest expected 288 rows, found {len(rows)}")
    keys = [(r.get("official_episode_id", ""), r.get("repeat_id", "")) for r in rows]
    if set(keys) != expected_keys() or len(keys) != len(set(keys)):
        errors.append("manifest key set is not exactly 12 x 24 without duplicates")
    for row in rows:
        try:
            ep = int(row["official_episode_id"])
            rep = int(row["repeat_id"])
            if row.get("policy") != POLICY or row.get("task") != task:
                errors.append(f"manifest wrong policy/task row: {row}")
                break
            if int(row["seed"]) != seed_for(task, rep, ep):
                errors.append(f"manifest seed mismatch for repeat={rep} episode={ep}")
                break
            if int(row["horizon"]) != horizon_for(task):
                errors.append(f"manifest horizon mismatch for repeat={rep} episode={ep}")
                break
        except Exception as exc:
            errors.append(f"manifest row parse error: {exc}; row={row}")
            break
    return errors, rows


def validate_task_rows(
    *,
    task: str,
    rows: list[dict[str, str]],
    header: list[str],
    manifest_rows: list[dict[str, str]] | None,
    allow_partial: bool,
    allow_errors: bool,
    allow_timeouts: bool,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    if header != RESULT_COLUMNS:
        errors.append(f"header mismatch: {header}")
    expected = expected_keys()
    keys = [(row.get("official_episode_id", ""), row.get("repeat_id", "")) for row in rows]
    unique = set(keys)
    if len(unique) != len(keys):
        errors.append(f"duplicate (episode, repeat) rows detected: rows={len(keys)} unique={len(unique)}")
    if not allow_partial:
        if len(rows) != len(expected):
            errors.append(f"expected 288 rows, found {len(rows)}")
        if unique != expected:
            missing = sorted(expected - unique, key=lambda x: (int(x[1]), int(x[0])))[:20]
            extra = sorted(unique - expected)[:20]
            errors.append(f"key mismatch missing_first20={missing} extra_first20={extra}")
    else:
        extra = sorted(unique - expected)[:20]
        if extra:
            errors.append(f"unexpected keys in partial result: {extra}")
    error_rows = [row for row in rows if row.get("error", "")]
    timeout_rows = [row for row in rows if row.get("timeout", "") not in {"", "0"}]
    if error_rows and not allow_errors:
        errors.append(f"found {len(error_rows)} rows with nonempty error")
    if timeout_rows and not allow_timeouts:
        errors.append(f"found {len(timeout_rows)} timeout rows")
    manifest_lookup = None
    if manifest_rows is not None:
        manifest_lookup = {(r.get("official_episode_id"), r.get("repeat_id")): r for r in manifest_rows}
    for row in rows:
        try:
            ep = int(row["official_episode_id"])
            rep = int(row["repeat_id"])
            if row.get("policy") != POLICY or row.get("task") != task:
                errors.append(f"wrong policy/task row: {row}")
                break
            if (str(ep), str(rep)) not in expected:
                errors.append(f"unexpected key repeat={rep} episode={ep}")
                break
            expected_seed = seed_for(task, rep, ep)
            if int(row["seed"]) != expected_seed:
                errors.append(f"seed mismatch for repeat={rep} episode={ep}: {row['seed']} != {expected_seed}")
                break
            expected_horizon = horizon_for(task)
            if int(row["horizon"]) != expected_horizon:
                errors.append(f"horizon mismatch for repeat={rep} episode={ep}: {row['horizon']} != {expected_horizon}")
                break
            if row.get("success") not in {"0", "1"}:
                errors.append(f"success must be 0/1 for repeat={rep} episode={ep}: {row.get('success')}")
                break
            if row.get("job_id", "") == "":
                errors.append(f"empty job_id for repeat={rep} episode={ep}")
                break
            if row.get("error", "") == "":
                if row.get("steps", "") == "":
                    errors.append(f"missing steps for non-error row repeat={rep} episode={ep}")
                    break
                steps = int(row["steps"])
                if steps < 0 or steps > expected_horizon:
                    errors.append(f"steps out of horizon for repeat={rep} episode={ep}: {steps}")
                    break
                video = row.get("video_path", "")
                if not video:
                    errors.append(f"missing video_path for non-error row repeat={rep} episode={ep}")
                    break
                if not Path(video).is_file():
                    errors.append(f"video_path does not exist: {video}")
                    break
            if manifest_lookup is not None:
                manifest_row = manifest_lookup.get((str(ep), str(rep)))
                if manifest_row is None:
                    errors.append(f"missing manifest row for repeat={rep} episode={ep}")
                    break
                if row.get("seed") != manifest_row.get("seed") or row.get("horizon") != manifest_row.get("horizon"):
                    errors.append(f"result row does not match manifest for repeat={rep} episode={ep}")
                    break
        except Exception as exc:
            errors.append(f"row parse error: {exc}; row={row}")
            break
    successes = sum(1 for row in rows if row.get("success") == "1")
    return {
        "policy": POLICY,
        "task": task,
        "expected_rows": len(expected),
        "actual_rows": len(rows),
        "unique_keys": len(unique),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "error_rows": len(error_rows),
        "timeout_rows": len(timeout_rows),
        "validation_status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
    }
