#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from seed_utils import OFFICIAL_EPISODES, REPEATS, RESULT_COLUMNS, horizon_for, seed_for, validate_policy_task


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--expected-reset-count", type=int, default=None)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--allow-timeouts", action="store_true")
    args = parser.parse_args()
    validate_policy_task(args.policy, args.task)
    errors: List[str] = []
    warnings: List[str] = []
    if not args.results.is_file():
        raise SystemExit(f"missing results CSV: {args.results}")
    if not args.manifest.is_file():
        raise SystemExit(f"missing manifest CSV: {args.manifest}")
    header, rows = read_csv(args.results)
    if header != RESULT_COLUMNS:
        errors.append(f"header mismatch: {header}")
    expected_keys = {(str(ep), str(rep)) for rep in REPEATS for ep in OFFICIAL_EPISODES}
    keys = [(row.get("official_episode_id", ""), row.get("repeat_id", "")) for row in rows]
    unique_keys = set(keys)
    if len(rows) != 288:
        errors.append(f"expected 288 rows, found {len(rows)}")
    if len(unique_keys) != len(keys):
        errors.append(f"duplicate (episode, repeat) rows detected: rows={len(keys)} unique={len(unique_keys)}")
    if unique_keys != expected_keys:
        missing = sorted(expected_keys - unique_keys, key=lambda x: (int(x[1]), int(x[0])))[:20]
        extra = sorted(unique_keys - expected_keys)[:20]
        errors.append(f"manifest key mismatch missing_first20={missing} extra_first20={extra}")
    error_rows = [row for row in rows if row.get("error", "")]
    timeout_rows = [row for row in rows if row.get("timeout", "") not in {"0", ""}]
    if error_rows and not args.allow_errors:
        errors.append(f"found {len(error_rows)} rows with nonempty error")
    if timeout_rows and not args.allow_timeouts:
        errors.append(f"found {len(timeout_rows)} timeout rows")
    for row in rows:
        if row.get("policy") != args.policy or row.get("task") != args.task:
            errors.append(f"wrong policy/task row: {row}")
            break
        try:
            ep = int(row["official_episode_id"])
            rep = int(row["repeat_id"])
            expected_seed = seed_for(args.policy, args.task, rep, ep)
            if int(row["seed"]) != expected_seed:
                errors.append(f"seed mismatch for repeat={rep} episode={ep}: {row['seed']} != {expected_seed}")
                break
            if int(row["horizon"]) != horizon_for(args.task):
                errors.append(f"horizon mismatch for repeat={rep} episode={ep}: {row['horizon']}")
                break
            if row.get("success") not in {"0", "1"}:
                errors.append(f"success must be 0/1 for repeat={rep} episode={ep}: {row.get('success')}")
                break
            if row.get("error", "") == "":
                if not row.get("video_path", ""):
                    errors.append(f"missing video_path for non-error row repeat={rep} episode={ep}")
                    break
                if not Path(row["video_path"]).is_file():
                    errors.append(f"video_path does not exist: {row['video_path']}")
                    break
                if not row.get("steps", ""):
                    errors.append(f"missing steps for non-error row repeat={rep} episode={ep}")
                    break
                steps = int(row["steps"])
                if steps < 0 or steps > horizon_for(args.task):
                    errors.append(f"steps out of horizon for repeat={rep} episode={ep}: {steps}")
                    break
        except Exception as exc:
            errors.append(f"row parse error: {exc}; row={row}")
            break
    m_header, manifest_rows = read_csv(args.manifest)
    expected_manifest_header = ["policy", "task", "official_episode_id", "repeat_id", "seed", "horizon"]
    if m_header != expected_manifest_header:
        errors.append(f"manifest header mismatch: {m_header}")
    if len(manifest_rows) != 288:
        errors.append(f"manifest expected 288 rows, found {len(manifest_rows)}")
    manifest_lookup = {(r.get("official_episode_id"), r.get("repeat_id")): r for r in manifest_rows}
    if set(manifest_lookup) != expected_keys:
        errors.append("manifest key set is not exactly 12 x 24")
    for row in rows:
        key = (row.get("official_episode_id"), row.get("repeat_id"))
        manifest_row = manifest_lookup.get(key)
        if manifest_row and (row.get("seed") != manifest_row.get("seed") or row.get("horizon") != manifest_row.get("horizon")):
            errors.append(f"result row does not match manifest for key={key}")
            break
    reset_count = None
    reset_count_loose = None
    if args.expected_reset_count is not None:
        if args.server_log is None or not args.server_log.is_file():
            errors.append(f"missing server log for reset validation: {args.server_log}")
        else:
            text = args.server_log.read_text(errors="replace")
            reset_count = len(re.findall(r"\*\*\s*reset memory\s*\*\*", text))
            reset_count_loose = text.lower().count("reset memory")
            if reset_count != args.expected_reset_count:
                errors.append(f"Dexbotic reset count mismatch: expected {args.expected_reset_count}, found {reset_count}")
    successes = sum(1 for row in rows if row.get("success") == "1")
    report = {
        "policy": args.policy,
        "task": args.task,
        "expected_rows": 288,
        "actual_rows": len(rows),
        "unique_keys": len(unique_keys),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "error_rows": len(error_rows),
        "timeout_rows": len(timeout_rows),
        "allow_errors": args.allow_errors,
        "allow_timeouts": args.allow_timeouts,
        "reset_count": reset_count,
        "reset_count_loose": reset_count_loose,
        "expected_reset_count": args.expected_reset_count,
        "seed_column_semantics": "host/client/repeat identifier and client/simulator seed; not full model-server RNG control",
        "validation_status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
