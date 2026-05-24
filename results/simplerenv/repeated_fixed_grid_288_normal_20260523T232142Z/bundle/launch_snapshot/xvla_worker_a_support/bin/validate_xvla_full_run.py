#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import POLICY, TASK_INDEX
from validation_common import read_csv, validate_task_rows


def parse_task_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected TASK=/path/to/per_episode_results.csv")
    task, path = value.split("=", 1)
    if task not in TASK_INDEX:
        raise argparse.ArgumentTypeError(f"unsupported task {task!r}")
    return task, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-result", action="append", required=True, type=parse_task_result)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--allow-timeouts", action="store_true")
    args = parser.parse_args()
    provided = dict(args.task_result)
    errors: list[str] = []
    if set(provided) != set(TASK_INDEX):
        errors.append(f"expected exactly tasks {sorted(TASK_INDEX)}, got {sorted(provided)}")
    task_reports = {}
    all_keys = set()
    total_rows = 0
    total_successes = 0
    for task in sorted(TASK_INDEX, key=TASK_INDEX.get):
        path = provided.get(task)
        if path is None:
            continue
        try:
            header, rows = read_csv(path)
            report = validate_task_rows(
                task=task,
                rows=rows,
                header=header,
                manifest_rows=None,
                allow_partial=False,
                allow_errors=args.allow_errors,
                allow_timeouts=args.allow_timeouts,
            )
            task_reports[task] = report
            errors.extend(f"{task}: {err}" for err in report["errors"])
            total_rows += len(rows)
            total_successes += report["successes"]
            for row in rows:
                key = (
                    row.get("policy", ""),
                    row.get("task", ""),
                    row.get("official_episode_id", ""),
                    row.get("repeat_id", ""),
                )
                if key in all_keys:
                    errors.append(f"duplicate full-run key: {key}")
                all_keys.add(key)
        except Exception as exc:
            errors.append(f"{task}: {exc}")
    if total_rows != 1152:
        errors.append(f"expected 1152 total X-VLA rows, found {total_rows}")
    if len(all_keys) != total_rows:
        errors.append(f"full-run unique key count {len(all_keys)} does not match rows {total_rows}")
    report = {
        "policy": POLICY,
        "expected_rows": 1152,
        "actual_rows": total_rows,
        "successes": total_successes,
        "success_rate": total_successes / total_rows if total_rows else None,
        "task_reports": task_reports,
        "validation_status": "passed" if not errors else "failed",
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
