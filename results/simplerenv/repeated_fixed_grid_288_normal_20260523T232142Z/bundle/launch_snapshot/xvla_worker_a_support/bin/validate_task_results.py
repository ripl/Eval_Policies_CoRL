#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import validate_task
from validation_common import read_csv, validate_manifest, validate_task_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--allow-timeouts", action="store_true")
    args = parser.parse_args()
    validate_task(args.task)
    manifest_errors, manifest_rows = validate_manifest(args.task, args.manifest)
    header, rows = read_csv(args.results)
    report = validate_task_rows(
        task=args.task,
        rows=rows,
        header=header,
        manifest_rows=manifest_rows,
        allow_partial=args.allow_partial,
        allow_errors=args.allow_errors,
        allow_timeouts=args.allow_timeouts,
    )
    report["errors"] = manifest_errors + report["errors"]
    report["validation_status"] = "passed" if not report["errors"] else "failed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
