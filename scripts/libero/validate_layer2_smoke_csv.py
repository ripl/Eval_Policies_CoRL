#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--summary-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.episode_csv)
    if not csv_path.exists():
        raise SystemExit(f"missing episode CSV: {csv_path}")
    rows = list(csv.DictReader(csv_path.open()))
    if len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, got {len(rows)} in {csv_path}")
    for idx, row in enumerate(rows):
        if row.get("suite") != args.suite:
            raise SystemExit(f"row {idx} has suite {row.get('suite')}, expected {args.suite}")
        if row.get("policy_name") != args.policy_name:
            raise SystemExit(f"row {idx} has policy {row.get('policy_name')}, expected {args.policy_name}")
        if row.get("instance_id", "").startswith(args.suite + "/") is False:
            raise SystemExit(f"row {idx} has unstable instance_id: {row.get('instance_id')}")
        if row.get("success") not in {"0", "1", 0, 1}:
            raise SystemExit(f"row {idx} has invalid success value: {row.get('success')}")
        if row.get("error_type", ""):
            raise SystemExit(f"row {idx} has error_type: {row.get('error_type')}")
    videos = list(csv_path.parent.rglob("*.mp4"))
    if videos:
        raise SystemExit(f"smoke should not write videos, found {len(videos)} under {csv_path.parent}")
    summary = {
        "policy_name": args.policy_name,
        "suite": args.suite,
        "episode_csv": str(csv_path),
        "rows": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "videos": 0,
        "validation_status": "passed",
    }
    Path(args.summary_path).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
