#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from protocol import POLICY, RESULT_COLUMNS, horizon_for, validate_policy, validate_task


def safe_error(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").replace("\r", " ").split())[:1500]


def tail_text(path: Path, max_chars: int = 800) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


def parse_xvla(episode_dir: Path, horizon: int) -> tuple[Dict[str, Any], List[str]]:
    result_file = episode_dir / "widowx_results.txt"
    if not result_file.is_file():
        return {}, [f"missing_result_file={result_file}"]
    try:
        rows = [json.loads(line) for line in result_file.read_text().splitlines() if line.strip()]
    except Exception as exc:
        return {}, [f"result_json_parse_error={exc}"]
    if len(rows) != 1:
        return {}, [f"expected_one_result_row_found={len(rows)}"]
    source = rows[0]
    errors: List[str] = []
    if source.get("error"):
        errors.append(f"episode_error={source['error']}")
        return {"success": 0, "steps": "", "video_path": str(source.get("output", ""))}, errors
    try:
        steps = int(source["steps"])
    except Exception as exc:
        return {}, [f"steps_parse_error={exc}"]
    video_path = Path(str(source.get("output", "")))
    if steps < 0 or steps > horizon:
        errors.append(f"steps_out_of_horizon={steps}>{horizon}")
    if not str(video_path):
        errors.append("missing_video_path")
    elif not video_path.is_file():
        errors.append(f"video_path_missing={video_path}")
    return {"success": int(bool(source.get("done", False))), "steps": steps, "video_path": str(video_path)}, errors


def append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    if not csv_path.is_file():
        raise RuntimeError(f"results CSV does not exist: {csv_path}")
    with csv_path.open("r", newline="") as f:
        header = next(csv.reader(f))
    if header != RESULT_COLUMNS:
        raise RuntimeError(f"results CSV header mismatch: {header}")
    fd = os.open(csv_path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(fd, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--policy", default=POLICY)
    parser.add_argument("--task", required=True)
    parser.add_argument("--official-episode-id", required=True, type=int)
    parser.add_argument("--repeat-id", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--timeout", required=True, choices=["0", "1"])
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--stderr-log", type=Path, default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    validate_policy(args.policy)
    validate_task(args.task)
    horizon = horizon_for(args.task)
    row: Dict[str, Any] = {
        "policy": args.policy,
        "task": args.task,
        "official_episode_id": args.official_episode_id,
        "repeat_id": args.repeat_id,
        "seed": args.seed,
        "success": 0,
        "steps": "",
        "error": "",
        "timeout": args.timeout,
        "horizon": horizon,
        "job_id": args.job_id,
        "video_path": "",
    }
    errors: List[str] = []
    parsed, parse_errors = parse_xvla(args.episode_dir, horizon)
    row.update(parsed)
    errors.extend(parse_errors)
    if args.timeout == "1":
        errors.append("timeout")
    if args.exit_code != 0:
        errors.append(f"exit_code={args.exit_code}")
    if args.stderr_log is not None:
        stderr_tail = safe_error(tail_text(args.stderr_log))
        if stderr_tail and (args.exit_code != 0 or errors):
            errors.append(f"stderr_tail={stderr_tail}")
    row["error"] = safe_error("; ".join(errors))
    append_row(args.csv, row)
    print(json.dumps(row, sort_keys=True))
    if args.fail_on_error and (row["error"] or row["timeout"] not in {"", "0"}):
        raise SystemExit(10)


if __name__ == "__main__":
    main()
