#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from seed_utils import RESULT_COLUMNS, horizon_for, validate_policy_task


def safe_error(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").replace("\r", " ").split())[:1000]


def count_video_steps(video_path: Path, subtract_initial_frame: bool) -> str:
    if not video_path.is_file():
        return ""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(video_path)]
    try:
        proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        frames = int(proc.stdout.strip().splitlines()[0])
        return str(max(frames - 1 if subtract_initial_frame else frames, 0))
    except Exception:
        return ""


def parse_dexbotic(episode_dir: Path, official_episode_id: int) -> Dict[str, Any]:
    eval_root = episode_dir / "eval"
    result_files = sorted(eval_root.glob("**/results.json"), key=lambda p: p.stat().st_mtime)
    if len(result_files) != 1:
        raise RuntimeError(f"expected one Dexbotic results.json under {eval_root}, found {len(result_files)}")
    result_file = result_files[0]
    run_dir = result_file.parent
    data = json.loads(result_file.read_text())
    successes = data.get("success_array")
    if not isinstance(successes, list) or len(successes) != 1:
        raise RuntimeError(f"expected one success_array entry in {result_file}, got {successes!r}")
    mp4s = sorted(run_dir.glob(f"videos/**/*obj_episode_{official_episode_id}_*.mp4"))
    if len(mp4s) != 1:
        raise RuntimeError(f"expected one Dexbotic video for episode {official_episode_id} under {run_dir}, found {len(mp4s)}")
    video = mp4s[0]
    return {"success": int(bool(successes[0])), "steps": count_video_steps(video, True), "video_path": str(video)}


def append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    if not csv_path.is_file():
        raise RuntimeError(f"results CSV does not exist: {csv_path}")
    with csv_path.open("r", newline="") as f:
        header = next(csv.reader(f))
    if header != RESULT_COLUMNS:
        raise RuntimeError(f"results CSV header mismatch: {header}")
    fd = os.open(csv_path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(fd, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULT_COLUMNS).writerow({key: row.get(key, "") for key in RESULT_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=["dexbotic"])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--official-episode-id", required=True, type=int)
    parser.add_argument("--repeat-id", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--timeout", required=True, choices=["0", "1"])
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--error-message", default="")
    args = parser.parse_args()
    validate_policy_task(args.policy, args.task)
    row: Dict[str, Any] = {"policy": args.policy, "task": args.task, "official_episode_id": args.official_episode_id, "repeat_id": args.repeat_id, "seed": args.seed, "success": 0, "steps": "", "error": "", "timeout": args.timeout, "horizon": horizon_for(args.task), "job_id": args.job_id, "video_path": ""}
    errors = []
    if args.timeout == "1":
        errors.append("timeout")
    if args.exit_code != 0:
        errors.append(f"exit_code={args.exit_code}")
    if args.error_message:
        errors.append(args.error_message)
    if args.exit_code == 0 and args.timeout == "0":
        try:
            row.update(parse_dexbotic(args.episode_dir, args.official_episode_id))
        except Exception as exc:
            errors.append(f"parse_error={exc}")
    row["error"] = safe_error("; ".join(errors))
    append_row(args.csv, row)
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
