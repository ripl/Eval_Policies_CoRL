#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from seed_utils import OFFICIAL_EPISODES, REPEATS, horizon_for, seed_for, validate_policy_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate_policy_task(args.policy, args.task)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["policy", "task", "official_episode_id", "repeat_id", "seed", "horizon"]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for repeat_id in REPEATS:
            for episode_id in OFFICIAL_EPISODES:
                writer.writerow({
                    "policy": args.policy,
                    "task": args.task,
                    "official_episode_id": episode_id,
                    "repeat_id": repeat_id,
                    "seed": seed_for(args.policy, args.task, repeat_id, episode_id),
                    "horizon": horizon_for(args.task),
                })
    print(f"wrote manifest {args.output} with {len(REPEATS) * len(OFFICIAL_EPISODES)} rows")


if __name__ == "__main__":
    main()
