#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from protocol import HORIZONS, OFFICIAL_EPISODES, POLICY, REPEATS, seed_for, validate_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate_task(args.task)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["policy", "task", "official_episode_id", "repeat_id", "seed", "horizon"]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for repeat_id in REPEATS:
            for episode_id in OFFICIAL_EPISODES:
                writer.writerow(
                    {
                        "policy": POLICY,
                        "task": args.task,
                        "official_episode_id": episode_id,
                        "repeat_id": repeat_id,
                        "seed": seed_for(args.task, repeat_id, episode_id),
                        "horizon": HORIZONS[args.task],
                    }
                )
    print(f"wrote {args.output} rows={len(REPEATS) * len(OFFICIAL_EPISODES)}")


if __name__ == "__main__":
    main()
