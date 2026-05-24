#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--official-episode-id", required=True, type=int)
    parser.add_argument("--repeat-id", required=True, type=int)
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(1)
    with args.csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [
        row
        for row in rows
        if row.get("official_episode_id") == str(args.official_episode_id)
        and row.get("repeat_id") == str(args.repeat_id)
    ]
    if len(matches) == 0:
        raise SystemExit(1)
    if len(matches) == 1:
        raise SystemExit(0)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
