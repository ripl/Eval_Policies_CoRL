#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from seed_utils import RESULT_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing results CSV: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULT_COLUMNS).writeheader()
        f.flush()
    print(f"initialized {args.output}")


if __name__ == "__main__":
    main()
