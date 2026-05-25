#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from protocol_abcde_common import set_global_seeds
from widowx_protocol1 import install_from_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--start-script", required=True, type=Path)
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not args.start_script.is_file():
        raise SystemExit(f"missing start script: {args.start_script}")
    set_global_seeds(args.seed)
    install_from_env(required=True)
    sys.argv = [str(args.start_script), *rest]
    runpy.run_path(str(args.start_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
