#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from protocol_abcde_common import set_global_seeds
from widowx_protocol1 import install_from_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    script_args = list(ns.args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    if not ns.script.is_file():
        raise SystemExit(f"missing script: {ns.script}")
    set_global_seeds(ns.seed)
    install_from_env(required=True)
    sys.argv = [str(ns.script), *script_args]
    runpy.run_path(str(ns.script), run_name="__main__")


if __name__ == "__main__":
    main()
