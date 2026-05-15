#!/usr/bin/env python3
"""Run an existing SimplerEnv main script after installing protocol patches."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from widowx_protocol1 import install_from_env


def main() -> None:
    target = os.environ.get("SIMPLERENV_TARGET_MAIN", "").strip()
    if not target:
        raise RuntimeError("SIMPLERENV_TARGET_MAIN must point to the policy main_inference.py")

    target_path = Path(target).expanduser()
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing target inference script: {target_path}")

    install_from_env(required=bool(os.environ.get("SIMPLERENV_PROTOCOL_CONFIG")))
    sys.argv[0] = str(target_path)
    runpy.run_path(str(target_path), run_name="__main__")


if __name__ == "__main__":
    main()
