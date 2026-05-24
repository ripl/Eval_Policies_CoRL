#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

from protocol import TASK_CLIENT, horizon_for, set_global_seeds, validate_task


def load_client_module(client_path: Path):
    spec = importlib.util.spec_from_file_location(f"xvla_v2_{client_path.stem}", client_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load client module from {client_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--official-episode-id", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--client-dir", type=Path, default=None)
    args = parser.parse_args()

    validate_task(args.task)
    if not 0 <= args.official_episode_id <= 23:
        raise SystemExit(f"official episode id must be 0..23, got {args.official_episode_id}")
    set_global_seeds(args.seed)

    client_dir = args.client_dir
    if client_dir is None:
        x_vla_repo = os.environ.get("X_VLA_REPO")
        if not x_vla_repo:
            raise SystemExit("set X_VLA_REPO or pass --client-dir")
        client_dir = Path(x_vla_repo) / "evaluation" / "simpler" / "WidowX"
    client_path = client_dir / TASK_CLIENT[args.task]
    if not client_path.is_file():
        raise SystemExit(f"missing X-VLA client: {client_path}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_file = args.output_dir / "widowx_results.txt"
    if result_file.exists():
        raise SystemExit(f"refusing to append to existing result file: {result_file}")

    module = load_client_module(client_path)
    client = module.XVLAClient(args.server_ip, args.server_port)
    module.evaluate_policy_widowx(
        client,
        str(args.output_dir),
        args.official_episode_id,
        max_steps=horizon_for(args.task),
    )

    if not result_file.is_file():
        raise SystemExit(f"X-VLA client did not write {result_file}")
    rows = [json.loads(line) for line in result_file.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise SystemExit(f"expected one X-VLA result row in {result_file}, found {len(rows)}")
    if rows[0].get("error"):
        raise SystemExit(f"X-VLA episode error: {rows[0]['error']}")


if __name__ == "__main__":
    main()
