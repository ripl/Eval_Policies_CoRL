#!/usr/bin/env python3
"""Protocol-aware X-VLA WidowX client wrapper."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from widowx_protocol1 import install_from_env


CLIENT_BY_TASK = {
    "stack": "client_blocks.py",
    "blocks": "client_blocks.py",
    "carrot": "client_carrot.py",
    "spoon": "client_spoon.py",
}


def load_client_module(client_path: Path):
    spec = importlib.util.spec_from_file_location(f"xvla_{client_path.stem}", client_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load client module from {client_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", required=True)
    parser.add_argument("--server_port", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task-label", required=True, choices=sorted(CLIENT_BY_TASK))
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument(
        "--client-dir",
        type=Path,
        default=None,
        help="Directory containing X-VLA WidowX client_*.py files",
    )
    args = parser.parse_args()

    if args.episode_start < 0 or args.episode_end <= args.episode_start:
        raise ValueError(f"Invalid episode range: {args.episode_start}:{args.episode_end}")

    install_from_env(required=True)

    x_vla_repo = Path(os.environ.get("X_VLA_REPO", "")).expanduser()
    client_dir = args.client_dir
    if client_dir is None:
        if not x_vla_repo:
            raise RuntimeError("Set X_VLA_REPO or pass --client-dir")
        client_dir = x_vla_repo / "evaluation/simpler/WidowX"
    client_path = client_dir / CLIENT_BY_TASK[args.task_label]
    if not client_path.is_file():
        raise FileNotFoundError(f"Missing X-VLA client: {client_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "widowx_results.txt"
    if result_path.exists():
        raise RuntimeError(f"Refusing to append to existing result file: {result_path}")

    module = load_client_module(client_path)
    client = module.XVLAClient(args.server_ip, args.server_port)

    protocol_name = os.environ.get("RUN_PROTOCOL_NAME", "widowx_protocol1_random_positions")
    total = args.episode_end - args.episode_start
    for offset, proc_id in enumerate(range(args.episode_start, args.episode_end), start=1):
        print(f"{protocol_name} {args.task_label}: episode {proc_id} ({offset}/{total})", flush=True)
        module.evaluate_policy_widowx(
            client,
            str(output_dir),
            proc_id,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
