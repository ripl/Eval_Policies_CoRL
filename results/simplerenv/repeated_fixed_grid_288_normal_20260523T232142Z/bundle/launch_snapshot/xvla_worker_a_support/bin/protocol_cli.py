#!/usr/bin/env python3
from __future__ import annotations

import argparse

from protocol import horizon_for, seed_for, server_start_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seed = sub.add_parser("seed")
    p_seed.add_argument("task")
    p_seed.add_argument("repeat_id", type=int)
    p_seed.add_argument("official_episode_id", type=int)
    p_horizon = sub.add_parser("horizon")
    p_horizon.add_argument("task")
    p_server = sub.add_parser("server-seed")
    p_server.add_argument("task")
    args = parser.parse_args()
    if args.cmd == "seed":
        print(seed_for(args.task, args.repeat_id, args.official_episode_id))
    elif args.cmd == "horizon":
        print(horizon_for(args.task))
    elif args.cmd == "server-seed":
        print(server_start_seed(args.task))


if __name__ == "__main__":
    main()
