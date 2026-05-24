#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from typing import Dict

BASE_SEED = 20260523
POLICY_INDEX: Dict[str, int] = {"dexbotic": 1}
TASK_INDEX: Dict[str, int] = {"stack": 0, "carrot": 1, "spoon": 2, "eggplant": 3}
HORIZONS: Dict[str, int] = {"stack": 60, "carrot": 60, "spoon": 60, "eggplant": 120}
REPEATS = range(12)
OFFICIAL_EPISODES = range(24)
RESULT_COLUMNS = [
    "policy", "task", "official_episode_id", "repeat_id", "seed", "success",
    "steps", "error", "timeout", "horizon", "job_id", "video_path",
]


def validate_policy_task(policy: str, task: str) -> None:
    if policy not in POLICY_INDEX:
        raise ValueError(f"unsupported policy {policy!r}; expected one of {sorted(POLICY_INDEX)}")
    if task not in TASK_INDEX:
        raise ValueError(f"unsupported task {task!r}; expected one of {sorted(TASK_INDEX)}")


def horizon_for(task: str) -> int:
    if task not in HORIZONS:
        raise ValueError(f"unsupported task {task!r}")
    return HORIZONS[task]


def seed_for(policy: str, task: str, repeat_id: int, official_episode_id: int) -> int:
    validate_policy_task(policy, task)
    if repeat_id not in REPEATS:
        raise ValueError(f"repeat_id must be 0..11, got {repeat_id}")
    if official_episode_id not in OFFICIAL_EPISODES:
        raise ValueError(f"official_episode_id must be 0..23, got {official_episode_id}")
    return BASE_SEED + POLICY_INDEX[policy] * 1_000_000 + TASK_INDEX[task] * 10_000 + repeat_id * 100 + official_episode_id


def server_start_seed(policy: str, task: str) -> int:
    validate_policy_task(policy, task)
    return BASE_SEED + POLICY_INDEX[policy] * 1_000_000 + TASK_INDEX[task] * 10_000 + 9_900


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["EVAL_ROLLOUT_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seed = sub.add_parser("seed")
    p_seed.add_argument("policy")
    p_seed.add_argument("task")
    p_seed.add_argument("repeat_id", type=int)
    p_seed.add_argument("official_episode_id", type=int)
    p_horizon = sub.add_parser("horizon")
    p_horizon.add_argument("task")
    p_server = sub.add_parser("server-seed")
    p_server.add_argument("policy")
    p_server.add_argument("task")
    args = parser.parse_args()
    if args.cmd == "seed":
        print(seed_for(args.policy, args.task, args.repeat_id, args.official_episode_id))
    elif args.cmd == "horizon":
        print(horizon_for(args.task))
    elif args.cmd == "server-seed":
        print(server_start_seed(args.policy, args.task))


if __name__ == "__main__":
    main()
