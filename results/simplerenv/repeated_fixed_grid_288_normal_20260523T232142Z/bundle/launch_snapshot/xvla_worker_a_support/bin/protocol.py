#!/usr/bin/env python3
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

BASE_SEED = 20260523
POLICY = "xvla"
POLICY_DISPLAY = "X-VLA-WidowX"

TASK_INDEX: Dict[str, int] = {
    "stack": 0,
    "carrot": 1,
    "spoon": 2,
    "eggplant": 3,
}

TASK_CLIENT: Dict[str, str] = {
    "stack": "client_blocks.py",
    "carrot": "client_carrot.py",
    "spoon": "client_spoon.py",
    "eggplant": "client_eggplant.py",
}

HORIZONS: Dict[str, int] = {
    "stack": 60,
    "carrot": 60,
    "spoon": 60,
    "eggplant": 120,
}

REPEATS = tuple(range(12))
OFFICIAL_EPISODES = tuple(range(24))

RESULT_COLUMNS = [
    "policy",
    "task",
    "official_episode_id",
    "repeat_id",
    "seed",
    "success",
    "steps",
    "error",
    "timeout",
    "horizon",
    "job_id",
    "video_path",
]


def validate_task(task: str) -> None:
    if task not in TASK_INDEX:
        raise ValueError(f"unsupported task {task!r}; expected one of {sorted(TASK_INDEX)}")


def validate_policy(policy: str) -> None:
    if policy != POLICY:
        raise ValueError(f"unsupported policy {policy!r}; this bundle only supports {POLICY!r}")


def horizon_for(task: str) -> int:
    validate_task(task)
    return HORIZONS[task]


def seed_for(task: str, repeat_id: int, official_episode_id: int) -> int:
    validate_task(task)
    if repeat_id not in REPEATS:
        raise ValueError(f"repeat_id must be 0..11, got {repeat_id}")
    if official_episode_id not in OFFICIAL_EPISODES:
        raise ValueError(f"official_episode_id must be 0..23, got {official_episode_id}")
    return BASE_SEED + TASK_INDEX[task] * 10_000 + repeat_id * 100 + official_episode_id


def server_start_seed(task: str) -> int:
    validate_task(task)
    return BASE_SEED + TASK_INDEX[task] * 10_000 + 9_900


def expected_keys() -> set[tuple[str, str]]:
    return {(str(ep), str(rep)) for rep in REPEATS for ep in OFFICIAL_EPISODES}


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


def project_root() -> Path:
    return Path("/share/data/ripl/tianchong/projects/Eval_Policies_CoRL")
