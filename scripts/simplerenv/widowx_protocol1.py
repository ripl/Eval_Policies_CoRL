"""Runtime patch for SimplerEnv WidowX Protocol 1 random positions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


_CONFIG: dict[str, Any] | None = None
_INSTALLED = False
_ORIGINAL_RESET = None

_TASK_BY_MODEL_PAIR = {
    ("baked_green_cube_3cm", "baked_yellow_cube_3cm"): "stack",
    ("bridge_carrot_generated_modified", "bridge_plate_objaverse_larger"): "carrot",
    ("bridge_spoon_generated_modified", "table_cloth_generated_shorter"): "spoon",
}


def _load_config() -> dict[str, Any]:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    path_value = os.environ.get("SIMPLERENV_PROTOCOL_CONFIG", "").strip()
    if not path_value:
        raise RuntimeError("SIMPLERENV_PROTOCOL_CONFIG is not set")

    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Protocol config does not exist: {path}")

    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    expected_sha256 = os.environ.get("SIMPLERENV_PROTOCOL_SHA256", "").strip()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Protocol config SHA mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    config = json.loads(raw)
    if config.get("name") != "widowx_protocol1_random_positions":
        raise RuntimeError(f"Unexpected protocol name: {config.get('name')!r}")

    tasks = config.get("tasks")
    if set(tasks or {}) != {"stack", "carrot", "spoon"}:
        raise RuntimeError(f"Protocol must contain stack/carrot/spoon tasks, got {sorted((tasks or {}).keys())}")

    for task_name, task in tasks.items():
        episodes = task.get("episodes", [])
        if len(episodes) != int(task.get("num_episodes", -1)):
            raise RuntimeError(f"{task_name} episode count mismatch in protocol config")
        if len(episodes) != int(config.get("num_episodes_per_task", -1)):
            raise RuntimeError(f"{task_name} does not match num_episodes_per_task")
        for idx, episode in enumerate(episodes):
            if int(episode.get("episode_id", -1)) != idx:
                raise RuntimeError(f"{task_name} episode id mismatch at index {idx}")

    _CONFIG = config
    print(f"[widowx_protocol1] loaded {path} sha256={actual_sha256}", flush=True)
    return config


def _resolve_task_key(env: Any) -> str:
    pair = (getattr(env, "_source_obj_name", None), getattr(env, "_target_obj_name", None))
    task_key = _TASK_BY_MODEL_PAIR.get(pair)
    if task_key is None:
        raise RuntimeError(
            "Protocol 1 only applies to stack/carrot/spoon. "
            f"Got source/target pair {pair!r}"
        )
    return task_key


def _as_xy(episode: dict[str, Any]) -> np.ndarray:
    return np.array([episode["source_xy"], episode["target_xy"]], dtype=float)


def _as_quat(episode: dict[str, Any]) -> np.ndarray:
    return np.array([episode["source_quat_wxyz"], episode["target_quat_wxyz"]], dtype=float)


def _validate_xy(task_key: str, task: dict[str, Any], xys: np.ndarray, *, settled: bool) -> None:
    center = np.array(task["square_center"], dtype=float)
    half_edge = float(task["half_edge_m"])
    min_separation = float(task["min_separation_m"])
    tol = 0.01 if settled else 1e-9

    if xys.shape != (2, 2):
        raise RuntimeError(f"{task_key} protocol xy has invalid shape: {xys.shape}")
    if np.any(xys < center - half_edge - tol) or np.any(xys > center + half_edge + tol):
        phase = "settled" if settled else "initial"
        raise RuntimeError(f"{task_key} {phase} xy outside Protocol 1 square: {xys.tolist()}")

    distance = float(np.linalg.norm(xys[0] - xys[1]))
    min_allowed = min_separation - (0.02 if settled else 1e-9)
    if distance < min_allowed:
        phase = "settled" if settled else "initial"
        raise RuntimeError(
            f"{task_key} {phase} source-target distance {distance:.4f} "
            f"is below allowed minimum {min_allowed:.4f}"
        )


def install_from_env(*, required: bool = False) -> dict[str, Any] | None:
    """Install the Protocol 1 reset patch when SIMPLERENV_PROTOCOL_CONFIG is set."""

    global _INSTALLED, _ORIGINAL_RESET
    if not os.environ.get("SIMPLERENV_PROTOCOL_CONFIG"):
        if required:
            raise RuntimeError("Protocol injection requested but SIMPLERENV_PROTOCOL_CONFIG is not set")
        return None

    config = _load_config()
    if _INSTALLED:
        return config

    from mani_skill2_real2sim.envs.custom_scenes import put_on_in_scene

    cls = put_on_in_scene.PutOnBridgeInSceneEnv
    _ORIGINAL_RESET = cls.reset

    def protocol_reset(self, seed=None, options=None):
        task_key = _resolve_task_key(self)
        task = config["tasks"][task_key]

        if options is None:
            options = {}
        else:
            options = options.copy()
        obj_init_options = dict(options.get("obj_init_options") or {})

        if "episode_id" not in obj_init_options:
            raise RuntimeError(f"Protocol 1 requires obj_init_options.episode_id for {task_key}")
        episode_id = int(obj_init_options["episode_id"])
        episodes = task["episodes"]
        if episode_id < 0 or episode_id >= len(episodes):
            raise RuntimeError(
                f"{task_key} Protocol 1 episode_id {episode_id} out of range [0, {len(episodes)})"
            )

        episode = episodes[episode_id]
        xys = _as_xy(episode)
        quats = _as_quat(episode)
        _validate_xy(task_key, task, xys, settled=False)

        self._xy_configs = [xys]
        self._quat_configs = [quats]
        options["obj_init_options"] = obj_init_options

        obs, info = _ORIGINAL_RESET(self, seed=seed, options=options)

        settled_xys = np.array(
            [
                self.episode_source_obj.pose.p[:2],
                self.episode_target_obj.pose.p[:2],
            ],
            dtype=float,
        )
        _validate_xy(task_key, task, settled_xys, settled=True)

        info.update(
            {
                "protocol": config["name"],
                "protocol_version": config["version"],
                "protocol_task": task_key,
                "protocol_episode_id": episode_id,
            }
        )
        return obs, info

    cls.reset = protocol_reset
    _INSTALLED = True
    print("[widowx_protocol1] reset patch installed", flush=True)
    return config
