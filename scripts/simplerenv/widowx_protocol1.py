"""Runtime patch for SimplerEnv WidowX randomized-position protocols."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat


_PROTOCOL3_FLAG = "SIMPLERENV_PROTOCOL3_STACK_YELLOW_ON_GREEN"
_PROTOCOL3_NAME = "widowx_protocol3_stack_yellow_on_green_protocol1_positions"
_PROTOCOL3_INSTRUCTION = "stack the yellow block on the green block"
_PROTOCOL3_STACK_PAIR = ("baked_green_cube_3cm", "baked_yellow_cube_3cm")

_CONFIG: dict[str, Any] | None = None
_INSTALLED = False
_PROTOCOL3_INSTALLED = False
_ORIGINAL_RESET = None

_TASK_BY_MODEL_PAIR = {
    ("baked_green_cube_3cm", "baked_yellow_cube_3cm"): "stack",
    ("bridge_carrot_generated_modified", "bridge_plate_objaverse_larger"): "carrot",
    ("bridge_spoon_generated_modified", "table_cloth_generated_shorter"): "spoon",
}

_SUPPORTED_PROTOCOLS = {
    "widowx_protocol1_random_positions",
    "widowx_protocol2_random_positions_robot_initial_states",
}

_MAT_TRANSFORM = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)


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
    protocol_name = config.get("name")
    if protocol_name not in _SUPPORTED_PROTOCOLS:
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
            if protocol_name == "widowx_protocol2_random_positions_robot_initial_states":
                state = episode.get("robot_initial_state")
                if not isinstance(state, dict):
                    raise RuntimeError(f"{task_name} episode {idx} is missing robot_initial_state")
                vec = state.get("state_xyz_rpy_pad_gripper")
                if len(vec or []) != 8:
                    raise RuntimeError(f"{task_name} episode {idx} has invalid robot_initial_state vector")

    _CONFIG = config
    print(f"[widowx_protocol] loaded {path} sha256={actual_sha256}", flush=True)
    return config


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_task_key(env: Any) -> str:
    pair = (getattr(env, "_source_obj_name", None), getattr(env, "_target_obj_name", None))
    task_key = _TASK_BY_MODEL_PAIR.get(pair)
    if task_key is None:
        raise RuntimeError(
            "Protocol 1 only applies to stack/carrot/spoon. "
            f"Got source/target pair {pair!r}"
        )
    return task_key


def _require_protocol3_config(config: dict[str, Any]) -> None:
    if config.get("name") != "widowx_protocol1_random_positions":
        raise RuntimeError(
            f"{_PROTOCOL3_NAME} must use the Protocol 1 randomized-position config; "
            f"got {config.get('name')!r}"
        )


def _require_protocol3_stack_pair(env: Any) -> None:
    pair = (getattr(env, "_source_obj_name", None), getattr(env, "_target_obj_name", None))
    if pair != _PROTOCOL3_STACK_PAIR:
        raise RuntimeError(
            f"{_PROTOCOL3_NAME} only supports the baked stack-cube env with "
            f"source/target pair {_PROTOCOL3_STACK_PAIR!r}; got {pair!r}"
        )


def _protocol3_actor(env: Any, actor_name: str) -> tuple[int, Any]:
    objs = getattr(env, "episode_objs", None)
    if objs is None:
        raise RuntimeError(f"{_PROTOCOL3_NAME} could not read env.episode_objs")
    matches = [(idx, obj) for idx, obj in enumerate(objs) if getattr(obj, "name", None) == actor_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"{_PROTOCOL3_NAME} expected exactly one actor named {actor_name!r}; "
            f"found {len(matches)} among {[getattr(obj, 'name', None) for obj in objs]!r}"
        )
    return matches[0]


def _protocol3_actor_bbox(env: Any, actor_name: str) -> np.ndarray:
    if actor_name == getattr(getattr(env, "episode_source_obj", None), "name", None):
        bbox = getattr(env, "episode_source_obj_bbox_world", None)
    elif actor_name == getattr(getattr(env, "episode_target_obj", None), "name", None):
        bbox = getattr(env, "episode_target_obj_bbox_world", None)
    else:
        raise RuntimeError(f"{_PROTOCOL3_NAME} cannot resolve bbox for actor {actor_name!r}")

    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape != (3,) or not np.isfinite(bbox).all():
        raise RuntimeError(f"{_PROTOCOL3_NAME} invalid bbox for {actor_name!r}: {bbox}")
    return bbox


def _protocol3_actor_pose(actor: Any) -> sapien.Pose:
    return actor.pose.transform(actor.cmass_local_pose)


def _protocol3_src_on_target(
    env: Any,
    source_name: str,
    target_name: str,
    *,
    success_require_src_completely_on_target: bool,
    z_flag_required_offset: float,
) -> bool:
    _, source_obj = _protocol3_actor(env, source_name)
    _, target_obj = _protocol3_actor(env, target_name)
    source_pose = _protocol3_actor_pose(source_obj)
    target_pose = _protocol3_actor_pose(target_obj)

    tgt_obj_half_length_bbox = _protocol3_actor_bbox(env, target_name) / 2
    src_obj_half_length_bbox = _protocol3_actor_bbox(env, source_name) / 2

    offset = source_pose.p - target_pose.p
    xy_flag = (
        np.linalg.norm(offset[:2])
        <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
    )
    z_flag = (offset[2] > 0) and (
        offset[2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
        <= z_flag_required_offset
    )
    src_on_target = bool(xy_flag and z_flag)

    if success_require_src_completely_on_target:
        contacts = env._scene.get_contacts()
        flag = True
        robot_link_names = [x.name for x in env.agent.robot.get_links()]
        ignore_actor_names = [target_obj.name] + robot_link_names
        for contact in contacts:
            actor_0, actor_1 = contact.actor0, contact.actor1
            other_obj_contact_actor_name = None
            if actor_0.name == source_obj.name:
                other_obj_contact_actor_name = actor_1.name
            elif actor_1.name == source_obj.name:
                other_obj_contact_actor_name = actor_0.name
            if other_obj_contact_actor_name is not None:
                contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
                if (other_obj_contact_actor_name not in ignore_actor_names) and (
                    np.linalg.norm(contact_impulse) > 1e-6
                ):
                    flag = False
                    break
        src_on_target = bool(src_on_target and flag)

    return src_on_target


def _protocol3_moved_flags(env: Any, source_name: str) -> tuple[bool, bool]:
    source_idx, source_obj = _protocol3_actor(env, source_name)
    settled_xyzs = getattr(env, "episode_obj_xyzs_after_settle", None)
    objs = getattr(env, "episode_objs", None)
    if settled_xyzs is None or objs is None or len(settled_xyzs) != len(objs):
        raise RuntimeError(f"{_PROTOCOL3_NAME} invalid settled object state")

    source_obj_xy_move_dist = np.linalg.norm(
        np.asarray(settled_xyzs[source_idx], dtype=float)[:2] - source_obj.pose.p[:2]
    )
    other_obj_xy_move_dist = []
    for idx, obj in enumerate(objs):
        if idx == source_idx:
            continue
        other_obj_xy_move_dist.append(
            np.linalg.norm(np.asarray(settled_xyzs[idx], dtype=float)[:2] - obj.pose.p[:2])
        )
    moved_correct_obj = (source_obj_xy_move_dist > 0.03) and all(
        x < source_obj_xy_move_dist for x in other_obj_xy_move_dist
    )
    moved_wrong_obj = any(x > 0.03 for x in other_obj_xy_move_dist) and any(
        x > source_obj_xy_move_dist for x in other_obj_xy_move_dist
    )
    return bool(moved_correct_obj), bool(moved_wrong_obj)


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


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0)))


def _robot_pose_from_state(robot_state: dict[str, Any]) -> sapien.Pose:
    state = np.array(robot_state["state_xyz_rpy_pad_gripper"], dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid Protocol 2 robot state: {state}")
    quat_wxyz = mat2quat(euler2mat(*state[3:6]) @ _MAT_TRANSFORM)
    return sapien.Pose(p=state[:3], q=quat_wxyz)


def _apply_robot_initial_state(env: Any, robot_state: dict[str, Any]) -> tuple[dict[str, Any], float, float]:
    controller = env.agent.controller.controllers["arm"]
    target_pose = _robot_pose_from_state(robot_state)
    init_arm_qpos = controller.compute_ik(target_pose)
    if init_arm_qpos is None:
        raise RuntimeError(
            "Protocol 2 IK failed for robot_initial_state "
            f"candidate_index={robot_state.get('candidate_index')} "
            f"episode_index={robot_state.get('episode_index')}"
        )

    cur_qpos = env.agent.robot.get_qpos()
    cur_qpos[controller.joint_indices] = init_arm_qpos
    env.agent.reset(cur_qpos)

    actual_pose = controller.ee_pose_at_base
    pos_error = float(np.linalg.norm(actual_pose.p - target_pose.p))
    rot_error = _quat_angle(np.asarray(actual_pose.q, dtype=float), np.asarray(target_pose.q, dtype=float))
    if pos_error > 0.035 or rot_error > 0.60:
        raise RuntimeError(
            "Protocol 2 IK error too large for robot_initial_state "
            f"candidate_index={robot_state.get('candidate_index')} "
            f"episode_index={robot_state.get('episode_index')}: "
            f"pos_error={pos_error:.6f}, rot_error={rot_error:.6f}"
        )

    return env.get_obs(), pos_error, rot_error


def _install_protocol3_stack_patch(put_on_in_scene: Any, config: dict[str, Any]) -> None:
    global _PROTOCOL3_INSTALLED
    if _PROTOCOL3_INSTALLED:
        return

    _require_protocol3_config(config)
    stack_cls = put_on_in_scene.StackGreenCubeOnYellowCubeInScene

    def protocol3_language(self, **kwargs):
        _require_protocol3_stack_pair(self)
        return _PROTOCOL3_INSTRUCTION

    def protocol3_evaluate(
        self,
        success_require_src_completely_on_target=True,
        z_flag_required_offset=0.02,
        **kwargs,
    ):
        _require_protocol3_stack_pair(self)
        moved_correct_obj, moved_wrong_obj = _protocol3_moved_flags(self, "baked_yellow_cube_3cm")
        _, yellow_obj = _protocol3_actor(self, "baked_yellow_cube_3cm")

        is_src_obj_grasped = self.agent.check_grasp(yellow_obj)
        if is_src_obj_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        yellow_on_green = _protocol3_src_on_target(
            self,
            "baked_yellow_cube_3cm",
            "baked_green_cube_3cm",
            success_require_src_completely_on_target=success_require_src_completely_on_target,
            z_flag_required_offset=z_flag_required_offset,
        )
        green_on_yellow = _protocol3_src_on_target(
            self,
            "baked_green_cube_3cm",
            "baked_yellow_cube_3cm",
            success_require_src_completely_on_target=success_require_src_completely_on_target,
            z_flag_required_offset=z_flag_required_offset,
        )
        success = yellow_on_green

        self.episode_stats["moved_correct_obj"] = moved_correct_obj
        self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = yellow_on_green
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] or is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] or consecutive_grasp
        )
        self.episode_stats["protocol3_success_yellow_on_green"] = yellow_on_green
        self.episode_stats["env_success_green_on_yellow"] = green_on_yellow

        return dict(
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            is_src_obj_grasped=is_src_obj_grasped,
            consecutive_grasp=consecutive_grasp,
            src_on_target=yellow_on_green,
            protocol3_success_yellow_on_green=yellow_on_green,
            env_success_green_on_yellow=green_on_yellow,
            episode_stats=self.episode_stats,
            success=success,
        )

    stack_cls.get_language_instruction = protocol3_language
    stack_cls.evaluate = protocol3_evaluate
    _PROTOCOL3_INSTALLED = True
    print(f"[widowx_protocol] {_PROTOCOL3_NAME} patch installed", flush=True)


def install_from_env(*, required: bool = False) -> dict[str, Any] | None:
    """Install the WidowX protocol reset patch when SIMPLERENV_PROTOCOL_CONFIG is set."""

    global _INSTALLED, _ORIGINAL_RESET
    if not os.environ.get("SIMPLERENV_PROTOCOL_CONFIG"):
        if required:
            raise RuntimeError("Protocol injection requested but SIMPLERENV_PROTOCOL_CONFIG is not set")
        return None

    config = _load_config()
    if _INSTALLED:
        if _env_flag_enabled(_PROTOCOL3_FLAG):
            from mani_skill2_real2sim.envs.custom_scenes import put_on_in_scene

            _install_protocol3_stack_patch(put_on_in_scene, config)
        return config

    from mani_skill2_real2sim.envs.custom_scenes import put_on_in_scene

    cls = put_on_in_scene.PutOnBridgeInSceneEnv
    _ORIGINAL_RESET = cls.reset

    def protocol_reset(self, seed=None, options=None):
        task_key = _resolve_task_key(self)
        if _env_flag_enabled(_PROTOCOL3_FLAG) and task_key != "stack":
            raise RuntimeError(f"{_PROTOCOL3_NAME} only supports stack, got task {task_key!r}")
        task = config["tasks"][task_key]

        if options is None:
            options = {}
        else:
            options = options.copy()
        obj_init_options = dict(options.get("obj_init_options") or {})

        if "episode_id" not in obj_init_options:
            # ManiSkill performs a constructor-time reconfigure reset before evaluators pick an episode.
            if options.get("reconfigure") is True:
                return _ORIGINAL_RESET(self, seed=seed, options=options)
            raise RuntimeError(f"{config['name']} requires obj_init_options.episode_id for {task_key}")
        episode_id = int(obj_init_options["episode_id"])
        episodes = task["episodes"]
        if episode_id < 0 or episode_id >= len(episodes):
            raise RuntimeError(
                f"{task_key} {config['name']} episode_id {episode_id} out of range [0, {len(episodes)})"
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

        if config["name"] == "widowx_protocol2_random_positions_robot_initial_states":
            robot_state = episode["robot_initial_state"]
            obs, ik_pos_error, ik_rot_error = _apply_robot_initial_state(self, robot_state)
            info.update(
                {
                    "protocol_robot_state_candidate_index": int(robot_state["candidate_index"]),
                    "protocol_robot_state_episode_index": int(robot_state["episode_index"]),
                    "protocol_robot_state_task": robot_state["task"],
                    "protocol_robot_ik_pos_error_m": ik_pos_error,
                    "protocol_robot_ik_rot_error_rad": ik_rot_error,
                }
            )

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
    if _env_flag_enabled(_PROTOCOL3_FLAG):
        _install_protocol3_stack_patch(put_on_in_scene, config)

    print("[widowx_protocol] reset patch installed", flush=True)
    return config
