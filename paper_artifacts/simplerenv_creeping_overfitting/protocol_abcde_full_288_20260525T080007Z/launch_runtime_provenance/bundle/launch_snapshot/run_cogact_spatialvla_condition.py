#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from transforms3d.euler import quat2euler

from protocol_abcde_common import (
    CSV_COLUMNS,
    EPISODE_IDS,
    EXPECTED_PROTOCOL_SHA256,
    HORIZON,
    PROJECT_ROOT,
    append_result,
    base_row,
    load_protocol_config,
    seed_for,
    set_global_seeds,
    validate_condition,
    validate_policy,
    validate_results_csv,
    write_manifest,
    write_runtime_metadata,
)

POLICY_SPECS = {
    "cogact": {
        "env_prefix": PROJECT_ROOT / "envs/simplerenv_cogact_py310_np126",
        "ckpt_path": "CogACT/CogACT-Base",
        "policy_model": "cogact",
        "pythonpath": [PROJECT_ROOT / "scripts/simplerenv", PROJECT_ROOT / "third_party/cogact", PROJECT_ROOT / "third_party/simpler_env"],
        "overlay_dir": PROJECT_ROOT / "third_party/simpler_env/ManiSkill2_real2sim/data/real_inpainting",
    },
    "spatialvla": {
        "env_prefix": PROJECT_ROOT / "envs/simplerenv_spatialvla_py310",
        "ckpt_path": "IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
        "policy_model": "spatialvla",
        "pythonpath": [
            PROJECT_ROOT / "scripts/simplerenv",
            PROJECT_ROOT / "scratch/repeated_fixed_grid_calibration_288_20260523_v2/sources/simplerenv_openvla_ccfe380",
        ],
        "overlay_dir": PROJECT_ROOT / "scratch/repeated_fixed_grid_calibration_288_20260523_v2/sources/simplerenv_openvla_ccfe380/ManiSkill2_real2sim/data/real_inpainting",
    },
}


def install_cache_defaults() -> None:
    defaults = {
        "HF_HOME": PROJECT_ROOT / "cache/huggingface",
        "HF_HUB_CACHE": PROJECT_ROOT / "cache/huggingface/hub",
        "TRANSFORMERS_CACHE": PROJECT_ROOT / "cache/huggingface/hub",
        "TORCH_HOME": PROJECT_ROOT / "cache/torch",
        "XDG_CACHE_HOME": PROJECT_ROOT / "cache/xdg",
        "PIP_CACHE_DIR": PROJECT_ROOT / "cache/pip",
        "CONDA_PKGS_DIRS": PROJECT_ROOT / "cache/conda_pkgs",
        "WANDB_DIR": PROJECT_ROOT / "artifacts/wandb",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    if not os.environ.get("VK_ICD_FILENAMES"):
        for candidate in ("/etc/vulkan/icd.d/nvidia_icd.json", "/usr/share/vulkan/icd.d/nvidia_icd.json"):
            if Path(candidate).is_file():
                os.environ["VK_ICD_FILENAMES"] = candidate
                break


def configure_paths(policy: str) -> None:
    paths = [str(path) for path in POLICY_SPECS[policy]["pythonpath"]]
    for path in reversed(paths):
        sys.path.insert(0, path)
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + ([old] if old else []))


def configure_tensorflow_memory() -> None:
    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ.get("TF_MEMORY_LIMIT", "3072")))],
            )
    except Exception as exc:
        print(f"tensorflow_memory_warning={exc}", file=sys.stderr)


def patch_hf_token_for_cogact() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    token_file = os.environ.get("HF_TOKEN_FILE", "").strip()
    if not token and token_file:
        token = Path(token_file).expanduser().read_text().strip()
    if not token:
        return
    import sim_cogact.cogact_policy as cogact_policy

    original_load_vla = cogact_policy.load_vla

    def load_vla_with_token(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("hf_token", token)
        return original_load_vla(*args, **kwargs)

    cogact_policy.load_vla = load_vla_with_token


def make_model(policy: str) -> Any:
    configure_paths(policy)
    configure_tensorflow_memory()
    if policy == "cogact":
        patch_hf_token_for_cogact()
        from sim_cogact import CogACTInference

        return CogACTInference(
            saved_model_path=POLICY_SPECS[policy]["ckpt_path"],
            policy_setup="widowx_bridge",
            action_scale=1.0,
            action_model_type=os.environ.get("COGACT_ACTION_MODEL_TYPE", "DiT-B"),
            cfg_scale=float(os.environ.get("COGACT_CFG_SCALE", "1.5")),
            use_bf16=os.environ.get("COGACT_USE_BF16", "false").lower() in {"1", "true", "yes", "on"},
            use_ddim=os.environ.get("COGACT_USE_DDIM", "true").lower() in {"1", "true", "yes", "on"},
            num_ddim_steps=int(os.environ.get("COGACT_NUM_DDIM_STEPS", "10")),
        )
    if policy == "spatialvla":
        from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference

        return SpatialVLAInference(
            saved_model_path=POLICY_SPECS[policy]["ckpt_path"],
            policy_setup="widowx_bridge",
            action_scale=1.0,
        )
    raise ValueError(policy)


def env_method(env: Any, method_name: str) -> Any:
    if hasattr(env, "get_wrapper_attr"):
        try:
            return env.get_wrapper_attr(method_name)
        except AttributeError:
            pass
    return getattr(getattr(env, "unwrapped", env), method_name)


def eef_pos(obs: Any) -> Any:
    if isinstance(obs, dict):
        agent = obs.get("agent", {})
        if isinstance(agent, dict):
            return agent.get("eef_pos")
    return None


def run_episode(model: Any, policy: str, condition: str, episode_id: int, output_root: Path) -> dict[str, Any]:
    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from simpler_env.utils.visualization import write_video
    from widowx_protocol1 import install_from_env

    install_from_env(required=True)
    spec = POLICY_SPECS[policy]
    overlay_path = spec["overlay_dir"] / "bridge_real_eval_1.png"
    control_mode = get_robot_control_mode("widowx", spec["policy_model"])
    env = build_maniskill2_env(
        "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        obs_mode="rgbd",
        robot="widowx",
        sim_freq=500,
        control_mode=control_mode,
        control_freq=5,
        max_episode_steps=HORIZON,
        scene_name="bridge_table_1_v1",
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=str(overlay_path),
    )
    robot_init_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs, _ = env.reset(
        options={
            "robot_init_options": {"init_xy": np.array([0.147, 0.028]), "init_rot_quat": robot_init_quat},
            "obj_init_options": {"episode_id": episode_id},
        }
    )
    is_final_subtask = env_method(env, "is_final_subtask")()
    task_description = env_method(env, "get_language_instruction")()
    obs_env = getattr(env, "unwrapped", env)
    image = get_image_from_maniskill2_obs_dict(obs_env, obs, camera_name=None)
    images = [image]
    predicted_actions: list[Any] = []
    predicted_terminated = False
    truncated = False
    done = False
    timestep = 0
    info: dict[str, Any] = {}
    success_text = "failure"
    model.reset(task_description)
    while not (predicted_terminated or truncated):
        if policy == "spatialvla":
            raw_action, action = model.step(image, task_description, eef_pos=eef_pos(obs))
        else:
            raw_action, action = model.step(image, task_description)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated and not is_final_subtask:
            predicted_terminated = False
            env_method(env, "advance_to_next_subtask")()
        obs, _reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )
        success_text = "success" if done else "failure"
        task_description = env_method(env, "get_language_instruction")()
        is_final_subtask = env_method(env, "is_final_subtask")()
        image = get_image_from_maniskill2_obs_dict(obs_env, obs, camera_name=None)
        images.append(image)
        timestep += 1
    episode_stats = info.get("episode_stats", {}) if isinstance(info, dict) else {}
    video_name = f"{success_text}_protocol_{condition}_episode_{episode_id}"
    for key, value in episode_stats.items():
        video_name += f"_{key}_{value}"
    video_name += ".mp4"
    roll, pitch, yaw = quat2euler(robot_init_quat)
    video_path = (
        output_root
        / "videos"
        / f"episode_{episode_id:03d}"
        / "bridge_table_1_v1"
        / control_mode
        / "StackGreenCubeOnYellowCubeBakedTexInScene-v0"
        / f"rob_0.147_0.028_rot_{roll:.3f}_{pitch:.3f}_{yaw:.3f}_rgb_overlay_bridge_real_eval_1"
        / video_name
    )
    write_video(str(video_path), images, fps=5)
    action_path = Path(str(video_path).replace(".mp4", ".png"))
    action_root = action_path.parent / "actions"
    action_root.mkdir(parents=True, exist_ok=True)
    model.visualize_epoch(predicted_actions, images, save_path=str(action_root / action_path.name))
    return {"success": int(success_text == "success"), "steps": timestep, "video_path": str(video_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=["cogact", "spatialvla"])
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    policy = validate_policy(args.policy)
    condition = validate_condition(args.condition)
    install_cache_defaults()
    os.environ["SIMPLERENV_PROTOCOL_SHA256"] = EXPECTED_PROTOCOL_SHA256
    load_protocol_config()
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_manifest(args.output_root / "manifest.csv", policy, condition, args.run_id)
    write_runtime_metadata(
        args.output_root / "runtime_metadata.json",
        policy=policy,
        condition=condition,
        run_id=args.run_id,
        checkpoint_identity=str(POLICY_SPECS[policy]["ckpt_path"]),
        extra={
            "policy_model": POLICY_SPECS[policy]["policy_model"],
            "env_prefix": str(POLICY_SPECS[policy]["env_prefix"]),
            "env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
            "overlay_dir": str(POLICY_SPECS[policy]["overlay_dir"]),
            "time_utc_epoch": int(time.time()),
        },
    )
    result_csv = args.output_root / "per_episode_results.csv"
    model = make_model(policy)
    for episode_id in EPISODE_IDS:
        seed = seed_for(policy, condition, episode_id)
        set_global_seeds(seed)
        row = base_row(policy, condition, episode_id, args.run_id)
        try:
            row.update(run_episode(model, policy, condition, episode_id, args.output_root))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            append_result(result_csv, row)
            traceback.print_exc()
            raise
        append_result(result_csv, row)
    validate_results_csv(result_csv, policy, condition, args.output_root / "validation_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
