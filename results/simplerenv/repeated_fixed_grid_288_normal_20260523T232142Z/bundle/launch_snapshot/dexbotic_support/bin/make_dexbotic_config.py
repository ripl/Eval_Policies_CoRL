#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from seed_utils import horizon_for, validate_policy_task

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "stack": {"env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "scene_name": "bridge_table_1_v1", "robot": "widowx", "rgb_overlay_path": "simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png", "robot_init_x_range": [0.147, 0.147, 1], "robot_init_y_range": [0.028, 0.028, 1]},
    "carrot": {"env_name": "PutCarrotOnPlateInScene-v0", "scene_name": "bridge_table_1_v1", "robot": "widowx", "rgb_overlay_path": "simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png", "robot_init_x_range": [0.147, 0.147, 1], "robot_init_y_range": [0.028, 0.028, 1]},
    "spoon": {"env_name": "PutSpoonOnTableClothInScene-v0", "scene_name": "bridge_table_1_v1", "robot": "widowx", "rgb_overlay_path": "simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png", "robot_init_x_range": [0.147, 0.147, 1], "robot_init_y_range": [0.028, 0.028, 1]},
    "eggplant": {"env_name": "PutEggplantInBasketScene-v0", "scene_name": "bridge_table_1_v2", "robot": "widowx_sink_camera_setup", "rgb_overlay_path": "simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png", "robot_init_x_range": [0.127, 0.127, 1], "robot_init_y_range": [0.06, 0.06, 1]},
}
BASE_CONFIG: Dict[str, Any] = {
    "policy_model": "vla", "policy_setup": "widowx_bridge", "ckpt_path": "Dexmal/simpler-db-memvla",
    "enable_raytracing": False, "obs_camera_name": None, "action_scale": 1.0,
    "control_freq": 5, "sim_freq": 500, "robot_init_rot_quat_center": [0, 0, 0, 1],
    "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, 0, 0, 1], "obj_variation_mode": "episode",
    "additional_env_build_kwargs": {}, "results_file": "results.json", "video_dir": "videos", "log_dir": "logs",
    "log_level": "INFO", "tf_memory_limit": 3072, "octo_init_rng": 0, "image_size": [224, 224],
    "replan_step": 5, "use_delta": True, "action_ensemble_horizon": 7, "adaptive_ensemble_alpha": 0.1,
    "action_ensemble": False, "verbose": False,
}


def yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if not value:
            return "{}"
        raise ValueError("nested non-empty dicts are not used")
    if isinstance(value, list):
        return "[" + ", ".join(yaml_value(v) for v in value) + "]"
    raise TypeError(f"unsupported YAML value type: {type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--official-episode-id", required=True, type=int)
    parser.add_argument("--repeat-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config-out", required=True, type=Path)
    parser.add_argument("--model-id", default="Dexmal/simpler-db-memvla")
    args = parser.parse_args()
    validate_policy_task("dexbotic", args.task)
    if args.official_episode_id < 0 or args.official_episode_id > 23:
        raise SystemExit(f"official episode id must be 0..23, got {args.official_episode_id}")
    if args.repeat_id < 0 or args.repeat_id > 11:
        raise SystemExit(f"repeat id must be 0..11, got {args.repeat_id}")
    config: Dict[str, Any] = dict(BASE_CONFIG)
    config.update(TASK_CONFIGS[args.task])
    config["ckpt_path"] = args.model_id
    config["max_episode_steps"] = horizon_for(args.task)
    config["obj_episode_range"] = [args.official_episode_id, args.official_episode_id + 1]
    config["additional_env_save_tags"] = f"repeat_{args.repeat_id}_episode_{args.official_episode_id}"
    config["output_dir"] = args.output_dir
    config["base_url"] = args.base_url
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    if args.config_out.exists():
        raise SystemExit(f"refusing to overwrite existing config: {args.config_out}")
    args.config_out.write_text("\n".join(f"{k}: {yaml_value(v)}" for k, v in config.items()) + "\n")
    print(args.config_out)


if __name__ == "__main__":
    main()
