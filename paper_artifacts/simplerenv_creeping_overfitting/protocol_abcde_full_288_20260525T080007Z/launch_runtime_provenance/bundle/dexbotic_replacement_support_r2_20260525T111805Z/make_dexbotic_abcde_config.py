#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from protocol_abcde_common import HORIZON


def yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "[" + ", ".join(yaml_value(v) for v in value) + "]"
    if isinstance(value, dict) and not value:
        return "{}"
    raise TypeError(f"unsupported YAML value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config-out", required=True, type=Path)
    parser.add_argument("--model-id", default="Dexmal/simpler-db-memvla")
    args = parser.parse_args()
    config = {
        "policy_model": "vla",
        "policy_setup": "widowx_bridge",
        "ckpt_path": args.model_id,
        "env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "scene_name": "bridge_table_1_v1",
        "robot": "widowx",
        "enable_raytracing": False,
        "obs_camera_name": None,
        "action_scale": 1.0,
        "control_freq": 5,
        "sim_freq": 500,
        "max_episode_steps": HORIZON,
        "rgb_overlay_path": "simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "robot_init_x_range": [0.147, 0.147, 1],
        "robot_init_y_range": [0.028, 0.028, 1],
        "robot_init_rot_quat_center": [0, 0, 0, 1],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
        "obj_variation_mode": "episode",
        "obj_episode_range": [0, 288],
        "additional_env_save_tags": "protocol_abcde_full_288",
        "additional_env_build_kwargs": {},
        "output_dir": args.output_dir,
        "results_file": "results.json",
        "video_dir": "videos",
        "log_dir": "logs",
        "log_level": "INFO",
        "tf_memory_limit": 3072,
        "octo_init_rng": 0,
        "image_size": [224, 224],
        "replan_step": 5,
        "use_delta": True,
        "action_ensemble_horizon": 7,
        "adaptive_ensemble_alpha": 0.1,
        "action_ensemble": False,
        "verbose": False,
        "base_url": args.base_url,
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    if args.config_out.exists():
        raise SystemExit(f"refusing to overwrite existing config: {args.config_out}")
    args.config_out.write_text("\n".join(f"{k}: {yaml_value(v)}" for k, v in config.items()) + "\n")
    print(args.config_out)


if __name__ == "__main__":
    main()
