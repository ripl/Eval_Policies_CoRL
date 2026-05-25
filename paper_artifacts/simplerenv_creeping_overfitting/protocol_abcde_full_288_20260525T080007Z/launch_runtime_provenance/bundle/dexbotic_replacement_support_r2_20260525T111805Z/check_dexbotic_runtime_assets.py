#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

EXPECTED_ASSET_IDS = (
    "render_candidate_blue_hybrid_v4",
    "render_candidate_red_corrected_v6e",
    "render_candidate_white_offwhite_hybrid_v4",
)
RERUN_CONDITIONS = (
    "protocol_C2_blue_on_red",
    "protocol_C3_red_on_blue",
    "protocol_D",
    "protocol_E",
)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def condition_model_ids(protocol_config: Path) -> dict[str, list[str]]:
    data = json.loads(protocol_config.read_text())
    out: dict[str, list[str]] = {}
    for condition in RERUN_CONDITIONS:
        episode = data["conditions"][condition]["episodes"][0]
        out[condition] = list(episode["model_ids"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--asset-dir",
        default="/workspace/simpler/ManiSkill2_real2sim/data",
        type=Path,
    )
    parser.add_argument(
        "--protocol-config",
        default=os.environ.get(
            "SIMPLERENV_PROTOCOL_CONFIG",
            "/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/configs/simplerenv/protocol_abcde/simplerenv_protocol_abcde_stack_v1.json",
        ),
        type=Path,
    )
    args = parser.parse_args()
    payload: dict[str, Any] = {
        "status": "unknown",
        "expected_asset_ids": list(EXPECTED_ASSET_IDS),
        "rerun_conditions": list(RERUN_CONDITIONS),
        "requested_asset_dir": str(args.asset_dir),
        "ms2_real2sim_asset_dir_env": os.environ.get("MS2_REAL2SIM_ASSET_DIR", ""),
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }
    try:
        from mani_skill2_real2sim import ASSET_DIR
        from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode

        payload["resolved_asset_dir"] = str(ASSET_DIR)
        info_json = Path(ASSET_DIR) / "custom/info_bridge_custom_baked_tex_v0.json"
        info = json.loads(info_json.read_text())
        payload["info_json"] = str(info_json)
        payload["info_json_has_expected"] = {asset_id: asset_id in info for asset_id in EXPECTED_ASSET_IDS}
        payload["condition_episode0_model_ids"] = condition_model_ids(args.protocol_config)

        env = build_maniskill2_env(
            "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
            obs_mode="rgbd",
            robot="widowx",
            sim_freq=500,
            control_mode=get_robot_control_mode("widowx", "vla"),
            control_freq=5,
            max_episode_steps=60,
            scene_name="bridge_table_1_v1",
            camera_cfgs={"add_segmentation": True},
            rgb_overlay_path="simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        )
        try:
            unwrapped = env.unwrapped
            model_db = getattr(unwrapped, "model_db")
            payload["env_class"] = f"{type(unwrapped).__module__}.{type(unwrapped).__name__}"
            payload["model_db_size"] = len(model_db)
            payload["model_db_has_expected"] = {asset_id: asset_id in model_db for asset_id in EXPECTED_ASSET_IDS}
            condition_missing = {}
            for condition, model_ids in payload["condition_episode0_model_ids"].items():
                condition_missing[condition] = [model_id for model_id in model_ids if model_id not in model_db]
            payload["condition_episode0_missing_model_ids"] = condition_missing
        finally:
            env.close()

        errors = []
        if str(ASSET_DIR) != str(args.asset_dir):
            errors.append(f"ASSET_DIR mismatch: {ASSET_DIR} != {args.asset_dir}")
        for asset_id, present in payload["info_json_has_expected"].items():
            if not present:
                errors.append(f"info_json missing {asset_id}")
        for asset_id, present in payload["model_db_has_expected"].items():
            if not present:
                errors.append(f"env.model_db missing {asset_id}")
        for condition, missing in payload["condition_episode0_missing_model_ids"].items():
            if missing:
                errors.append(f"{condition} episode 0 missing from env.model_db: {missing}")
        payload["errors"] = errors
        payload["status"] = "passed" if not errors else "failed"
        write_report(args.report, payload)
        return 0 if not errors else 1
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = repr(exc)
        payload["traceback"] = traceback.format_exc()
        write_report(args.report, payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
