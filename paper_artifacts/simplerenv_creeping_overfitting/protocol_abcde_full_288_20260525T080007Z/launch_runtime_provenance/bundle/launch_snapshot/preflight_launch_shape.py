#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from protocol_abcde_common import (
    CONDITIONS,
    EPISODE_IDS,
    EXPECTED_PROTOCOL_SHA256,
    HORIZON,
    POLICIES,
    PROTOCOL_NAME,
    load_protocol_config,
)


def text(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing launch file: {path}")
    return path.read_text()


def no_array(path: Path) -> bool:
    return "--array" not in text(path) and "\n#SBATCH -a" not in text(path)


def contains(path: Path, needle: str) -> bool:
    return needle in text(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-root", required=True, type=Path)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = load_protocol_config()
    errors: list[str] = []
    matrix = [{"policy": policy, "condition": condition} for policy in POLICIES for condition in CONDITIONS]
    if len(matrix) != 35:
        errors.append(f"job matrix count is {len(matrix)}, expected 35")
    for condition in CONDITIONS:
        episodes = config["conditions"][condition]["episodes"]
        if len(episodes) != len(EPISODE_IDS):
            errors.append(f"{condition} episode count {len(episodes)} != 288")
        ids = [episode.get("episode_id", idx) for idx, episode in enumerate(episodes)]
        if ids != list(EPISODE_IDS):
            errors.append(f"{condition} episode IDs are not exactly 0..287")

    files = {
        "common": args.support_root / "protocol_abcde_common.py",
        "cogact_spatial_runner": args.support_root / "run_cogact_spatialvla_condition.py",
        "cogact_spatial_sbatch": args.support_root / "run_cogact_spatialvla_condition.sbatch",
        "intern_runner": args.support_root / "run_internvla_condition.py",
        "intern_sbatch": args.support_root / "run_internvla_condition.sbatch",
        "xvla_runner": args.support_root / "run_xvla_condition.sh",
        "xvla_sbatch": args.support_root / "run_xvla_condition.sbatch",
        "dex_runner": args.support_root / "run_dexbotic_condition.sh",
        "dex_sbatch": args.support_root / "run_dexbotic_condition.sbatch",
        "dex_config": args.support_root / "make_dexbotic_abcde_config.py",
    }
    for label, path in files.items():
        if not path.is_file():
            errors.append(f"missing {label}: {path}")

    for label in ("cogact_spatial_sbatch", "intern_sbatch", "xvla_sbatch", "dex_sbatch"):
        path = files[label]
        if path.is_file() and not no_array(path):
            errors.append(f"{label} contains a Slurm array directive")

    stack_needles = {
        "cogact_spatial_runner": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "intern_runner": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "xvla_runner": "--task-label stack",
        "dex_config": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
    }
    for label, needle in stack_needles.items():
        path = files[label]
        if path.is_file() and not contains(path, needle):
            errors.append(f"{label} does not prove stack-only wiring with {needle!r}")

    for label in ("intern_runner", "intern_sbatch", "xvla_runner", "dex_runner"):
        path = files[label]
        if path.is_file():
            body = text(path)
            for needle in ("SIMPLERENV_PROTOCOL_CONDITION", "SIMPLERENV_PROTOCOL_CONFIG", EXPECTED_PROTOCOL_SHA256):
                if needle not in body:
                    errors.append(f"{label} missing explicit Protocol A-E wiring: {needle}")

    if HORIZON != 60:
        errors.append(f"HORIZON={HORIZON}, expected 60")
    for label in ("intern_runner", "dex_config"):
        path = files[label]
        if path.is_file() and "HORIZON" not in text(path):
            errors.append(f"{label} does not use shared HORIZON")
    if not args.asset_manifest.exists():
        errors.append(f"asset manifest was not written: {args.asset_manifest}")
    else:
        try:
            asset_manifest = json.loads(args.asset_manifest.read_text())
        except Exception as exc:
            errors.append(f"asset manifest is not valid JSON: {exc}")
        else:
            if set(asset_manifest.get("accepted_asset_ids") or []) != {
                "render_candidate_blue_hybrid_v4",
                "render_candidate_red_corrected_v6e",
                "render_candidate_white_offwhite_hybrid_v4",
            }:
                errors.append("asset manifest missing final-validator accepted_asset_ids")
            runtime_paths = asset_manifest.get("runtime_paths")
            if not isinstance(runtime_paths, list) or len(runtime_paths) != len(POLICIES):
                errors.append("asset manifest missing final-validator runtime_paths")
            else:
                seen = {entry.get("policy") for entry in runtime_paths if isinstance(entry, dict)}
                if seen != set(POLICIES):
                    errors.append(f"asset manifest runtime_paths policies mismatch: {sorted(seen)}")
                for entry in runtime_paths:
                    if not isinstance(entry, dict):
                        continue
                    if not entry.get("maniskill2_real2sim") or not entry.get("info_json"):
                        errors.append(f"asset manifest runtime path lacks root/info_json: {entry.get('policy')}")
                    if not entry.get("asset_files"):
                        errors.append(f"asset manifest runtime path lacks asset_files: {entry.get('policy')}")

    report: dict[str, Any] = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "job_count": len(matrix),
        "policies": list(POLICIES),
        "conditions": list(CONDITIONS),
        "episode_ids": [0, 287],
        "episodes_per_job": len(EPISODE_IDS),
        "horizon": HORIZON,
        "stack_only": True,
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "asset_manifest": str(args.asset_manifest),
        "matrix": matrix,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
