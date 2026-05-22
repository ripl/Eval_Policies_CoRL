#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import torch


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        default="/share/data/ripl/tianchong/projects/Policy_Eval_Done_Right/LIBERO/libero/libero/init_files",
    )
    parser.add_argument(
        "--target-root",
        default="/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/artifacts/libero/init_state_official_4suite_50",
    )
    parser.add_argument(
        "--config-source",
        default="/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/artifacts/libero/libero_config_layer2_4suite_2500_seed20260519/config.yaml",
    )
    parser.add_argument(
        "--config-root",
        default="/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/artifacts/libero/libero_config_official_4suite_50",
    )
    parser.add_argument(
        "--comparison-root",
        default="/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/artifacts/libero/init_state_layer2_4suite_2500_seed20260519",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def validate_suite(source_suite: Path, compare_suite: Path) -> list[dict]:
    files = sorted(source_suite.glob("*.pruned_init"))
    if len(files) != 10:
        raise SystemExit(f"{source_suite} expected 10 .pruned_init files, found {len(files)}")

    if not compare_suite.exists():
        raise SystemExit(f"missing comparison suite directory: {compare_suite}")
    expected_names = sorted(path.name for path in compare_suite.glob("*.pruned_init"))
    actual_names = [path.name for path in files]
    if actual_names != expected_names:
        raise SystemExit(
            f"task filename mismatch for {source_suite.name}:\n"
            f"official={actual_names}\ncomparison={expected_names}"
        )

    records = []
    for task_id, path in enumerate(files):
        state = load_state(path)
        shape = tuple(getattr(state, "shape", ()))
        dtype = str(getattr(state, "dtype", ""))
        if len(shape) != 2 or shape[0] != 50 or shape[1] <= 0:
            raise SystemExit(f"{path} expected shape (50, positive_width), got {shape}")
        if dtype != "float64":
            raise SystemExit(f"{path} expected dtype float64, got {dtype}")
        records.append(
            {
                "task_id": task_id,
                "filename": path.name,
                "path": str(path),
                "shape": list(shape),
                "dtype": dtype,
                "sha256": sha256(path),
            }
        )
    return records


def ensure_symlink(source: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        if not dest.is_symlink():
            raise SystemExit(f"refusing to replace non-symlink artifact path: {dest}")
        resolved = dest.resolve()
        if resolved != source.resolve():
            raise SystemExit(f"existing symlink {dest} points to {resolved}, expected {source.resolve()}")
        return
    os.symlink(source, dest, target_is_directory=True)


def write_config(config_source: Path, config_root: Path, target_root: Path) -> Path:
    if not config_source.exists():
        raise SystemExit(f"missing config source: {config_source}")
    lines = config_source.read_text().splitlines()
    out_lines = []
    saw_init_states = False
    for line in lines:
        if line.startswith("init_states:"):
            out_lines.append(f"init_states: {target_root}")
            saw_init_states = True
        else:
            out_lines.append(line)
    if not saw_init_states:
        raise SystemExit(f"config source has no init_states line: {config_source}")
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"
    config_path.write_text("\n".join(out_lines) + "\n")
    return config_path


def main() -> None:
    args = parse_args()
    official_root = Path(args.official_root)
    target_root = Path(args.target_root)
    config_source = Path(args.config_source)
    config_root = Path(args.config_root)
    comparison_root = Path(args.comparison_root)

    if not official_root.exists():
        raise SystemExit(f"missing official root: {official_root}")
    if not comparison_root.exists():
        raise SystemExit(f"missing comparison root: {comparison_root}")

    target_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_type": "libero_official_4suite_50_init_states",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "official_root": str(official_root),
        "target_root": str(target_root),
        "comparison_root": str(comparison_root),
        "suites": {},
    }

    total_states = 0
    for suite in SUITES:
        source_suite = official_root / suite
        compare_suite = comparison_root / suite
        if not source_suite.exists():
            raise SystemExit(f"missing official suite directory: {source_suite}")
        records = validate_suite(source_suite, compare_suite)
        ensure_symlink(source_suite, target_root / suite)
        manifest["suites"][suite] = {
            "source_dir": str(source_suite),
            "target_dir": str(target_root / suite),
            "tasks": records,
            "num_tasks": len(records),
            "states_per_task": 50,
        }
        total_states += 50 * len(records)

    config_path = write_config(config_source, config_root, target_root)
    manifest["config_path"] = str(config_path)
    manifest["num_suites"] = len(SUITES)
    manifest["total_task_files"] = len(SUITES) * 10
    manifest["total_states"] = total_states
    manifest["validation_status"] = "passed"

    manifest_path = target_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "config": str(config_path), "total_states": total_states}, indent=2))


if __name__ == "__main__":
    main()
