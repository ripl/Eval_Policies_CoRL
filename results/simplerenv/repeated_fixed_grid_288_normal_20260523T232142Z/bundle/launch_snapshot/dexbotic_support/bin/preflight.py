#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from seed_utils import horizon_for, server_start_seed, validate_policy_task

PROJECT_ROOT = Path("/share/data/ripl/tianchong/projects/Eval_Policies_CoRL")
SUPPORT_ROOT = Path(os.environ.get("SUPPORT_ROOT", Path(__file__).resolve().parents[1]))
STRICT_BENCHMARK_SRC = PROJECT_ROOT / "scratch" / "strict_calibrations_20260523" / "sources" / "dexbotic-benchmark_54a6c749_simpler4ab717_clean_20260523T0152Z"
STRICT_NORM_STATS = PROJECT_ROOT / "results" / "simplerenv" / "strict_calibrations_20260523" / "dexbotic_official_workspace_strict_20260523T015147Z" / "run" / "norm_stats.json"
EXPECTED_NORM_STATS_SHA256 = "ddbeb68786543c68f8fa198c33fd9a265025d35c5f7ebba144d6ec9c655693e6"
EXPECTED_SERVER_IMAGE = "docker://dexmal/dexbotic@sha256:7851cf8ed236dc18b5df3df6e8ea8ff5a543d154c03ac637a6dc6bd4e9eda654"
EXPECTED_BENCHMARK_IMAGE = "docker://dexmal/dexbotic_benchmark@sha256:2e6938be25991c43e5261f91a14abcfaad68d6f1a664643ef69a2e3628b60fef"
EXPECTED_COMMITS = {
    "dexbotic": "ccb2cc14a56fd61b5914ca654f97df47ac4d1f13",
    "benchmark": "54a6c749a9fe36e72654efe899ad9dc3712b849c",
    "simpler": "4ab7178e83e84ee06894034ec6dbf9e7aad1e882",
    "maniskill2_real2sim": "ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682",
}


def git_status(path: Path) -> str:
    proc = subprocess.run(["git", "-C", str(path), "status", "--short"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git status failed for {path}")
    return proc.stdout.strip()


def git_head(path: Path) -> str:
    proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git rev-parse failed for {path}")
    return proc.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="dexbotic")
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    errors: list[str] = []
    try:
        validate_policy_task(args.policy, args.task)
    except Exception as exc:
        errors.append(str(exc))
    if args.policy != "dexbotic":
        errors.append(f"this v2 support tree only owns dexbotic, got policy={args.policy!r}")
    require(SUPPORT_ROOT.is_dir(), f"missing support root {SUPPORT_ROOT}", errors)
    sbatch_path = SUPPORT_ROOT / "launch" / "run_repeated_fixed_grid_task.sbatch"
    require(sbatch_path.is_file(), f"missing sbatch template {sbatch_path}", errors)
    if sbatch_path.is_file() and "--array" in sbatch_path.read_text():
        errors.append(f"sbatch template must not contain --array: {sbatch_path}")
    run_script = SUPPORT_ROOT / "launch" / "run_dexbotic_task.sh"
    require(run_script.is_file(), f"missing Dexbotic run script {run_script}", errors)
    if run_script.is_file() and "huggingface.co" in run_script.read_text():
        errors.append(f"run script must not contain mutable network norm_stats fallback: {run_script}")
    require(shutil.which("apptainer") is not None, "apptainer is not on PATH", errors)
    require(shutil.which("ffprobe") is not None, "ffprobe is not on PATH", errors)
    run_script_text = run_script.read_text(errors="replace") if run_script.is_file() else ""
    require(EXPECTED_SERVER_IMAGE in run_script_text, f"run script does not pin expected server image digest {EXPECTED_SERVER_IMAGE}", errors)
    require(EXPECTED_BENCHMARK_IMAGE in run_script_text, f"run script does not pin expected benchmark image digest {EXPECTED_BENCHMARK_IMAGE}", errors)
    dexbotic_src = PROJECT_ROOT / "third_party" / "dexbotic"
    repos = {
        "dexbotic": dexbotic_src,
        "benchmark": STRICT_BENCHMARK_SRC,
        "simpler": STRICT_BENCHMARK_SRC / "simpler",
        "maniskill2_real2sim": STRICT_BENCHMARK_SRC / "simpler" / "ManiSkill2_real2sim",
    }
    require(dexbotic_src.is_dir(), f"missing Dexbotic source {dexbotic_src}", errors)
    require(STRICT_BENCHMARK_SRC.is_dir(), f"missing strict clean benchmark source {STRICT_BENCHMARK_SRC}", errors)
    require((STRICT_BENCHMARK_SRC / "evaluation" / "run_simpler_evaluation.py").is_file(), "missing evaluation/run_simpler_evaluation.py", errors)
    require(STRICT_NORM_STATS.is_file(), f"missing strict norm_stats file {STRICT_NORM_STATS}", errors)
    if STRICT_NORM_STATS.is_file():
        actual = sha256_file(STRICT_NORM_STATS)
        require(actual == EXPECTED_NORM_STATS_SHA256, f"norm_stats sha256 mismatch: {actual} != {EXPECTED_NORM_STATS_SHA256}", errors)
    for name, path in repos.items():
        try:
            head = git_head(path)
            require(head == EXPECTED_COMMITS[name], f"{name} commit mismatch: {head} != {EXPECTED_COMMITS[name]}", errors)
            status = git_status(path)
            require(status == "", f"{name} source is dirty: {status}", errors)
        except Exception as exc:
            errors.append(str(exc))
    print(f"policy={args.policy}")
    print(f"task={args.task}")
    print(f"horizon={horizon_for(args.task)}")
    print(f"server_start_seed={server_start_seed(args.policy, args.task)}")
    print("rollout_seed_formula=20260523 + policy_index*1000000 + task_index*10000 + repeat_id*100 + official_episode_id")
    print("seed_column_semantics=host/client/repeat identifier and client/simulator seed; not full model-server RNG control")
    print(f"strict_norm_stats={STRICT_NORM_STATS}")
    print(f"strict_norm_stats_sha256={EXPECTED_NORM_STATS_SHA256}")
    if errors:
        print("preflight_status=failed")
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("preflight_status=passed")


if __name__ == "__main__":
    main()
