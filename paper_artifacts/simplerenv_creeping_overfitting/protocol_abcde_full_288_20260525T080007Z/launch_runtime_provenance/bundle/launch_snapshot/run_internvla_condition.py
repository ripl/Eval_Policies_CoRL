#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from protocol_abcde_common import (
    EPISODE_IDS,
    EXPECTED_PROTOCOL_SHA256,
    HORIZON,
    PROJECT_ROOT,
    append_result,
    base_row,
    count_video_steps,
    load_protocol_config,
    seed_for,
    server_seed_for,
    validate_condition,
    validate_results_csv,
    write_manifest,
    write_runtime_metadata,
)

POLICY = "internvla_m1"
EXPECTED_SHA_LITERAL = "1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b"
CLONE = PROJECT_ROOT / "third_party/InternVLA-M1"
CKPT = Path("/share/data/ripl/tianchong/projects/InternVLA-M1/cache/models/InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt")
SIMPLER = Path("/share/data/ripl/tianchong/projects/InternVLA-M1/cache/Projects/SimplerEnv")
SERVER_PY = Path("/share/data/ripl/tianchong/conda/envs/internvla_m1_widowx_repro/bin/python")
SIM_PY = Path("/share/data/ripl/tianchong/conda/envs/internvla_simplerenv_repro/bin/python")


def wait_for_port(port: int, proc: subprocess.Popen, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before accepting connections: {proc.returncode}")
        with socket.socket() as sock:
            sock.settimeout(2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(5)
    raise RuntimeError(f"server did not open 127.0.0.1:{port}")


def stop_process_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=30)


def env_for(seed: int, support_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    cache = PROJECT_ROOT / "cache"
    env.update(
        {
            "HF_HOME": str(cache / "huggingface"),
            "HF_HUB_CACHE": str(cache / "huggingface/hub"),
            "TRANSFORMERS_CACHE": str(cache / "huggingface/transformers"),
            "TORCH_HOME": str(cache / "torch"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "PIP_CACHE_DIR": str(cache / "pip"),
            "WANDB_DIR": str(cache / "wandb"),
            "TOKENIZERS_PARALLELISM": "false",
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "MUJOCO_GL": "egl",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": str(seed),
            "INTERNVLA_EVAL_SEED": str(seed),
            "DISPLAY": "",
            "SIMPLERENV_PROTOCOL_CONDITION": os.environ["SIMPLERENV_PROTOCOL_CONDITION"],
            "SIMPLERENV_PROTOCOL_CONFIG": os.environ["SIMPLERENV_PROTOCOL_CONFIG"],
            "SIMPLERENV_PROTOCOL_SHA256": EXPECTED_PROTOCOL_SHA256,
        }
    )
    for key in ["HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "WANDB_DIR"]:
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    env["PYTHONPATH"] = ":".join([str(support_root), str(PROJECT_ROOT / "scripts/simplerenv"), str(CLONE), str(SIMPLER), env.get("PYTHONPATH", "")])
    return env


def start_server(condition: str, port: int, output_root: Path, support_root: Path) -> subprocess.Popen:
    seed = server_seed_for(POLICY, condition)
    cmd = [
        str(SERVER_PY),
        "-u",
        str(PROJECT_ROOT / "scratch/repeated_fixed_grid_calibration_288_20260523_v2/run_seeded_policy_server.py"),
        "--seed",
        str(seed),
        "--server-script",
        str(CLONE / "deployment/model_server/server_policy_M1.py"),
        "--",
        "--ckpt_path",
        str(CKPT),
        "--port",
        str(port),
        "--use_bf16",
    ]
    with (output_root / "logs/server.log").open("ab") as log:
        proc = subprocess.Popen(cmd, cwd=str(CLONE), env=env_for(seed, support_root), stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    wait_for_port(port, proc, 1200)
    return proc


def steps_from_client_log(client_log: Path) -> tuple[int | None, str]:
    if not client_log.exists():
        return None, "missing_client_log"
    text = client_log.read_text(errors="replace")
    step_lines = sum(1 for line in text.splitlines() if re.match(r"^\s*\d+\s+\{", line))
    if step_lines:
        return step_lines, "client_log_step_lines"
    elapsed = [int(match.group(1)) for match in re.finditer(r"['\"]elapsed_steps['\"]\s*:\s*(\d+)", text)]
    if elapsed:
        return max(elapsed), "client_log_elapsed_steps"
    return None, "client_log_no_step_match"


def parse_summary(episode_root: Path, episode_id: int) -> tuple[int, int, str, str]:
    summary_files = sorted((episode_root / "simpler_results").rglob("episode_summary.csv"))
    if len(summary_files) != 1:
        raise RuntimeError(f"expected one episode_summary.csv, found {len(summary_files)}")
    with summary_files[0].open(newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    if len(rows) != 1 or int(rows[0][1]) != episode_id:
        raise RuntimeError(f"malformed episode_summary.csv for episode {episode_id}: {rows}")
    success = 1 if rows[0][2].strip().lower() == "success" else 0
    videos = sorted((episode_root / "simpler_results").rglob(f"*obj_episode_{episode_id}*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"expected one video for episode {episode_id}, found {len(videos)}")
    steps, steps_source = steps_from_client_log(episode_root / "logs/client.log")
    if steps is None:
        steps = count_video_steps(videos[0], subtract_initial_frame=True)
        steps_source = "video_frame_count_minus_initial" if steps is not None else steps_source
    if steps is None:
        steps = HORIZON
        steps_source = f"{steps_source}_runtime_horizon_fallback"
    if steps < 0 or steps > HORIZON:
        raise RuntimeError(f"steps outside 0..{HORIZON} for episode {episode_id}: {steps} source={steps_source}")
    return success, steps, str(videos[0]), steps_source


def run_episode(condition: str, episode_id: int, port: int, output_root: Path, support_root: Path) -> dict[str, str | int]:
    seed = seed_for(POLICY, condition, episode_id)
    episode_root = output_root / "episodes" / f"episode_{episode_id:03d}"
    episode_root.mkdir(parents=True, exist_ok=False)
    log_dir = episode_root / "logs"
    log_dir.mkdir()
    cmd = [
        str(SIM_PY),
        "-u",
        str(support_root / "run_seeded_simpler_episode_with_protocol.py"),
        "--seed",
        str(seed),
        "--start-script",
        str(CLONE / "examples/SimplerEnv/start_simpler_env.py"),
        "--",
        "--ckpt-path",
        str(CKPT),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--robot",
        "widowx",
        "--policy-setup",
        "widowx_bridge",
        "--control-freq",
        "5",
        "--sim-freq",
        "500",
        "--max-episode-steps",
        str(HORIZON),
        "--env-name",
        "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "--scene-name",
        "bridge_table_1_v1",
        "--rgb-overlay-path",
        str(SIMPLER / "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"),
        "--robot-init-x-range",
        "0.147",
        "0.147",
        "1",
        "--robot-init-y-range",
        "0.028",
        "0.028",
        "1",
        "--obj-variation-mode",
        "episode",
        "--obj-episode-range",
        str(episode_id),
        str(episode_id + 1),
        "--robot-init-rot-quat-center",
        "0",
        "0",
        "0",
        "1",
        "--robot-init-rot-rpy-range",
        "0",
        "0",
        "1",
        "0",
        "0",
        "1",
        "0",
        "0",
        "1",
        "--logging-dir",
        str(episode_root / "simpler_results"),
    ]
    with (log_dir / "client.log").open("ab") as log:
        proc = subprocess.run(cmd, cwd=str(CLONE), env=env_for(seed, support_root), stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"InternVLA episode subprocess exited {proc.returncode}")
    success, steps, video_path, steps_source = parse_summary(episode_root, episode_id)
    return {"success": success, "steps": steps, "video_path": video_path, "_steps_source": steps_source}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--support-root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if EXPECTED_SHA_LITERAL != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"internal SHA guard mismatch: {EXPECTED_SHA_LITERAL} != {EXPECTED_PROTOCOL_SHA256}")
    condition = validate_condition(args.condition)
    load_protocol_config()
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "logs").mkdir()
    write_manifest(args.output_root / "manifest.csv", POLICY, condition, args.run_id)
    write_runtime_metadata(
        args.output_root / "runtime_metadata.json",
        policy=POLICY,
        condition=condition,
        run_id=args.run_id,
        checkpoint_identity=str(CKPT),
        checkpoint_path=str(CKPT),
        extra={
            "policy_model": "internvla_m1",
            "internvla_repo": str(CLONE),
            "simplerenv_repo": str(SIMPLER),
            "server_python": str(SERVER_PY),
            "sim_python": str(SIM_PY),
        },
    )
    port = args.port or (20000 + (int(os.environ.get("SLURM_JOB_ID", os.getpid())) % 20000))
    server_proc = None
    try:
        server_proc = start_server(condition, port, args.output_root, args.support_root)
        result_csv = args.output_root / "per_episode_results.csv"
        steps_source_counts: dict[str, int] = {}
        fallback_episodes: list[int] = []
        for episode_id in EPISODE_IDS:
            row = base_row(POLICY, condition, episode_id, args.run_id)
            try:
                episode_result = run_episode(condition, episode_id, port, args.output_root, args.support_root)
                steps_source = str(episode_result.pop("_steps_source", ""))
                steps_source_counts[steps_source] = steps_source_counts.get(steps_source, 0) + 1
                if "runtime_horizon_fallback" in steps_source:
                    fallback_episodes.append(episode_id)
                row.update(episode_result)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                append_result(result_csv, row)
                raise
            append_result(result_csv, row)
        (args.output_root / "steps_metadata.json").write_text(
            json.dumps(
                {
                    "steps_source_counts": steps_source_counts,
                    "runtime_horizon_fallback_episode_ids": fallback_episodes,
                    "runtime_horizon_fallback_caveat": (
                        "Fallback steps equal the configured horizon and are not success timing; "
                        "they are used only when client logs and video frame counts are unavailable."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        validate_results_csv(result_csv, POLICY, condition, args.output_root / "validation_report.json")
    finally:
        stop_process_tree(server_proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
