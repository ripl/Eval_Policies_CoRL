#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from huggingface_hub import snapshot_download

import lerobot.envs.libero as libero_env_mod
import lerobot.policies.pi05.processor_pi05  # noqa: F401 - registers Pi05 processor steps
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.env_processor import LiberoProcessorStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--initial-states-path", required=True)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--selected-task-id", type=int, default=0)
    parser.add_argument("--episode-start-idx", type=int, default=0)
    parser.add_argument("--episode-end-idx", type=int, default=5)
    parser.add_argument("--max-total-episodes", type=int, default=5)
    parser.add_argument("--episode-results-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--policy-name", default="pi05_lerobot")
    parser.add_argument("--policy-tracker-id", type=int, default=869)
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--control-mode", default="relative", choices=["relative", "absolute"])
    parser.add_argument("--observation-size", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def install_init_state_loader(initial_states_path: Path) -> None:
    def get_task_init_states(task_suite, i: int):
        task = task_suite.tasks[i]
        path = initial_states_path / task.problem_folder / task.init_states_file
        if not path.exists():
            raise FileNotFoundError(f"missing init states for task {i}: {path}")
        return torch.load(path, weights_only=False)

    libero_env_mod.get_task_init_states = get_task_init_states


def load_policy_and_processors(args: argparse.Namespace):
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "model.safetensors").exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=checkpoint_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    config = PreTrainedConfig.from_pretrained(
        checkpoint_dir,
        local_files_only=args.local_files_only,
    )
    config.compile_model = False
    config.gradient_checkpointing = False
    config.device = args.device

    policy = PI05Policy.from_pretrained(
        checkpoint_dir,
        config=config,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    policy.to(args.device).eval()

    preprocessor_overrides = {"device_processor": {"device": args.device}}
    if args.tokenizer_name:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": args.tokenizer_name}

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides=preprocessor_overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        local_files_only=args.local_files_only,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        local_files_only=args.local_files_only,
    )
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    return policy, preprocessor, postprocessor, env_preprocessor


def make_vec_env(args: argparse.Namespace, suite, episode_idx: int):
    def factory():
        return LiberoEnv(
            task_suite=suite,
            task_id=args.selected_task_id,
            task_suite_name=args.task_suite,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            init_states=True,
            episode_index=episode_idx,
            n_envs=1,
            num_steps_wait=args.num_steps_wait,
            control_mode=args.control_mode,
            observation_width=args.observation_size,
            observation_height=args.observation_size,
        )

    return gym.vector.SyncVectorEnv([factory])


def run_episode(args: argparse.Namespace, suite, policy, preprocessor, postprocessor, env_preprocessor, episode_idx: int):
    env = make_vec_env(args, suite, episode_idx)
    task = suite.get_task(args.selected_task_id)
    start = time.time()
    num_steps = 0
    success = False
    error_type = ""
    try:
        policy.reset()
        observation, _ = env.reset(seed=[args.seed + episode_idx])
        max_steps = int(env.call("_max_episode_steps")[0])
        for step in range(max_steps):
            observation_batch = preprocess_observation(observation)
            observation_batch = add_envs_task(env, observation_batch)
            observation_batch = env_preprocessor(observation_batch)
            policy_input = preprocessor(observation_batch)
            with torch.inference_mode():
                action = policy.select_action(policy_input)
            action = postprocessor(action)
            action_np = action.detach().cpu().numpy()
            if action_np.ndim != 2 or action_np.shape[0] != 1:
                raise ValueError(f"unexpected action shape: {action_np.shape}")
            observation, _reward, terminated, truncated, info = env.step(action_np)
            num_steps = step + 1
            final_info = info.get("final_info") if isinstance(info, dict) else None
            if final_info is not None and "is_success" in final_info:
                success = bool(np.asarray(final_info["is_success"]).reshape(-1)[0])
            if bool(np.asarray(terminated | truncated).reshape(-1)[0]):
                break
    except Exception as exc:  # keep a row so pairwise matching can detect failed rollouts
        error_type = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    task_name = getattr(task, "name", "")
    return {
        "suite": args.task_suite,
        "task_id": args.selected_task_id,
        "task_name": task_name,
        "task_description": getattr(task, "language", ""),
        "episode_idx": episode_idx,
        "instance_id": f"{args.task_suite}/{task_name}/{episode_idx:04d}",
        "total_episode_idx": args.selected_task_id * 500 + episode_idx,
        "policy_name": args.policy_name,
        "checkpoint_ref": str(Path(args.checkpoint_dir)),
        "initial_states_path": str(Path(args.initial_states_path)),
        "success": int(success),
        "num_env_steps": num_steps,
        "error_type": error_type,
        "elapsed_s": f"{time.time() - start:.3f}",
    }


def main() -> None:
    args = parse_args()
    initial_states_path = Path(args.initial_states_path)
    install_init_state_loader(initial_states_path)

    policy, preprocessor, postprocessor, env_preprocessor = load_policy_and_processors(args)
    suite = libero_env_mod._get_suite(args.task_suite)

    episode_csv = Path(args.episode_results_path)
    episode_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "suite",
        "task_id",
        "task_name",
        "task_description",
        "episode_idx",
        "instance_id",
        "total_episode_idx",
        "policy_name",
        "checkpoint_ref",
        "initial_states_path",
        "success",
        "num_env_steps",
        "error_type",
        "elapsed_s",
    ]

    rows = []
    episode_indices = list(range(args.episode_start_idx, args.episode_end_idx))[: args.max_total_episodes]
    with episode_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        for episode_idx in episode_indices:
            row = run_episode(args, suite, policy, preprocessor, postprocessor, env_preprocessor, episode_idx)
            rows.append(row)
            writer.writerow(row)
            f.flush()

    result_dir = episode_csv.parent
    videos = list(result_dir.rglob("*.mp4"))
    summary = {
        "policy_name": args.policy_name,
        "policy_tracker_id": args.policy_tracker_id,
        "suite": args.task_suite,
        "task_id": args.selected_task_id,
        "model_id": args.model_id,
        "revision": args.revision or "",
        "tokenizer_name": args.tokenizer_name or "checkpoint_default",
        "checkpoint_dir": str(Path(args.checkpoint_dir)),
        "init_state_root": str(initial_states_path),
        "episode_csv": str(episode_csv),
        "rows": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "errors": sum(1 for row in rows if row["error_type"]),
        "videos": len(videos),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }
    manifest = {
        **summary,
        "init_state_manifest": str(initial_states_path / "MANIFEST.json"),
        "save_videos": False,
        "command": "python scripts/libero/run_pi05_goal_smoke.py ...",
    }
    Path(args.summary_path).write_text(json.dumps(summary, indent=2) + "\n")
    Path(args.manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if len(rows) != args.max_total_episodes:
        raise SystemExit(f"expected {args.max_total_episodes} rows, got {len(rows)}")
    if summary["errors"]:
        raise SystemExit(f"expected no episode errors, got {summary['errors']}")
    if summary["videos"] != 0:
        raise SystemExit(f"expected no videos, found {summary['videos']}")


if __name__ == "__main__":
    main()
