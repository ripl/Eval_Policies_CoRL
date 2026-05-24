#!/usr/bin/env python3
"""Run one repeated fixed-grid SimplerEnv job for one policy/task.

This script is scratch-only support for the 2026-05-23 repeated calibration.
One invocation owns exactly one (policy, task) job and appends one CSV row after
an episode finishes or errors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/share/data/ripl/tianchong/projects/Eval_Policies_CoRL"))
SCRATCH_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ID = SCRATCH_DIR.name
DEFAULT_BASE_SEED = 202605230
CSV_COLUMNS = [
    "policy",
    "task",
    "official_episode_id",
    "repeat_id",
    "seed",
    "success",
    "steps",
    "error",
    "timeout",
    "horizon",
    "job_id",
    "video_path",
]

EXPECTED_COMMITS = {
    "cogact_repo": "b174a1b86deedfab4d198d935207e7bb0527994e",
    "simpler_env_repo": "06accaca93535902d408da4855f21cece12bceb7",
    "simpler_env_maniskill2": "ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682",
    "spatialvla_repo": "ccfe3809766839a2fcfb7a3d3c9abff585189188",
    "spatialvla_maniskill2": "cd45dd27dc6bb26d048cb6570cdab4e3f935cc37",
}
EXPECTED_HF_REVISIONS = {
    "CogACT/CogACT-Base": "6550bf0992f162fc5d74f14ffee30771a9433363",
    "IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge": "cfb5ff76f7545d07c134f33810d8bf4070b5d517",
}
MAX_HASH_BYTES = 64 * 1024 * 1024

POLICY_OFFSETS = {"cogact": 100_000, "spatialvla": 200_000}
TASK_OFFSETS = {"stack": 1_000, "carrot": 2_000, "spoon": 3_000, "eggplant": 4_000}


@dataclass(frozen=True)
class TaskSpec:
    env_name: str
    scene_name: str
    robot: str
    overlay_name: str
    robot_init_x: float
    robot_init_y: float
    horizon: int


TASKS = {
    "stack": TaskSpec(
        env_name="StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        overlay_name="bridge_real_eval_1.png",
        robot_init_x=0.147,
        robot_init_y=0.028,
        horizon=60,
    ),
    "carrot": TaskSpec(
        env_name="PutCarrotOnPlateInScene-v0",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        overlay_name="bridge_real_eval_1.png",
        robot_init_x=0.147,
        robot_init_y=0.028,
        horizon=60,
    ),
    "spoon": TaskSpec(
        env_name="PutSpoonOnTableClothInScene-v0",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        overlay_name="bridge_real_eval_1.png",
        robot_init_x=0.147,
        robot_init_y=0.028,
        horizon=60,
    ),
    "eggplant": TaskSpec(
        env_name="PutEggplantInBasketScene-v0",
        scene_name="bridge_table_1_v2",
        robot="widowx_sink_camera_setup",
        overlay_name="bridge_sink.png",
        robot_init_x=0.127,
        robot_init_y=0.06,
        horizon=120,
    ),
}


@dataclass(frozen=True)
class PolicySpec:
    label: str
    env_prefix: Path
    ckpt_path: str
    policy_model: str
    repo_paths: dict[str, Path]
    overlay_dir: Path


def build_policy_specs() -> dict[str, PolicySpec]:
    return {
        "cogact": PolicySpec(
            label="cogact",
            env_prefix=Path(os.environ.get("COGACT_ENV_PREFIX", PROJECT_ROOT / "envs/simplerenv_cogact_py310_np126")),
            ckpt_path=os.environ.get("COGACT_CKPT_PATH", "CogACT/CogACT-Base"),
            policy_model="cogact",
            repo_paths={
                "cogact_repo": PROJECT_ROOT / "third_party/cogact",
                "simpler_env_repo": PROJECT_ROOT / "third_party/simpler_env",
                "simpler_env_maniskill2": PROJECT_ROOT / "third_party/simpler_env/ManiSkill2_real2sim",
            },
            overlay_dir=PROJECT_ROOT / "third_party/simpler_env/ManiSkill2_real2sim/data/real_inpainting",
        ),
        "spatialvla": PolicySpec(
            label="spatialvla",
            env_prefix=Path(os.environ.get("SPATIALVLA_ENV_PREFIX", PROJECT_ROOT / "envs/simplerenv_spatialvla_py310")),
            ckpt_path=os.environ.get("SPATIALVLA_CKPT_PATH", "IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge"),
            policy_model="spatialvla",
            repo_paths={
                "spatialvla_repo": Path(os.environ.get("SPATIALVLA_REPO", SCRATCH_DIR / "sources/simplerenv_openvla_ccfe380")),
                "spatialvla_maniskill2": Path(os.environ.get("SPATIALVLA_REPO", SCRATCH_DIR / "sources/simplerenv_openvla_ccfe380")) / "ManiSkill2_real2sim",
            },
            overlay_dir=Path(os.environ.get("SPATIALVLA_REPO", SCRATCH_DIR / "sources/simplerenv_openvla_ccfe380")) / "ManiSkill2_real2sim/data/real_inpainting",
        ),
    }


def run_cmd(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def git_head(path: Path) -> str:
    code, out, err = run_cmd(["git", "rev-parse", "HEAD"], cwd=path)
    if code != 0:
        return f"ERROR: {err or out}"
    return out


def git_status(path: Path) -> str:
    code, out, err = run_cmd(["git", "status", "--short"], cwd=path)
    if code != 0:
        return f"ERROR: {err or out}"
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hf_cache_identity(model_path: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    info: dict[str, Any] = {"model_path": model_path}
    path = Path(model_path)
    if path.exists():
        info.update({"source_type": "local_path", "local_path": str(path.resolve())})
        return info, errors
    if "/" not in model_path:
        errors.append(f"checkpoint path {model_path!r} is neither a local path nor an HF repo id")
        return info, errors
    owner, repo = model_path.split("/", 1)
    cache_dir = Path(os.environ["HF_HUB_CACHE"]) / f"models--{owner}--{repo}"
    expected_revision = EXPECTED_HF_REVISIONS.get(model_path)
    info.update({"source_type": "hf_repo_id", "cache_dir": str(cache_dir), "expected_revision": expected_revision})
    if expected_revision is None:
        errors.append(f"no expected HF revision pinned for mutable checkpoint id {model_path}")
        return info, errors
    refs_dir = cache_dir / "refs"
    revision = (refs_dir / "main").read_text().strip() if (refs_dir / "main").is_file() else None
    info["resolved_revision"] = revision
    if revision != expected_revision:
        errors.append(f"HF cached revision mismatch for {model_path}: {revision} != {expected_revision}")
    snapshot_path = cache_dir / "snapshots" / expected_revision
    info["snapshot_path"] = str(snapshot_path)
    if not snapshot_path.is_dir():
        errors.append(f"missing expected HF snapshot for {model_path}: {snapshot_path}")
        return info, errors
    files = []
    for name in [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "dataset_statistics.json",
    ]:
        file_path = snapshot_path / name
        if file_path.exists():
            record: dict[str, Any] = {"path": str(file_path), "size_bytes": file_path.stat().st_size}
            if record["size_bytes"] <= MAX_HASH_BYTES:
                record["sha256"] = sha256_file(file_path)
            files.append(record)
    info["critical_files"] = files
    return info, errors


def output_root_for(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root)
    run_id = args.run_id or DEFAULT_RUN_ID
    return SCRATCH_DIR / "outputs" / run_id / f"{args.policy}_{args.task}"


def deterministic_seed(policy: str, task: str, repeat_id: int, official_episode_id: int, base_seed: int) -> int:
    return base_seed + POLICY_OFFSETS[policy] + TASK_OFFSETS[task] + repeat_id * 24 + official_episode_id


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception as exc:  # pragma: no cover - import availability is environment-specific.
        print(f"WARNING: could not set NumPy seed: {exc}", file=sys.stderr)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not set torch seed: {exc}", file=sys.stderr)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not set TensorFlow seed: {exc}", file=sys.stderr)


def install_cache_defaults() -> None:
    defaults = {
        "HF_HOME": PROJECT_ROOT / "cache/huggingface",
        "HF_HUB_CACHE": PROJECT_ROOT / "cache/huggingface/hub",
        "TRANSFORMERS_CACHE": PROJECT_ROOT / "cache/huggingface/hub",
        "TORCH_HOME": PROJECT_ROOT / "cache/torch",
        "XDG_CACHE_HOME": PROJECT_ROOT / "cache/xdg",
        "PIP_CACHE_DIR": PROJECT_ROOT / "cache/pip",
        "CONDA_PKGS_DIRS": PROJECT_ROOT / "cache/conda_pkgs",
        "APPTAINER_CACHEDIR": PROJECT_ROOT / "cache/apptainer",
        "SINGULARITY_CACHEDIR": PROJECT_ROOT / "cache/singularity",
        "WANDB_DIR": PROJECT_ROOT / "artifacts/wandb",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not os.environ.get("VK_ICD_FILENAMES"):
        for candidate in ("/etc/vulkan/icd.d/nvidia_icd.json", "/usr/share/vulkan/icd.d/nvidia_icd.json"):
            if Path(candidate).is_file():
                os.environ["VK_ICD_FILENAMES"] = candidate
                break


def preflight(policy: str, task: str, output_root: Path) -> dict[str, Any]:
    policies = build_policy_specs()
    spec = policies[policy]
    task_spec = TASKS[task]
    errors: list[str] = []
    report: dict[str, Any] = {
        "policy": policy,
        "task": task,
        "project_root": str(PROJECT_ROOT),
        "scratch_dir": str(SCRATCH_DIR),
        "output_root": str(output_root),
        "expected_commits": {},
        "current_commits": {},
        "dirty_status": {},
        "checkpoint_identity": {},
        "checks": {},
    }

    required_paths = {
        "project_root": PROJECT_ROOT,
        "env_python": spec.env_prefix / "bin/python",
        "overlay": spec.overlay_dir / task_spec.overlay_name,
    }
    required_paths.update(spec.repo_paths)
    for label, path in required_paths.items():
        exists = path.exists()
        report["checks"][label] = {"path": str(path), "exists": exists}
        if not exists:
            errors.append(f"missing {label}: {path}")

    parent_head = git_head(PROJECT_ROOT)
    parent_status = git_status(PROJECT_ROOT)
    report["current_commits"]["project_root"] = parent_head
    report["dirty_status"]["project_root"] = parent_status

    for label, path in spec.repo_paths.items():
        expected = EXPECTED_COMMITS[label]
        current = git_head(path) if path.exists() else "MISSING"
        status = git_status(path) if path.exists() else "MISSING"
        report["expected_commits"][label] = expected
        report["current_commits"][label] = current
        report["dirty_status"][label] = status
        if current != expected:
            errors.append(f"{label} commit mismatch: expected {expected}, current {current}")
        if status:
            errors.append(f"{label} has dirty status: {status}")

    checkpoint_identity, checkpoint_errors = hf_cache_identity(spec.ckpt_path)
    report["checkpoint_identity"] = checkpoint_identity
    errors.extend(checkpoint_errors)

    report["status"] = "failed" if errors else "passed"
    report["errors"] = errors
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if errors:
        detail = "\n".join(f"- {err}" for err in errors)
        raise RuntimeError(f"preflight failed for {policy}/{task}:\n{detail}")
    return report


def manifest_rows(policy: str, task: str, base_seed: int, output_root: Path) -> list[dict[str, Any]]:
    spec = build_policy_specs()[policy]
    task_spec = TASKS[task]
    overlay_path = spec.overlay_dir / task_spec.overlay_name
    rows: list[dict[str, Any]] = []
    for repeat_id in range(12):
        for official_episode_id in range(24):
            seed = deterministic_seed(policy, task, repeat_id, official_episode_id, base_seed)
            rows.append(
                {
                    "policy": policy,
                    "task": task,
                    "official_episode_id": official_episode_id,
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "horizon": task_spec.horizon,
                    "env_name": task_spec.env_name,
                    "scene_name": task_spec.scene_name,
                    "robot": task_spec.robot,
                    "robot_init_x": task_spec.robot_init_x,
                    "robot_init_y": task_spec.robot_init_y,
                    "overlay_path": str(overlay_path),
                    "ckpt_path": spec.ckpt_path,
                    "output_root": str(output_root),
                }
            )
    return rows


def write_manifest(policy: str, task: str, base_seed: int, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows(policy, task, base_seed, output_root)
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "policy": policy,
        "task": task,
        "row_count": len(rows),
        "official_episode_ids": [0, 23],
        "repeat_ids": [0, 11],
        "seed_formula": "base_seed + policy_offset + task_offset + repeat_id * 24 + official_episode_id",
        "base_seed": base_seed,
        "policy_offset": POLICY_OFFSETS[policy],
        "task_offset": TASK_OFFSETS[task],
        "task_spec": asdict(TASKS[task]),
        "csv_columns": CSV_COLUMNS,
    }
    (output_root / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return manifest_path


def append_result(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="raise")
        if needs_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def existing_keys(csv_path: Path, policy: str, task: str) -> set[tuple[int, int]]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise RuntimeError(f"existing CSV has wrong columns: {reader.fieldnames}")
        keys: set[tuple[int, int]] = set()
        for row in reader:
            if row["policy"] != policy or row["task"] != task:
                raise RuntimeError(f"existing CSV row has wrong job key: {row}")
            if row.get("error", "").strip() or row.get("timeout", "").strip() not in {"", "0"}:
                raise RuntimeError(f"existing CSV contains error/timeout row; refusing unsafe resume: {row}")
            if row.get("success") not in {"0", "1"} or not row.get("steps") or not row.get("video_path"):
                raise RuntimeError(f"existing CSV contains incomplete row; refusing unsafe resume: {row}")
            keys.add((int(row["repeat_id"]), int(row["official_episode_id"])))
        return keys


def configure_python_path(policy: str) -> None:
    spec = build_policy_specs()[policy]
    if policy == "cogact":
        paths = [spec.repo_paths["cogact_repo"], spec.repo_paths["simpler_env_repo"]]
    else:
        paths = [spec.repo_paths["spatialvla_repo"]]
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    old = os.environ.get("PYTHONPATH", "")
    prefix = os.pathsep.join(str(p) for p in paths)
    os.environ["PYTHONPATH"] = prefix + (os.pathsep + old if old else "")


def configure_tensorflow_memory(memory_limit: int) -> None:
    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
            except RuntimeError as exc:
                print(f"WARNING: TensorFlow GPU memory was already initialized: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: TensorFlow memory configuration skipped: {exc}", file=sys.stderr)


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
    spec = build_policy_specs()[policy]
    configure_python_path(policy)
    configure_tensorflow_memory(int(os.environ.get("TF_MEMORY_LIMIT", "3072")))
    if policy == "cogact":
        patch_hf_token_for_cogact()
        from sim_cogact import CogACTInference

        return CogACTInference(
            saved_model_path=spec.ckpt_path,
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

        return SpatialVLAInference(saved_model_path=spec.ckpt_path, policy_setup="widowx_bridge", action_scale=1.0)
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


def run_single_episode(
    model: Any,
    policy: str,
    task: str,
    official_episode_id: int,
    repeat_id: int,
    output_root: Path,
) -> dict[str, Any]:
    import numpy as np
    from transforms3d.euler import quat2euler

    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from simpler_env.utils.visualization import write_video

    spec = build_policy_specs()[policy]
    task_spec = TASKS[task]
    overlay_path = spec.overlay_dir / task_spec.overlay_name
    control_mode = get_robot_control_mode(task_spec.robot, spec.policy_model)
    kwargs = dict(
        obs_mode="rgbd",
        robot=task_spec.robot,
        sim_freq=500,
        control_mode=control_mode,
        control_freq=5,
        max_episode_steps=task_spec.horizon,
        scene_name=task_spec.scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=str(overlay_path),
    )
    env = build_maniskill2_env(task_spec.env_name, **kwargs)
    robot_init_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    reset_options = {
        "robot_init_options": {
            "init_xy": np.array([task_spec.robot_init_x, task_spec.robot_init_y]),
            "init_rot_quat": robot_init_quat,
        },
        "obj_init_options": {"episode_id": official_episode_id},
    }
    obs, _ = env.reset(options=reset_options)
    is_final_subtask = env_method(env, "is_final_subtask")()
    task_description = env_method(env, "get_language_instruction")()
    print(task_description)

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

        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )
        success_text = "success" if done else "failure"
        new_task_description = env_method(env, "get_language_instruction")()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env_method(env, "is_final_subtask")()
        print(timestep, info)
        image = get_image_from_maniskill2_obs_dict(obs_env, obs, camera_name=None)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {}) if isinstance(info, dict) else {}
    env_save_name = task_spec.env_name
    ckpt_basename = spec.ckpt_path[:-1] if spec.ckpt_path.endswith("/") else spec.ckpt_path
    ckpt_basename = ckpt_basename.split("/")[-1]
    video_name = f"{success_text}_obj_episode_{official_episode_id}"
    for key, value in episode_stats.items():
        video_name += f"_{key}_{value}"
    video_name += ".mp4"
    overlay_stem = overlay_path.stem
    roll, pitch, yaw = quat2euler(robot_init_quat)
    relative_path = (
        f"{task_spec.scene_name}/{control_mode}/{env_save_name}/"
        f"rob_{task_spec.robot_init_x}_{task_spec.robot_init_y}_rot_{roll:.3f}_{pitch:.3f}_{yaw:.3f}_"
        f"rgb_overlay_{overlay_stem}/{video_name}"
    )
    if policy == "cogact":
        relative_path = f"{ckpt_basename}/{relative_path}"
    video_path = output_root / "videos" / f"repeat_{repeat_id:02d}" / relative_path
    write_video(str(video_path), images, fps=5)

    action_path = Path(str(video_path).replace(".mp4", ".png"))
    action_root = action_path.parent / "actions"
    action_root.mkdir(parents=True, exist_ok=True)
    model.visualize_epoch(predicted_actions, images, save_path=str(action_root / action_path.name))

    return {
        "success": bool(success_text == "success"),
        "steps": timestep,
        # SimplerEnv horizon truncation is an ordinary failed rollout, not an
        # infrastructure timeout. External subprocess timeouts are recorded by
        # wrappers that can distinguish wall-clock failures.
        "timeout": False,
        "video_path": str(video_path),
    }


def write_runtime_metadata(output_root: Path, policy: str, task: str, base_seed: int) -> None:
    stochastic_action_sampling = {
        "cogact": "active: CogACT-Base uses a diffusion action head seeded before each rollout",
        "spatialvla": "not intentionally active in the evaluated path: greedy action-token decoding",
    }
    metadata = {
        "policy": policy,
        "task": task,
        "base_seed": base_seed,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "hostname": os.uname().nodename,
        "python": sys.executable,
        "python_version": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "ckpt_path": build_policy_specs()[policy].ckpt_path,
        "preflight_report": str(output_root / "preflight_report.json"),
        "stochastic_action_sampling": stochastic_action_sampling[policy],
        "policy_state_reset": "model.reset(task_description) is called at the start of every rollout",
        "rollout_seed_control": "PYTHONHASHSEED, Python random, NumPy, torch, and TensorFlow seeds are set from the manifest seed before each rollout",
        "time_utc_epoch": int(time.time()),
    }
    (output_root / "runtime_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICY_OFFSETS), default=os.environ.get("POLICY"))
    parser.add_argument("--task", choices=sorted(TASKS), default=os.environ.get("TASK"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", DEFAULT_RUN_ID))
    parser.add_argument("--output-root", default=os.environ.get("RESULTS_ROOT"))
    parser.add_argument("--base-seed", type=int, default=int(os.environ.get("BASE_SEED", str(DEFAULT_BASE_SEED))))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run preflight and write manifest/header, but do not run episodes.")
    parser.add_argument("--resume", action="store_true", default=os.environ.get("RESUME", "0") == "1")
    args = parser.parse_args()
    if not args.policy or not args.task:
        parser.error("--policy/--task or POLICY/TASK are required")
    return args


def main() -> int:
    args = parse_args()
    output_root = output_root_for(args)
    install_cache_defaults()
    preflight(args.policy, args.task, output_root)
    if args.preflight_only:
        print(f"preflight passed: {output_root / 'preflight_report.json'}")
        return 0

    manifest_path = write_manifest(args.policy, args.task, args.base_seed, output_root)
    results_csv = output_root / "per_episode_results.csv"
    if results_csv.exists() and not args.resume and not args.dry_run:
        raise RuntimeError(f"refusing to append to existing CSV without --resume: {results_csv}")
    if args.dry_run:
        if not results_csv.exists():
            append_result(results_csv, {key: "" for key in CSV_COLUMNS})
            text = results_csv.read_text().splitlines()[0] + "\n"
            results_csv.write_text(text)
        print(f"dry run wrote manifest: {manifest_path}")
        print(f"dry run wrote CSV header: {results_csv}")
        return 0

    write_runtime_metadata(output_root, args.policy, args.task, args.base_seed)
    completed = existing_keys(results_csv, args.policy, args.task) if args.resume else set()
    model = make_model(args.policy)
    job_id = os.environ.get("SLURM_JOB_ID", "")

    for repeat_id in range(12):
        for official_episode_id in range(24):
            if (repeat_id, official_episode_id) in completed:
                print(f"skipping existing repeat={repeat_id} episode={official_episode_id}")
                continue
            seed = deterministic_seed(args.policy, args.task, repeat_id, official_episode_id, args.base_seed)
            set_all_seeds(seed)
            row = {
                "policy": args.policy,
                "task": args.task,
                "official_episode_id": official_episode_id,
                "repeat_id": repeat_id,
                "seed": seed,
                "success": "",
                "steps": "",
                "error": "",
                "timeout": "",
                "horizon": TASKS[args.task].horizon,
                "job_id": job_id,
                "video_path": "",
            }
            try:
                result = run_single_episode(model, args.policy, args.task, official_episode_id, repeat_id, output_root)
                row.update(
                    {
                        "success": int(result["success"]),
                        "steps": result["steps"],
                        "timeout": int(result["timeout"]),
                        "video_path": result["video_path"],
                    }
                )
                append_result(results_csv, row)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                append_result(results_csv, row)
                traceback.print_exc()
                raise
    print(f"completed {args.policy}/{args.task}: {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
