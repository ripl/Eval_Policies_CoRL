#!/usr/bin/env python3
"""Run one InternVLA-M1 SimplerEnv/WidowX task for 12 x 24 calibration."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

POLICY = "InternVLA-M1"
ROOT = Path("/share/data/ripl/tianchong/projects/Eval_Policies_CoRL")
SCRATCH = Path(os.environ.get("SCRATCH", ROOT / "scratch/repeated_fixed_grid_calibration_288_20260523_v2"))
DEFAULT_OUTPUT_ROOT = ROOT / "results/simplerenv/repeated_fixed_grid_calibration_288_20260523_v2/internvla_m1_third_party_standard_horizon"
CLONE = ROOT / "third_party/InternVLA-M1"
CKPT_ROOT = Path("/share/data/ripl/tianchong/projects/InternVLA-M1/cache/models/InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge")
CKPT = CKPT_ROOT / "checkpoints/steps_50000_pytorch_model.pt"
SIMPLER = Path("/share/data/ripl/tianchong/projects/InternVLA-M1/cache/Projects/SimplerEnv")
SERVER_PY = Path("/share/data/ripl/tianchong/conda/envs/internvla_m1_widowx_repro/bin/python")
SIM_PY = Path("/share/data/ripl/tianchong/conda/envs/internvla_simplerenv_repro/bin/python")
CACHE = ROOT / "cache"
ACCEPTED_INTERNVLA_COMMIT = "21e6e8f4bd42bee269dd021a118133501e9d8ede"
ACCEPTED_SIMPLER_COMMIT = "4ab7178e83e84ee06894034ec6dbf9e7aad1e882"
ACCEPTED_INTERNVLA_PATCH_PATH = "examples/SimplerEnv/start_simpler_env.py"
ACCEPTED_INTERNVLA_PATCH_SHA256 = "251f8e7e0e89dbd5ffa14b0ef5d6e89a9f54eb1806de15780f2925093cb5b733"
ACCEPTED_INTERNVLA_PATCH_CLASSIFICATION = "launch/host/debugpy-only"
ACCEPTED_SIMPLER_PATCH_PATHS = [
    "simpler_env/evaluation/maniskill2_evaluator.py",
    "simpler_env/utils/env/env_builder.py",
]
ACCEPTED_SIMPLER_PATCH_SHA256 = "cd3bdf02b2e5ac9949001fbbcb042e24505aa29868bc0396065b89b67c1522bc"
ACCEPTED_SIMPLER_PATCH_CLASSIFICATION = "evaluation-launch compatibility: fixed task reset/logging/result capture and renderer path plumbing"
EXPECTED_WEIGHT_SHA256 = {
    "checkpoints/steps_50000_pytorch_model.pt": "383fd5357fe193396027bc3c93b64312680d6ba98207fb433aee07b22f74acc1",
    "config.yaml": "552a1da418b6ecf1e829dd11c3a7603f8b8d046461320bf1a35c98873069727c",
    "dataset_statistics.json": "0de823f284562492fac29c3057f01c6dbe4b3c82b0091f73824ad1d9d9de7cae",
}
BASE_SEED = 20260523
REPEATS = range(12)
OFFICIAL_EPISODES = range(24)
RESULT_COLUMNS = [
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

TASK_ORDER = ["stack", "carrot", "spoon", "eggplant"]
TASKS = {
    "stack": {
        "env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "scene": "bridge_table_1_v1",
        "robot": "widowx",
        "overlay": SIMPLER / "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "init_x": "0.147",
        "init_y": "0.028",
        "horizon": 60,
    },
    "carrot": {
        "env_name": "PutCarrotOnPlateInScene-v0",
        "scene": "bridge_table_1_v1",
        "robot": "widowx",
        "overlay": SIMPLER / "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "init_x": "0.147",
        "init_y": "0.028",
        "horizon": 60,
    },
    "spoon": {
        "env_name": "PutSpoonOnTableClothInScene-v0",
        "scene": "bridge_table_1_v1",
        "robot": "widowx",
        "overlay": SIMPLER / "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "init_x": "0.147",
        "init_y": "0.028",
        "horizon": 60,
    },
    "eggplant": {
        "env_name": "PutEggplantInBasketScene-v0",
        "scene": "bridge_table_1_v2",
        "robot": "widowx_sink_camera_setup",
        "overlay": SIMPLER / "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "init_x": "0.127",
        "init_y": "0.06",
        "horizon": 120,
    },
}


def seed_for(task, repeat_id, official_episode_id):
    return BASE_SEED + TASK_ORDER.index(task) * 10000 + repeat_id * 24 + official_episode_id


def server_seed_for(task):
    return BASE_SEED + TASK_ORDER.index(task) * 10000 + 9000


def run_text(cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, text=True, stderr=subprocess.STDOUT).strip()


def git_head(path):
    return run_text(["git", "-C", str(path), "rev-parse", "HEAD"])


def check_path(path, label, executable=False):
    path = Path(path)
    if not path.exists():
        raise RuntimeError("missing {}: {}".format(label, path))
    if executable and not os.access(str(path), os.X_OK):
        raise RuntimeError("{} is not executable: {}".format(label, path))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(task):
    cfg = TASKS[task]
    required = [
        (CLONE / "examples/SimplerEnv/start_simpler_env.py", "InternVLA SimplerEnv client", False),
        (CLONE / "deployment/model_server/server_policy_M1.py", "InternVLA server", False),
        (CKPT, "InternVLA checkpoint", False),
        (SIMPLER / "simpler_env", "SimplerEnv package", False),
        (cfg["overlay"], "task rgb overlay", False),
        (SERVER_PY, "server python", True),
        (SIM_PY, "sim python", True),
        (SCRATCH / "run_seeded_policy_server.py", "seeded server wrapper", True),
        (SCRATCH / "run_seeded_simpler_episode.py", "seeded episode wrapper", True),
        (SCRATCH / "validate_internvla_task_results.py", "validator", True),
    ]
    for path, label, executable in required:
        check_path(path, label, executable=executable)

    internvla_head = git_head(CLONE)
    simpler_head = git_head(SIMPLER)
    if internvla_head != ACCEPTED_INTERNVLA_COMMIT:
        raise RuntimeError("InternVLA-M1 commit mismatch: {} != {}".format(internvla_head, ACCEPTED_INTERNVLA_COMMIT))
    if simpler_head != ACCEPTED_SIMPLER_COMMIT:
        raise RuntimeError("SimplerEnv commit mismatch: {} != {}".format(simpler_head, ACCEPTED_SIMPLER_COMMIT))
    simpler_status = run_text(["git", "-C", str(SIMPLER), "status", "--short"])
    expected_simpler_status = "\n".join(" M {}".format(path) for path in ACCEPTED_SIMPLER_PATCH_PATHS)
    if simpler_status.strip() != expected_simpler_status.strip():
        raise RuntimeError("unexpected SimplerEnv dirty status: {!r}".format(simpler_status))
    simpler_patch = subprocess.check_output(
        ["git", "-C", str(SIMPLER), "diff", "--"] + ACCEPTED_SIMPLER_PATCH_PATHS,
        text=True,
        stderr=subprocess.STDOUT,
    )
    simpler_patch_hash = hashlib.sha256(simpler_patch.encode()).hexdigest()
    if simpler_patch_hash != ACCEPTED_SIMPLER_PATCH_SHA256:
        raise RuntimeError("SimplerEnv patch hash mismatch: {} != {}".format(simpler_patch_hash, ACCEPTED_SIMPLER_PATCH_SHA256))
    internvla_status = run_text(["git", "-C", str(CLONE), "status", "--short"])
    allowed_status = " M {}".format(ACCEPTED_INTERNVLA_PATCH_PATH)
    if internvla_status.strip() != allowed_status.strip():
        raise RuntimeError("unexpected InternVLA dirty status: {!r}".format(internvla_status))
    patch_text = run_text(["git", "-C", str(CLONE), "diff", "--", ACCEPTED_INTERNVLA_PATCH_PATH])
    patch_hash = hashlib.sha256((patch_text + "\n").encode()).hexdigest()
    if patch_hash != ACCEPTED_INTERNVLA_PATCH_SHA256:
        raise RuntimeError("InternVLA patch hash mismatch: {} != {}".format(patch_hash, ACCEPTED_INTERNVLA_PATCH_SHA256))

    weight_hashes = {}
    for rel_path, expected_hash in EXPECTED_WEIGHT_SHA256.items():
        path = CKPT_ROOT / rel_path
        if not path.is_file():
            raise RuntimeError("missing pinned weight/config artifact: {}".format(path))
        actual_hash = sha256_file(path)
        weight_hashes[rel_path] = actual_hash
        if actual_hash != expected_hash:
            raise RuntimeError("weight/config hash mismatch for {}: {} != {}".format(path, actual_hash, expected_hash))

    horizons = {name: TASKS[name]["horizon"] for name in TASK_ORDER}
    if horizons != {"stack": 60, "carrot": 60, "spoon": 60, "eggplant": 120}:
        raise RuntimeError("standard horizon table was modified: {}".format(horizons))

    sbatch = SCRATCH / "internvla_m1_task.sbatch"
    if sbatch.exists() and "#SBATCH --array" in sbatch.read_text():
        raise RuntimeError("sbatch template contains a Slurm array directive")

    return {
        "status": "ok",
        "task": task,
        "policy": POLICY,
        "internvla_commit": internvla_head,
        "simpler_env_commit": simpler_head,
        "internvla_patch_path": ACCEPTED_INTERNVLA_PATCH_PATH,
        "internvla_patch_sha256": patch_hash,
        "internvla_patch_classification": ACCEPTED_INTERNVLA_PATCH_CLASSIFICATION,
        "simpler_env_patch_paths": ACCEPTED_SIMPLER_PATCH_PATHS,
        "simpler_env_patch_sha256": simpler_patch_hash,
        "simpler_env_patch_classification": ACCEPTED_SIMPLER_PATCH_CLASSIFICATION,
        "checkpoint": str(CKPT),
        "expected_weight_sha256": EXPECTED_WEIGHT_SHA256,
        "actual_weight_sha256": weight_hashes,
        "horizon": cfg["horizon"],
        "expected_rows": 12 * 24,
    }


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_manifest(task_root, task):
    cfg = TASKS[task]
    path = task_root / "manifest.csv"
    rows = []
    for repeat_id in REPEATS:
        for episode_id in OFFICIAL_EPISODES:
            rows.append({
                "policy": POLICY,
                "task": task,
                "official_episode_id": episode_id,
                "repeat_id": repeat_id,
                "seed": seed_for(task, repeat_id, episode_id),
                "horizon": cfg["horizon"],
                "env_name": cfg["env_name"],
                "scene": cfg["scene"],
                "robot": cfg["robot"],
                "rgb_overlay_path": str(cfg["overlay"]),
                "robot_init_x": cfg["init_x"],
                "robot_init_y": cfg["init_y"],
                "obj_variation_mode": "episode",
                "obj_episode_start": episode_id,
                "obj_episode_end": episode_id + 1,
            })
    fieldnames = list(rows[0].keys())
    tmp = path.with_suffix(".csv.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))
    return path


def ensure_results_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header != RESULT_COLUMNS:
            raise RuntimeError("unexpected per_episode_results.csv header: {}".format(header))
        return
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(RESULT_COLUMNS)
        f.flush()
        os.fsync(f.fileno())


def load_existing_keys(path):
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != RESULT_COLUMNS:
            raise RuntimeError("unexpected per_episode_results.csv header: {}".format(reader.fieldnames))
        keys = []
        for row in reader:
            keys.append((int(row["repeat_id"]), int(row["official_episode_id"])))
    seen = set()
    duplicates = []
    for key in keys:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise RuntimeError("duplicate existing result keys: {}".format(sorted(set(duplicates))[:10]))
    return seen


def append_result_row(path, row):
    if list(row.keys()) != RESULT_COLUMNS:
        raise RuntimeError("result row columns changed: {}".format(list(row.keys())))
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def write_seed_caveat(task_root, task):
    text = "".join([
        "policy=InternVLA-M1\n",
        "task={}\n".format(task),
        "seed_formula=BASE_SEED({}) + task_index*10000 + repeat_id*24 + official_episode_id\n".format(BASE_SEED),
        "server_seed_formula=BASE_SEED({}) + task_index*10000 + 9000\n".format(BASE_SEED),
        "client_seed_control=Each episode is run in a fresh simulator/client Python subprocess with PYTHONHASHSEED, random.seed, numpy.random.seed, and torch.manual_seed/torch.cuda.manual_seed_all set before importing the InternVLA client.\n",
        "server_seed_control=The policy server Python subprocess is seeded once at server startup with PYTHONHASHSEED, random.seed, numpy.random.seed, and torch.manual_seed/torch.cuda.manual_seed_all before loading InternVLA-M1.\n",
        "server_rng_caveat=InternVLA-M1 exposes no accepted per-episode server seed/reset RPC in this integration. Server-side diffusion/action RNG state advances across requests and is not reset to each manifest seed. Treat per-episode seeds as simulator/client seeds plus recorded rollout identifiers, not full control of server RNG.\n",
        "policy_state_reset=The accepted SimplerEnv evaluator calls M1Inference.reset(task_description) at the start of each episode; this launcher also isolates every official episode in a fresh client subprocess and logging directory.\n",
    ])
    atomic_write_text(task_root / "seed_caveat.txt", text)


def command_for_episode(task, repeat_id, episode_id, port, episode_root):
    cfg = TASKS[task]
    return [
        str(SIM_PY),
        "-u",
        str(SCRATCH / "run_seeded_simpler_episode.py"),
        "--seed",
        str(seed_for(task, repeat_id, episode_id)),
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
        cfg["robot"],
        "--policy-setup",
        "widowx_bridge",
        "--control-freq",
        "5",
        "--sim-freq",
        "500",
        "--max-episode-steps",
        str(cfg["horizon"]),
        "--env-name",
        cfg["env_name"],
        "--scene-name",
        cfg["scene"],
        "--rgb-overlay-path",
        str(cfg["overlay"]),
        "--robot-init-x-range",
        cfg["init_x"],
        cfg["init_x"],
        "1",
        "--robot-init-y-range",
        cfg["init_y"],
        cfg["init_y"],
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


def base_env(seed, include_simpler=True):
    env = os.environ.copy()
    env.update({
        "HF_HOME": str(CACHE / "huggingface"),
        "HF_HUB_CACHE": str(CACHE / "huggingface/hub"),
        "TRANSFORMERS_CACHE": str(CACHE / "huggingface/transformers"),
        "TORCH_HOME": str(CACHE / "torch"),
        "XDG_CACHE_HOME": str(CACHE / "xdg"),
        "PIP_CACHE_DIR": str(CACHE / "pip"),
        "WANDB_DIR": str(CACHE / "wandb"),
        "TOKENIZERS_PARALLELISM": "false",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": str(seed),
        "INTERNVLA_EVAL_SEED": str(seed),
        "DISPLAY": "",
    })
    for path in [env["HF_HOME"], env["TORCH_HOME"], env["XDG_CACHE_HOME"], env["PIP_CACHE_DIR"], env["WANDB_DIR"]]:
        Path(path).mkdir(parents=True, exist_ok=True)
    pythonpath_parts = [str(CLONE)]
    if include_simpler:
        pythonpath_parts.append(str(SIMPLER))
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return env


def wait_for_port(host, port, proc, timeout_sec):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited before accepting connections with code {}".format(proc.returncode))
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(5)
    raise RuntimeError("server did not accept connections on {}:{} within {}s".format(host, port, timeout_sec))


def start_server(task, port, task_root, timeout_sec):
    seed = server_seed_for(task)
    logs = task_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(SERVER_PY),
        "-u",
        str(SCRATCH / "run_seeded_policy_server.py"),
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
    with (logs / "server.log").open("ab") as log:
        proc = subprocess.Popen(cmd, cwd=str(CLONE), env=base_env(seed, include_simpler=False), stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    wait_for_port("127.0.0.1", port, proc, timeout_sec)
    return proc, cmd, seed


def stop_process_tree(proc):
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


def parse_summary(episode_root, episode_id):
    summary_files = sorted((episode_root / "simpler_results").rglob("episode_summary.csv"))
    if len(summary_files) != 1:
        raise RuntimeError("expected one episode_summary.csv, found {}".format(len(summary_files)))
    rows = []
    with summary_files[0].open(newline="") as f:
        for row in csv.reader(f):
            if row:
                rows.append(row)
    if len(rows) != 1:
        raise RuntimeError("expected one summary row, found {} in {}".format(len(rows), summary_files[0]))
    row = rows[0]
    if len(row) < 5:
        raise RuntimeError("summary row is malformed: {}".format(row))
    if int(row[1]) != episode_id:
        raise RuntimeError("summary episode id mismatch: {} != {}".format(row[1], episode_id))
    status = row[2].strip().lower()
    if status not in ("success", "failure"):
        raise RuntimeError("summary status is malformed: {}".format(status))
    return 1 if status == "success" else 0


def parse_steps(log_path):
    if not log_path.exists():
        return ""
    count = 0
    step_line = re.compile(r"^\s*\d+\s+\{")
    with log_path.open(errors="replace") as f:
        for line in f:
            if step_line.match(line):
                count += 1
    return str(count) if count else ""


def find_video_path(episode_root, episode_id):
    token = "_obj_episode_{}".format(episode_id)
    videos = sorted(p for p in (episode_root / "simpler_results").rglob("*.mp4") if token in p.name)
    if len(videos) != 1:
        raise RuntimeError("expected one video containing {}, found {}".format(token, len(videos)))
    return str(videos[0])


def tail_text(path, max_chars=4000):
    if not path.exists():
        return ""
    data = path.read_text(errors="replace")
    return data[-max_chars:]


def truncate_error(text, max_len=1000):
    text = " ".join(str(text).split())
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


def run_episode(task, repeat_id, episode_id, port, task_root, timeout_sec):
    episode_root = task_root / "episodes" / "repeat_{:02d}".format(repeat_id) / "episode_{:03d}".format(episode_id)
    logs = episode_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = command_for_episode(task, repeat_id, episode_id, port, episode_root)
    atomic_write_text(episode_root / "command.txt", " ".join(cmd) + "\n")
    seed = seed_for(task, repeat_id, episode_id)
    timeout_flag = 0
    rc = None
    log_path = logs / "client.log"
    with log_path.open("ab") as log:
        proc = subprocess.Popen(cmd, cwd=str(CLONE), env=base_env(seed, include_simpler=True), stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        try:
            rc = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timeout_flag = 1
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
            rc = proc.returncode

    success = ""
    steps = parse_steps(log_path)
    video_path = ""
    error = ""
    if timeout_flag:
        error = "episode subprocess timeout after {}s".format(timeout_sec)
    elif rc != 0:
        error = "episode subprocess exited with code {}; client_log_tail={}".format(rc, tail_text(log_path))

    try:
        if not error:
            success = str(parse_summary(episode_root, episode_id))
            video_path = find_video_path(episode_root, episode_id)
        else:
            try:
                success = str(parse_summary(episode_root, episode_id))
            except Exception:
                success = ""
            try:
                video_path = find_video_path(episode_root, episode_id)
            except Exception:
                video_path = ""
    except Exception as exc:
        error = "{}; parse_error={}".format(error, exc) if error else "parse_error={}".format(exc)

    return {
        "policy": POLICY,
        "task": task,
        "official_episode_id": str(episode_id),
        "repeat_id": str(repeat_id),
        "seed": str(seed),
        "success": success,
        "steps": steps,
        "error": truncate_error(error),
        "timeout": str(timeout_flag),
        "horizon": str(TASKS[task]["horizon"]),
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "video_path": video_path,
    }
def record_bundle(output_root, task, hash_weights):
    bundle = output_root / "bundle" / task
    bundle.mkdir(parents=True, exist_ok=True)
    scripts = bundle / "scripts_snapshot"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in ["internvla_m1_task_driver.py", "run_seeded_simpler_episode.py", "run_seeded_policy_server.py", "validate_internvla_task_results.py", "internvla_m1_task.sbatch"]:
        src = SCRATCH / name
        if src.exists():
            shutil.copy2(str(src), str(scripts / name))

    repos = []
    for label, path in [("Eval_Policies_CoRL", ROOT), ("InternVLA-M1", CLONE), ("SimplerEnv", SIMPLER)]:
        section = ["# {}".format(label), "path={}".format(path)]
        for key, cmd in [
            ("remote", ["git", "-C", str(path), "remote", "get-url", "origin"]),
            ("branch", ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"]),
            ("commit", ["git", "-C", str(path), "rev-parse", "HEAD"]),
        ]:
            try:
                section.append("{}={}".format(key, run_text(cmd)))
            except Exception as exc:
                section.append("{}=ERROR:{}".format(key, exc))
        try:
            section.append("status_start")
            section.append(run_text(["git", "-C", str(path), "status", "--short"]) or "<clean>")
            section.append("status_end")
        except Exception as exc:
            section.append("status_error={}".format(exc))
        repos.append("\n".join(section))
    atomic_write_text(bundle / "code_repos.txt", "\n\n".join(repos) + "\n")

    try:
        internvla_patch = subprocess.check_output(["git", "-C", str(CLONE), "diff", "--", ACCEPTED_INTERNVLA_PATCH_PATH], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        internvla_patch = "diff_error={}\n".format(exc)
    atomic_write_text(bundle / "internvla_m1_local.patch", internvla_patch)
    atomic_write_text(
        bundle / "internvla_m1_local_patch_classification.txt",
        "\n".join([
            "path={}".format(ACCEPTED_INTERNVLA_PATCH_PATH),
            "sha256={}".format(hashlib.sha256(internvla_patch.encode()).hexdigest()),
            "expected_sha256={}".format(ACCEPTED_INTERNVLA_PATCH_SHA256),
            "classification={}".format(ACCEPTED_INTERNVLA_PATCH_CLASSIFICATION),
            "scope=launch/host/debugpy-only; not evaluation-semantic",
        ]) + "\n",
    )
    try:
        simpler_patch = subprocess.check_output(["git", "-C", str(SIMPLER), "diff", "--"] + ACCEPTED_SIMPLER_PATCH_PATHS, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        simpler_patch = "diff_error={}\n".format(exc)
    atomic_write_text(bundle / "simpler_env_local.patch", simpler_patch)
    atomic_write_text(
        bundle / "simpler_env_local_patch_classification.txt",
        "\n".join([
            "paths={}".format(",".join(ACCEPTED_SIMPLER_PATCH_PATHS)),
            "sha256={}".format(hashlib.sha256(simpler_patch.encode()).hexdigest()),
            "expected_sha256={}".format(ACCEPTED_SIMPLER_PATCH_SHA256),
            "classification={}".format(ACCEPTED_SIMPLER_PATCH_CLASSIFICATION),
        ]) + "\n",
    )

    env_lines = [
        "server_python={}".format(SERVER_PY),
        "server_python_version={}".format(run_text([str(SERVER_PY), "--version"])),
        "sim_python={}".format(SIM_PY),
        "sim_python_version={}".format(run_text([str(SIM_PY), "--version"])),
    ]
    atomic_write_text(bundle / "environment.txt", "\n".join(env_lines) + "\n")

    weights = [
        "checkpoint_source=InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge",
        "checkpoint_root={}".format(CKPT_ROOT),
        "checkpoint_path={}".format(CKPT),
        "hash_weights={}".format(int(bool(hash_weights))),
    ]
    atomic_write_text(bundle / "weights.txt", "\n".join(weights) + "\n")
    if hash_weights:
        lines = []
        for path in [CKPT, CKPT_ROOT / "config.yaml", CKPT_ROOT / "dataset_statistics.json", CKPT_ROOT / "README.md"]:
            if path.exists():
                lines.append("{}  {}".format(sha256_file(path), path))
        atomic_write_text(bundle / "weights.sha256", "\n".join(lines) + "\n")

    cfg = TASKS[task]
    write_json(bundle / "task_protocol_{}.json".format(task), {
        "policy": POLICY,
        "task": task,
        "repeat_ids": [0, 11],
        "official_episode_ids": [0, 23],
        "rows": 288,
        "standard_horizon": cfg["horizon"],
        "env_name": cfg["env_name"],
        "scene": cfg["scene"],
        "robot": cfg["robot"],
        "rgb_overlay_path": str(cfg["overlay"]),
        "robot_init_x": cfg["init_x"],
        "robot_init_y": cfg["init_y"],
        "obj_variation_mode": "episode",
        "control_freq": 5,
        "sim_freq": 500,
        "policy_setup": "widowx_bridge",
    })


def validate_task(task_root, task, allow_error_rows):
    cmd = [str(SIM_PY), str(SCRATCH / "validate_internvla_task_results.py"), "--task-root", str(task_root), "--task", task]
    if allow_error_rows:
        cmd.append("--allow-error-rows")
    subprocess.check_call(cmd)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASK_ORDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--episode-timeout-sec", type=int, default=1800)
    parser.add_argument("--server-timeout-sec", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true", help="write manifest/header and print commands without starting server or running episodes")
    parser.add_argument("--preflight", action="store_true", help="check paths, commits, horizons, and wrappers without writing output")
    parser.add_argument("--no-skip-existing", action="store_true", help="fail if per_episode_results.csv already has completed keys")
    parser.add_argument("--continue-on-error", action="store_true", help="append error rows and continue instead of stopping at first error")
    parser.add_argument("--allow-error-rows-in-validation", action="store_true", help="let final validator pass coverage despite error/timeout rows")
    parser.add_argument("--hash-weights", action="store_true", help="hash checkpoint and critical config/stat files into bundle/weights.sha256")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    if args.preflight:
        print(json.dumps(preflight(args.task), indent=2, sort_keys=True))
        return 0

    preflight_report = preflight(args.task)
    output_root = args.output_root
    task_root = output_root / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "logs").mkdir(parents=True, exist_ok=True)
    write_json(task_root / "preflight.json", preflight_report)
    manifest_path = write_manifest(task_root, args.task)
    write_seed_caveat(task_root, args.task)
    results_path = task_root / "per_episode_results.csv"
    ensure_results_header(results_path)
    record_bundle(output_root, args.task, args.hash_weights and not args.dry_run)

    existing = load_existing_keys(results_path)
    if args.no_skip_existing and existing:
        raise RuntimeError("existing result rows found and --no-skip-existing was set")

    port = args.port or (20000 + (int(os.environ.get("SLURM_JOB_ID", os.getpid())) % 20000))
    planned = []
    for repeat_id in REPEATS:
        for episode_id in OFFICIAL_EPISODES:
            if (repeat_id, episode_id) in existing:
                continue
            planned.append((repeat_id, episode_id, seed_for(args.task, repeat_id, episode_id)))

    dry_payload = {
        "status": "dry_run" if args.dry_run else "planned",
        "task": args.task,
        "output_root": str(output_root),
        "task_root": str(task_root),
        "manifest": str(manifest_path),
        "per_episode_results": str(results_path),
        "port": port,
        "total_manifest_rows": 288,
        "existing_result_rows": len(existing),
        "episodes_to_run": len(planned),
        "first_planned": planned[0] if planned else None,
        "last_planned": planned[-1] if planned else None,
    }
    write_json(task_root / ("dry_run_plan.json" if args.dry_run else "run_plan.json"), dry_payload)
    print(json.dumps(dry_payload, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    server_proc = None
    try:
        if planned:
            server_proc, server_cmd, server_seed = start_server(args.task, port, task_root, args.server_timeout_sec)
            write_json(task_root / "server_start.json", {
                "server_seed": server_seed,
                "port": port,
                "pid": server_proc.pid,
                "command": server_cmd,
            })
        for repeat_id, episode_id, _seed in planned:
            row = run_episode(args.task, repeat_id, episode_id, port, task_root, args.episode_timeout_sec)
            append_result_row(results_path, row)
            if (row["error"] or row["timeout"] == "1") and not args.continue_on_error:
                raise RuntimeError("episode failed; appended row then stopping: repeat_id={} official_episode_id={} error={}".format(repeat_id, episode_id, row["error"]))
        validate_task(task_root, args.task, args.allow_error_rows_in_validation)
        atomic_write_text(task_root / "status.txt", "completed_at={}\n".format(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        return 0
    finally:
        stop_process_tree(server_proc)


if __name__ == "__main__":
    sys.exit(main())
