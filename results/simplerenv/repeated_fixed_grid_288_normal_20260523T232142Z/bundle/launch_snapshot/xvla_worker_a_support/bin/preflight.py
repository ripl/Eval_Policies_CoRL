#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from protocol import POLICY, POLICY_DISPLAY, TASK_CLIENT, horizon_for, project_root, server_start_seed, validate_task

EXPECTED_X_VLA_COMMIT = "ccd1992f3ecce554e3ebe68e21c759acf111f2b0"
EXPECTED_XVLA_SIMPLER_ENV_COMMIT = "4233e3fcf006f0bd0e951c190db3b209dc3f3543"
SUPPORT_ROOT = Path(os.environ.get("SUPPORT_ROOT", Path(__file__).resolve().parents[1]))
EXPECTED_HF_REVISION = "8d7ea1aaa948665d44129a3ff488629b955fc0f9"
MAX_HASH_BYTES = 64 * 1024 * 1024


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"command failed: {' '.join(cmd)}")
    return proc.stdout.strip()


def git_record(path: Path, expected_commit: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path)}
    record["head"] = run(["git", "-C", str(path), "rev-parse", "HEAD"])
    record["status_short"] = run(["git", "-C", str(path), "status", "--short"])
    if expected_commit is not None:
        record["expected_head"] = expected_commit
        record["head_matches_expected"] = record["head"] == expected_commit
    return record


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    rec: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "resolved_path": str(resolved),
        "is_symlink": path.is_symlink(),
        "symlink_target": os.readlink(path) if path.is_symlink() else "",
        "size_bytes": resolved.stat().st_size,
    }
    if rec["size_bytes"] <= MAX_HASH_BYTES:
        rec["sha256"] = sha256_file(resolved)
    else:
        rec["sha256"] = None
        rec["sha256_note"] = f"skipped because file exceeds {MAX_HASH_BYTES} bytes"
    return rec


def hf_cache_identity(model_path: str, hf_hub_cache: Path, expected_revision: str | None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    info: dict[str, Any] = {"model_path": model_path, "hf_hub_cache": str(hf_hub_cache)}
    if Path(model_path).exists():
        local = Path(model_path).resolve()
        info.update({"source_type": "local_path", "local_path": str(local)})
        return info, errors
    if "/" not in model_path:
        errors.append(f"model path {model_path!r} is neither a local path nor an HF repo id")
        return info, errors
    owner, repo = model_path.split("/", 1)
    cache_dir = hf_hub_cache / f"models--{owner}--{repo}"
    info.update({"source_type": "hf_repo_id", "cache_dir": str(cache_dir)})
    if not cache_dir.is_dir():
        errors.append(f"HF cache directory missing for mutable model id: {cache_dir}")
        return info, errors
    refs = {}
    refs_dir = cache_dir / "refs"
    if refs_dir.is_dir():
        for ref_file in sorted(refs_dir.iterdir()):
            if ref_file.is_file():
                refs[ref_file.name] = ref_file.read_text().strip()
    snapshots_dir = cache_dir / "snapshots"
    snapshots = sorted(p.name for p in snapshots_dir.iterdir() if p.is_dir()) if snapshots_dir.is_dir() else []
    info["refs"] = refs
    info["snapshots"] = snapshots
    revision = refs.get("main")
    if revision is None and len(snapshots) == 1:
        revision = snapshots[0]
    info["resolved_revision"] = revision
    info["expected_revision"] = expected_revision
    if expected_revision is not None and revision != expected_revision:
        errors.append(f"HF cached revision mismatch for {model_path}: {revision} != {expected_revision}")
    if revision is None:
        errors.append(f"could not resolve cached revision for HF model id {model_path}")
        return info, errors
    snapshot_path = snapshots_dir / revision
    info["snapshot_path"] = str(snapshot_path)
    if not snapshot_path.is_dir():
        errors.append(f"resolved HF snapshot path is missing: {snapshot_path}")
        return info, errors
    files = []
    for name in [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "model.safetensors",
    ]:
        path = snapshot_path / name
        if path.exists():
            files.append(file_record(path))
    info["snapshot_files"] = files
    if not files:
        errors.append(f"no expected model/config files found in {snapshot_path}")
    return info, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-prefix", default=str(project_root() / "envs" / "simplerenv_xvla_py310"))
    parser.add_argument("--model-path", default="2toINF/X-VLA-WidowX")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    errors: list[str] = []
    try:
        validate_task(args.task)
    except Exception as exc:
        errors.append(str(exc))
    root = project_root()
    x_vla_repo = root / "third_party" / "x_vla"
    xvla_simpler_env = root / "third_party" / "xvla_simpler_env"
    env_prefix = Path(args.env_prefix)
    if not (env_prefix / "bin" / "python").is_file():
        errors.append(f"missing env python: {env_prefix / 'bin' / 'python'}")
    for script in sorted((SUPPORT_ROOT / "launch").glob("*.sbatch")) if (SUPPORT_ROOT / "launch").is_dir() else []:
        if "--array" in script.read_text(errors="replace"):
            errors.append(f"Slurm arrays are forbidden; found --array in {script}")
    for path in [x_vla_repo / "deploy.py", xvla_simpler_env]:
        if not path.exists():
            errors.append(f"missing required path: {path}")
    for client in TASK_CLIENT.values():
        path = x_vla_repo / "evaluation" / "simpler" / "WidowX" / client
        if not path.is_file():
            errors.append(f"missing X-VLA client: {path}")
    git_records: dict[str, Any] = {}
    try:
        git_records["eval_repo"] = git_record(root)
        git_records["x_vla"] = git_record(x_vla_repo, EXPECTED_X_VLA_COMMIT)
        git_records["xvla_simpler_env"] = git_record(xvla_simpler_env, EXPECTED_XVLA_SIMPLER_ENV_COMMIT)
        if not git_records["x_vla"]["head_matches_expected"]:
            errors.append(f"x_vla HEAD mismatch: {git_records['x_vla']['head']} != {EXPECTED_X_VLA_COMMIT}")
        if git_records["x_vla"]["status_short"]:
            errors.append(f"x_vla checkout is dirty: {git_records['x_vla']['status_short']}")
        if not git_records["xvla_simpler_env"]["head_matches_expected"]:
            errors.append(
                "xvla_simpler_env HEAD mismatch: "
                f"{git_records['xvla_simpler_env']['head']} != {EXPECTED_XVLA_SIMPLER_ENV_COMMIT}"
            )
        if git_records["xvla_simpler_env"]["status_short"]:
            errors.append(f"xvla_simpler_env checkout is dirty: {git_records['xvla_simpler_env']['status_short']}")
    except Exception as exc:
        errors.append(f"git evidence collection failed: {exc}")
    hf_home = Path(os.environ.get("HF_HOME", str(root / "cache" / "huggingface")))
    hf_hub_cache = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))
    expected_revision = EXPECTED_HF_REVISION if args.model_path == "2toINF/X-VLA-WidowX" else None
    if expected_revision is None and "/" in args.model_path and not Path(args.model_path).exists():
        errors.append(f"no expected HF revision pinned for mutable model id {args.model_path}")
    model_identity, model_errors = hf_cache_identity(args.model_path, hf_hub_cache, expected_revision)
    errors.extend(model_errors)
    report = {
        "policy": POLICY,
        "policy_display": POLICY_DISPLAY,
        "task": args.task,
        "horizon": horizon_for(args.task),
        "server_start_seed": server_start_seed(args.task),
        "support_root": str(SUPPORT_ROOT),
        "env_prefix": str(env_prefix),
        "model_identity": model_identity,
        "git_records": git_records,
        "preflight_status": "passed" if not errors else "failed",
        "errors": errors,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
