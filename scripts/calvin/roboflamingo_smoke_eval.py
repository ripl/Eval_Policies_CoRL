#!/usr/bin/env python3
import argparse
import builtins
import json
import os
from pathlib import Path
import sys
import types

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calvin_reset_override import patch_module_from_env
from calvin_sequence_manifest import (
    load_manifest,
    validate_bank_metadata_against_manifest,
    validate_sequences_against_manifest,
)


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def reset_bank_metadata():
    path = os.environ.get("CALVIN_RESET_BANK")
    if not path:
        return None, None
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"].item()))
    return data, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--calvin-conf-path", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--openflamingo-checkpoint", required=True)
    parser.add_argument("--num-sequences", type=int, default=50)
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--eval-end", type=int, default=None)
    parser.add_argument("--llm-name", default="mpt_dolly_3b")
    parser.add_argument("--lang-encoder", default="anas-awadalla/mpt-1b-redpajama-200b-dolly")
    parser.add_argument("--tokenizer", default="anas-awadalla/mpt-1b-redpajama-200b-dolly")
    parser.add_argument("--precision", default="fp32")
    args = parser.parse_args()
    if args.eval_end is None:
        args.eval_end = args.num_sequences
    if args.eval_start < 0 or args.eval_start > args.eval_end:
        raise ValueError(f"invalid eval range: eval_start={args.eval_start}, eval_end={args.eval_end}")

    repo = Path(args.repo).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "open_flamingo"))
    sys.path.insert(0, str(repo / "robot_flamingo" / "eval"))
    os.chdir(repo)

    if "zarr" not in sys.modules:
        zarr_stub = types.ModuleType("zarr")
        zarr_stub.Array = type("Array", (), {})
        sys.modules["zarr"] = zarr_stub
    try:
        import pyhash  # noqa: F401
    except ImportError:
        pyhash_stub = types.ModuleType("pyhash")

        def fnv1_32():
            def hash_fn(value):
                data = str(value).encode("utf-8")
                hashed = 2166136261
                for byte in data:
                    hashed = (hashed * 16777619) & 0xFFFFFFFF
                    hashed ^= byte
                return hashed
            return hash_fn

        pyhash_stub.fnv1_32 = fnv1_32
        sys.modules["pyhash"] = pyhash_stub

    if "robot_flamingo.data.real_dataset_hdf5" not in sys.modules:
        real_dataset_stub = types.ModuleType("robot_flamingo.data.real_dataset_hdf5")
        real_dataset_stub.RealDatasetHDF5 = type("RealDatasetHDF5", (), {})
        sys.modules["robot_flamingo.data.real_dataset_hdf5"] = real_dataset_stub
    if "robot_flamingo.utils" not in sys.modules:
        utils_stub = types.ModuleType("robot_flamingo.utils")
        utils_stub.world_to_tcp_frame = lambda action, _robot_obs: action
        utils_stub.tcp_to_world_frame = lambda action, _robot_obs: action
        sys.modules["robot_flamingo.utils"] = utils_stub

    from robot_flamingo.models import factory

    factory.mpt_dict[args.llm_name] = {
        "lang_encoder_path": args.lang_encoder,
        "tokenizer_path": args.tokenizer,
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": str(Path(args.openflamingo_checkpoint).resolve()),
    }

    os.environ["PYOPENGL_PLATFORM"] = "egl"
    original_import = builtins.__import__

    def import_with_egl(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyrender" or name.startswith("pyrender.") or name.startswith("OpenGL"):
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = import_with_egl
    try:
        import eval_utils

        import robot_flamingo.eval.eval_calvin as ec
    finally:
        builtins.__import__ = original_import

    os.environ["PYOPENGL_PLATFORM"] = "egl"
    manifest = load_manifest(os.environ["CALVIN_SEQUENCE_MANIFEST"]) if os.environ.get("CALVIN_SEQUENCE_MANIFEST") else None
    reset_bank, metadata = reset_bank_metadata()
    sequence_count = int(manifest["num_sequences"]) if manifest is not None else int(args.num_sequences)
    workers = int(manifest.get("sequence_workers", 4)) if manifest is not None else 4
    if metadata is not None:
        if manifest is not None:
            validate_bank_metadata_against_manifest(metadata, manifest, "RoboFlamingo")
        sequence_count = int(metadata["num_sequences"])
        workers = int(metadata.get("sequence_workers", 4))
    if args.eval_end > sequence_count:
        raise RuntimeError(f"eval_end={args.eval_end} exceeds canonical sequence count {sequence_count}")
    eval_sequences = eval_utils.get_sequences(sequence_count, num_workers=workers)
    if manifest is not None:
        validate_sequences_against_manifest(eval_sequences, manifest, "RoboFlamingo")
    if reset_bank is not None:
        expected = str(reset_bank["initial_state_json"][args.eval_start])
        actual = stable_json(eval_sequences[args.eval_start][0])
        if actual != expected:
            raise RuntimeError(f"canonical RoboFlamingo sequence mismatch at {args.eval_start}: actual={actual}, expected={expected}")
    selected_sequences = eval_sequences[args.eval_start : args.eval_end]
    if len(selected_sequences) != args.eval_end - args.eval_start:
        raise RuntimeError(f"bad selected sequence range: got {len(selected_sequences)}")
    eval_utils.NUM_SEQUENCES = len(selected_sequences)
    patch_module_from_env(eval_utils, eval_start=args.eval_start)

    def fixed_log_dir(log_dir):
        eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"logging to {eval_dir}")
        return eval_dir

    eval_utils.get_log_dir = fixed_log_dir

    local_sequences = eval_dir / "eval_sequences_protocol1.json"
    local_sequences.write_text(json.dumps(selected_sequences) + "\n")

    original_print_and_save = eval_utils.print_and_save

    def print_and_save_with_rows(results, eval_sequences_for_save, eval_log_dir, epoch):
        rows = []
        for offset, (result, (initial_state, eval_sequence)) in enumerate(zip(results, eval_sequences_for_save)):
            rows.append({
                "global_index": int(args.eval_start + offset),
                "success": int(result),
                "initial_state_json": stable_json(initial_state),
                "eval_sequence": eval_sequence,
            })
        Path(eval_log_dir, "per_sequence_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        return original_print_and_save(results, eval_sequences_for_save, eval_log_dir, epoch)

    eval_utils.print_and_save = print_and_save_with_rows

    original_open = builtins.open

    def redirected_open(file, *open_args, **open_kwargs):
        if str(file) == "/mnt/bn/robotics/lxh/robot-flamingo/eval_sequences.json":
            return original_open(local_sequences, *open_args, **open_kwargs)
        return original_open(file, *open_args, **open_kwargs)

    builtins.open = redirected_open

    sys.argv = [
        "eval_calvin.py",
        "--precision",
        args.precision,
        "--use_gripper",
        "--window_size",
        "12",
        "--fusion_mode",
        "post",
        "--run_name",
        "RoboFlamingoCALVINSmoke",
        "--calvin_dataset",
        str(Path(args.dataset_dir).resolve()),
        "--evaluate_from_checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--calvin_conf_path",
        str(Path(args.calvin_conf_path).resolve()),
        "--workers",
        "1",
        "--llm_name",
        args.llm_name,
    ]
    ec.main()

    results = eval_dir / "results.json"
    if results.exists():
        data = json.loads(results.read_text())
        payload = data.get("0", data.get("null", data))
        if isinstance(payload, dict):
            payload.update({
                "eval_start": int(args.eval_start),
                "eval_end": int(args.eval_end),
                "num_sequences": int(args.eval_end - args.eval_start),
                "calvin_sequence_manifest": os.environ.get("CALVIN_SEQUENCE_MANIFEST", ""),
                "calvin_reset_bank": os.environ.get("CALVIN_RESET_BANK", ""),
            })
        (eval_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
