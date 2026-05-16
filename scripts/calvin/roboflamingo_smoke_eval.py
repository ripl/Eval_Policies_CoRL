#!/usr/bin/env python3
import argparse
import builtins
import json
import os
from pathlib import Path
import sys
import types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--calvin-conf-path", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--openflamingo-checkpoint", required=True)
    parser.add_argument("--num-sequences", type=int, default=50)
    parser.add_argument("--llm-name", default="mpt_dolly_3b")
    parser.add_argument("--lang-encoder", default="anas-awadalla/mpt-1b-redpajama-200b-dolly")
    parser.add_argument("--tokenizer", default="anas-awadalla/mpt-1b-redpajama-200b-dolly")
    parser.add_argument("--precision", default="fp32")
    args = parser.parse_args()

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
    if "pyhash" not in sys.modules:
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
    eval_utils.NUM_SEQUENCES = args.num_sequences

    def fixed_log_dir(log_dir):
        eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"logging to {eval_dir}")
        return eval_dir

    eval_utils.get_log_dir = fixed_log_dir

    original_open = builtins.open
    local_sequences = repo / "eval_sequences.json"

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
        (eval_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
