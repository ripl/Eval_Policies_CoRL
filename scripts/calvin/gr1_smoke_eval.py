#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

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
    parser.add_argument("--gr1-repo", required=True)
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--policy-ckpt", required=True)
    parser.add_argument("--mae-ckpt", required=True)
    parser.add_argument("--num-sequences", type=int, default=50)
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--eval-end", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.eval_end is None:
        args.eval_end = args.num_sequences
    if args.eval_start < 0 or args.eval_start > args.eval_end:
        raise ValueError(f"invalid eval range: eval_start={args.eval_start}, eval_end={args.eval_end}")

    gr1_repo = Path(args.gr1_repo).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CALVIN_ROOT"] = str(Path(args.calvin_root).resolve())
    sys.path.insert(0, str(gr1_repo))
    os.chdir(gr1_repo)

    import transformers.file_utils as file_utils

    def no_op_docstring_decorator(*_args, **_kwargs):
        def decorate(fn):
            return fn
        return decorate

    file_utils.add_code_sample_docstrings = no_op_docstring_decorator

    from transformers.models.gpt2.configuration_gpt2 import GPT2Config

    if not hasattr(GPT2Config, "n_ctx"):
        GPT2Config.n_ctx = property(lambda self: self.n_positions)

    import evaluate_calvin as ec

    manifest = load_manifest(os.environ["CALVIN_SEQUENCE_MANIFEST"]) if os.environ.get("CALVIN_SEQUENCE_MANIFEST") else None
    reset_bank, reset_metadata = reset_bank_metadata()
    sequence_count = int(manifest["num_sequences"]) if manifest is not None else int(args.num_sequences)
    workers = int(manifest.get("sequence_workers", 4)) if manifest is not None else 4
    if reset_metadata is not None:
        if manifest is not None:
            validate_bank_metadata_against_manifest(reset_metadata, manifest, "GR-1")
        sequence_count = int(reset_metadata["num_sequences"])
        workers = int(reset_metadata.get("sequence_workers", 4))
    if args.eval_end > sequence_count:
        raise RuntimeError(f"eval_end={args.eval_end} exceeds canonical sequence count {sequence_count}")
    original_get_sequences = ec.get_sequences
    canonical_sequences = list(original_get_sequences(sequence_count, num_workers=workers))
    if manifest is not None:
        validate_sequences_against_manifest(canonical_sequences, manifest, "GR-1")
    if reset_bank is not None:
        actual = stable_json(canonical_sequences[args.eval_start][0])
        expected = str(reset_bank["initial_state_json"][args.eval_start])
        if actual != expected:
            raise RuntimeError(f"canonical GR-1 sequence mismatch at {args.eval_start}: actual={actual}, expected={expected}")
    selected_sequences = canonical_sequences[args.eval_start : args.eval_end]
    if len(selected_sequences) != args.eval_end - args.eval_start:
        raise RuntimeError(f"bad selected sequence range: got {len(selected_sequences)}")

    def get_sequences_with_range(num_sequences=len(selected_sequences), num_workers=None):
        if int(num_sequences) == len(selected_sequences):
            return selected_sequences
        if int(num_sequences) == sequence_count:
            return canonical_sequences
        return original_get_sequences(num_sequences, num_workers=workers if num_workers is None else num_workers)

    ec.NUM_SEQUENCES = len(selected_sequences)
    ec.get_sequences = get_sequences_with_range
    original_evaluate_policy = ec.evaluate_policy

    def evaluate_policy_with_rows(*eval_args, **eval_kwargs):
        results = original_evaluate_policy(*eval_args, **eval_kwargs)
        eval_output_dir = eval_kwargs.get("eval_dir")
        if eval_output_dir is None and len(eval_args) >= 5:
            eval_output_dir = eval_args[4]
        if eval_output_dir is None:
            eval_output_dir = eval_dir
        rows = []
        for offset, (result, (initial_state, eval_sequence)) in enumerate(zip(results, selected_sequences)):
            rows.append({
                "global_index": int(args.eval_start + offset),
                "success": int(result),
                "initial_state_json": stable_json(initial_state),
                "eval_sequence": eval_sequence,
            })
        Path(eval_output_dir, "per_sequence_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        return results

    ec.evaluate_policy = evaluate_policy_with_rows
    patch_module_from_env(ec, eval_start=args.eval_start)
    sys.argv = [
        "evaluate_calvin.py",
        "--eval_dir",
        str(eval_dir),
        "--mae_ckpt_path",
        str(Path(args.mae_ckpt).resolve()),
        "--policy_ckpt_path",
        str(Path(args.policy_ckpt).resolve()),
        "--configs_path",
        str(gr1_repo / "logs" / "configs.json"),
        "--dataset_dir",
        str(Path(args.dataset_dir).resolve()),
        "--device",
        str(args.device),
    ]
    ec.main()

    result_path = eval_dir / "result.txt"
    summary_path = eval_dir / "summary.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            payload = data.get("null", data.get("None", data))
            if isinstance(payload, dict):
                payload.update({
                    "eval_start": int(args.eval_start),
                    "eval_end": int(args.eval_end),
                    "num_sequences": int(args.eval_end - args.eval_start),
                    "calvin_sequence_manifest": os.environ.get("CALVIN_SEQUENCE_MANIFEST", ""),
                    "calvin_reset_bank": os.environ.get("CALVIN_RESET_BANK", ""),
                })
            summary_path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
        except Exception as exc:
            summary_path.write_text(json.dumps({"parse_error": str(exc), "result_path": str(result_path)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
