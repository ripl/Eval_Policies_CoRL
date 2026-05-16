#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gr1-repo", required=True)
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--policy-ckpt", required=True)
    parser.add_argument("--mae-ckpt", required=True)
    parser.add_argument("--num-sequences", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

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

    ec.NUM_SEQUENCES = args.num_sequences
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
            summary_path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
        except Exception as exc:
            summary_path.write_text(json.dumps({"parse_error": str(exc), "result_path": str(result_path)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
