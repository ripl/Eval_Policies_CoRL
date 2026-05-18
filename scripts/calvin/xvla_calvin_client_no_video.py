#!/usr/bin/env python3
import argparse
import importlib.util
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


def reset_bank_context():
    path = os.environ.get("CALVIN_RESET_BANK")
    if not path:
        return None, None
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"].item()))
    return data, metadata


def sequence_manifest_context():
    path = os.environ.get("CALVIN_SEQUENCE_MANIFEST")
    return load_manifest(path) if path else None


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-client", required=True)
    known, rest = parser.parse_known_args()

    source = Path(known.source_client).resolve()
    spec = importlib.util.spec_from_file_location("xvla_calvin_client_upstream", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def skip_video(path, frames, fps=30):
        return None

    upstream_conf_dir = source.parent / "ABC_D" / "validation"
    original_load = module.OmegaConf.load

    def load_with_upstream_yaml(path):
        p = Path(path)
        if p.name in {"new_playtable_tasks.yaml", "new_playtable_validation.yaml"}:
            return original_load(upstream_conf_dir / p.name)
        return original_load(path)

    module.save_video = skip_video
    module.OmegaConf.load = load_with_upstream_yaml
    reset_bank, reset_metadata = reset_bank_context()
    manifest = sequence_manifest_context()
    sequence_count = int(manifest["num_sequences"]) if manifest is not None else None
    workers = int(manifest.get("sequence_workers", 4)) if manifest is not None else 4
    if reset_metadata is not None:
        if manifest is not None:
            validate_bank_metadata_against_manifest(reset_metadata, manifest, "X-VLA")
        sequence_count = int(reset_metadata["num_sequences"])
        workers = int(reset_metadata.get("sequence_workers", 4))
    if sequence_count is not None:
        eval_end = int(rest[rest.index("--eval_end") + 1]) if "--eval_end" in rest else sequence_count
        eval_start = int(rest[rest.index("--eval_start") + 1]) if "--eval_start" in rest else 0
        if eval_start < 0 or eval_start > eval_end:
            raise RuntimeError(f"invalid eval range: eval_start={eval_start}, eval_end={eval_end}")
        if eval_end > sequence_count:
            raise RuntimeError(f"eval_end={eval_end} exceeds canonical sequence count {sequence_count}")
        original_get_sequences = module.get_sequences
        canonical_sequences = list(original_get_sequences(sequence_count, num_workers=workers))
        if manifest is not None:
            validate_sequences_against_manifest(canonical_sequences, manifest, "X-VLA")
        if reset_bank is not None:
            expected = str(reset_bank["initial_state_json"][0])
            actual = stable_json(canonical_sequences[0][0])
            if actual != expected:
                raise RuntimeError(f"canonical X-VLA sequence mismatch: actual={actual}, expected={expected}")

        def get_sequences_with_bank_order(num_sequences=sequence_count, num_workers=None):
            if int(num_sequences) == sequence_count:
                return canonical_sequences
            return original_get_sequences(num_sequences, num_workers=workers if num_workers is None else num_workers)

        module.NUM_SEQUENCES = sequence_count
        module.get_sequences = get_sequences_with_bank_order
    ep_len = int(os.environ.get("XVLA_EP_LEN", os.environ.get("EP_LEN", str(module.EP_LEN))))
    module.EP_LEN = ep_len
    required_ep_len = os.environ.get("REQUIRE_XVLA_EP_LEN")
    if required_ep_len is not None and int(required_ep_len) != int(module.EP_LEN):
        raise RuntimeError(f"X-VLA EP_LEN mismatch: got {module.EP_LEN}, expected {required_ep_len}")
    original_evaluate_policy = module.evaluate_policy

    def evaluate_policy_with_rows(model, env, output_dir, debug=False, eval_start=0, eval_end=module.NUM_SEQUENCES):
        results = original_evaluate_policy(model, env, output_dir, debug=debug, eval_start=eval_start, eval_end=eval_end)
        eval_sequences = list(module.get_sequences(module.NUM_SEQUENCES))
        rows = []
        for global_idx, result in zip(range(int(eval_start), int(eval_end)), results):
            initial_state, eval_sequence = eval_sequences[global_idx]
            rows.append({
                "global_index": int(global_idx),
                "success": int(result),
                "initial_state_json": stable_json(initial_state),
                "eval_sequence": eval_sequence,
            })
        Path(output_dir, "per_sequence_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        return results

    module.evaluate_policy = evaluate_policy_with_rows
    patch_module_from_env(module)
    sys.argv = [str(source), *rest]
    module.main()


if __name__ == "__main__":
    main()
