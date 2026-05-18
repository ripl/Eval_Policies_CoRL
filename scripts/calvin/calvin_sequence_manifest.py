#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROTOCOL = "calvin_official_sequence_manifest_v1"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_json(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def hash_rows(rows: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_calvin_get_sequences(calvin_root: Path):
    for rel in ("calvin_models", "calvin_env"):
        path = calvin_root / rel
        if not path.exists():
            raise FileNotFoundError(f"missing CALVIN path: {path}")
        sys.path.insert(0, str(path))
    from calvin_agent.evaluation.multistep_sequences import get_sequences

    return get_sequences


def sequence_rows(eval_sequences: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for index, (initial_state, eval_sequence) in enumerate(eval_sequences):
        rows.append({
            "index": index,
            "initial_state": jsonable(initial_state),
            "eval_sequence": jsonable(eval_sequence),
            "initial_state_json": stable_json(initial_state),
            "eval_sequence_json": stable_json(eval_sequence),
        })
    return rows


def build_manifest(calvin_root: Path, num_sequences: int, sequence_workers: int) -> dict[str, Any]:
    get_sequences = load_calvin_get_sequences(calvin_root)
    eval_sequences = list(get_sequences(int(num_sequences), num_workers=int(sequence_workers)))
    if len(eval_sequences) != int(num_sequences):
        raise RuntimeError(f"get_sequences returned {len(eval_sequences)}, expected {num_sequences}")
    rows = sequence_rows(eval_sequences)
    initial_rows = [row["initial_state_json"] for row in rows]
    eval_rows = [row["eval_sequence_json"] for row in rows]
    return {
        "protocol": PROTOCOL,
        "calvin_root": str(calvin_root.resolve()),
        "num_sequences": int(num_sequences),
        "sequence_workers": int(sequence_workers),
        "initial_state_sha256": hash_rows(initial_rows),
        "eval_sequence_sha256": hash_rows(eval_rows),
        "sequences": rows,
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing CALVIN sequence manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError(f"manifest protocol mismatch in {manifest_path}: {manifest.get('protocol')!r}")
    num_sequences = int(manifest.get("num_sequences", -1))
    rows = manifest.get("sequences")
    if not isinstance(rows, list) or len(rows) != num_sequences:
        raise RuntimeError(f"manifest sequence count mismatch in {manifest_path}: rows={len(rows) if isinstance(rows, list) else type(rows)}, num_sequences={num_sequences}")
    return manifest


def validate_sequences_against_manifest(eval_sequences: list[Any], manifest: dict[str, Any], context: str) -> None:
    rows = sequence_rows(eval_sequences)
    expected_n = int(manifest["num_sequences"])
    if len(rows) != expected_n:
        raise RuntimeError(f"{context}: sequence count mismatch: got {len(rows)}, expected {expected_n}")
    initial_hash = hash_rows([row["initial_state_json"] for row in rows])
    eval_hash = hash_rows([row["eval_sequence_json"] for row in rows])
    if initial_hash != manifest["initial_state_sha256"]:
        raise RuntimeError(f"{context}: initial-state sequence hash mismatch: got {initial_hash}, expected {manifest['initial_state_sha256']}")
    if eval_hash != manifest["eval_sequence_sha256"]:
        raise RuntimeError(f"{context}: eval-sequence hash mismatch: got {eval_hash}, expected {manifest['eval_sequence_sha256']}")


def validate_bank_metadata_against_manifest(metadata: dict[str, Any], manifest: dict[str, Any], context: str) -> None:
    for key in ("num_sequences", "sequence_workers", "initial_state_sha256", "eval_sequence_sha256"):
        actual = metadata.get(key)
        expected = manifest.get(key)
        if str(actual) != str(expected):
            raise RuntimeError(f"{context}: reset-bank/manifest mismatch for {key}: got {actual!r}, expected {expected!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a fail-loud CALVIN 1000-sequence manifest.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--sequence-workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_sequences <= 0:
        raise ValueError(f"--num-sequences must be positive, got {args.num_sequences}")
    if args.sequence_workers <= 0:
        raise ValueError(f"--sequence-workers must be positive, got {args.sequence_workers}")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(Path(args.calvin_root).resolve(), args.num_sequences, args.sequence_workers)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(output)
    print(json.dumps({
        "manifest": str(output),
        "num_sequences": manifest["num_sequences"],
        "sequence_workers": manifest["sequence_workers"],
        "initial_state_sha256": manifest["initial_state_sha256"],
        "eval_sequence_sha256": manifest["eval_sequence_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
