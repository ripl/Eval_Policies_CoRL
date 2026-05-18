#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calvin_sequence_manifest import load_manifest

POLICY_DIRS = {
    "xvla": "xvla_abc_d",
    "gr1": "gr1_abc_d",
    "roboflamingo": "roboflamingo_abc_d",
}


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise RuntimeError(f"{path} did not contain a list")
        for row in payload:
            row = dict(row)
            row["_source"] = str(path)
            rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]], manifest: dict[str, Any], num_sequences: int, context: str) -> None:
    if len(rows) != num_sequences:
        raise RuntimeError(f"{context}: got {len(rows)} rows, expected {num_sequences}")
    seen = {}
    for row in rows:
        idx = int(row["global_index"])
        if idx in seen:
            raise RuntimeError(f"{context}: duplicate global_index={idx}: {seen[idx]} and {row['_source']}")
        seen[idx] = row["_source"]
        if idx < 0 or idx >= num_sequences:
            raise RuntimeError(f"{context}: global_index={idx} outside 0..{num_sequences - 1}")
        expected = manifest["sequences"][idx]["initial_state_json"]
        if row.get("initial_state_json") != expected:
            raise RuntimeError(f"{context}: initial-state mismatch at {idx}: source={row['_source']}")
    missing = [idx for idx in range(num_sequences) if idx not in seen]
    if missing:
        preview = missing[:10]
        raise RuntimeError(f"{context}: missing {len(missing)} sequence rows, first missing={preview}")


def summarize(rows: list[dict[str, Any]], policy: str, condition: str, num_sequences: int, manifest_path: Path, array_job_id: str) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row["global_index"]))
    successes = [int(row["success"]) for row in rows]
    chain_sr = {str(k): sum(s >= k for s in successes) / num_sequences for k in range(1, 6)}
    task_info: dict[str, dict[str, int]] = {}
    for row in rows:
        success_count = int(row["success"])
        sequence = list(row["eval_sequence"])
        attempts = min(len(sequence), success_count + (0 if success_count >= len(sequence) else 1))
        for task_idx in range(attempts):
            task = str(sequence[task_idx])
            stats = task_info.setdefault(task, {"success": 0, "total": 0})
            stats["total"] += 1
            if task_idx < success_count:
                stats["success"] += 1
    return {
        "policy": policy,
        "condition": condition,
        "array_job_id": str(array_job_id),
        "num_sequences": int(num_sequences),
        "calvin_sequence_manifest": str(manifest_path),
        "avg_seq_len": sum(successes) / num_sequences,
        "chain_sr": chain_sr,
        "task_info": task_info,
        "per_sequence_sources": sorted({row["_source"] for row in rows}),
    }


def discover(results_root: Path, policy: str, condition_tag: str, num_sequences: int, array_job_id: str) -> list[Path]:
    policy_dir = results_root / POLICY_DIRS[policy]
    pattern = f"{policy}_abc_d_{condition_tag}_{num_sequences}seq_chunk*_*_*_{array_job_id}_*/per_sequence_results.json"
    client_pattern = f"{policy}_abc_d_{condition_tag}_{num_sequences}seq_chunk*_*_*_{array_job_id}_*/client/per_sequence_results.json"
    return sorted([*policy_dir.glob(pattern), *policy_dir.glob(client_pattern)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CALVIN chunked 1000-sequence rollouts.")
    parser.add_argument("--project-root", default="/share/data/ripl/tianchong/projects/Eval_Policies_CoRL")
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--reset-seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    results_root = project_root / "results" / "calvin"
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    if int(manifest["num_sequences"]) != int(args.num_sequences):
        raise RuntimeError(f"manifest has {manifest['num_sequences']} sequences, expected {args.num_sequences}")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root / "aggregates" / f"calvin_full_1000_{args.array_job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_tags = {
        "calibration": "calibration",
        "protocol1": f"official_d_table_resets_seed{args.reset_seed}",
    }
    combined: dict[str, Any] = {
        "array_job_id": str(args.array_job_id),
        "num_sequences": int(args.num_sequences),
        "calvin_sequence_manifest": str(manifest_path),
        "summaries": {},
        "opensource_artifacts": {
            "sequence_manifest": str(manifest_path),
            "reset_bank": str(results_root / "reset_banks" / f"abc_d_official_d_table_resets_{args.num_sequences}seq_seed{args.reset_seed}.npz"),
            "aggregate_dir": str(output_dir),
            "slurm_logs": str(project_root / "logs" / "slurm"),
            "source_files": [
                "scripts/calvin/calvin_sequence_manifest.py",
                "scripts/calvin/calvin_reset_bank.py",
                "scripts/calvin/calvin_reset_override.py",
                "scripts/calvin/calvin_aggregate_chunks.py",
                "scripts/calvin/xvla_calvin_client_no_video.py",
                "scripts/calvin/gr1_smoke_eval.py",
                "scripts/calvin/roboflamingo_smoke_eval.py",
                "scripts/calvin/run_xvla_calvin_smoke.sh",
                "scripts/calvin/run_gr1_calvin_smoke.sh",
                "scripts/calvin/run_roboflamingo_calvin_smoke.sh",
                "scripts/slurm/calvin_sequence_manifest.sbatch",
                "scripts/slurm/calvin_train_pose_reset_bank.sbatch",
                "scripts/slurm/calvin_1seq_validation_array.sbatch",
                "scripts/slurm/calvin_full_1000_array.sbatch",
            ],
        },
    }

    for condition, condition_tag in condition_tags.items():
        for policy in POLICY_DIRS:
            paths = discover(results_root, policy, condition_tag, args.num_sequences, str(args.array_job_id))
            context = f"{policy}/{condition}"
            if not paths:
                raise RuntimeError(f"{context}: no per_sequence_results.json files found")
            rows = load_rows(paths)
            validate_rows(rows, manifest, args.num_sequences, context)
            summary = summarize(rows, policy, condition, args.num_sequences, manifest_path, str(args.array_job_id))
            summary_path = output_dir / f"{policy}_{condition}_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            combined["summaries"][f"{policy}_{condition}"] = {
                "summary": str(summary_path),
                "avg_seq_len": summary["avg_seq_len"],
                "chain_sr": summary["chain_sr"],
            }

    combined_path = output_dir / "combined_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"combined_summary": str(combined_path), "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
