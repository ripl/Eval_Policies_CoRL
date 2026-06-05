#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL = "calvin_official_sequence_manifest_v1"
POLICY_DIRS = {
    "xvla": "xvla",
    "gr1": "gr1",
    "roboflamingo": "roboflamingo",
}
CONDITION_SPECS = {
    "layer2_fresh_seed2026052601": {
        "sha256": "f2a339b2ec42320cac1b001aa6763d477015f26f55089528914f9ecb5a137e93",
        "seed": 2026052601,
        "filename": "abc_d_layer2_fresh_1000seq_seed2026052601.json",
    },
    "layer2_fresh_seed2026052602": {
        "sha256": "bf2b9a19c73b8a54c5c2aa5f85ef26210df508c41e9a37385da16b5da1f73ee6",
        "seed": 2026052602,
        "filename": "abc_d_layer2_fresh_1000seq_seed2026052602.json",
    },
}
SAMPLE_KIND = "layer2_fresh_stratified"
NUM_SEQUENCES = 1000
NUM_CHUNKS = 4
CHUNK_SIZE = 250
XVLA_EP_LEN = 360
CURATED_PACKAGE_REL = "paper_artifacts/calvin_creeping_overfitting/layer2_fresh_1000_20260526"
MANIFEST_LIST_REL = (
    f"{CURATED_PACKAGE_REL}/layer2_fresh_manifest_list.json"
)
BASELINE_REL = "results/calvin/aggregates/calvin_full_1000_2048456/combined_summary.json"
BASELINE_SHA256 = "40961c86c07b49a7dac80ac1003db3f7c048544abc6d1acbf6a1c0efdf13b798"
SHA_FIELDS = ("sha256", "manifest_sha256", "sequence_manifest_sha256")
TAG_FIELDS = ("condition_tag", "condition", "tag", "name")
FORBIDDEN_METADATA_KEYS = {
    "calvin_reset_bank",
    "reset_bank",
    "calvin_reset_protocol",
    "CALVIN_RESET_BANK",
    "RESET_BANK",
    "CALVIN_RESET_PROTOCOL",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def truthy_metadata(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null"}
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return bool(value)


def iter_manifest_entries(payload: Any):
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield dict(item)
        return
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest list must be a dict or list, got {type(payload).__name__}")
    for key in ("manifests", "manifest_list", "layer2_fresh_manifests"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield dict(item)
            return
        if isinstance(value, dict):
            for tag, item in value.items():
                row = dict(item) if isinstance(item, dict) else {"manifest_path": item}
                row.setdefault("condition_tag", tag)
                yield row
            return
    for tag, item in payload.items():
        if str(tag).startswith("layer2_fresh_seed"):
            row = dict(item) if isinstance(item, dict) else {"manifest_path": item}
            row.setdefault("condition_tag", tag)
            yield row


def first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def require_under(path: Path, parent: Path, context: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(f"{context}: selected manifest is outside curated package: {path}") from exc


def load_manifest_paths(manifest_list_path: Path, project_root: Path) -> dict[str, Path]:
    if not manifest_list_path.is_file():
        raise RuntimeError(
            f"Missing Worker A Layer 2 manifest list: {manifest_list_path}. "
            "Refusing to fall back to ignored results/calvin/sequence_manifests files."
        )
    payload = json.loads(manifest_list_path.read_text())
    entries = list(iter_manifest_entries(payload))
    curated_dir = (project_root / CURATED_PACKAGE_REL).resolve()
    resolved: dict[str, Path] = {}
    for condition_tag, spec in CONDITION_SPECS.items():
        selected = None
        for row in entries:
            tag = first_present(row, TAG_FIELDS)
            seed = row.get("sample_seed", row.get("seed"))
            if tag == condition_tag or str(seed) == str(spec["seed"]):
                selected = row
                break
        if selected is None:
            raise RuntimeError(f"{manifest_list_path} does not contain required condition tag {condition_tag}")
        manifest_path_raw = selected.get("curated_repo_relative_path")
        if not manifest_path_raw:
            raise RuntimeError(
                f"{condition_tag}: manifest-list entry lacks curated_repo_relative_path; "
                "source_path/source_absolute_path are provenance-only"
            )
        listed_sha = first_present(selected, SHA_FIELDS)
        if not listed_sha:
            raise RuntimeError(f"{condition_tag}: manifest-list entry lacks sha256")
        expected_sha = str(spec["sha256"])
        if str(listed_sha) != expected_sha:
            raise RuntimeError(f"{condition_tag}: listed sha256 {listed_sha} != expected {expected_sha}")
        manifest_path_rel = Path(str(manifest_path_raw))
        if manifest_path_rel.is_absolute():
            raise RuntimeError(f"{condition_tag}: curated_repo_relative_path must be repo-relative, got {manifest_path_raw}")
        manifest_path = (project_root / manifest_path_rel).resolve()
        require_under(manifest_path, curated_dir, condition_tag)
        if manifest_path.name != spec["filename"]:
            raise RuntimeError(f"{condition_tag}: selected manifest filename {manifest_path.name} != expected {spec['filename']}")
        if not manifest_path.is_file():
            raise RuntimeError(f"{condition_tag}: manifest file does not exist: {manifest_path}")
        actual_sha = file_sha256(manifest_path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"{condition_tag}: actual manifest sha256 {actual_sha} != expected {expected_sha}: {manifest_path}")
        resolved[condition_tag] = manifest_path
    return resolved


def validate_manifest(condition_tag: str, manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    spec = CONDITION_SPECS[condition_tag]
    if int(manifest.get("num_sequences", -1)) != NUM_SEQUENCES:
        raise RuntimeError(f"{condition_tag}: manifest num_sequences is not {NUM_SEQUENCES}")
    if manifest.get("sample_kind") != SAMPLE_KIND:
        raise RuntimeError(f"{condition_tag}: manifest sample_kind {manifest.get('sample_kind')!r} is not {SAMPLE_KIND}")
    if int(manifest.get("generation", {}).get("sample_seed", -1)) != int(spec["seed"]):
        raise RuntimeError(f"{condition_tag}: manifest generation.sample_seed mismatch")
    if manifest.get("calvin_reset_bank_used") is not False:
        raise RuntimeError(f"{condition_tag}: manifest indicates reset-bank use")
    rows = manifest.get("sequences")
    if not isinstance(rows, list) or len(rows) != NUM_SEQUENCES:
        raise RuntimeError(f"{condition_tag}: manifest does not contain {NUM_SEQUENCES} rows")
    indices = sorted(int(row.get("global_index", row.get("index", -1))) for row in rows)
    if indices != list(range(NUM_SEQUENCES)):
        raise RuntimeError(f"{condition_tag}: manifest global_index values are not exactly 0..{NUM_SEQUENCES - 1}")
    return manifest


def parse_key_value_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def check_forbidden_metadata(mapping: dict[str, Any], context: str) -> None:
    for key, value in mapping.items():
        if key in FORBIDDEN_METADATA_KEYS and truthy_metadata(value):
            raise RuntimeError(f"{context}: forbidden reset metadata {key}={value!r}")
    nested = mapping.get("forbidden_reset_envs")
    if isinstance(nested, dict):
        for key, value in nested.items():
            if truthy_metadata(value):
                raise RuntimeError(f"{context}: forbidden reset env {key}={value!r}")


def load_json(path: Path, context: str) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"{context}: failed to read JSON {path}: {exc}") from exc


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path, f"manifest {path}")
    if not isinstance(manifest, dict):
        raise RuntimeError(f"manifest is not a JSON object: {path}")
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError(f"manifest protocol mismatch in {path}: {manifest.get('protocol')!r}")
    num_sequences = int(manifest.get("num_sequences", -1))
    rows = manifest.get("sequences")
    if not isinstance(rows, list) or len(rows) != num_sequences:
        raise RuntimeError(f"manifest sequence count mismatch in {path}")
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"manifest row is not an object in {path}")
        items = row.get("initial_state_items")
        if not isinstance(items, list) or not all(isinstance(item, list) and len(item) == 2 for item in items):
            raise RuntimeError(f"invalid ordered initial_state_items in {path} row {row.get('index')}")
        ordered_state = {str(key): value for key, value in items}
        expected_values_repr = row.get("initial_state_values_repr")
        if expected_values_repr is not None and expected_values_repr != str(ordered_state.values()):
            raise RuntimeError(f"initial-state value order drift in {path} row {row.get('index')}")
    return manifest


def result_dir_for_rows(path: Path) -> Path:
    if path.parent.name == "client":
        return path.parent.parent
    return path.parent


def discover_row_files(results_root: Path, condition_tag: str, policy: str, array_job_id: str) -> list[Path]:
    search_root = results_root / condition_tag / POLICY_DIRS[policy]
    if not search_root.exists():
        raise RuntimeError(f"{policy}/{condition_tag}: missing result root {search_root}")
    patterns = [
        f"*{condition_tag}*{array_job_id}*/per_sequence_results.json",
        f"*{condition_tag}*{array_job_id}*/client/per_sequence_results.json",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(search_root.glob(pattern))
    return sorted(set(path.resolve() for path in paths))


def validate_chunk_metadata(
    metadata: dict[str, Any],
    *,
    context: str,
    policy: str,
    condition_tag: str,
    array_job_id: str,
    manifest_path: Path,
) -> tuple[int, int, int]:
    check_forbidden_metadata(metadata, context)
    if metadata.get("run_kind") != "paper":
        raise RuntimeError(f"{context}: run_kind must be paper, got {metadata.get('run_kind')!r}")
    tag_text = " ".join(str(metadata.get(key, "")) for key in ("run_tag", "results_dir", "condition_tag")).lower()
    if "smoke" in tag_text or "debug" in tag_text:
        raise RuntimeError(f"{context}: paper aggregate refuses smoke/debug tag metadata: {tag_text}")
    if metadata.get("policy") != policy:
        raise RuntimeError(f"{context}: metadata policy {metadata.get('policy')!r} != {policy!r}")
    if metadata.get("condition_tag") != condition_tag:
        raise RuntimeError(f"{context}: metadata condition {metadata.get('condition_tag')!r} != {condition_tag!r}")
    if str(metadata.get("array_job_id")) != str(array_job_id):
        raise RuntimeError(f"{context}: metadata array_job_id {metadata.get('array_job_id')!r} != {array_job_id!r}")
    if int(metadata.get("num_sequences", -1)) != NUM_SEQUENCES:
        raise RuntimeError(f"{context}: metadata num_sequences mismatch")
    if int(metadata.get("chunk_count", -1)) != CHUNK_SIZE:
        raise RuntimeError(f"{context}: metadata chunk_count mismatch")
    if int(metadata.get("chunk_size", -1)) != CHUNK_SIZE:
        raise RuntimeError(f"{context}: metadata chunk_size mismatch")
    if policy == "xvla":
        if "xvla_ep_len" not in metadata:
            raise RuntimeError(f"{context}: X-VLA metadata lacks xvla_ep_len")
        if int(metadata["xvla_ep_len"]) != XVLA_EP_LEN:
            raise RuntimeError(f"{context}: X-VLA xvla_ep_len {metadata['xvla_ep_len']} != {XVLA_EP_LEN}")
    if metadata.get("calvin_sequence_sample_kind") != SAMPLE_KIND:
        raise RuntimeError(f"{context}: metadata sample kind mismatch")
    expected_sha = CONDITION_SPECS[condition_tag]["sha256"]
    if metadata.get("calvin_sequence_manifest_sha256") != expected_sha:
        raise RuntimeError(f"{context}: metadata manifest sha mismatch")
    metadata_manifest = Path(str(metadata.get("calvin_sequence_manifest", ""))).resolve()
    if metadata_manifest != manifest_path.resolve():
        raise RuntimeError(f"{context}: metadata manifest path {metadata_manifest} != {manifest_path.resolve()}")
    chunk_id = int(metadata.get("chunk_id", -1))
    start = int(metadata.get("global_index_start", metadata.get("eval_start", -1)))
    end = int(metadata.get("global_index_end_exclusive", metadata.get("eval_end", -1)))
    if chunk_id < 0 or chunk_id >= NUM_CHUNKS:
        raise RuntimeError(f"{context}: invalid chunk_id={chunk_id}")
    if start != chunk_id * CHUNK_SIZE or end != (chunk_id + 1) * CHUNK_SIZE:
        raise RuntimeError(f"{context}: chunk range {start}:{end} does not match chunk_id={chunk_id}")
    return chunk_id, start, end


def validate_row_identity(row: dict[str, Any], expected: dict[str, Any], context: str) -> None:
    if list(map(str, row.get("eval_sequence", []))) != list(map(str, expected.get("eval_sequence", []))):
        raise RuntimeError(f"{context}: eval_sequence mismatch")
    if "initial_state_json" in row and row["initial_state_json"] != expected.get("initial_state_json"):
        raise RuntimeError(f"{context}: initial_state_json mismatch")
    if "initial_state_ordered_json" in row:
        if row["initial_state_ordered_json"] != expected.get("initial_state_ordered_json"):
            raise RuntimeError(f"{context}: initial_state_ordered_json mismatch")
        return
    if "initial_state_items" in row:
        if row["initial_state_items"] != expected.get("initial_state_items"):
            raise RuntimeError(f"{context}: initial_state_items mismatch")
        return
    if "initial_state_values_repr" in row:
        if row["initial_state_values_repr"] != expected.get("initial_state_values_repr"):
            raise RuntimeError(f"{context}: initial_state_values_repr mismatch")
        return
    raise RuntimeError(f"{context}: row lacks ordered initial-state identity")


def load_and_validate_chunk(
    row_path: Path,
    *,
    policy: str,
    condition_tag: str,
    array_job_id: str,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[int, list[dict[str, Any]], Path]:
    result_dir = result_dir_for_rows(row_path)
    context = f"{policy}/{condition_tag}/{result_dir.name}"
    if "calibration" in str(row_path).lower() or "protocol1" in str(row_path).lower():
        raise RuntimeError(f"{context}: refusing calibration/protocol1 path {row_path}")
    metadata_path = result_dir / "layer2_chunk_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"{context}: missing required layer2_chunk_metadata.json")
    metadata = load_json(metadata_path, context)
    chunk_id, start, end = validate_chunk_metadata(
        metadata,
        context=context,
        policy=policy,
        condition_tag=condition_tag,
        array_job_id=array_job_id,
        manifest_path=manifest_path,
    )
    text_metadata = parse_key_value_file(result_dir / "metadata.txt")
    check_forbidden_metadata(text_metadata, f"{context}/metadata.txt")
    for summary_name in ("summary.json", "client/summary.json"):
        summary_path = result_dir / summary_name
        if summary_path.exists():
            summary = load_json(summary_path, f"{context}/{summary_name}")
            if isinstance(summary, dict):
                check_forbidden_metadata(summary, f"{context}/{summary_name}")
    rows = load_json(row_path, context)
    if not isinstance(rows, list):
        raise RuntimeError(f"{context}: {row_path} is not a row list")
    if len(rows) != CHUNK_SIZE:
        raise RuntimeError(f"{context}: got {len(rows)} rows, expected chunk size {CHUNK_SIZE}")
    seen: set[int] = set()
    manifest_rows = manifest["sequences"]
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"{context}: row is not a dict")
        check_forbidden_metadata(row, f"{context}/row")
        idx = int(row["global_index"])
        if idx in seen:
            raise RuntimeError(f"{context}: duplicate chunk global_index={idx}")
        if idx < start or idx >= end:
            raise RuntimeError(f"{context}: global_index={idx} outside chunk range {start}:{end}")
        success = int(row["success"])
        if success < 0 or success > 5:
            raise RuntimeError(f"{context}: invalid success count {success} at global_index={idx}")
        validate_row_identity(row, manifest_rows[idx], f"{context}/global_index={idx}")
        copied = dict(row)
        copied["_source"] = str(row_path)
        copied["_result_dir"] = str(result_dir)
        seen.add(idx)
        clean_rows.append(copied)
    missing = [idx for idx in range(start, end) if idx not in seen]
    if missing:
        raise RuntimeError(f"{context}: missing indices inside chunk, first={missing[:10]}")
    return chunk_id, clean_rows, result_dir


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    condition_tag: str,
    array_job_id: str,
    manifest_path: Path,
) -> dict[str, Any]:
    if len(rows) != NUM_SEQUENCES:
        raise RuntimeError(f"{policy}/{condition_tag}: got {len(rows)} combined rows, expected {NUM_SEQUENCES}")
    by_idx: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = int(row["global_index"])
        if idx in by_idx:
            raise RuntimeError(f"{policy}/{condition_tag}: duplicate global_index={idx}")
        by_idx[idx] = row
    missing = [idx for idx in range(NUM_SEQUENCES) if idx not in by_idx]
    if missing:
        raise RuntimeError(f"{policy}/{condition_tag}: missing {len(missing)} rows, first={missing[:10]}")
    ordered = [by_idx[idx] for idx in range(NUM_SEQUENCES)]
    successes = [int(row["success"]) for row in ordered]
    chain_sr = {str(k): sum(success >= k for success in successes) / NUM_SEQUENCES for k in range(1, 6)}
    task_info: dict[str, dict[str, int]] = {}
    for row in ordered:
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
        "condition_tag": condition_tag,
        "array_job_id": str(array_job_id),
        "num_sequences": NUM_SEQUENCES,
        "num_chunks": NUM_CHUNKS,
        "chunk_size": CHUNK_SIZE,
        "calvin_sequence_manifest": str(manifest_path),
        "calvin_sequence_manifest_sha256": CONDITION_SPECS[condition_tag]["sha256"],
        "sample_kind": SAMPLE_KIND,
        "sample_seed": CONDITION_SPECS[condition_tag]["seed"],
        "avg_seq_len": sum(successes) / NUM_SEQUENCES,
        "chain_sr": chain_sr,
        "success_count_histogram": {str(k): int(v) for k, v in sorted(Counter(successes).items())},
        "task_info": task_info,
        "per_sequence_sources": sorted({row["_source"] for row in ordered}),
        "result_dirs": sorted({row["_result_dir"] for row in ordered}),
    }


def load_baseline(path: Path, expected_sha: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing baseline combined summary: {path}")
    actual = file_sha256(path)
    if actual != expected_sha:
        raise RuntimeError(f"baseline summary sha256 {actual} != expected {expected_sha}: {path}")
    payload = json.loads(path.read_text())
    summaries = payload.get("summaries")
    if not isinstance(summaries, dict):
        raise RuntimeError(f"baseline summary lacks summaries dict: {path}")
    return payload


def baseline_for_policy(baseline: dict[str, Any], policy: str) -> dict[str, Any]:
    key = f"{policy}_calibration"
    item = baseline["summaries"].get(key)
    if not isinstance(item, dict) or "avg_seq_len" not in item:
        raise RuntimeError(f"baseline summary lacks {key}.avg_seq_len")
    return item


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CALVIN Layer 2 fresh 1000-sequence rollouts.")
    parser.add_argument("--project-root", default="/share/data/ripl/tianchong/projects/Eval_Policies_CoRL")
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--manifest-list", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--baseline-sha256", default=BASELINE_SHA256)
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty aggregate directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    manifest_list_path = Path(args.manifest_list).resolve() if args.manifest_list else project_root / MANIFEST_LIST_REL
    results_root = Path(args.results_root).resolve() if args.results_root else project_root / "results/calvin/layer2_fresh_1000/paper"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else project_root / "results" / "calvin" / "aggregates" / f"calvin_layer2_fresh_1000_{args.array_job_id}"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise RuntimeError(f"output aggregate directory exists and is non-empty: {output_dir}; pass --force to overwrite known JSON outputs")

    manifest_paths = load_manifest_paths(manifest_list_path, project_root)
    manifests = {condition: validate_manifest(condition, path) for condition, path in manifest_paths.items()}
    baseline_path = Path(args.baseline_summary).resolve() if args.baseline_summary else project_root / BASELINE_REL
    baseline = load_baseline(baseline_path, args.baseline_sha256)
    summaries_to_write: list[tuple[Path, dict[str, Any]]] = []

    combined: dict[str, Any] = {
        "array_job_id": str(args.array_job_id),
        "num_sequences": NUM_SEQUENCES,
        "num_chunks": NUM_CHUNKS,
        "chunk_size": CHUNK_SIZE,
        "conditions": list(CONDITION_SPECS.keys()),
        "policies": list(POLICY_DIRS.keys()),
        "sample_kind": SAMPLE_KIND,
        "manifest_list": str(manifest_list_path),
        "results_root": str(results_root),
        "aggregate_dir": str(output_dir),
        "baseline_comparison": {
            "reference_combined_summary": str(baseline_path),
            "reference_sha256": args.baseline_sha256,
            "comparison_condition": "calibration",
            "note": "Existing calvin_full_1000_2048456 aggregate is read-only and was not modified.",
        },
        "summaries": {},
    }

    for condition_tag in CONDITION_SPECS:
        manifest_path = manifest_paths[condition_tag]
        manifest = manifests[condition_tag]
        for policy in POLICY_DIRS:
            row_files = discover_row_files(results_root, condition_tag, policy, str(args.array_job_id))
            context = f"{policy}/{condition_tag}"
            if not row_files:
                raise RuntimeError(f"{context}: no per_sequence_results.json files found under {results_root}")
            chunks: dict[int, list[dict[str, Any]]] = {}
            result_dirs: dict[int, Path] = {}
            for row_path in row_files:
                chunk_id, rows, result_dir = load_and_validate_chunk(
                    row_path,
                    policy=policy,
                    condition_tag=condition_tag,
                    array_job_id=str(args.array_job_id),
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
                if chunk_id in chunks:
                    raise RuntimeError(f"{context}: duplicate chunk_id={chunk_id}: {result_dirs[chunk_id]} and {result_dir}")
                chunks[chunk_id] = rows
                result_dirs[chunk_id] = result_dir
            if sorted(chunks) != list(range(NUM_CHUNKS)):
                raise RuntimeError(f"{context}: expected chunks 0..{NUM_CHUNKS - 1}, got {sorted(chunks)}")
            rows = [row for chunk_id in range(NUM_CHUNKS) for row in chunks[chunk_id]]
            summary = summarize_rows(
                rows,
                policy=policy,
                condition_tag=condition_tag,
                array_job_id=str(args.array_job_id),
                manifest_path=manifest_path,
            )
            baseline_item = baseline_for_policy(baseline, policy)
            baseline_avg = float(baseline_item["avg_seq_len"])
            summary["baseline_reference"] = {
                "combined_summary": str(baseline_path),
                "combined_summary_sha256": args.baseline_sha256,
                "baseline_key": f"{policy}_calibration",
                "baseline_avg_seq_len": baseline_avg,
                "layer2_minus_baseline_avg_seq_len": float(summary["avg_seq_len"]) - baseline_avg,
                "baseline_chain_sr": baseline_item.get("chain_sr", {}),
            }
            summary_path = output_dir / f"{policy}_{condition_tag}_summary.json"
            summaries_to_write.append((summary_path, summary))
            combined["summaries"][f"{policy}_{condition_tag}"] = {
                "summary": str(summary_path),
                "avg_seq_len": summary["avg_seq_len"],
                "chain_sr": summary["chain_sr"],
                "baseline_avg_seq_len": baseline_avg,
                "layer2_minus_baseline_avg_seq_len": summary["baseline_reference"]["layer2_minus_baseline_avg_seq_len"],
                "calvin_sequence_manifest": str(manifest_path),
                "calvin_sequence_manifest_sha256": CONDITION_SPECS[condition_tag]["sha256"],
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    for summary_path, summary in summaries_to_write:
        write_json(summary_path, summary)
    combined_path = output_dir / "combined_summary.json"
    write_json(combined_path, combined)
    print(json.dumps({"combined_summary": str(combined_path), "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
