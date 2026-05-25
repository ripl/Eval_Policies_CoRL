# SimplerEnv Protocol A-E Full 288 Paper Artifact Package

## Purpose

This package is the curated lightweight paper artifact bundle for the completed SimplerEnv/WidowX creeping-overfitting Protocol A-E full run at 288 episodes per policy/condition.

It supports the paper-facing Protocol A-E result matrix for five policies across seven conditions. It intentionally contains summaries, manifests, provenance, configs, accepted lightweight custom assets, and small logs only.

## Source Run

- Source run root: `/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/results/simplerenv/protocol_abcde_full_288/protocol_abcde_full_288_20260525T080007Z`
- Source final summary: `final_summary/` under the source run root
- Eval_Policies_CoRL repo commit observed during packaging: `36b558d4c78f0e3b0351f51de91ad2b28d416da0`
- Frozen protocol config: `config/simplerenv_protocol_abcde_stack_v1.json`
- Protocol config SHA256: `1f2b4ea48e38df7d25304638d998c40902d79634433f725f0895e26f04ad810b`

## Validation Identity

- Final validator status: `passed`
- Rows: `10080/10080`
- Matrix: `5` policies x `7` conditions x `288` rows
- Active policy-condition cells: `35`
- Horizon: standard stack horizon `60` for all seven Protocol A-E conditions
- Error rows: `0`
- Timeout rows: `0`

The validation evidence is copied in `final_summary/validation_report.json`. The row-level evidence is `final_summary/per_episode_results_all.csv`, with aggregate summaries beside it.

## Dexbotic Replacement Caveat

Dexbotic / DB-MemVLA required replacement accounting. The original Dexbotic jobs `2075427`-`2075433` failed before valid rollout output because the seeded MemVLA server launcher path was wrong. Replacement r1 jobs `2076512`-`2076514` are the active A/B/C1 results.

Replacement r1 jobs `2076515`-`2076518` failed at episode `0` for C2/C3/D/E because the benchmark container did not register the accepted custom assets. Replacement r2 jobs `2076709`-`2076712` are the active C2/C3/D/E results after `MS2_REAL2SIM_ASSET_DIR` was pinned to the staged asset tree.

The supersession chains are preserved in `final_summary/replacement_job_records.csv`, `final_summary/validation_report.json`, and the Dexbotic replacement support/preflight/submission files under `launch_runtime_provenance/bundle/`.

## Included Files

- `final_summary/`: final validator report, contract, per-episode CSV, aggregate summaries, job records, replacement job records, and test-set manifest.
- `launch_runtime_provenance/`: launch snapshot, runtime asset manifest, submission support hashes, preflight reports, shared runtime script provenance JSON, Dexbotic replacement support/preflight/submission files, initial `submission.tsv`, and small Slurm logs.
- `config/`: frozen Protocol A-E config JSON and its SHA256 sidecar copied from `configs/simplerenv/protocol_abcde/`.
- `assets/`: accepted blue, red, and white/off-white custom asset files plus `info_bridge_custom_baked_tex_v0.json`, copied from one source tree: `third_party/simpler_env/ManiSkill2_real2sim/data/custom`.
- `checksums.sha256`: SHA256 digest for every package file except itself.

## Excluded Raw Artifacts

This package deliberately excludes raw rollout result directories, videos, observations, model weights, model caches, scratch directories, full third-party repositories, full source run directories, and large policy/runtime server logs. Those remain external to the paper package and should be referenced by exact source paths when needed.
