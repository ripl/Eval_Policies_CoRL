# Eval Policies CoRL

Evaluation code for the CoRL manipulation benchmark audit, focused on task-preserving test-set changes that probe overfitting to benchmark idiosyncrasies.

Current first target:

- Benchmark: SimplerEnv WidowX / Bridge
- Policy: `CogACT-Base`
- Calibration: official fixed-grid `24 x 4` object episodes
- Altered test set: randomized valid object positions on the same table

## Layout

- `eval_policies_corl/`: reusable Python helpers.
- `configs/`: experiment and policy configs.
- `scripts/`: launch, parsing, and Slurm entry points.
- `third_party/`: external policy or benchmark repos as pinned git submodules.
- `results/`, `artifacts/`, `checkpoints/`, `cache/`, `envs/`: local-only directories ignored by git.

## Open-Source Boundary

Commit source code, configs, scripts, submodule pins, and small metadata. Do not commit model weights, datasets, rollout videos, simulator caches, conda envs, container images, or large generated artifacts.

Prefer upstream policy repos as pinned submodules. Use a `ripl` fork only when we need patches, a frozen evaluation branch, or paper-critical reproducibility.

License is TBD before public release.
