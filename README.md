# Eval Policies CoRL

Evaluation code for the CoRL manipulation benchmark audit, focused on task-preserving test-set changes that probe overfitting to benchmark idiosyncrasies.

Current first target:

- Benchmark: SimplerEnv WidowX / Bridge
- Policy: `CogACT-Base`
- Calibration: official fixed-grid `24 x 4` object episodes
- Altered test set: randomized valid object positions on the same table

The first SimplerEnv run is a calibration run against the official fixed-grid
setting. Treat differences from the paper as possible environment/package
version effects until the local evaluator is checked against the reported
CogACT WidowX/Bridge number.

## Layout

- `eval_policies_corl/`: reusable Python helpers.
- `configs/`: experiment and policy configs.
- `scripts/`: launch, parsing, and Slurm entry points.
- `third_party/`: external policy or benchmark repos as pinned git submodules.
- `results/`, `artifacts/`, `checkpoints/`, `cache/`, `envs/`: local-only directories ignored by git.

## SimplerEnv / CogACT Setup

This repo expects external code as submodules:

- `third_party/simpler_env`: `simpler-env/SimplerEnv`
- `third_party/cogact`: `microsoft/CogACT`

Build the runtime on the cluster with:

```bash
sbatch scripts/slurm/setup_simplerenv_cogact.sbatch
```

The conda env is created at `envs/simplerenv_cogact`, with package caches under
`cache/`, so `/home-nfs/tianchong` is not used for large environment state.

Run a one-episode smoke test:

```bash
EPISODE_END=1 TASK_FILTER=stack sbatch scripts/slurm/simplerenv_cogact_bridge.sbatch
```

Run the official WidowX/Bridge calibration:

```bash
sbatch scripts/slurm/simplerenv_cogact_bridge.sbatch
```

Each run writes videos under `results/` and version metadata under `artifacts/`.

## Open-Source Boundary

Commit source code, configs, scripts, submodule pins, and small metadata. Do not commit model weights, datasets, rollout videos, simulator caches, conda envs, container images, or large generated artifacts.

Prefer upstream policy repos as pinned submodules. Use a `ripl` fork only when we need patches, a frozen evaluation branch, or paper-critical reproducibility.

License is TBD before public release.
