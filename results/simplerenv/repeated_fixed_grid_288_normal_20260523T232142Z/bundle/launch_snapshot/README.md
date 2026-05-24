# Repeated Fixed-Grid Calibration v2

## Purpose

This directory is the clean v2 launch and finalization layer for the
SimplerEnv/WidowX repeated fixed-grid calibration on 2026-05-23. Put only the
20-job launcher, copied worker scripts needed by those jobs, clean pinned
source snapshots, validation scripts, and small run metadata here. Do not use
the mixed old scratch directory as a launch target.

## Supported Policy/Task Matrix

The supported policies are `cogact`, `spatialvla`, `internvla_m1`, `xvla`, and
`dexbotic`. The supported tasks are `stack`, `carrot`, `spoon`, and `eggplant`.
The run shape is exactly `5 x 4 = 20` independent Slurm jobs. Slurm arrays are
not supported.

Each job must produce exactly `288` rows: `12` repeats of official fixed-grid
episode IDs `0..23`. Per-policy totals are `1152` rows; the full run total is
`5760` rows. Standard horizons are `stack=60`, `carrot=60`, `spoon=60`, and
`eggplant=120`.

## Script Manifest

- `prepare_spatialvla_source.sh`: creates or verifies the clean pinned
  SpatialVLA source at `sources/simplerenv_openvla_ccfe380`. It never
  modifies or reverts `third_party/simplerenv_openvla`.
- `submit_all_20.sh`: preflights all supported policy/task workers, then
  submits exactly 20 independent jobs and writes `submission_jobs.csv`. Use
  `--dry-run` or `--preflight-only` for review without submitting.
- `run_one_policy_task.sbatch`, `run_one_policy_task.py`, `validate_job_csv.py`:
  CogACT-Base and SpatialVLA standard-horizon worker path. SpatialVLA uses the
  clean pinned source under this v2 directory.
- `internvla_m1_task.sbatch`, `internvla_m1_task_driver.py`,
  `run_seeded_policy_server.py`, `run_seeded_simpler_episode.py`,
  `validate_internvla_task_results.py`: InternVLA-M1 worker path. The driver
  requires the only dirty InternVLA file to be
  `examples/SimplerEnv/start_simpler_env.py` with diff sha256
  `251f8e7e0e89dbd5ffa14b0ef5d6e89a9f54eb1806de15780f2925093cb5b733`,
  classified as `launch/host/debugpy-only`.
- `xvla_worker_a_support/`: X-VLA-WidowX worker path copied into v2 with v2
  paths. Its Slurm template uses `--gpus=1 -c 4` and no array directive.
- `dexbotic_support/`: Dexbotic / DB-MemVLA worker path copied into v2 with v2
  paths. Its Slurm template uses `--gpus=1 -c 4` and no array directive.
- `final_validate_and_summarize.py`: the single finalizer. It validates
  `submission_jobs.csv`, all 20 per-task `per_episode_results.csv` files,
  deterministic seed formulas, row/key counts, horizons, errors/timeouts,
  videos, and duplicate artifact paths, then emits
  `per_episode_results_all.csv`, `per_task_summary.csv`, and
  `per_policy_summary.csv`.

## Seed Formulas

- `cogact` and `spatialvla`:
  `202605230 + policy_offset + task_offset + repeat_id * 24 + official_episode_id`,
  where policy offsets are `cogact=100000`, `spatialvla=200000`, and task
  offsets are `stack=1000`, `carrot=2000`, `spoon=3000`, `eggplant=4000`.
- `internvla_m1`:
  `20260523 + task_index * 10000 + repeat_id * 24 + official_episode_id`.
- `xvla` and `dexbotic`:
  `20260523 + policy_index * 1000000 + task_index * 10000 + repeat_id * 100 + official_episode_id`,
  where policy indices are `xvla=0`, `dexbotic=1`.

## Review Commands

```bash
bash prepare_spatialvla_source.sh --verify-only
bash submit_all_20.sh --dry-run
python3 -m py_compile run_one_policy_task.py validate_job_csv.py internvla_m1_task_driver.py validate_internvla_task_results.py final_validate_and_summarize.py xvla_worker_a_support/bin/*.py dexbotic_support/bin/*.py
bash -n submit_all_20.sh prepare_spatialvla_source.sh run_one_policy_task.sbatch internvla_m1_task.sbatch xvla_worker_a_support/run_one_task.sh xvla_worker_a_support/launch/*.sbatch dexbotic_support/run_one_task.sh dexbotic_support/launch/*.sh dexbotic_support/launch/*.sbatch
```

Do not run plain `bash submit_all_20.sh` unless Tianchong explicitly approves
job submission.
