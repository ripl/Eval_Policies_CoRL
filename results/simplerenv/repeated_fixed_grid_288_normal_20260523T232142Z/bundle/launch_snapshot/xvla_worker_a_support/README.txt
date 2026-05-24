Repeated fixed-grid calibration v2 support for X-VLA-WidowX only.

This Worker A bundle lives under:
  /share/data/ripl/tianchong/projects/Eval_Policies_CoRL/scratch/repeated_fixed_grid_calibration_288_20260523_v2/xvla_worker_a_support

Scope:
- One invocation runs exactly one task: stack, carrot, spoon, or eggplant.
- Policy column value is xvla.
- Each normal task run covers 12 repeats x official episode IDs 0..23.
- Standard horizons are stack=60, carrot=60, spoon=60, eggplant=120.
- Slurm arrays are not used.

Dry-run/preflight:
  bash run_one_task.sh --task stack --dry-run

Normal run shape, for review only:
  bash run_one_task.sh --task stack --run-tag xvla_stack_manual_test

Sbatch shape, for review only; do not submit unless explicitly approved:
  sbatch -J repcal_xvla_stack launch/run_xvla_task.sbatch stack

Resume and error handling:
- Existing output is rejected unless --resume is passed.
- --resume only skips rows already present in per_episode_results.csv; an
  episode directory without a matching row is stale output and fails.
- After every episode, one row is appended to per_episode_results.csv.
- By default any episode error or timeout row is appended and then the task job
  exits nonzero.
- --continue-on-error keeps running after error rows, but final validation still
  fails on any error/timeout row unless the validator is explicitly run with
  diagnostic allow flags.

Validation:
- validate_task_results.py checks one task and expects exactly 288 rows by
  default.
- validate_xvla_full_run.py checks all four X-VLA tasks and expects exactly
  1152 rows.
