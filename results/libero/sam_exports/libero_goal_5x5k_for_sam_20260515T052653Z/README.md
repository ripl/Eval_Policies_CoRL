# LIBERO-Goal 5 x 5k rollout artifacts for Sam

Created: 20260515T052653Z

This package contains the final per-episode success/failure artifacts for five policies evaluated on the same 5000 LIBERO-Goal init states, for the empirical paired-disagreement D estimate in the statistical-significance section.

Included policies: Spatial Forcing, OpenVLA-OFT, HiF-VLA, SimVLA, and Pi0.5/LeRobot with n_action_steps=10.

Use `policy_success_summary.csv` for aggregate success counts and `pairwise_disagreement.csv` for pairwise D. Each policy also has `policies/<policy_slug>/episodes_combined.csv` and task-level CSV/manifest/summary files under `policies/<policy_slug>/tasks/task_*`.

Excluded: checkpoint weights, videos, image dumps, full observations, and per-step action traces. The earlier Pi0.5 diagnostic run with the checkpoint-default action horizon is not included.

Validation errors: 0
