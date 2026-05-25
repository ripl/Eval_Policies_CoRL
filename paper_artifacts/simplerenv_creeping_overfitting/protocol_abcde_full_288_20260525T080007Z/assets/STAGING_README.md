# Custom Asset Staging Note

## Purpose

This note explains how the accepted custom SimplerEnv assets in this package relate to runtime trees. It is a staging note only; it does not modify any third-party checkout.

The files under `assets/custom/` were copied from one source tree:

`/share/data/ripl/tianchong/projects/Eval_Policies_CoRL/third_party/simpler_env/ManiSkill2_real2sim/data/custom`

Accepted asset IDs included here:

- `render_candidate_blue_hybrid_v4`
- `render_candidate_red_corrected_v6e`
- `render_candidate_white_offwhite_hybrid_v4`

For reproduction, stage these files into each runtime tree that evaluates Protocol C/D/E so that the runtime has:

- `ManiSkill2_real2sim/data/custom/info_bridge_custom_baked_tex_v0.json`
- `ManiSkill2_real2sim/data/custom/models/render_candidate_blue_hybrid_v4/`
- `ManiSkill2_real2sim/data/custom/models/render_candidate_red_corrected_v6e/`
- `ManiSkill2_real2sim/data/custom/models/render_candidate_white_offwhite_hybrid_v4/`

Do not treat this package as an instruction to edit `third_party/` in place. Stage the files into the intended runtime tree explicitly and verify the runtime asset manifest/preflight before launching rollouts.
