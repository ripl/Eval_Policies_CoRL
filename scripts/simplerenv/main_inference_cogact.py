"""CogACT entry point for SimplerEnv evaluation.

This mirrors SimplerEnv's upstream ``main_inference.py`` and applies the
CogACT README's policy-model hook without modifying the SimplerEnv submodule.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
import sim_cogact.cogact_policy as cogact_policy
from sim_cogact import CogACTInference


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _patch_hf_token_for_cogact() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    token_file = os.environ.get("HF_TOKEN_FILE", "").strip()
    if not token and token_file:
        token = Path(token_file).expanduser().read_text().strip()
    if not token:
        return

    original_load_vla = cogact_policy.load_vla

    def load_vla_with_token(*args, **kwargs):
        kwargs.setdefault("hf_token", token)
        return original_load_vla(*args, **kwargs)

    cogact_policy.load_vla = load_vla_with_token


if __name__ == "__main__":
    args = get_args()
    if args.policy_model != "cogact":
        raise ValueError(f"main_inference_cogact.py only supports --policy-model cogact, got {args.policy_model!r}")

    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    if args.ckpt_path is None or args.ckpt_path == "None":
        raise ValueError("--ckpt-path must point to a CogACT checkpoint or Hugging Face model id")

    _patch_hf_token_for_cogact()

    model = CogACTInference(
        saved_model_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        action_scale=args.action_scale,
        action_model_type=os.environ.get("COGACT_ACTION_MODEL_TYPE", "DiT-B"),
        cfg_scale=float(os.environ.get("COGACT_CFG_SCALE", "1.5")),
        use_bf16=_env_bool("COGACT_USE_BF16", False),
        use_ddim=_env_bool("COGACT_USE_DDIM", True),
        num_ddim_steps=int(os.environ.get("COGACT_NUM_DDIM_STEPS", "10")),
    )

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
