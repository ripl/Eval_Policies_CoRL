#!/usr/bin/env python3
"""Seed a simulator/client process, then run InternVLA SimplerEnv entrypoint."""

import argparse
import os
import random
import runpy
import sys


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["INTERNVLA_EVAL_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception as exc:
        print("seed_warning=numpy:{}".format(exc), flush=True)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
    except Exception as exc:
        print("seed_warning=torch:{}".format(exc), flush=True)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start-script", required=True)
    args, rest = parser.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    return args, rest


def main(argv=None):
    args, rest = parse_args(argv or sys.argv[1:])
    seed_everything(args.seed)
    print("seeded_process=client seed={}".format(args.seed), flush=True)
    sys.argv = [args.start_script] + rest
    runpy.run_path(args.start_script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
