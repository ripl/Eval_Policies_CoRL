#!/usr/bin/env python3
from __future__ import annotations

import os
from seed_utils import set_global_seeds


def main() -> None:
    seed = int(os.environ["DEXBOTIC_SERVER_SEED"])
    set_global_seeds(seed)
    from playground.benchmarks.simpler.simpler_memvla import SimplerMemVLAExp
    exp = SimplerMemVLAExp()
    exp.inference_config.model_name_or_path = os.environ["DEXBOTIC_MODEL_ID"]
    exp.inference_config.port = int(os.environ["DEXBOTIC_PORT"])
    exp.inference_config.save_image = False
    exp.inference_config.norm_stats = os.environ["DEXBOTIC_NORM_STATS"]
    print(f"Starting seeded Dexbotic Simpler MemVLA server seed={seed} model={exp.inference_config.model_name_or_path} port={exp.inference_config.port} norm_stats={exp.inference_config.norm_stats}", flush=True)
    exp.inference()


if __name__ == "__main__":
    main()
