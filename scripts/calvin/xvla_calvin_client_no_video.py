#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-client", required=True)
    known, rest = parser.parse_known_args()

    source = Path(known.source_client).resolve()
    spec = importlib.util.spec_from_file_location("xvla_calvin_client_upstream", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def skip_video(path, frames, fps=30):
        return None

    upstream_conf_dir = source.parent / "ABC_D" / "validation"
    original_load = module.OmegaConf.load

    def load_with_upstream_yaml(path):
        p = Path(path)
        if p.name in {"new_playtable_tasks.yaml", "new_playtable_validation.yaml"}:
            return original_load(upstream_conf_dir / p.name)
        return original_load(path)

    module.save_video = skip_video
    module.OmegaConf.load = load_with_upstream_yaml
    sys.argv = [str(source), *rest]
    module.main()


if __name__ == "__main__":
    main()
