#!/usr/bin/env python3
"""Protocol-aware X-VLA WidowX client wrapper."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from widowx_protocol1 import install_from_env


CLIENT_BY_TASK = {
    "stack": "client_blocks.py",
    "blocks": "client_blocks.py",
    "carrot": "client_carrot.py",
    "spoon": "client_spoon.py",
}


def load_client_module(client_path: Path):
    spec = importlib.util.spec_from_file_location(f"xvla_{client_path.stem}", client_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load client module from {client_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _infer_success_from_video_path(video_path: str | os.PathLike[str]) -> bool:
    suffix = Path(video_path).stem.rsplit("_", 1)[-1]
    try:
        return float(suffix) >= 0.5
    except ValueError as exc:
        raise ValueError(f"Cannot infer success from video filename: {video_path}") from exc


def _as_uint8_rgb(frame):
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected image frame with 2 or 3 dims, got shape {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError(f"Expected RGB/RGBA frame, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.copy()
    return arr


def _overlay_visual_inspect_text(frame, *, step: int, total_steps: int, success: bool):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(_as_uint8_rgb(frame))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"step {step}/{total_steps} | success={str(success).lower()}"
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 5
    draw.rectangle(
        (0, 0, bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad),
        fill=(0, 0, 0),
    )
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def install_visual_inspect_video_format(module) -> None:
    original_write_video = getattr(module, "write_video", None)
    if original_write_video is None:
        raise RuntimeError(f"Loaded client module {module.__name__} has no write_video")

    def write_video_with_overlay(video_path, images, fps=10, *args, **kwargs):
        if float(fps) != 10.0:
            raise ValueError(f"Visual-inspection videos must be 10 fps, got fps={fps}")
        frames = list(images)
        if not frames:
            raise ValueError(f"Refusing to write empty visual-inspection video: {video_path}")
        final_success = _infer_success_from_video_path(video_path)
        total_steps = len(frames)
        rendered = [
            _overlay_visual_inspect_text(
                frame,
                step=idx,
                total_steps=total_steps,
                success=(final_success and idx == total_steps),
            )
            for idx, frame in enumerate(frames, start=1)
        ]
        rendered.extend([rendered[-1].copy() for _ in range(10)])
        return original_write_video(video_path, rendered, fps=10, *args, **kwargs)

    module.write_video = write_video_with_overlay


def parse_episode_ids_text(text: str, *, source: str) -> list[int]:
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError(f"No episode IDs found in {source}")

    episode_ids: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if not token.isdecimal():
            raise ValueError(
                f"Invalid episode ID token {token!r} in {source}; "
                "use non-negative integer IDs separated by commas or whitespace"
            )
        episode_id = int(token)
        if episode_id in seen:
            raise ValueError(f"Duplicate episode ID {episode_id} in {source}")
        seen.add(episode_id)
        episode_ids.append(episode_id)
    return episode_ids


def resolve_episode_ids(args: argparse.Namespace) -> list[int]:
    exact_sources = [bool(args.episode_ids), args.episode_ids_file is not None]
    if sum(exact_sources) > 1:
        raise ValueError("Pass only one of --episode-ids or --episode-ids-file")

    if args.episode_ids:
        if args.episode_start != 0 or args.episode_end != 500:
            raise ValueError("--episode-start/--episode-end cannot be combined with --episode-ids")
        return parse_episode_ids_text(args.episode_ids, source="--episode-ids")

    if args.episode_ids_file is not None:
        if args.episode_start != 0 or args.episode_end != 500:
            raise ValueError("--episode-start/--episode-end cannot be combined with --episode-ids-file")
        path = args.episode_ids_file.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Missing episode IDs file: {path}")
        return parse_episode_ids_text(path.read_text(), source=str(path))

    if args.require_explicit_episode_ids:
        raise ValueError(
            "Explicit episode IDs are required; refusing to fall back to "
            "contiguous --episode-start/--episode-end"
        )

    if args.episode_start < 0 or args.episode_end <= args.episode_start:
        raise ValueError(f"Invalid episode range: {args.episode_start}:{args.episode_end}")
    return list(range(args.episode_start, args.episode_end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", required=True)
    parser.add_argument("--server_port", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task-label", required=True, choices=sorted(CLIENT_BY_TASK))
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=500)
    parser.add_argument(
        "--episode-ids",
        default=None,
        help="Exact episode IDs, separated by commas or whitespace",
    )
    parser.add_argument(
        "--episode-ids-file",
        type=Path,
        default=None,
        help="File containing exact episode IDs separated by commas or whitespace",
    )
    parser.add_argument(
        "--require-explicit-episode-ids",
        action="store_true",
        help="Fail instead of using the contiguous episode range fallback",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument(
        "--client-dir",
        type=Path,
        default=None,
        help="Directory containing X-VLA WidowX client_*.py files",
    )
    parser.add_argument(
        "--visual-inspect-video-format",
        action="store_true",
        help="Write 10 fps videos with step/success overlay and a 1 second final-frame hold",
    )
    args = parser.parse_args()
    episode_ids = resolve_episode_ids(args)

    install_from_env(required=True)

    x_vla_repo = Path(os.environ.get("X_VLA_REPO", "")).expanduser()
    client_dir = args.client_dir
    if client_dir is None:
        if not x_vla_repo:
            raise RuntimeError("Set X_VLA_REPO or pass --client-dir")
        client_dir = x_vla_repo / "evaluation/simpler/WidowX"
    client_path = client_dir / CLIENT_BY_TASK[args.task_label]
    if not client_path.is_file():
        raise FileNotFoundError(f"Missing X-VLA client: {client_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "widowx_results.txt"
    if result_path.exists():
        raise RuntimeError(f"Refusing to append to existing result file: {result_path}")

    module = load_client_module(client_path)
    if args.visual_inspect_video_format:
        install_visual_inspect_video_format(module)
    client = module.XVLAClient(args.server_ip, args.server_port)

    protocol_name = os.environ.get("RUN_PROTOCOL_NAME", "widowx_protocol1_random_positions")
    total = len(episode_ids)
    for offset, proc_id in enumerate(episode_ids, start=1):
        print(f"{protocol_name} {args.task_label}: episode {proc_id} ({offset}/{total})", flush=True)
        module.evaluate_policy_widowx(
            client,
            str(output_dir),
            proc_id,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
