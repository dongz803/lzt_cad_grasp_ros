#!/usr/bin/env python3
"""Create a non-copying one-image T-LESS BOP dataset for smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"existing symlink points elsewhere: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=20)
    parser.add_argument("--image-id", type=int, default=3)
    parser.add_argument("--object-id", type=int, default=1)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    source_tless = source / "tless"
    destination_tless = destination / "tless"

    target = {
        "im_id": args.image_id,
        "inst_count": 1,
        "obj_id": args.object_id,
        "scene_id": args.scene_id,
    }
    source_targets = json.loads(
        (source_tless / "test_targets_bop19.json").read_text(encoding="utf-8")
    )
    sample_exists = any(
        item["scene_id"] == args.scene_id
        and item["im_id"] == args.image_id
        and item["obj_id"] == args.object_id
        for item in source_targets
    )
    if not sample_exists:
        raise RuntimeError(f"requested sample is not a BOP19 target: {target}")

    ensure_symlink(
        destination_tless / "camera_primesense.json",
        source_tless / "camera_primesense.json",
    )
    ensure_symlink(
        destination_tless / "dataset_info.md", source_tless / "dataset_info.md"
    )
    ensure_symlink(
        destination_tless / "models_cad", source_tless / "models_cad"
    )
    ensure_symlink(
        destination_tless / "test_primesense" / f"{args.scene_id:06d}",
        source_tless / "test_primesense" / f"{args.scene_id:06d}",
    )
    ensure_symlink(
        destination
        / "detections"
        / "cnos-fastsam"
        / "cnos-fastsam_tless-test.json",
        source / "detections" / "cnos-fastsam" / "cnos-fastsam_tless-test.json",
    )

    target_path = destination_tless / "test_targets_bop19.json"
    expected = json.dumps([target], indent=2) + "\n"
    if target_path.exists() and target_path.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"refusing to replace different target file: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(expected, encoding="utf-8")
    print(f"T-LESS smoke dataset ready: {destination}")


if __name__ == "__main__":
    main()
