#!/usr/bin/env python3
"""Validate the local T-LESS BOP19 test subset and write a machine report."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bop-root",
        type=Path,
        default=Path(os.environ.get("BOP_PATH", workspace / "datasets/bop")),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bop_root = args.bop_root.resolve()
    dataset = bop_root / "tless"
    targets_path = dataset / "test_targets_bop19.json"
    targets = _load_json(targets_path)
    if not isinstance(targets, list) or not targets:
        raise SystemExit("test_targets_bop19.json is empty or invalid")

    model_ids = list(range(1, 31))
    missing_models = [
        object_id
        for object_id in model_ids
        if not (dataset / "models_cad" / f"obj_{object_id:06d}.ply").is_file()
    ]
    scene_ids = list(range(1, 21))
    missing_scene_metadata = []
    missing_target_images = []
    checked_images: set[tuple[int, int]] = set()
    for target in targets:
        scene_id = int(target["scene_id"])
        image_id = int(target["im_id"])
        scene = dataset / "test_primesense" / f"{scene_id:06d}"
        if (scene_id, image_id) not in checked_images:
            checked_images.add((scene_id, image_id))
            required = [
                scene / "rgb" / f"{image_id:06d}.png",
                scene / "depth" / f"{image_id:06d}.png",
                scene / "scene_camera.json",
                scene / "scene_gt.json",
                scene / "scene_gt_info.json",
            ]
            for path in required:
                if not path.is_file():
                    missing_target_images.append(str(path))
    for scene_id in scene_ids:
        scene = dataset / "test_primesense" / f"{scene_id:06d}"
        if not scene.is_dir():
            missing_scene_metadata.append(str(scene))

    detection_path = (
        bop_root / "detections/cnos-fastsam/cnos-fastsam_tless-test.json"
    )
    detections = _load_json(detection_path) if detection_path.is_file() else None
    object_target_counts = Counter(int(item["obj_id"]) for item in targets)
    total_instances = sum(int(item["inst_count"]) for item in targets)
    report = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bop_root": str(bop_root),
        "dataset": "tless",
        "target_file": str(targets_path),
        "target_entry_count": len(targets),
        "target_instance_count": total_instances,
        "unique_target_image_count": len(checked_images),
        "scene_count": len(scene_ids) - len(missing_scene_metadata),
        "cad_model_count": len(model_ids) - len(missing_models),
        "target_entries_per_object": {
            str(key): object_target_counts[key] for key in sorted(object_target_counts)
        },
        "detection_file": str(detection_path),
        "detection_count": len(detections) if isinstance(detections, list) else None,
        "missing_models": missing_models,
        "missing_scene_paths": missing_scene_metadata,
        "missing_target_files": missing_target_images,
        "complete": not (
            missing_models or missing_scene_metadata or missing_target_images
        ),
    }
    output = (
        args.output.resolve()
        if args.output
        else Path(__file__).resolve().parents[1]
        / "outputs/foundpose_bop19/dataset_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
