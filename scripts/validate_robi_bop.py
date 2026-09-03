#!/usr/bin/env python3
"""Validate converted ROBI BOP structure and quantify reflective depth loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--estimates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-masks", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    required_root = [
        root / "camera.json",
        root / "models/models_info.json",
        root / "test_targets_bop19.json",
        root / "robi_scene_map.json",
    ]
    missing = [str(path) for path in required_root if not path.is_file()]
    targets = load(root / "test_targets_bop19.json") if not missing else []
    image_count = 0
    instance_count = 0
    depth_valid_fractions = []
    gt_depths_mm = []
    missing_scene_files = []
    missing_masks = []
    for scene in sorted((root / "test").glob("[0-9]*")):
        files = [
            scene / "scene_camera.json",
            scene / "scene_gt.json",
            scene / "scene_gt_info.json",
            scene / "robi_image_map.json",
        ]
        missing_scene_files.extend(str(path) for path in files if not path.is_file())
        if any(not path.is_file() for path in files):
            continue
        cameras = load(scene / "scene_camera.json")
        ground_truth = load(scene / "scene_gt.json")
        info = load(scene / "scene_gt_info.json")
        for image_key, instances in ground_truth.items():
            image_id = int(image_key)
            rgb = scene / "rgb" / f"{image_id:06d}.png"
            depth_path = scene / "depth" / f"{image_id:06d}.png"
            if not rgb.is_file() or not depth_path.is_file():
                missing_scene_files.append(str(rgb if not rgb.is_file() else depth_path))
                continue
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                missing_scene_files.append(str(depth_path))
                continue
            depth_valid_fractions.append(float(np.count_nonzero(depth) / depth.size))
            image_count += 1
            instance_count += len(instances)
            if image_key not in cameras or len(info.get(image_key, [])) != len(instances):
                missing_scene_files.append(f"annotation mismatch: {scene.name}/{image_key}")
            for gt_id, instance in enumerate(instances):
                gt_depths_mm.append(float(instance["cam_t_m2c"][2]))
                if args.require_masks:
                    for folder in ("mask", "mask_visib"):
                        path = scene / folder / f"{image_id:06d}_{gt_id:06d}.png"
                        if not path.is_file():
                            missing_masks.append(str(path))

    detections = []
    if args.detections is not None:
        if args.detections.is_file():
            detections = load(args.detections)
        else:
            missing.append(str(args.detections))
    estimates = []
    if args.estimates is not None:
        if args.estimates.is_file():
            estimates = load(args.estimates)
        else:
            missing.append(str(args.estimates))
    target_instances = sum(int(item["inst_count"]) for item in targets)
    complete = not missing and not missing_scene_files and not missing_masks
    mssd_mm = np.asarray(
        [float(item["mssd"]) for item in estimates if "mssd" in item],
        dtype=np.float64,
    )
    mssd_normalized = np.asarray(
        [float(item["mssd_n"]) for item in estimates if "mssd_n" in item],
        dtype=np.float64,
    )
    mspd_px = np.asarray(
        [float(item["mspd"]) for item in estimates if "mspd" in item],
        dtype=np.float64,
    )
    mssd_thresholds = np.arange(0.05, 0.51, 0.05)
    mspd_thresholds = np.arange(5.0, 51.0, 5.0) * 1280.0 / 640.0

    def average_recall(values: np.ndarray, thresholds: np.ndarray) -> float | None:
        if target_instances <= 0 or values.size == 0:
            return None
        return float(
            np.mean(
                [
                    np.count_nonzero(values < threshold) / target_instances
                    for threshold in thresholds
                ]
            )
        )

    summary = {
        "schema_version": 1,
        "complete": complete,
        "dataset_root": str(root),
        "scene_count": len(list((root / "test").glob("[0-9]*"))),
        "image_count": image_count,
        "instance_count": instance_count,
        "target_rows": len(targets),
        "target_instances": target_instances,
        "detection_count": len(detections),
        "pose_estimate_count": len(estimates),
        "pose_prediction_coverage": (
            len(estimates) / target_instances if target_instances else None
        ),
        "preliminary_ar_mssd": average_recall(mssd_normalized, mssd_thresholds),
        "preliminary_ar_mspd": average_recall(mspd_px, mspd_thresholds),
        "median_mssd_mm": float(np.median(mssd_mm)) if mssd_mm.size else None,
        "median_mspd_px": float(np.median(mspd_px)) if mspd_px.size else None,
        "mssd_below_0_5d_count": (
            int(np.count_nonzero(mssd_normalized < 0.5))
            if mssd_normalized.size
            else None
        ),
        "mean_depth_valid_fraction": (
            float(np.mean(depth_valid_fractions)) if depth_valid_fractions else None
        ),
        "median_depth_valid_fraction": (
            float(np.median(depth_valid_fractions)) if depth_valid_fractions else None
        ),
        "minimum_gt_depth_mm": min(gt_depths_mm) if gt_depths_mm else None,
        "maximum_gt_depth_mm": max(gt_depths_mm) if gt_depths_mm else None,
        "missing_root_files": missing,
        "missing_scene_files": missing_scene_files,
        "missing_masks": missing_masks,
        "metric_scope": (
            "preliminary MSSD/MSPD on a local ROBI smoke subset; not an official BOP score"
            if estimates
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
