#!/usr/bin/env python3
"""Refine BOP-format instance masks using RGB-D support without GT labels."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)

from lzt_perception.mask_evaluation import decode_coco_rle  # noqa: E402
from lzt_perception.mask_refinement import (  # noqa: E402
    RGBDMaskRefinementConfig,
    binary_mask_to_rle,
    mask_bbox_xywh,
    refine_rgbd_mask,
)


def _align_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    source_height, source_width = mask.shape
    height, width = shape
    if source_height < height or source_width < width:
        raise ValueError(f"mask {mask.shape} is smaller than image {shape}")
    offset_y = (source_height - height) // 2
    offset_x = (source_width - width) // 2
    cropped = mask[offset_y:offset_y + height, offset_x:offset_x + width]
    if cropped.shape != shape:
        raise ValueError(f"cannot align mask {mask.shape} to image {shape}")
    return cropped


def _preview(
    path: Path,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    original: np.ndarray,
    refined: np.ndarray,
) -> None:
    before = rgb.copy()
    after = rgb.copy()
    before[original] = np.rint(
        0.45 * before[original] + 0.55 * np.array([255, 40, 40])
    ).astype(np.uint8)
    after[refined] = np.rint(
        0.45 * after[refined] + 0.55 * np.array([40, 255, 40])
    ).astype(np.uint8)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.quantile(depth_m[valid], [0.02, 0.98])
        if high > low:
            normalized[valid] = np.clip(
                255.0 * (depth_m[valid] - low) / (high - low), 0, 255
            ).astype(np.uint8)
    depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    canvas = np.hstack((before, after, depth_color))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    default_dataset = WORKSPACE_ROOT / "datasets/bop/tless"
    default_detections = (
        WORKSPACE_ROOT
        / "datasets/bop/detections/cnos-fastsam"
        / "cnos-fastsam_tless-test.json"
    )
    parser = argparse.ArgumentParser(
        description="Conservatively refine BOP masks with RGB-D support."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--detections", type=Path, default=default_detections)
    parser.add_argument("--object-ids", type=int, nargs="+")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/mask_refinement/tless",
    )
    parser.add_argument("--seed-erosion-px", type=int, default=2)
    parser.add_argument("--search-dilation-px", type=int, default=1)
    parser.add_argument("--depth-margin-m", type=float, default=0.008)
    parser.add_argument("--depth-mad-scale", type=float, default=3.0)
    parser.add_argument("--opening-kernel", type=int, default=1)
    parser.add_argument("--closing-kernel", type=int, default=1)
    parser.add_argument("--minimum-correction-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-correction-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-growth-fraction", type=float, default=0.08)
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = RGBDMaskRefinementConfig(
        seed_erosion_px=args.seed_erosion_px,
        search_dilation_px=args.search_dilation_px,
        depth_margin_m=args.depth_margin_m,
        depth_mad_scale=args.depth_mad_scale,
        opening_kernel=args.opening_kernel,
        closing_kernel=args.closing_kernel,
        minimum_correction_fraction=args.minimum_correction_fraction,
        maximum_correction_fraction=args.maximum_correction_fraction,
        maximum_growth_fraction=args.maximum_growth_fraction,
    )
    records = json.loads(args.detections.read_text(encoding="utf-8"))
    selected_objects = None if args.object_ids is None else set(args.object_ids)
    selected = [
        record
        for record in records
        if selected_objects is None
        or int(record["category_id"]) in selected_objects
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    split_root = args.dataset_root / args.split

    @lru_cache(maxsize=24)
    def load_observation(scene_id: int, image_id: int):
        scene_root = split_root / f"{scene_id:06d}"
        camera_payload = json.loads(
            (scene_root / "scene_camera.json").read_text(encoding="utf-8")
        )[str(image_id)]
        bgr = cv2.imread(
            str(scene_root / "rgb" / f"{image_id:06d}.png"),
            cv2.IMREAD_COLOR,
        )
        raw_depth = cv2.imread(
            str(scene_root / "depth" / f"{image_id:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if bgr is None or raw_depth is None:
            raise FileNotFoundError(
                f"missing RGB-D observation for {scene_id}/{image_id}"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_m = (
            raw_depth.astype(np.float64)
            * float(camera_payload.get("depth_scale", 1.0))
            / 1000.0
        )
        return rgb, depth_m

    output_records = []
    status_counts: Counter[str] = Counter()
    object_counts: Counter[int] = Counter()
    total_added = 0
    total_removed = 0
    previews_written = 0
    start = time.perf_counter()
    for index, record in enumerate(selected):
        scene_id = int(record["scene_id"])
        image_id = int(record["image_id"])
        object_id = int(record["category_id"])
        rgb, depth_m = load_observation(scene_id, image_id)
        original = _align_mask(
            decode_coco_rle(record["segmentation"]), depth_m.shape
        )
        refine_start = time.perf_counter()
        result = refine_rgbd_mask(rgb, depth_m, original, config)
        refine_seconds = time.perf_counter() - refine_start
        updated = dict(record)
        updated["bbox"] = mask_bbox_xywh(result.mask)
        updated["segmentation"] = binary_mask_to_rle(result.mask)
        updated["refinement"] = {
            **result.diagnostics(),
            "runtime_s": refine_seconds,
        }
        output_records.append(updated)
        status_counts[result.status] += 1
        object_counts[object_id] += 1
        total_added += result.added_pixels
        total_removed += result.removed_pixels
        if previews_written < args.preview_limit:
            name = (
                f"{index:05d}_s{scene_id:06d}_i{image_id:06d}_"
                f"o{object_id:06d}.png"
            )
            _preview(
                args.output_dir / "previews" / name,
                rgb,
                depth_m,
                original,
                result.mask,
            )
            previews_written += 1

    elapsed = time.perf_counter() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detections_path = args.output_dir / "detections_rgbd_refined.json"
    summary_path = args.output_dir / "summary.json"
    detections_path.write_text(
        json.dumps(output_records, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "method": "conservative_rgbd_mask_refinement",
        "input": str(args.detections.resolve()),
        "output": str(detections_path.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "object_ids": sorted(object_counts),
        "config": config.__dict__,
        "processed_predictions": len(output_records),
        "status_counts": dict(status_counts),
        "per_object_prediction_counts": {
            str(key): value for key, value in sorted(object_counts.items())
        },
        "added_pixels": total_added,
        "removed_pixels": total_removed,
        "runtime_s": elapsed,
        "ground_truth_used": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
