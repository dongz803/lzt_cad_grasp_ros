#!/usr/bin/env python3
"""Render BOP masks/info and GT detections for converted ROBI scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = PROJECT_ROOT / "third_party/bop_toolkit_eval"
sys.path.insert(0, str(TOOLKIT_ROOT))

from bop_toolkit_lib import misc, visibility  # noqa: E402
from bop_toolkit_lib.rendering import renderer  # noqa: E402


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [-1, -1, -1, -1]
    return [int(value) for value in misc.calc_2d_bbox(xs, ys, mask.shape[::-1])]


def _rle(mask: np.ndarray) -> dict:
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts = []
    current = 0
    run = 0
    for value in pixels:
        bit = int(value != 0)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = bit
    counts.append(run)
    return {"counts": counts, "size": list(mask.shape)}


def main() -> None:
    workspace = PROJECT_ROOT.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=workspace / "datasets/bop/robi",
    )
    parser.add_argument(
        "--detections-output",
        type=Path,
        default=(
            workspace
            / "datasets/bop/detections/cnos-fastsam/cnos-fastsam_robi-test.json"
        ),
    )
    parser.add_argument("--visibility-delta-mm", type=float, default=15.0)
    args = parser.parse_args()

    camera_info = _load(args.dataset_root / "camera.json")
    width = int(camera_info["width"])
    height = int(camera_info["height"])
    models = args.dataset_root / "models"
    render = renderer.create_renderer(width, height, "vispy", mode="depth")
    model_ids = sorted(
        int(path.stem.split("_")[-1]) for path in models.glob("obj_*.ply")
    )
    for object_id in model_ids:
        render.add_object(object_id, str(models / f"obj_{object_id:06d}.ply"))

    detections = []
    rendered_instances = 0
    for scene in sorted((args.dataset_root / "test").glob("[0-9]*")):
        scene_id = int(scene.name)
        cameras = _load(scene / "scene_camera.json")
        ground_truth = _load(scene / "scene_gt.json")
        scene_info = {}
        mask_dir = scene / "mask"
        mask_visib_dir = scene / "mask_visib"
        mask_dir.mkdir(exist_ok=True)
        mask_visib_dir.mkdir(exist_ok=True)
        for image_key, instances in ground_truth.items():
            image_id = int(image_key)
            camera = cameras[image_key]
            intrinsics = np.asarray(camera["cam_K"]).reshape(3, 3)
            parameters = (
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            )
            scene_depth = cv2.imread(
                str(scene / "depth" / f"{image_id:06d}.png"),
                cv2.IMREAD_UNCHANGED,
            ).astype(np.float32) * float(camera.get("depth_scale", 1.0))
            image_info = []
            for gt_id, instance in enumerate(instances):
                object_id = int(instance["obj_id"])
                rotation = np.asarray(instance["cam_R_m2c"]).reshape(3, 3)
                translation = np.asarray(instance["cam_t_m2c"]).reshape(3, 1)
                rendered_depth = render.render_object(
                    object_id,
                    rotation,
                    translation,
                    *parameters,
                )["depth"]
                mask = rendered_depth > 0.0
                mask_visib = visibility.estimate_visib_mask_gt(
                    scene_depth,
                    rendered_depth,
                    args.visibility_delta_mm,
                    visib_mode="bop19",
                )
                mask_path = mask_dir / f"{image_id:06d}_{gt_id:06d}.png"
                visib_path = mask_visib_dir / f"{image_id:06d}_{gt_id:06d}.png"
                cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
                cv2.imwrite(str(visib_path), mask_visib.astype(np.uint8) * 255)
                count_all = int(np.count_nonzero(mask))
                count_valid = int(np.count_nonzero(mask & (scene_depth > 0.0)))
                count_visib = int(np.count_nonzero(mask_visib))
                image_info.append(
                    {
                        "bbox_obj": _bbox(mask),
                        "bbox_visib": _bbox(mask_visib),
                        "px_count_all": count_all,
                        "px_count_valid": count_valid,
                        "px_count_visib": count_visib,
                        "visib_fract": (
                            float(count_visib / count_all) if count_all else 0.0
                        ),
                    }
                )
                if count_visib:
                    detections.append(
                        {
                            "scene_id": scene_id,
                            "image_id": image_id,
                            "category_id": object_id,
                            "bbox": _bbox(mask_visib),
                            "score": 1.0,
                            "time": 0.0,
                            "segmentation": _rle(mask_visib),
                        }
                    )
                rendered_instances += 1
            scene_info[image_key] = image_info
        (scene / "scene_gt_info.json").write_text(
            json.dumps(scene_info, indent=2) + "\n",
            encoding="utf-8",
        )
    args.detections_output.parent.mkdir(parents=True, exist_ok=True)
    args.detections_output.write_text(
        json.dumps(detections) + "\n",
        encoding="utf-8",
    )
    print(
        f"rendered {rendered_instances} instances; "
        f"wrote {len(detections)} GT-mask detections to {args.detections_output}"
    )


if __name__ == "__main__":
    main()
