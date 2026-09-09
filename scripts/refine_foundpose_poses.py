#!/usr/bin/env python3
"""Refine FoundPose candidates with CAD silhouette and observed RGB-D feedback."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.cad_grasp import load_ascii_ply_geometry  # noqa: E402
from lzt_cad_grasp_core.mask_evaluation import decode_coco_rle  # noqa: E402
from lzt_cad_grasp_core.object_config import load_object_config  # noqa: E402
from lzt_cad_grasp_core.pose_gating import (  # noqa: E402
    foundpose_record_transform,
    render_model_points,
    sample_model_points,
)
from lzt_cad_grasp_core.pose_refinement import (  # noqa: E402
    PoseRefinementConfig,
    refine_pose_rgbd,
)


STALE_GT_FIELDS = (
    "mssd",
    "mssd_n",
    "mspd",
    "mssd_id",
    "mspd_id",
    "inliers_gt",
    "inliers_gt_err",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _align_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    height, width = shape
    source_height, source_width = mask.shape
    if source_height < height or source_width < width:
        raise ValueError(f"mask {mask.shape} is smaller than image {shape}")
    offset_y = (source_height - height) // 2
    offset_x = (source_width - width) // 2
    aligned = mask[offset_y:offset_y + height, offset_x:offset_x + width]
    if aligned.shape != shape:
        raise ValueError(f"cannot align mask {mask.shape} to {shape}")
    return aligned


def _record_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["scene_id"]),
        int(record["img_id"]),
        int(record["obj_id"]),
        int(record.get("inst_id", 0)),
    )


def _preview(
    rgb_bgr: np.ndarray,
    observed_mask: np.ndarray,
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> np.ndarray:
    image = rgb_bgr.copy()
    layers = (
        (observed_mask, (255, 0, 0)),
        (initial_mask, (0, 0, 255)),
        (refined_mask, (0, 255, 0)),
    )
    for mask, color in layers:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, 2)
    cv2.putText(
        image,
        "observed=blue initial=red refined=green",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--object-config", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE_ROOT / "datasets/bop/tless",
    )
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--render-point-count", type=int, default=5000)
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--reweight-score", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("BOP_PATH", str(args.dataset_root.resolve().parent))
    object_config = load_object_config(args.object_config)
    object_suffix = object_config.cad_object.object_id.rsplit("_", 1)[-1]
    object_id = int(object_suffix) if object_suffix.isdigit() else None
    if object_id is None:
        raise SystemExit("object config ID must end in a numeric BOP object ID")
    records = _load_json(args.input)
    if not isinstance(records, list):
        raise SystemExit("input pose file must contain a JSON list")
    records = [item for item in records if int(item.get("obj_id", -1)) == object_id]
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_record_key(record)].append(record)
    selected_keys = sorted(groups)
    if args.limit_groups is not None:
        selected_keys = selected_keys[:args.limit_groups]

    detections: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for detection in _load_json(args.detections):
        key = (
            int(detection["scene_id"]),
            int(detection["image_id"]),
            int(detection["category_id"]),
        )
        if key[2] == object_id:
            detections[key].append(detection)
    for group in detections.values():
        group.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    geometry = load_ascii_ply_geometry(
        object_config.cad_object.mesh_path,
        units_to_m=object_config.cad_object.units_to_m,
    )
    model_points = sample_model_points(geometry.vertices_m, args.render_point_count)
    settings = PoseRefinementConfig()
    split_root = args.dataset_root / args.split

    @lru_cache(maxsize=24)
    def load_observation(scene_id: int, image_id: int):
        root = split_root / f"{scene_id:06d}"
        camera = _load_json(root / "scene_camera.json")[str(image_id)]
        depth_raw = cv2.imread(
            str(root / "depth" / f"{image_id:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        rgb = cv2.imread(str(root / "rgb" / f"{image_id:06d}.png"))
        if depth_raw is None or rgb is None:
            raise FileNotFoundError(f"missing RGB-D image {scene_id}/{image_id}")
        depth_m = (
            depth_raw.astype(np.float64)
            * float(camera.get("depth_scale", 1.0))
            / 1000.0
        )
        intrinsics = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
        return rgb, depth_m, intrinsics

    output_records = []
    audits = []
    reasons: Counter[str] = Counter()
    preview_count = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    for key in selected_keys:
        scene_id, image_id, _, instance_id = key
        detection_group = detections.get((scene_id, image_id, object_id), [])
        candidates = sorted(
            groups[key], key=lambda item: float(item.get("score", 0.0)), reverse=True
        )[:args.top_k]
        if instance_id >= len(detection_group):
            reasons["missing_observation_mask"] += len(candidates)
            continue
        rgb, depth_m, intrinsics = load_observation(scene_id, image_id)
        observed_mask = _align_mask(
            decode_coco_rle(detection_group[instance_id]["segmentation"]),
            depth_m.shape,
        )
        for record in candidates:
            initial = foundpose_record_transform(record)
            result = refine_pose_rgbd(
                model_points,
                initial,
                intrinsics,
                observed_mask,
                depth_m,
                settings,
            )
            output = dict(record)
            output["R"] = result.refined_transform_camera_object[:3, :3].tolist()
            output["t"] = (
                1000.0 * result.refined_transform_camera_object[:3, 3, None]
            ).tolist()
            input_gt = {field: output.pop(field) for field in STALE_GT_FIELDS if field in output}
            output["source"] = (
                "foundpose_cad_rgbd_refined"
                if result.accepted
                else "foundpose_cad_rgbd_refinement_no_change"
            )
            output["pose_refinement"] = result.to_dict()
            output["pose_refinement"]["input_evaluation_only"] = input_gt
            if args.reweight_score:
                pre_score = float(record.get("score", 0.0))
                quality = float(np.clip(1.0 - result.refined_metrics.data_loss, 0.0, 1.0))
                output["pre_refinement_score"] = pre_score
                output["score"] = pre_score * quality
            output_records.append(output)
            audits.append(
                {
                    "scene_id": scene_id,
                    "image_id": image_id,
                    "object_id": object_id,
                    "instance_id": instance_id,
                    "hypothesis_id": int(record.get("hypothesis_id", 0)),
                    **result.to_dict(),
                }
            )
            reasons["accepted" if result.accepted else "no_change"] += 1
            if result.accepted and preview_count < args.preview_limit:
                initial_mask, _, _ = render_model_points(
                    model_points,
                    initial,
                    intrinsics,
                    depth_m.shape,
                    splat_radius_px=settings.render_splat_radius_px,
                )
                refined_mask, _, _ = render_model_points(
                    model_points,
                    result.refined_transform_camera_object,
                    intrinsics,
                    depth_m.shape,
                    splat_radius_px=settings.render_splat_radius_px,
                )
                preview_dir.mkdir(parents=True, exist_ok=True)
                name = (
                    f"{scene_id}_{image_id}_{object_id}_{instance_id}_"
                    f"{int(record.get('hypothesis_id', 0))}.jpg"
                )
                cv2.imwrite(
                    str(preview_dir / name),
                    _preview(rgb, observed_mask, initial_mask, refined_mask),
                )
                preview_count += 1

    improvements = [item["objective_improvement"] for item in audits]
    summary = {
        "schema_version": 1,
        "method": "foundpose_cad_silhouette_depth_feedback_refinement",
        "object_id": object_id,
        "input": str(args.input.resolve()),
        "detections": str(args.detections.resolve()),
        "group_count": len(selected_keys),
        "candidate_count": len(output_records),
        "accepted_refinement_count": reasons["accepted"],
        "no_change_count": reasons["no_change"],
        "missing_mask_candidate_count": reasons["missing_observation_mask"],
        "mean_objective_improvement": (
            None if not improvements else float(np.mean(improvements))
        ),
        "median_objective_improvement": (
            None if not improvements else float(np.median(improvements))
        ),
        "config": settings.to_dict(),
        "ground_truth_used_for_refinement": False,
        "stale_input_gt_metrics_removed": True,
        "score_reweighted": args.reweight_score,
    }
    (args.output_dir / "refined-poses.json").write_text(
        json.dumps(output_records, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps({"summary": summary, "candidates": audits}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
