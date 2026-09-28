#!/usr/bin/env python3
"""Hard-gate impossible poses and soft-score RGB-D Top-K candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)

from lzt_grasp_planning.cad_grasp import load_ascii_ply_geometry  # noqa: E402
from lzt_perception.mask_evaluation import decode_coco_rle  # noqa: E402
from lzt_common.object_config import load_object_config  # noqa: E402
from lzt_perception.pose_gating import (  # noqa: E402
    PoseGateConfig,
    PoseGateDecision,
    evaluate_pose_candidate,
    reweighted_foundpose_record,
    sample_model_points,
)
from lzt_perception.pose_hypotheses import (  # noqa: E402
    foundpose_records_to_top_k,
    hypothesis_sets_to_jsonable,
)
from lzt_perception.pose_reliability import (  # noqa: E402
    FEATURE_NAMES as RELIABILITY_FEATURE_NAMES,
    PoseReliabilityModel,
    pose_reliability_features,
)


AUDIT_COLUMNS = [
    "scene_id",
    "image_id",
    "object_id",
    "instance_id",
    "hypothesis_id",
    "accepted",
    "rejection_reasons",
    "soft_quality_flags",
    "soft_pass",
    "original_score",
    "geometric_quality",
    "robust_score",
    "calibrated_probability",
    "translation_x_m",
    "translation_y_m",
    "translation_z_m",
    "pnp_inliers",
    "positive_model_fraction",
    "silhouette_iou",
    "reprojection_error_px",
    "depth_overlap_pixels",
    "median_depth_error_m",
    "depth_inlier_fraction",
    "mssd_mm_evaluation_only",
    "mspd_px_evaluation_only",
]


def _align_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    source_height, source_width = mask.shape
    height, width = shape
    if source_height < height or source_width < width:
        raise ValueError(f"mask {mask.shape} is smaller than image {shape}")
    offset_y = (source_height - height) // 2
    offset_x = (source_width - width) // 2
    result = mask[offset_y:offset_y + height, offset_x:offset_x + width]
    if result.shape != shape:
        raise ValueError(f"cannot align mask {mask.shape} to image {shape}")
    return result


def _opening(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size in (0, 1):
        return mask
    if kernel_size < 0 or kernel_size % 2 == 0:
        raise ValueError("opening kernel must be zero, one, or odd")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
    ).astype(bool)


def _load_robot_camera(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("transform_robot_camera")
    transform = np.asarray(payload, dtype=np.float64)
    if transform.shape != (4, 4):
        raise SystemExit("robot-camera transform must be a 4x4 JSON matrix")
    return transform


def _selection_metrics(
    records: list[dict[str, Any]],
    score_key: str,
    denominator: int,
) -> dict[str, Any]:
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            int(record["scene_id"]),
            int(record["img_id"]),
            int(record.get("obj_id", 0)),
            int(record.get("inst_id", 0)),
        )
        groups[key].append(record)
    selected = [
        max(group, key=lambda item: float(item.get(score_key, 0.0)))
        for group in groups.values()
    ]

    def values(field: str) -> list[float]:
        return [float(item[field]) for item in selected if field in item]

    result: dict[str, Any] = {
        "selected_groups": len(selected),
        "group_coverage": len(selected) / max(denominator, 1),
    }
    for field, output_name in (
        ("mssd", "median_mssd_mm"),
        ("mspd", "median_mspd_px"),
        ("mssd_n", "median_mssd_normalized"),
    ):
        measurements = values(field)
        result[output_name] = (
            None if not measurements else float(np.median(measurements))
        )
    return result


def _missing_mask_decision() -> PoseGateDecision:
    return PoseGateDecision(
        accepted=False,
        rejection_reasons=("missing_observation_mask",),
        soft_quality_flags=(),
        soft_pass=False,
        metrics={},
        geometric_quality=0.0,
        robust_score=0.0,
    )


def main() -> None:
    default_dataset = WORKSPACE_ROOT / "datasets/bop/tless"
    parser = argparse.ArgumentParser(
        description=(
            "Reject only nonphysical/non-executable FoundPose poses, convert "
            "PnP and RGB-D consistency into continuous soft scores, then "
            "recompute symmetry-aware weights and covariance."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--object-config", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mask-opening-kernel", type=int, default=3)
    parser.add_argument("--camera-min", type=float, nargs=3, default=[-1, -1, 0.05])
    parser.add_argument("--camera-max", type=float, nargs=3, default=[1, 1, 2.0])
    parser.add_argument("--robot-min", type=float, nargs=3)
    parser.add_argument("--robot-max", type=float, nargs=3)
    parser.add_argument("--robot-camera-transform", type=Path)
    parser.add_argument("--minimum-pnp-inliers", type=int, default=6)
    parser.add_argument("--minimum-pnp-score", type=float, default=0.05)
    parser.add_argument("--minimum-silhouette-iou", type=float, default=0.05)
    parser.add_argument(
        "--maximum-reprojection-error-px", type=float, default=20.0
    )
    parser.add_argument("--minimum-depth-overlap-pixels", type=int, default=12)
    parser.add_argument("--depth-inlier-tolerance-m", type=float, default=0.02)
    parser.add_argument(
        "--minimum-depth-inlier-fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--maximum-median-depth-error-m", type=float, default=0.05
    )
    parser.add_argument("--render-point-count", type=int, default=12000)
    parser.add_argument("--render-splat-radius-px", type=int, default=2)
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument(
        "--reliability-model",
        type=Path,
        help="optional learned JSON model replacing heuristic soft scores",
    )
    args = parser.parse_args()
    if (args.robot_min is None) != (args.robot_max is None):
        raise SystemExit("--robot-min and --robot-max must be provided together")

    os.environ.setdefault("BOP_PATH", str(args.dataset_root.resolve().parent))
    object_config = load_object_config(args.object_config)
    settings = PoseGateConfig(
        camera_min_xyz_m=tuple(args.camera_min),
        camera_max_xyz_m=tuple(args.camera_max),
        robot_min_xyz_m=(
            None if args.robot_min is None else tuple(args.robot_min)
        ),
        robot_max_xyz_m=(
            None if args.robot_max is None else tuple(args.robot_max)
        ),
        minimum_pnp_inliers=args.minimum_pnp_inliers,
        minimum_pnp_score=args.minimum_pnp_score,
        minimum_silhouette_iou=args.minimum_silhouette_iou,
        maximum_reprojection_error_px=args.maximum_reprojection_error_px,
        minimum_depth_overlap_pixels=args.minimum_depth_overlap_pixels,
        depth_inlier_tolerance_m=args.depth_inlier_tolerance_m,
        minimum_depth_inlier_fraction=args.minimum_depth_inlier_fraction,
        maximum_median_depth_error_m=args.maximum_median_depth_error_m,
        render_point_count=args.render_point_count,
        render_splat_radius_px=args.render_splat_radius_px,
    )
    robot_camera = _load_robot_camera(args.robot_camera_transform)
    if settings.robot_workspace_enabled and robot_camera is None:
        raise SystemExit("robot workspace gating requires --robot-camera-transform")

    raw_records = json.loads(args.input.read_text(encoding="utf-8"))
    object_suffix = object_config.cad_object.object_id.rsplit("_", 1)[-1]
    object_id = int(object_suffix) if object_suffix.isdigit() else None
    if object_id is not None:
        raw_records = [
            record
            for record in raw_records
            if int(record.get("obj_id", object_id)) == object_id
        ]
    grouped_raw: dict[
        tuple[int, int, int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for record in raw_records:
        key = (
            int(record["scene_id"]),
            int(record["img_id"]),
            int(record.get("obj_id", object_id or 0)),
            int(record.get("inst_id", 0)),
        )
        grouped_raw[key].append(record)
    selected_keys = sorted(grouped_raw)
    if args.limit_groups is not None:
        selected_keys = selected_keys[: args.limit_groups]
    selected_key_set = set(selected_keys)
    raw_records = [
        record
        for key in selected_keys
        for record in grouped_raw[key]
    ]

    detection_records = json.loads(args.detections.read_text(encoding="utf-8"))
    detections: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in detection_records:
        key = (
            int(record["scene_id"]),
            int(record["image_id"]),
            int(record["category_id"]),
        )
        if object_id is None or key[2] == object_id:
            detections[key].append(record)
    for group in detections.values():
        group.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    geometry = load_ascii_ply_geometry(
        object_config.cad_object.mesh_path,
        units_to_m=object_config.cad_object.units_to_m,
    )
    model_points = sample_model_points(
        geometry.vertices_m, settings.render_point_count
    )
    split_root = args.dataset_root / args.split

    @lru_cache(maxsize=24)
    def load_observation(scene_id: int, image_id: int):
        scene_root = split_root / f"{scene_id:06d}"
        camera_payload = json.loads(
            (scene_root / "scene_camera.json").read_text(encoding="utf-8")
        )[str(image_id)]
        raw_depth = cv2.imread(
            str(scene_root / "depth" / f"{image_id:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if raw_depth is None:
            raise FileNotFoundError(
                f"missing depth observation for {scene_id}/{image_id}"
            )
        depth_m = (
            raw_depth.astype(np.float64)
            * float(camera_payload.get("depth_scale", 1.0))
            / 1000.0
        )
        intrinsics = np.asarray(
            camera_payload["cam_K"], dtype=np.float64
        ).reshape(3, 3)
        return depth_m, intrinsics

    accepted_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    csv_rows = []
    rejection_counts: Counter[str] = Counter()
    soft_flag_counts: Counter[str] = Counter()
    start = time.perf_counter()
    for record in raw_records:
        scene_id = int(record["scene_id"])
        image_id = int(record["img_id"])
        current_object_id = int(record.get("obj_id", object_id or 0))
        instance_id = int(record.get("inst_id", 0))
        hypothesis_id = int(record.get("hypothesis_id", 0))
        depth_m, intrinsics = load_observation(scene_id, image_id)
        detection_group = detections.get(
            (scene_id, image_id, current_object_id), []
        )
        if instance_id >= len(detection_group):
            decision = _missing_mask_decision()
        else:
            mask = decode_coco_rle(
                detection_group[instance_id]["segmentation"]
            )
            mask = _opening(
                _align_mask(mask, depth_m.shape), args.mask_opening_kernel
            )
            decision = evaluate_pose_candidate(
                record,
                model_points,
                intrinsics,
                mask,
                depth_m,
                config=settings,
                transform_robot_camera=robot_camera,
            )
        if decision.accepted:
            accepted_records.append(reweighted_foundpose_record(record, decision))
        rejection_counts.update(decision.rejection_reasons)
        soft_flag_counts.update(decision.soft_quality_flags)
        identity = {
            "scene_id": scene_id,
            "image_id": image_id,
            "object_id": current_object_id,
            "instance_id": instance_id,
            "hypothesis_id": hypothesis_id,
        }
        audit = {
            **identity,
            "original_score": float(record.get("score", 0.0)),
            **decision.to_dict(),
            "evaluation_only": {
                "mssd_mm": record.get("mssd"),
                "mspd_px": record.get("mspd"),
                "mssd_normalized": record.get("mssd_n"),
            },
        }
        audit_records.append(audit)
        csv_rows.append(
            {
                **identity,
                "accepted": decision.accepted,
                "rejection_reasons": ";".join(decision.rejection_reasons),
                "soft_quality_flags": ";".join(decision.soft_quality_flags),
                "soft_pass": decision.soft_pass,
                "original_score": float(record.get("score", 0.0)),
                "geometric_quality": decision.geometric_quality,
                "robust_score": decision.robust_score,
                **{
                    key: decision.metrics.get(key)
                    for key in AUDIT_COLUMNS
                    if key in decision.metrics
                },
                "mssd_mm_evaluation_only": record.get("mssd"),
                "mspd_px_evaluation_only": record.get("mspd"),
            }
        )

    reliability_model = (
        None
        if args.reliability_model is None
        else PoseReliabilityModel.load(args.reliability_model)
    )
    if reliability_model is not None:
        cad_scale_m = float(
            np.linalg.norm(
                np.max(geometry.vertices_m, axis=0)
                - np.min(geometry.vertices_m, axis=0)
            )
        )
        audit_groups: dict[
            tuple[int, int, int, int], list[dict[str, Any]]
        ] = defaultdict(list)
        for audit in audit_records:
            audit_groups[
                (
                    int(audit["scene_id"]),
                    int(audit["image_id"]),
                    int(audit["object_id"]),
                    int(audit["instance_id"]),
                )
            ].append(audit)
        accepted_by_key = {
            (
                int(record["scene_id"]),
                int(record["img_id"]),
                int(record.get("obj_id", object_id or 0)),
                int(record.get("inst_id", 0)),
                int(record.get("hypothesis_id", 0)),
            ): record
            for record in accepted_records
        }
        for group_key, group in audit_groups.items():
            for audit in group:
                if not audit["accepted"]:
                    continue
                features = pose_reliability_features(audit, cad_scale_m, group)
                probability = float(reliability_model.predict_proba(features))
                feature_payload = dict(
                    zip(RELIABILITY_FEATURE_NAMES, features.tolist())
                )
                audit["calibrated_probability"] = probability
                audit["reliability_features"] = feature_payload
                record = accepted_by_key[(*group_key, int(audit["hypothesis_id"]))]
                record["heuristic_robust_score"] = float(record["score"])
                record["score"] = probability
                record["calibrated_probability"] = probability
                record["reliability_features"] = feature_payload
                record["source"] = "foundpose_coarse_rgbd_gated_calibrated"
        for row, audit in zip(csv_rows, audit_records):
            row["calibrated_probability"] = audit.get(
                "calibrated_probability"
            )

    gated_sets = foundpose_records_to_top_k(
        accepted_records,
        object_config.cad_object,
        top_k=args.top_k,
    )
    elapsed = time.perf_counter() - start
    accepted_group_keys = {
        (
            int(record["scene_id"]),
            int(record["img_id"]),
            int(record.get("obj_id", object_id or 0)),
            int(record.get("inst_id", 0)),
        )
        for record in accepted_records
    }
    summary = {
        "schema_version": 2,
        "method": (
            "foundpose_topk_minimal_hard_gate_continuous_rgbd_score"
            if reliability_model is None
            else "foundpose_topk_minimal_hard_gate_calibrated_reliability"
        ),
        "input": str(args.input.resolve()),
        "detections": str(args.detections.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "object_id": object_id,
        "config": settings.to_dict(),
        "robot_workspace_enabled": settings.robot_workspace_enabled,
        "raw_group_count": len(selected_key_set),
        "accepted_group_count": len(accepted_group_keys),
        "all_rejected_group_count": len(selected_key_set - accepted_group_keys),
        "raw_candidate_count": len(raw_records),
        "accepted_candidate_count": len(accepted_records),
        "rejected_candidate_count": len(raw_records) - len(accepted_records),
        "candidate_acceptance_rate": len(accepted_records) / max(len(raw_records), 1),
        "hard_gate_policy": (
            "reject only nonfinite/nonpositive/behind-camera/out-of-workspace "
            "poses and candidates without the required observation mask"
        ),
        "soft_feature_policy": (
            "PnP, silhouette, contour and depth thresholds emit diagnostic "
            "flags and continuous quality features; they do not reject"
        ),
        "rejection_reason_counts": dict(rejection_counts),
        "hard_rejection_reason_counts": dict(rejection_counts),
        "soft_quality_flag_counts": dict(soft_flag_counts),
        "soft_pass_candidate_count": sum(
            bool(record.get("soft_pass", False)) for record in audit_records
        ),
        "model_points_used": len(model_points),
        "runtime_s": elapsed,
        "gate_uses_ground_truth": False,
        "reliability_model": (
            None
            if args.reliability_model is None
            else {
                "path": str(args.reliability_model.resolve()),
                "sha256": hashlib.sha256(
                    args.reliability_model.read_bytes()
                ).hexdigest(),
                "ground_truth_used_at_runtime": False,
            }
        ),
        "evaluation_only": {
            "raw_score_top1": _selection_metrics(
                raw_records, "score", len(selected_key_set)
            ),
            (
                "gated_robust_top1"
                if reliability_model is None
                else "gated_calibrated_top1"
            ): _selection_metrics(
                accepted_records, "score", len(selected_key_set)
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gated-poses.json").write_text(
        json.dumps(accepted_records, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "pose-hypotheses-gated.json").write_text(
        json.dumps(hypothesis_sets_to_jsonable(gated_sets), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps({"summary": summary, "candidates": audit_records}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
