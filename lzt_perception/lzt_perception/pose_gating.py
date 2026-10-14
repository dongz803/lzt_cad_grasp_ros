"""Minimal physical gates and continuous RGB-D features for Top-K poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from .mask_evaluation import mask_pair_metrics


@dataclass(frozen=True)
class PoseGateConfig:
    """Hard physical bounds plus reference scales for continuous soft scores.

    Only impossible or non-executable poses are rejected.  The PnP, silhouette,
    contour and depth thresholds are retained as interpretable reference points
    for soft flags and normalized quality features; crossing one of them no
    longer removes a hypothesis.
    """

    minimum_depth_m: float = 0.05
    hard_minimum_positive_model_fraction: float = 0.50
    minimum_positive_model_fraction: float = 0.95
    camera_min_xyz_m: tuple[float, float, float] = (-1.0, -1.0, 0.05)
    camera_max_xyz_m: tuple[float, float, float] = (1.0, 1.0, 2.0)
    robot_min_xyz_m: tuple[float, float, float] | None = None
    robot_max_xyz_m: tuple[float, float, float] | None = None
    minimum_pnp_inliers: int = 6
    minimum_pnp_score: float = 0.05
    minimum_silhouette_iou: float = 0.05
    maximum_reprojection_error_px: float = 20.0
    minimum_depth_overlap_pixels: int = 12
    depth_inlier_tolerance_m: float = 0.02
    minimum_depth_inlier_fraction: float = 0.25
    maximum_median_depth_error_m: float = 0.05
    render_point_count: int = 12000
    render_splat_radius_px: int = 2

    def __post_init__(self) -> None:
        if self.minimum_depth_m <= 0.0:
            raise ValueError("minimum_depth_m must be positive")
        if not (
            0.0
            <= self.hard_minimum_positive_model_fraction
            <= self.minimum_positive_model_fraction
            <= 1.0
        ):
            raise ValueError(
                "positive-model fractions must lie in [0, 1] and the hard "
                "minimum must not exceed the soft reference"
            )
        camera_min = np.asarray(self.camera_min_xyz_m, dtype=np.float64)
        camera_max = np.asarray(self.camera_max_xyz_m, dtype=np.float64)
        if camera_min.shape != (3,) or camera_max.shape != (3,):
            raise ValueError("camera workspace bounds must contain three values")
        if np.any(camera_min >= camera_max):
            raise ValueError("camera workspace minimum must be below maximum")
        robot_bounds = (self.robot_min_xyz_m, self.robot_max_xyz_m)
        if (robot_bounds[0] is None) != (robot_bounds[1] is None):
            raise ValueError("both robot workspace bounds must be provided")
        if robot_bounds[0] is not None:
            robot_min = np.asarray(robot_bounds[0], dtype=np.float64)
            robot_max = np.asarray(robot_bounds[1], dtype=np.float64)
            if robot_min.shape != (3,) or robot_max.shape != (3,):
                raise ValueError("robot workspace bounds must contain three values")
            if np.any(robot_min >= robot_max):
                raise ValueError("robot workspace minimum must be below maximum")
        if self.minimum_pnp_inliers < 0:
            raise ValueError("minimum_pnp_inliers must be non-negative")
        unit_interval = (
            self.minimum_pnp_score,
            self.minimum_silhouette_iou,
            self.minimum_depth_inlier_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in unit_interval):
            raise ValueError("score, IoU, and inlier thresholds must lie in [0, 1]")
        if self.maximum_reprojection_error_px <= 0.0:
            raise ValueError("maximum_reprojection_error_px must be positive")
        if self.minimum_depth_overlap_pixels < 0:
            raise ValueError("minimum_depth_overlap_pixels must be non-negative")
        if self.depth_inlier_tolerance_m <= 0.0:
            raise ValueError("depth_inlier_tolerance_m must be positive")
        if self.maximum_median_depth_error_m <= 0.0:
            raise ValueError("maximum_median_depth_error_m must be positive")
        if self.render_point_count <= 0 or self.render_splat_radius_px < 0:
            raise ValueError("render point count/radius is invalid")

    @property
    def robot_workspace_enabled(self) -> bool:
        return self.robot_min_xyz_m is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class PoseGateDecision:
    """One hard-gate decision with traceable continuous quality evidence."""

    accepted: bool
    rejection_reasons: tuple[str, ...]
    soft_quality_flags: tuple[str, ...]
    soft_pass: bool
    metrics: Mapping[str, float | int | None]
    geometric_quality: float
    robust_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "hard_rejection_reasons": list(self.rejection_reasons),
            "soft_quality_flags": list(self.soft_quality_flags),
            "soft_pass": self.soft_pass,
            "metrics": dict(self.metrics),
            "geometric_quality": self.geometric_quality,
            "robust_score": self.robust_score,
        }


def foundpose_record_transform(
    record: Mapping[str, Any],
    translation_units_to_m: float = 0.001,
) -> np.ndarray:
    """Convert a FoundPose result record to ``T_camera_object``."""

    rotation = np.asarray(record["R"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(record["t"], dtype=np.float64).reshape(3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation * translation_units_to_m
    return transform


def sample_model_points(points_m: np.ndarray, maximum_points: int) -> np.ndarray:
    """Deterministically cap a dense CAD surface point set."""

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("model points must have shape (N, 3)")
    if len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError("model points must be non-empty and finite")
    if maximum_points <= 0:
        raise ValueError("maximum_points must be positive")
    if len(points) <= maximum_points:
        return points.copy()
    indices = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
    return points[indices]


def render_model_points(
    model_points_m: np.ndarray,
    transform_camera_object: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    splat_radius_px: int = 2,
    minimum_depth_m: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Render a deterministic point-splat silhouette and approximate z-buffer."""

    points = np.asarray(model_points_m, dtype=np.float64)
    transform = np.asarray(transform_camera_object, dtype=np.float64)
    camera = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("model points must have shape (N, 3)")
    if transform.shape != (4, 4) or camera.shape != (3, 3):
        raise ValueError("transform and intrinsics have invalid shapes")
    height, width = image_shape
    if height <= 0 or width <= 0 or splat_radius_px < 0:
        raise ValueError("image shape and splat radius must be valid")

    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    positive = camera_points[:, 2] > minimum_depth_m
    positive_count = int(np.count_nonzero(positive))
    positive_fraction = float(positive_count / max(len(points), 1))
    depth_map = np.full((height, width), np.inf, dtype=np.float64)
    if positive_count == 0:
        return (
            np.zeros((height, width), dtype=bool),
            depth_map,
            {
                "model_point_count": len(points),
                "positive_model_points": 0,
                "positive_model_fraction": positive_fraction,
                "projected_model_points": 0,
                "projected_model_fraction": 0.0,
            },
        )

    visible_points = camera_points[positive]
    projected = visible_points @ camera.T
    columns = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int64)
    rows = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int64)
    depths = visible_points[:, 2]
    inside = (
        (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    projected_count = int(np.count_nonzero(inside))
    columns = columns[inside]
    rows = rows[inside]
    depths = depths[inside]
    flat = depth_map.reshape(-1)
    for offset_y in range(-splat_radius_px, splat_radius_px + 1):
        for offset_x in range(-splat_radius_px, splat_radius_px + 1):
            if offset_x * offset_x + offset_y * offset_y > splat_radius_px**2:
                continue
            shifted_columns = columns + offset_x
            shifted_rows = rows + offset_y
            valid = (
                (shifted_columns >= 0)
                & (shifted_columns < width)
                & (shifted_rows >= 0)
                & (shifted_rows < height)
            )
            indices = shifted_rows[valid] * width + shifted_columns[valid]
            np.minimum.at(flat, indices, depths[valid])
    silhouette = np.isfinite(depth_map)
    if np.any(silhouette):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        silhouette = cv2.morphologyEx(
            silhouette.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    return (
        silhouette,
        depth_map,
        {
            "model_point_count": len(points),
            "positive_model_points": positive_count,
            "positive_model_fraction": positive_fraction,
            "projected_model_points": projected_count,
            "projected_model_fraction": float(
                projected_count / max(positive_count, 1)
            ),
        },
    )


def contour_reprojection_error_px(
    rendered_mask: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """Return symmetric contour Chamfer distance in pixels."""

    rendered = np.asarray(rendered_mask, dtype=bool)
    observed = np.asarray(observed_mask, dtype=bool)
    if rendered.shape != observed.shape:
        raise ValueError("rendered and observed masks must have the same shape")
    kernel = np.ones((3, 3), dtype=np.uint8)
    rendered_boundary = rendered & ~cv2.erode(
        rendered.astype(np.uint8), kernel
    ).astype(bool)
    observed_boundary = observed & ~cv2.erode(
        observed.astype(np.uint8), kernel
    ).astype(bool)
    if not np.any(rendered_boundary) or not np.any(observed_boundary):
        return float("inf")
    distance_to_observed = cv2.distanceTransform(
        (~observed_boundary).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_rendered = cv2.distanceTransform(
        (~rendered_boundary).astype(np.uint8), cv2.DIST_L2, 3
    )
    return float(
        0.5
        * (
            np.mean(distance_to_observed[rendered_boundary])
            + np.mean(distance_to_rendered[observed_boundary])
        )
    )


def _inside(point: np.ndarray, lower: Any, upper: Any) -> bool:
    return bool(
        np.all(point >= np.asarray(lower, dtype=np.float64))
        and np.all(point <= np.asarray(upper, dtype=np.float64))
    )


def _homogeneous(transform: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} must be homogeneous")
    return value


def _minimum_reference_quality(value: float, reference: float) -> float:
    """Map a larger-is-better measurement continuously into [0, 1]."""

    if reference <= 0.0:
        return 1.0
    return float(np.clip(value / reference, 0.0, 1.0))


def _maximum_reference_quality(value: float | None, reference: float) -> float:
    """Map a smaller-is-better error to a smooth positive quality score."""

    if value is None or not np.isfinite(value):
        return 0.0
    return float(np.exp(-max(value, 0.0) / reference))


def evaluate_pose_candidate(
    record: Mapping[str, Any],
    model_points_m: np.ndarray,
    intrinsics: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    config: PoseGateConfig | None = None,
    transform_robot_camera: np.ndarray | None = None,
) -> PoseGateDecision:
    """Reject impossible poses and score all other evidence continuously."""

    settings = config or PoseGateConfig()
    mask = np.asarray(observed_mask, dtype=bool)
    depth = np.asarray(observed_depth_m, dtype=np.float64)
    camera = np.asarray(intrinsics, dtype=np.float64)
    if mask.shape != depth.shape or camera.shape != (3, 3):
        raise ValueError("mask/depth shapes or intrinsics are invalid")
    try:
        transform = foundpose_record_transform(record)
    except (KeyError, TypeError, ValueError):
        return PoseGateDecision(
            accepted=False,
            rejection_reasons=("invalid_pose",),
            soft_quality_flags=(),
            soft_pass=False,
            metrics={},
            geometric_quality=0.0,
            robust_score=0.0,
        )
    if not np.all(np.isfinite(transform)):
        return PoseGateDecision(
            accepted=False,
            rejection_reasons=("nonfinite_pose",),
            soft_quality_flags=(),
            soft_pass=False,
            metrics={},
            geometric_quality=0.0,
            robust_score=0.0,
        )
    invalid_confidence = False
    try:
        confidence = float(record.get("score", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
        invalid_confidence = True
    try:
        pnp_inliers = int(record.get("inliers_est", 0))
    except (TypeError, ValueError):
        pnp_inliers = 0
    center_camera = transform[:3, 3]
    hard_reasons: list[str] = []
    soft_flags: list[str] = []
    if invalid_confidence or not np.isfinite(confidence):
        confidence = 0.0
        soft_flags.append("nonfinite_pnp_score")
    if center_camera[2] <= settings.minimum_depth_m:
        hard_reasons.append("nonpositive_depth")
    if not _inside(
        center_camera,
        settings.camera_min_xyz_m,
        settings.camera_max_xyz_m,
    ):
        hard_reasons.append("outside_camera_workspace")
    center_robot = None
    if settings.robot_workspace_enabled:
        if transform_robot_camera is None:
            hard_reasons.append("missing_robot_camera_transform")
        else:
            robot_camera = _homogeneous(
                transform_robot_camera, "transform_robot_camera"
            )
            center_robot = (robot_camera @ np.r_[center_camera, 1.0])[:3]
            if not _inside(
                center_robot,
                settings.robot_min_xyz_m,
                settings.robot_max_xyz_m,
            ):
                hard_reasons.append("outside_robot_workspace")
    if pnp_inliers < settings.minimum_pnp_inliers:
        soft_flags.append("insufficient_pnp_inliers")
    if confidence < settings.minimum_pnp_score:
        soft_flags.append("low_pnp_score")

    silhouette, rendered_depth, projection = render_model_points(
        model_points_m,
        transform,
        camera,
        mask.shape,
        splat_radius_px=settings.render_splat_radius_px,
        minimum_depth_m=settings.minimum_depth_m,
    )
    positive_fraction = float(projection["positive_model_fraction"])
    if positive_fraction < settings.hard_minimum_positive_model_fraction:
        hard_reasons.append("model_behind_camera")
    elif positive_fraction < settings.minimum_positive_model_fraction:
        soft_flags.append("low_positive_model_fraction")

    pair = mask_pair_metrics(silhouette, mask)
    silhouette_iou = float(pair["iou"])
    reprojection_error = contour_reprojection_error_px(silhouette, mask)
    if not np.any(silhouette):
        soft_flags.append("empty_model_projection")
    else:
        if silhouette_iou < settings.minimum_silhouette_iou:
            soft_flags.append("low_silhouette_iou")
        if reprojection_error > settings.maximum_reprojection_error_px:
            soft_flags.append("high_reprojection_error")

    valid_observed_depth = np.isfinite(depth) & (depth > 0.0)
    overlap = mask & silhouette & valid_observed_depth & np.isfinite(rendered_depth)
    depth_overlap_pixels = int(np.count_nonzero(overlap))
    if depth_overlap_pixels > 0:
        absolute_error = np.abs(depth[overlap] - rendered_depth[overlap])
        median_depth_error = float(np.median(absolute_error))
        depth_inlier_fraction = float(
            np.mean(absolute_error <= settings.depth_inlier_tolerance_m)
        )
    else:
        median_depth_error = None
        depth_inlier_fraction = 0.0
    if depth_overlap_pixels < settings.minimum_depth_overlap_pixels:
        soft_flags.append("insufficient_depth_overlap")
    else:
        if median_depth_error > settings.maximum_median_depth_error_m:
            soft_flags.append("high_depth_error")
        if depth_inlier_fraction < settings.minimum_depth_inlier_fraction:
            soft_flags.append("low_depth_inlier_fraction")

    pnp_inlier_quality = _minimum_reference_quality(
        float(pnp_inliers), float(settings.minimum_pnp_inliers)
    )
    silhouette_quality = float(np.clip(silhouette_iou, 0.0, 1.0))
    reprojection_quality = _maximum_reference_quality(
        reprojection_error, settings.maximum_reprojection_error_px
    )
    depth_overlap_quality = _minimum_reference_quality(
        float(depth_overlap_pixels), float(settings.minimum_depth_overlap_pixels)
    )
    depth_error_quality = _maximum_reference_quality(
        median_depth_error, settings.maximum_median_depth_error_m
    )
    depth_inlier_quality = float(np.clip(depth_inlier_fraction, 0.0, 1.0))
    positive_fraction_span = (
        settings.minimum_positive_model_fraction
        - settings.hard_minimum_positive_model_fraction
    )
    positive_model_quality = (
        1.0
        if positive_fraction_span <= 1e-12
        else float(
            np.clip(
                (
                    positive_fraction
                    - settings.hard_minimum_positive_model_fraction
                )
                / positive_fraction_span,
                0.0,
                1.0,
            )
        )
    )
    quality_terms = np.asarray(
        [
            positive_model_quality,
            pnp_inlier_quality,
            silhouette_quality,
            reprojection_quality,
            depth_overlap_quality,
            depth_error_quality,
            depth_inlier_quality,
        ],
        dtype=np.float64,
    )
    geometric_quality = float(
        np.exp(np.mean(np.log(np.clip(quality_terms, 1e-6, 1.0))))
    )
    accepted = len(hard_reasons) == 0
    soft_pass = accepted and len(soft_flags) == 0
    robust_score = (
        float(np.clip(confidence, 0.0, 1.0) * geometric_quality)
        if accepted
        else 0.0
    )
    metrics: dict[str, float | int | None] = {
        "translation_x_m": float(center_camera[0]),
        "translation_y_m": float(center_camera[1]),
        "translation_z_m": float(center_camera[2]),
        "robot_x_m": None if center_robot is None else float(center_robot[0]),
        "robot_y_m": None if center_robot is None else float(center_robot[1]),
        "robot_z_m": None if center_robot is None else float(center_robot[2]),
        "pnp_inliers": pnp_inliers,
        "pnp_score": confidence,
        "positive_model_fraction": positive_fraction,
        "projected_model_fraction": float(
            projection["projected_model_fraction"]
        ),
        "rendered_silhouette_pixels": int(np.count_nonzero(silhouette)),
        "observed_mask_pixels": int(np.count_nonzero(mask)),
        "silhouette_iou": silhouette_iou,
        "silhouette_precision": float(pair["pixel_precision"]),
        "silhouette_recall": float(pair["pixel_recall"]),
        "silhouette_boundary_f1": float(pair["boundary_f1"]),
        "silhouette_center_error_px": pair["center_error_px"],
        "reprojection_error_px": (
            reprojection_error if np.isfinite(reprojection_error) else None
        ),
        "depth_overlap_pixels": depth_overlap_pixels,
        "median_depth_error_m": median_depth_error,
        "depth_inlier_fraction": depth_inlier_fraction,
        "pnp_inlier_quality": pnp_inlier_quality,
        "silhouette_quality": silhouette_quality,
        "reprojection_quality": reprojection_quality,
        "depth_overlap_quality": depth_overlap_quality,
        "depth_error_quality": depth_error_quality,
        "depth_inlier_quality": depth_inlier_quality,
        "positive_model_quality": positive_model_quality,
    }
    return PoseGateDecision(
        accepted=accepted,
        rejection_reasons=tuple(hard_reasons),
        soft_quality_flags=tuple(soft_flags),
        soft_pass=soft_pass,
        metrics=metrics,
        geometric_quality=geometric_quality,
        robust_score=robust_score,
    )


def reweighted_foundpose_record(
    record: Mapping[str, Any], decision: PoseGateDecision
) -> dict[str, Any]:
    """Copy an accepted record and replace its score with the robust weight."""

    if not decision.accepted:
        raise ValueError("cannot reweight a rejected pose")
    result = dict(record)
    result["original_score"] = float(record.get("score", 0.0))
    result["geometric_quality"] = decision.geometric_quality
    result["pose_gate_soft_flags"] = list(decision.soft_quality_flags)
    result["pose_gate_soft_pass"] = decision.soft_pass
    result["score"] = decision.robust_score
    result["source"] = "foundpose_coarse_rgbd_gated"
    result["pose_gate_metrics"] = dict(decision.metrics)
    return result
