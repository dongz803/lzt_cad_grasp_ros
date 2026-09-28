"""CAD silhouette and RGB-D feedback refinement for coarse object poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pose_gating import contour_reprojection_error_px, render_model_points


def _homogeneous(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation must have determinant +1")
    return transform.copy()


def _axis_angle(axis: int, angle_rad: float) -> np.ndarray:
    vector = np.zeros(3, dtype=np.float64)
    vector[axis] = 1.0
    skew = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle_rad) * skew
        + (1.0 - np.cos(angle_rad)) * (skew @ skew)
    )


def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    left, _, right = np.linalg.svd(relative)
    relative = left @ right
    if np.linalg.det(relative) < 0.0:
        left[:, -1] *= -1.0
        relative = left @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


@dataclass(frozen=True)
class PoseRefinementConfig:
    """Multi-scale search and RGB-D objective settings."""

    translation_steps_m: tuple[float, ...] = (0.006, 0.002, 0.00075)
    rotation_steps_deg: tuple[float, ...] = (2.0, 0.75, 0.25)
    iterations_per_level: int = 2
    silhouette_weight: float = 0.30
    contour_weight: float = 0.20
    depth_error_weight: float = 0.35
    depth_inlier_weight: float = 0.15
    contour_error_scale_px: float = 30.0
    depth_error_scale_m: float = 0.05
    depth_inlier_tolerance_m: float = 0.02
    minimum_depth_overlap_pixels: int = 12
    render_splat_radius_px: int = 2
    maximum_translation_change_m: float = 0.025
    maximum_rotation_change_deg: float = 5.0
    regularization_weight: float = 0.01
    minimum_objective_improvement: float = 1e-4
    minimum_depth_m: float = 0.03

    def __post_init__(self) -> None:
        if len(self.translation_steps_m) != len(self.rotation_steps_deg):
            raise ValueError("translation and rotation schedules must have equal length")
        if not self.translation_steps_m:
            raise ValueError("refinement schedule must not be empty")
        if any(value <= 0.0 for value in self.translation_steps_m):
            raise ValueError("translation steps must be positive")
        if any(value <= 0.0 for value in self.rotation_steps_deg):
            raise ValueError("rotation steps must be positive")
        if self.iterations_per_level <= 0:
            raise ValueError("iterations_per_level must be positive")
        weights = (
            self.silhouette_weight,
            self.contour_weight,
            self.depth_error_weight,
            self.depth_inlier_weight,
        )
        if any(value < 0.0 for value in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("RGB-D objective weights must be non-negative and sum to 1")
        if self.contour_error_scale_px <= 0.0 or self.depth_error_scale_m <= 0.0:
            raise ValueError("error scales must be positive")
        if self.depth_inlier_tolerance_m <= 0.0:
            raise ValueError("depth_inlier_tolerance_m must be positive")
        if self.minimum_depth_overlap_pixels < 1:
            raise ValueError("minimum_depth_overlap_pixels must be positive")
        if self.render_splat_radius_px < 0:
            raise ValueError("render_splat_radius_px must be non-negative")
        if self.maximum_translation_change_m <= 0.0:
            raise ValueError("maximum_translation_change_m must be positive")
        if self.maximum_rotation_change_deg <= 0.0:
            raise ValueError("maximum_rotation_change_deg must be positive")
        if self.regularization_weight < 0.0:
            raise ValueError("regularization_weight must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation_steps_m": list(self.translation_steps_m),
            "rotation_steps_deg": list(self.rotation_steps_deg),
            "iterations_per_level": self.iterations_per_level,
            "silhouette_weight": self.silhouette_weight,
            "contour_weight": self.contour_weight,
            "depth_error_weight": self.depth_error_weight,
            "depth_inlier_weight": self.depth_inlier_weight,
            "contour_error_scale_px": self.contour_error_scale_px,
            "depth_error_scale_m": self.depth_error_scale_m,
            "depth_inlier_tolerance_m": self.depth_inlier_tolerance_m,
            "minimum_depth_overlap_pixels": self.minimum_depth_overlap_pixels,
            "render_splat_radius_px": self.render_splat_radius_px,
            "maximum_translation_change_m": self.maximum_translation_change_m,
            "maximum_rotation_change_deg": self.maximum_rotation_change_deg,
            "regularization_weight": self.regularization_weight,
            "minimum_objective_improvement": self.minimum_objective_improvement,
            "minimum_depth_m": self.minimum_depth_m,
        }


@dataclass(frozen=True)
class PoseAlignmentMetrics:
    """Traceable terms of the CAD/RGB-D alignment objective."""

    objective: float
    data_loss: float
    regularization: float
    silhouette_iou: float
    contour_error_px: float
    depth_overlap_pixels: int
    median_depth_error_m: float | None
    depth_inlier_fraction: float
    projected_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "data_loss": self.data_loss,
            "regularization": self.regularization,
            "silhouette_iou": self.silhouette_iou,
            "contour_error_px": self.contour_error_px,
            "depth_overlap_pixels": self.depth_overlap_pixels,
            "median_depth_error_m": self.median_depth_error_m,
            "depth_inlier_fraction": self.depth_inlier_fraction,
            "projected_pixels": self.projected_pixels,
        }


@dataclass(frozen=True)
class PoseRefinementResult:
    """Initial/refined pose pair and deterministic optimization audit."""

    initial_transform_camera_object: np.ndarray
    refined_transform_camera_object: np.ndarray
    initial_metrics: PoseAlignmentMetrics
    refined_metrics: PoseAlignmentMetrics
    accepted: bool
    evaluations: int
    accepted_steps: int
    translation_change_m: float
    rotation_change_deg: float

    @property
    def objective_improvement(self) -> float:
        return self.initial_metrics.objective - self.refined_metrics.objective

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "evaluations": self.evaluations,
            "accepted_steps": self.accepted_steps,
            "objective_improvement": self.objective_improvement,
            "translation_change_m": self.translation_change_m,
            "rotation_change_deg": self.rotation_change_deg,
            "initial_metrics": self.initial_metrics.to_dict(),
            "refined_metrics": self.refined_metrics.to_dict(),
            "initial_transform_camera_object": (
                self.initial_transform_camera_object.tolist()
            ),
            "refined_transform_camera_object": (
                self.refined_transform_camera_object.tolist()
            ),
        }


def evaluate_pose_alignment(
    model_points_m: np.ndarray,
    transform_camera_object: np.ndarray,
    camera_intrinsics: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    config: PoseRefinementConfig,
    initial_transform_camera_object: np.ndarray | None = None,
) -> PoseAlignmentMetrics:
    """Evaluate CAD silhouette/depth agreement for one object pose."""

    transform = _homogeneous(transform_camera_object, "transform_camera_object")
    initial = (
        transform
        if initial_transform_camera_object is None
        else _homogeneous(initial_transform_camera_object, "initial_transform")
    )
    points = np.asarray(model_points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("model_points_m must have shape (N, 3) and be finite")
    mask = np.asarray(observed_mask, dtype=bool)
    depth = np.asarray(observed_depth_m, dtype=np.float64)
    if mask.shape != depth.shape:
        raise ValueError("observed mask and depth must have equal image shape")

    rendered_mask, rendered_depth, _ = render_model_points(
        points,
        transform,
        np.asarray(camera_intrinsics, dtype=np.float64),
        mask.shape,
        splat_radius_px=config.render_splat_radius_px,
    )
    intersection = int(np.count_nonzero(rendered_mask & mask))
    union = int(np.count_nonzero(rendered_mask | mask))
    silhouette_iou = intersection / union if union else 0.0
    contour_error = contour_reprojection_error_px(rendered_mask, mask)
    if not np.isfinite(contour_error):
        contour_error = config.contour_error_scale_px

    valid_depth = np.isfinite(depth) & (depth >= config.minimum_depth_m)
    overlap = rendered_mask & mask & valid_depth & np.isfinite(rendered_depth)
    depth_errors = np.abs(rendered_depth[overlap] - depth[overlap])
    if depth_errors.size >= config.minimum_depth_overlap_pixels:
        median_depth_error = float(np.median(depth_errors))
        depth_inlier_fraction = float(
            np.mean(depth_errors <= config.depth_inlier_tolerance_m)
        )
    else:
        median_depth_error = None
        depth_inlier_fraction = 0.0

    silhouette_loss = 1.0 - silhouette_iou
    contour_loss = min(contour_error / config.contour_error_scale_px, 1.0)
    depth_error_loss = (
        1.0
        if median_depth_error is None
        else min(median_depth_error / config.depth_error_scale_m, 1.0)
    )
    depth_inlier_loss = 1.0 - depth_inlier_fraction
    data_loss = (
        config.silhouette_weight * silhouette_loss
        + config.contour_weight * contour_loss
        + config.depth_error_weight * depth_error_loss
        + config.depth_inlier_weight * depth_inlier_loss
    )
    translation_change = float(np.linalg.norm(transform[:3, 3] - initial[:3, 3]))
    rotation_change = _rotation_distance_rad(initial[:3, :3], transform[:3, :3])
    regularization = config.regularization_weight * (
        translation_change / config.maximum_translation_change_m
        + rotation_change / np.radians(config.maximum_rotation_change_deg)
    )
    return PoseAlignmentMetrics(
        objective=float(data_loss + regularization),
        data_loss=float(data_loss),
        regularization=float(regularization),
        silhouette_iou=float(silhouette_iou),
        contour_error_px=float(contour_error),
        depth_overlap_pixels=int(depth_errors.size),
        median_depth_error_m=median_depth_error,
        depth_inlier_fraction=depth_inlier_fraction,
        projected_pixels=int(np.count_nonzero(rendered_mask)),
    )


def refine_pose_rgbd(
    model_points_m: np.ndarray,
    initial_transform_camera_object: np.ndarray,
    camera_intrinsics: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    config: PoseRefinementConfig | None = None,
) -> PoseRefinementResult:
    """Refine a coarse pose by deterministic multi-scale SE(3) search."""

    settings = config or PoseRefinementConfig()
    initial = _homogeneous(
        initial_transform_camera_object, "initial_transform_camera_object"
    )
    best_transform = initial.copy()
    best_metrics = evaluate_pose_alignment(
        model_points_m,
        best_transform,
        camera_intrinsics,
        observed_mask,
        observed_depth_m,
        settings,
        initial,
    )
    initial_metrics = best_metrics
    evaluations = 1
    accepted_steps = 0

    # Use a bounded coarse-to-fine coordinate search.  This deliberately keeps
    # the module independent from a renderer-specific differentiable optimizer.
    for translation_step, rotation_step_deg in zip(
        settings.translation_steps_m, settings.rotation_steps_deg
    ):
        for _ in range(settings.iterations_per_level):
            proposals = []
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = best_transform.copy()
                    candidate[axis, 3] += sign * translation_step
                    proposals.append(candidate)
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = best_transform.copy()
                    candidate[:3, :3] = (
                        candidate[:3, :3]
                        @ _axis_angle(axis, np.radians(sign * rotation_step_deg))
                    )
                    proposals.append(candidate)

            improved_transform = best_transform
            improved_metrics = best_metrics
            for candidate in proposals:
                # A trust region prevents a noisy mask/depth patch from moving
                # an already plausible FoundPose hypothesis arbitrarily far.
                translation_change = float(
                    np.linalg.norm(candidate[:3, 3] - initial[:3, 3])
                )
                rotation_change_deg = float(
                    np.degrees(
                        _rotation_distance_rad(
                            initial[:3, :3], candidate[:3, :3]
                        )
                    )
                )
                if (
                    candidate[2, 3] < settings.minimum_depth_m
                    or translation_change > settings.maximum_translation_change_m
                    or rotation_change_deg > settings.maximum_rotation_change_deg
                ):
                    continue
                metrics = evaluate_pose_alignment(
                    model_points_m,
                    candidate,
                    camera_intrinsics,
                    observed_mask,
                    observed_depth_m,
                    settings,
                    initial,
                )
                evaluations += 1
                if metrics.objective < improved_metrics.objective:
                    improved_transform = candidate
                    improved_metrics = metrics
            if (
                best_metrics.objective - improved_metrics.objective
                < settings.minimum_objective_improvement
            ):
                break
            best_transform = improved_transform
            best_metrics = improved_metrics
            accepted_steps += 1

    accepted = (
        initial_metrics.objective - best_metrics.objective
        >= settings.minimum_objective_improvement
    )
    if not accepted:
        best_transform = initial
        best_metrics = initial_metrics
    translation_change = float(
        np.linalg.norm(best_transform[:3, 3] - initial[:3, 3])
    )
    rotation_change_deg = float(
        np.degrees(
            _rotation_distance_rad(initial[:3, :3], best_transform[:3, :3])
        )
    )
    return PoseRefinementResult(
        initial_transform_camera_object=initial,
        refined_transform_camera_object=best_transform,
        initial_metrics=initial_metrics,
        refined_metrics=best_metrics,
        accepted=accepted,
        evaluations=evaluations,
        accepted_steps=accepted_steps,
        translation_change_m=translation_change,
        rotation_change_deg=rotation_change_deg,
    )
