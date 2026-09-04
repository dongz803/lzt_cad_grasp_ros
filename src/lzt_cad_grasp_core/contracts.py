"""Method-neutral data contracts for learned pose and task grasp research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


def _transform(value: np.ndarray, field_name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{field_name} must have shape (4, 4)")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{field_name} must contain finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{field_name} must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{field_name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{field_name} rotation must have determinant +1")
    return transform.copy()


@dataclass(frozen=True)
class ContinuousSymmetry:
    """A continuous rotational symmetry expressed in the CAD object frame."""

    axis_xyz: np.ndarray
    offset_m: np.ndarray

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis_xyz, dtype=np.float64)
        offset = np.asarray(self.offset_m, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("axis_xyz must contain three finite values")
        norm = np.linalg.norm(axis)
        if norm <= 1e-12:
            raise ValueError("axis_xyz must not be zero")
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("offset_m must contain three finite values")
        object.__setattr__(self, "axis_xyz", axis / norm)
        object.__setattr__(self, "offset_m", offset.copy())


@dataclass(frozen=True)
class CameraObservation:
    """One calibrated RGB-D observation with an optional target ROI mask."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8 with shape (height, width, 3)")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth_m must match the RGB image size")
        if intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3)")
        if self.mask is not None and np.asarray(self.mask).shape != depth.shape:
            raise ValueError("mask must match the RGB-D image size")


@dataclass(frozen=True)
class CADObject:
    """An onboarded industrial object and its symmetry transforms."""

    object_id: str
    mesh_path: Path
    units_to_m: float = 1.0
    symmetry_transforms: tuple[np.ndarray, ...] = ()
    continuous_symmetries: tuple[ContinuousSymmetry, ...] = ()

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must not be empty")
        if self.units_to_m <= 0.0:
            raise ValueError("units_to_m must be positive")
        for index, symmetry in enumerate(self.symmetry_transforms):
            _transform(symmetry, f"symmetry_transforms[{index}]")


@dataclass(frozen=True)
class PoseHypothesis:
    """One object-to-camera pose with confidence and optional uncertainty."""

    transform_camera_object: np.ndarray
    confidence: float
    symmetry_class: str = "identity"
    covariance_se3: np.ndarray | None = None
    source: str = "unknown"
    hypothesis_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_camera_object",
            _transform(self.transform_camera_object, "transform_camera_object"),
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if self.hypothesis_id < 0:
            raise ValueError("hypothesis_id must be non-negative")
        if self.covariance_se3 is not None:
            covariance = np.asarray(self.covariance_se3, dtype=np.float64)
            if covariance.shape != (6, 6):
                raise ValueError("covariance_se3 must have shape (6, 6)")
            if not np.all(np.isfinite(covariance)):
                raise ValueError("covariance_se3 must contain finite values")
            if not np.allclose(covariance, covariance.T, atol=1e-10):
                raise ValueError("covariance_se3 must be symmetric")
            if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
                raise ValueError("covariance_se3 must be positive semidefinite")
            object.__setattr__(self, "covariance_se3", covariance.copy())


@dataclass(frozen=True)
class PoseHypothesisSet:
    """Confidence-ordered Top-K hypotheses for one observed object instance."""

    object_id: str
    scene_id: int
    image_id: int
    instance_id: int
    hypotheses: tuple[PoseHypothesis, ...]
    covariance_se3: np.ndarray | None = None
    uncertainty_method: str = "unavailable"
    symmetry_discretization_deg: float | None = None

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must not be empty")
        if self.scene_id < 0 or self.image_id < 0 or self.instance_id < 0:
            raise ValueError("scene, image and instance IDs must be non-negative")
        if len(self.hypotheses) == 0:
            raise ValueError("hypotheses must not be empty")
        if (
            self.symmetry_discretization_deg is not None
            and self.symmetry_discretization_deg <= 0.0
        ):
            raise ValueError("symmetry_discretization_deg must be positive")
        confidences = [item.confidence for item in self.hypotheses]
        if any(a < b for a, b in zip(confidences, confidences[1:])):
            raise ValueError("hypotheses must be ordered by decreasing confidence")
        if self.covariance_se3 is not None:
            covariance = np.asarray(self.covariance_se3, dtype=np.float64)
            if covariance.shape != (6, 6):
                raise ValueError("covariance_se3 must have shape (6, 6)")
            if not np.allclose(covariance, covariance.T, atol=1e-10):
                raise ValueError("covariance_se3 must be symmetric")
            if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
                raise ValueError("covariance_se3 must be positive semidefinite")
            object.__setattr__(self, "covariance_se3", covariance.copy())


@dataclass(frozen=True)
class GraspCandidate:
    """A parallel-jaw grasp expressed in the CAD object frame."""

    grasp_id: str
    transform_object_grasp: np.ndarray
    jaw_opening_m: float
    analytic_quality: float = 0.0
    approach_distance_m: float = 0.08
    antipodal_score: float = 0.0
    contact_points_object_m: np.ndarray | None = None
    contact_normals_object: np.ndarray | None = None
    center_of_mass_offset_m: float = 0.0
    minimum_edge_distance_m: float = 0.0
    opening_margin_m: float = 0.0
    finger_collision_free: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_object_grasp",
            _transform(self.transform_object_grasp, "transform_object_grasp"),
        )
        if self.jaw_opening_m <= 0.0:
            raise ValueError("jaw_opening_m must be positive")
        if not 0.0 <= self.analytic_quality <= 1.0:
            raise ValueError("analytic_quality must lie in [0, 1]")
        if self.approach_distance_m <= 0.0:
            raise ValueError("approach_distance_m must be positive")
        if not 0.0 <= self.antipodal_score <= 1.0:
            raise ValueError("antipodal_score must lie in [0, 1]")
        for field_name in ("center_of_mass_offset_m", "minimum_edge_distance_m"):
            value = float(getattr(self, field_name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not np.isfinite(self.opening_margin_m):
            raise ValueError("opening_margin_m must be finite")
        if self.contact_points_object_m is not None:
            points = np.asarray(self.contact_points_object_m, dtype=np.float64)
            if points.shape != (2, 3) or not np.all(np.isfinite(points)):
                raise ValueError("contact_points_object_m must have shape (2, 3)")
            object.__setattr__(self, "contact_points_object_m", points.copy())
        if self.contact_normals_object is not None:
            normals = np.asarray(self.contact_normals_object, dtype=np.float64)
            if normals.shape != (2, 3) or not np.all(np.isfinite(normals)):
                raise ValueError("contact_normals_object must have shape (2, 3)")
            lengths = np.linalg.norm(normals, axis=1)
            if np.any(lengths < 1e-8):
                raise ValueError("contact_normals_object must be non-zero")
            object.__setattr__(
                self, "contact_normals_object", normals / lengths[:, None]
            )


@dataclass(frozen=True)
class PickPlaceTask:
    """Desired object placement and its task-space tolerances."""

    transform_world_object_goal: np.ndarray
    position_tolerance_m: float
    angle_tolerance_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_world_object_goal",
            _transform(
                self.transform_world_object_goal,
                "transform_world_object_goal",
            ),
        )
        if self.position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive")
        if self.angle_tolerance_deg <= 0.0:
            raise ValueError("angle_tolerance_deg must be positive")


class PoseEstimator(Protocol):
    """Replaceable traditional, learned or fused pose-estimation backend."""

    def estimate(
        self,
        observation: CameraObservation,
        cad_object: CADObject,
    ) -> list[PoseHypothesis]:
        """Return confidence-ordered, symmetry-aware pose hypotheses."""


class TaskGraspPlanner(Protocol):
    """Rank CAD grasps over all pose hypotheses and pick-place constraints."""

    def rank(
        self,
        pose_hypotheses: list[PoseHypothesis],
        grasp_candidates: list[GraspCandidate],
        task: PickPlaceTask,
    ) -> list[GraspCandidate]:
        """Return feasible candidates from most to least robust."""
