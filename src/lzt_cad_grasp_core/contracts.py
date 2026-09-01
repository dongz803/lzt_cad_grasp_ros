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
    return transform.copy()


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_camera_object",
            _transform(self.transform_camera_object, "transform_camera_object"),
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if self.covariance_se3 is not None:
            covariance = np.asarray(self.covariance_se3, dtype=np.float64)
            if covariance.shape != (6, 6):
                raise ValueError("covariance_se3 must have shape (6, 6)")
            object.__setattr__(self, "covariance_se3", covariance.copy())


@dataclass(frozen=True)
class GraspCandidate:
    """A parallel-jaw grasp expressed in the CAD object frame."""

    grasp_id: str
    transform_object_grasp: np.ndarray
    jaw_opening_m: float
    analytic_quality: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_object_grasp",
            _transform(self.transform_object_grasp, "transform_object_grasp"),
        )
        if self.jaw_opening_m <= 0.0:
            raise ValueError("jaw_opening_m must be positive")


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
