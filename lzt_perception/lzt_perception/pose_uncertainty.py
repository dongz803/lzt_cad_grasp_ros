"""Calibrated pose-error uncertainty for symmetry-aware pose modes.

The covariance computed while clustering a pose mode describes only the spread
of the candidates that happened to enter that mode.  It is therefore a
``dispersion`` covariance and becomes zero for a singleton mode.  This module
keeps that quantity separate from a ``predictive`` covariance estimated from
ground-truth residuals on a calibration split.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lzt_common.contracts import CADObject

from .pose_hypotheses import _rotation_vector
from .pose_modes import symmetry_aware_pose_distance


@dataclass(frozen=True)
class PredictivePoseCovarianceModel:
    """Frozen predictive pose-error covariance with an explicit validity set."""

    object_ids: tuple[str, ...]
    covariance_se3: np.ndarray
    observable_projector_se3: np.ndarray
    method: str
    conditional_on: str
    training: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.object_ids or any(not item for item in self.object_ids):
            raise ValueError("predictive covariance requires object IDs")
        covariance = np.asarray(self.covariance_se3, dtype=np.float64)
        projector = np.asarray(self.observable_projector_se3, dtype=np.float64)
        if covariance.shape != (6, 6) or projector.shape != (6, 6):
            raise ValueError("covariance and projector must have shape (6, 6)")
        if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(projector)):
            raise ValueError("covariance and projector must be finite")
        covariance = 0.5 * (covariance + covariance.T)
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-9:
            raise ValueError("predictive covariance must be positive semidefinite")
        object.__setattr__(self, "covariance_se3", covariance)
        object.__setattr__(self, "observable_projector_se3", 0.5 * (projector + projector.T))

    def covariance_for(self, object_id: str) -> np.ndarray:
        if object_id not in self.object_ids:
            raise ValueError(
                f"predictive covariance is not calibrated for object {object_id}"
            )
        return self.covariance_se3.copy()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictivePoseCovarianceModel":
        return cls(
            object_ids=tuple(str(item) for item in payload["object_ids"]),
            covariance_se3=np.asarray(payload["predictive_covariance_se3"], dtype=np.float64),
            observable_projector_se3=np.asarray(
                payload["observable_projector_se3"], dtype=np.float64
            ),
            method=str(payload["method"]),
            conditional_on=str(payload["conditional_on"]),
            training=dict(payload.get("training", {})),
        )

    @classmethod
    def load(cls, path: Path | str) -> "PredictivePoseCovarianceModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": self.method,
            "conditional_on": self.conditional_on,
            "object_ids": list(self.object_ids),
            "predictive_covariance_se3": self.covariance_se3.tolist(),
            "observable_projector_se3": self.observable_projector_se3.tolist(),
            "training": dict(self.training),
        }

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination


def symmetry_observable_projector(cad_object: CADObject) -> np.ndarray:
    """Return the 6D projector that removes unobservable symmetry rotations.

    Translation is retained.  Continuous-symmetry axes are expressed in the
    object frame, which is also the frame used for the rotational residual.
    """

    rotation_projector = np.eye(3)
    if cad_object.continuous_symmetries:
        axes = np.column_stack(
            [item.axis_xyz for item in cad_object.continuous_symmetries]
        )
        left, singular_values, _ = np.linalg.svd(axes, full_matrices=False)
        rank = int(np.sum(singular_values > 1e-10))
        if rank:
            basis = left[:, :rank]
            rotation_projector -= basis @ basis.T
    projector = np.zeros((6, 6), dtype=np.float64)
    projector[:3, :3] = np.eye(3)
    projector[3:, 3:] = rotation_projector
    return 0.5 * (projector + projector.T)


def symmetry_aware_pose_residual(
    predicted: np.ndarray,
    target: np.ndarray,
    cad_object: CADObject,
    continuous_symmetry_samples: int = 360,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return target-minus-prediction residual in the observable SE(3) space.

    The target is first mapped to the symmetry-equivalent pose nearest to the
    prediction.  Translation is expressed in the camera frame and rotation in
    the predicted object frame, matching the convention used by pose modes.
    """

    _, _, equivalent, symmetry_class = symmetry_aware_pose_distance(
        predicted,
        target,
        cad_object,
        continuous_symmetry_samples=continuous_symmetry_samples,
    )
    predicted_array = np.asarray(predicted, dtype=np.float64)
    translation = equivalent[:3, 3] - predicted_array[:3, 3]
    rotation = _rotation_vector(
        predicted_array[:3, :3].T @ equivalent[:3, :3]
    )
    residual = np.concatenate((translation, rotation))
    residual = symmetry_observable_projector(cad_object) @ residual
    return residual, equivalent, symmetry_class


def fit_predictive_covariance(
    residuals: np.ndarray,
    observable_projector: np.ndarray | None = None,
    shrinkage: float = 0.1,
    numerical_variance_floor: float = 1e-12,
) -> np.ndarray:
    """Fit an uncentred error covariance with unit-safe block shrinkage.

    The uncentred second moment includes systematic bias in the predicted error
    budget.  Translation (metres) and rotation (radians) are shrunk separately
    so quantities with different units are never averaged together.
    """

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] < 2:
        raise ValueError("residuals must have shape (N, 6) with N >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("residuals must be finite")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if numerical_variance_floor <= 0.0:
        raise ValueError("numerical_variance_floor must be positive")

    projector = (
        np.eye(6)
        if observable_projector is None
        else np.asarray(observable_projector, dtype=np.float64)
    )
    if projector.shape != (6, 6) or not np.all(np.isfinite(projector)):
        raise ValueError("observable_projector must be a finite 6x6 matrix")
    projector = 0.5 * (projector + projector.T)
    projected = values @ projector.T
    second_moment = projected.T @ projected / projected.shape[0]

    shrinkage_target = np.zeros((6, 6), dtype=np.float64)
    numerical_floor = np.zeros((6, 6), dtype=np.float64)
    for block in (slice(0, 3), slice(3, 6)):
        block_projector = projector[block, block]
        rank = int(np.linalg.matrix_rank(block_projector, tol=1e-10))
        if rank == 0:
            continue
        block_moment = second_moment[block, block]
        scale = max(
            float(np.trace(block_projector @ block_moment)) / rank,
            numerical_variance_floor,
        )
        shrinkage_target[block, block] = scale * block_projector
        numerical_floor[block, block] = (
            numerical_variance_floor * block_projector
        )

    covariance = (
        (1.0 - shrinkage) * second_moment
        + shrinkage * shrinkage_target
        + numerical_floor
    )
    covariance = projector @ covariance @ projector
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def combine_predictive_and_dispersion(
    predictive_covariance: np.ndarray,
    dispersion_covariance: np.ndarray,
) -> np.ndarray:
    """Add calibrated error and within-mode dispersion without conflating them."""

    predictive = np.asarray(predictive_covariance, dtype=np.float64)
    dispersion = np.asarray(dispersion_covariance, dtype=np.float64)
    if predictive.shape != (6, 6) or dispersion.shape != (6, 6):
        raise ValueError("both covariances must have shape (6, 6)")
    if not np.all(np.isfinite(predictive)) or not np.all(np.isfinite(dispersion)):
        raise ValueError("covariances must be finite")
    combined = 0.5 * (predictive + predictive.T)
    combined += 0.5 * (dispersion + dispersion.T)
    eigenvalues, eigenvectors = np.linalg.eigh(combined)
    if float(np.min(eigenvalues)) < -1e-9:
        raise ValueError("covariances must be positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def squared_mahalanobis(
    residual: np.ndarray,
    covariance: np.ndarray,
    observable_projector: np.ndarray | None = None,
) -> float:
    """Return a stable squared Mahalanobis distance in observable dimensions."""

    value = np.asarray(residual, dtype=np.float64).reshape(-1)
    matrix = np.asarray(covariance, dtype=np.float64)
    if value.shape != (6,) or matrix.shape != (6, 6):
        raise ValueError("residual and covariance must have shapes (6,) and (6, 6)")
    projector = (
        np.eye(6)
        if observable_projector is None
        else np.asarray(observable_projector, dtype=np.float64)
    )
    value = projector @ value
    matrix = projector @ (0.5 * (matrix + matrix.T)) @ projector
    return float(value @ np.linalg.pinv(matrix, rcond=1e-10) @ value)
