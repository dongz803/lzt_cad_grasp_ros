"""Calibrated pose-error uncertainty for symmetry-aware pose modes.

The covariance computed while clustering a pose mode describes only the spread
of the candidates that happened to enter that mode.  It is therefore a
``dispersion`` covariance and becomes zero for a singleton mode.  This module
keeps that quantity separate from a ``predictive`` covariance estimated from
ground-truth residuals on a calibration split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from statistics import NormalDist
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
    schema_version: int = 1
    approved_object_ids: tuple[str, ...] = ()
    rejected_object_ids: tuple[str, ...] = ()
    covariance_scale: float = 1.0
    nominal_coverage_levels: tuple[float, ...] = ()
    empirical_mahalanobis_thresholds: Mapping[str, float] = field(
        default_factory=dict
    )
    calibration_semantics: str = "predictive_only"
    model_approved: bool = False
    acceptance_policy: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

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
        if self.schema_version not in (1, 2):
            raise ValueError(f"unsupported predictive covariance schema {self.schema_version}")
        if not np.isfinite(self.covariance_scale) or self.covariance_scale <= 0.0:
            raise ValueError("covariance_scale must be finite and positive")
        approved = tuple(str(item) for item in self.approved_object_ids)
        rejected = tuple(str(item) for item in self.rejected_object_ids)
        known = set(self.object_ids)
        if not set(approved).issubset(known) or not set(rejected).issubset(known):
            raise ValueError("approved and rejected object IDs must be known objects")
        if set(approved) & set(rejected):
            raise ValueError("approved and rejected object IDs must be disjoint")
        if self.calibration_semantics not in (
            "predictive_only",
            "predictive_plus_candidate_dispersion",
        ):
            raise ValueError("unsupported covariance calibration semantics")
        thresholds = {
            str(key): float(value)
            for key, value in self.empirical_mahalanobis_thresholds.items()
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in thresholds.values()):
            raise ValueError("empirical Mahalanobis thresholds must be positive")
        levels = tuple(float(value) for value in self.nominal_coverage_levels)
        if any(not 0.0 < value < 1.0 for value in levels):
            raise ValueError("nominal coverage levels must lie in (0, 1)")
        if self.schema_version == 2:
            missing = [
                level
                for level in levels
                if f"{level:.4f}" not in thresholds
            ]
            if missing:
                raise ValueError(
                    "schema v2 is missing empirical thresholds for levels "
                    f"{missing}"
                )
            if self.model_approved and not approved:
                raise ValueError("an approved model requires approved object IDs")
        object.__setattr__(self, "covariance_se3", covariance)
        object.__setattr__(self, "observable_projector_se3", 0.5 * (projector + projector.T))
        object.__setattr__(self, "approved_object_ids", approved)
        object.__setattr__(self, "rejected_object_ids", rejected)
        object.__setattr__(self, "nominal_coverage_levels", levels)
        object.__setattr__(self, "empirical_mahalanobis_thresholds", thresholds)
        object.__setattr__(self, "limitations", tuple(self.limitations))

    def covariance_for(
        self,
        object_id: str,
        *,
        calibration_semantics: str | None = None,
        allow_unapproved: bool = False,
    ) -> np.ndarray:
        if object_id not in self.object_ids:
            raise ValueError(
                f"predictive covariance is not calibrated for object {object_id}"
            )
        if self.schema_version < 2 and not allow_unapproved:
            raise ValueError(
                "legacy schema v1 covariance has no coverage-based approval; "
                "explicit research override is required"
            )
        if self.schema_version >= 2 and not allow_unapproved:
            if not self.model_approved:
                raise ValueError("predictive covariance model is not approved")
            if object_id not in self.approved_object_ids:
                raise ValueError(
                    f"predictive covariance is not approved for object {object_id}"
                )
        if (
            calibration_semantics is not None
            and calibration_semantics != self.calibration_semantics
        ):
            raise ValueError(
                "online covariance semantics do not match offline calibration"
            )
        return self.covariance_scale * self.covariance_se3.copy()

    def empirical_threshold(self, nominal_coverage: float) -> float:
        key = f"{float(nominal_coverage):.4f}"
        if key not in self.empirical_mahalanobis_thresholds:
            raise ValueError(
                f"no empirical Mahalanobis threshold for coverage {key}"
            )
        return float(self.empirical_mahalanobis_thresholds[key])

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictivePoseCovarianceModel":
        schema_version = int(payload.get("schema_version", 1))
        validation = dict(payload.get("validation", {}))
        return cls(
            object_ids=tuple(str(item) for item in payload["object_ids"]),
            covariance_se3=np.asarray(payload["predictive_covariance_se3"], dtype=np.float64),
            observable_projector_se3=np.asarray(
                payload["observable_projector_se3"], dtype=np.float64
            ),
            method=str(payload["method"]),
            conditional_on=str(payload["conditional_on"]),
            training=dict(payload.get("training", {})),
            schema_version=schema_version,
            approved_object_ids=tuple(
                str(item) for item in payload.get("approved_object_ids", ())
            ),
            rejected_object_ids=tuple(
                str(item) for item in payload.get("rejected_object_ids", ())
            ),
            covariance_scale=float(payload.get("covariance_scale", 1.0)),
            nominal_coverage_levels=tuple(
                float(item) for item in payload.get("nominal_coverage_levels", ())
            ),
            empirical_mahalanobis_thresholds=dict(
                payload.get("empirical_mahalanobis_thresholds", {})
            ),
            calibration_semantics=str(
                payload.get("calibration_semantics", "predictive_only")
            ),
            model_approved=bool(
                payload.get("model_approved", validation.get("passed", False))
            ),
            acceptance_policy=dict(payload.get("acceptance_policy", {})),
            validation=validation,
            limitations=tuple(str(item) for item in payload.get("limitations", ())),
        )

    @classmethod
    def load(cls, path: Path | str) -> "PredictivePoseCovarianceModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "method": self.method,
            "conditional_on": self.conditional_on,
            "object_ids": list(self.object_ids),
            "predictive_covariance_se3": self.covariance_se3.tolist(),
            "observable_projector_se3": self.observable_projector_se3.tolist(),
            "training": dict(self.training),
        }
        if self.schema_version >= 2:
            payload.update(
                {
                    "approved_object_ids": list(self.approved_object_ids),
                    "rejected_object_ids": list(self.rejected_object_ids),
                    "covariance_scale": self.covariance_scale,
                    "nominal_coverage_levels": list(self.nominal_coverage_levels),
                    "empirical_mahalanobis_thresholds": dict(
                        self.empirical_mahalanobis_thresholds
                    ),
                    "calibration_semantics": self.calibration_semantics,
                    "model_approved": self.model_approved,
                    "acceptance_policy": dict(self.acceptance_policy),
                    "validation": dict(self.validation),
                    "limitations": list(self.limitations),
                }
            )
        return payload

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


def chi_square_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Approximate a chi-square quantile without a SciPy runtime dependency.

    The Wilson-Hilferty transform is accurate for the observable ranks used by
    this project (five and six).  Empirical conformal thresholds are stored as
    the authoritative online confidence-region thresholds; this approximation
    is used only to choose a covariance scale on the calibration split.
    """

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    dof = float(degrees_of_freedom)
    z_score = NormalDist().inv_cdf(probability)
    transformed = 1.0 - 2.0 / (9.0 * dof) + z_score * math.sqrt(2.0 / (9.0 * dof))
    return float(dof * max(transformed, 1e-12) ** 3)


def conformal_upper_quantile(values: np.ndarray, coverage: float) -> float:
    """Return the finite-sample split-conformal upper quantile."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("conformal values must be non-empty and finite")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    rank = min(int(math.ceil((len(array) + 1) * coverage)), len(array))
    return float(np.partition(array, rank - 1)[rank - 1])


def wilson_coverage_interval(
    successes: int,
    sample_count: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for empirical coverage."""

    if sample_count < 1 or not 0 <= successes <= sample_count:
        raise ValueError("invalid success and sample counts")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    estimate = successes / sample_count
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z_score * z_score / sample_count
    center = (
        estimate + z_score * z_score / (2.0 * sample_count)
    ) / denominator
    radius = z_score / denominator * math.sqrt(
        estimate * (1.0 - estimate) / sample_count
        + z_score * z_score / (4.0 * sample_count * sample_count)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def empirical_coverage_statistics(
    squared_distances: np.ndarray,
    threshold: float,
    nominal_coverage: float,
) -> dict[str, float | int]:
    """Summarize empirical coverage without deleting out-of-ellipsoid samples."""

    distances = np.asarray(squared_distances, dtype=np.float64).reshape(-1)
    if len(distances) == 0 or not np.all(np.isfinite(distances)):
        raise ValueError("squared distances must be non-empty and finite")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("coverage threshold must be finite and positive")
    successes = int(np.sum(distances <= threshold))
    observed = successes / len(distances)
    lower, upper = wilson_coverage_interval(successes, len(distances))
    return {
        "sample_count": len(distances),
        "covered_count": successes,
        "threshold_d2": float(threshold),
        "nominal_coverage": float(nominal_coverage),
        "observed_coverage": float(observed),
        "coverage_error": float(observed - nominal_coverage),
        "undercoverage": float(max(nominal_coverage - observed, 0.0)),
        "wilson_95_lower": float(lower),
        "wilson_95_upper": float(upper),
    }


def covariance_log_pseudodeterminant(
    covariance: np.ndarray,
    observable_projector: np.ndarray | None = None,
) -> float:
    """Return log pseudo-determinant in the observable covariance subspace."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a finite 6x6 matrix")
    projector = (
        np.eye(6)
        if observable_projector is None
        else np.asarray(observable_projector, dtype=np.float64)
    )
    matrix = projector @ (0.5 * (matrix + matrix.T)) @ projector
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive = eigenvalues[eigenvalues > 1e-14]
    if len(positive) == 0:
        raise ValueError("covariance has no observable positive eigenvalues")
    return float(np.sum(np.log(positive)))


def calibrate_covariance_scale(
    residuals: np.ndarray,
    base_covariance: np.ndarray,
    observable_projector: np.ndarray,
    *,
    dispersion_covariances: np.ndarray | None = None,
    nominal_coverage: float = 0.9,
    minimum_coverage: float | None = None,
    maximum_scale: float = 1e4,
    iterations: int = 60,
) -> tuple[float, dict[str, float | int]]:
    """Choose the smallest inflation scale meeting calibration coverage.

    Coverage is measured against the nominal chi-square threshold in the
    observable rank.  All calibration residuals remain in the calculation;
    points outside the ellipsoid count as failures.
    """

    values = np.asarray(residuals, dtype=np.float64)
    base = np.asarray(base_covariance, dtype=np.float64)
    projector = np.asarray(observable_projector, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
        raise ValueError("residuals must have shape (N, 6)")
    if base.shape != (6, 6) or projector.shape != (6, 6):
        raise ValueError("covariance and projector must have shape (6, 6)")
    dispersions = None
    if dispersion_covariances is not None:
        dispersions = np.asarray(dispersion_covariances, dtype=np.float64)
        if dispersions.shape != (len(values), 6, 6):
            raise ValueError("dispersion covariances must have shape (N, 6, 6)")
    target = nominal_coverage if minimum_coverage is None else minimum_coverage
    if not 0.0 < target <= nominal_coverage < 1.0:
        raise ValueError("invalid nominal or minimum coverage")
    if maximum_scale < 1.0 or iterations < 1:
        raise ValueError("invalid covariance scale search configuration")
    rank = int(np.linalg.matrix_rank(projector, tol=1e-10))
    threshold = chi_square_quantile(nominal_coverage, rank)

    def distances(scale: float) -> np.ndarray:
        result = []
        predictive = scale * base
        for index, residual in enumerate(values):
            covariance = predictive
            if dispersions is not None:
                covariance = combine_predictive_and_dispersion(
                    predictive, dispersions[index]
                )
            result.append(squared_mahalanobis(residual, covariance, projector))
        return np.asarray(result, dtype=np.float64)

    lower = 1.0
    lower_distances = distances(lower)
    if float(np.mean(lower_distances <= threshold)) >= target:
        return lower, empirical_coverage_statistics(
            lower_distances, threshold, nominal_coverage
        )
    upper = 2.0
    while upper <= maximum_scale:
        upper_distances = distances(upper)
        if float(np.mean(upper_distances <= threshold)) >= target:
            break
        lower = upper
        upper *= 2.0
    else:
        raise ValueError("no covariance inflation scale meets calibration coverage")
    if upper > maximum_scale:
        upper = maximum_scale
        upper_distances = distances(upper)
        if float(np.mean(upper_distances <= threshold)) < target:
            raise ValueError("no covariance inflation scale meets calibration coverage")

    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        middle_distances = distances(middle)
        if float(np.mean(middle_distances <= threshold)) >= target:
            upper = middle
        else:
            lower = middle
    final_distances = distances(upper)
    return upper, empirical_coverage_statistics(
        final_distances, threshold, nominal_coverage
    )
