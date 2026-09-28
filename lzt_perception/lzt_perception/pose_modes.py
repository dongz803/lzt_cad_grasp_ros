"""Symmetry-aware multi-modal clustering for Top-K 6D pose hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lzt_common.contracts import CADObject, PoseHypothesis, PoseHypothesisSet

from .pose_hypotheses import _closest_equivalent, _rotation_vector


def _transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation must have determinant +1")
    return transform.copy()


def _covariance(value: np.ndarray) -> np.ndarray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance_se3 must be a finite 6x6 matrix")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) < -1e-9:
        raise ValueError("covariance_se3 must be positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


@dataclass(frozen=True)
class PoseModeClusteringConfig:
    """Thresholds for deterministic symmetry-quotient pose clustering."""

    maximum_translation_distance_m: float = 0.03
    maximum_rotation_distance_deg: float = 15.0
    continuous_symmetry_samples: int = 360

    def __post_init__(self) -> None:
        if self.maximum_translation_distance_m <= 0.0:
            raise ValueError("maximum_translation_distance_m must be positive")
        if not 0.0 < self.maximum_rotation_distance_deg <= 180.0:
            raise ValueError(
                "maximum_rotation_distance_deg must lie in (0, 180]"
            )
        if self.continuous_symmetry_samples < 2:
            raise ValueError("continuous_symmetry_samples must be at least two")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_translation_distance_m": self.maximum_translation_distance_m,
            "maximum_rotation_distance_deg": self.maximum_rotation_distance_deg,
            "continuous_symmetry_samples": self.continuous_symmetry_samples,
        }


@dataclass(frozen=True)
class PoseMode:
    """One non-equivalent pose mode with local SE(3) uncertainty."""

    mode_id: int
    weight: float
    transform_camera_object: np.ndarray
    covariance_se3: np.ndarray
    representative_hypothesis_id: int
    member_hypothesis_ids: tuple[int, ...]
    member_weights: tuple[float, ...]
    symmetry_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.mode_id < 0 or self.representative_hypothesis_id < 0:
            raise ValueError("mode and hypothesis IDs must be non-negative")
        if not -1e-12 <= self.weight <= 1.0 + 1e-12:
            raise ValueError("mode weight must lie in [0, 1]")
        object.__setattr__(self, "weight", float(np.clip(self.weight, 0.0, 1.0)))
        if not self.member_hypothesis_ids:
            raise ValueError("a pose mode must contain at least one hypothesis")
        if not (
            len(self.member_hypothesis_ids)
            == len(self.member_weights)
            == len(self.symmetry_classes)
        ):
            raise ValueError("mode member fields must have equal lengths")
        if any(weight < 0.0 for weight in self.member_weights):
            raise ValueError("member weights must be non-negative")
        if not np.isclose(sum(self.member_weights), 1.0, atol=1e-8):
            raise ValueError("member weights must sum to one within each mode")
        object.__setattr__(
            self,
            "transform_camera_object",
            _transform(self.transform_camera_object, "transform_camera_object"),
        )
        object.__setattr__(self, "covariance_se3", _covariance(self.covariance_se3))


@dataclass(frozen=True)
class PoseMixture:
    """A confidence-ordered mixture of non-equivalent object-pose modes."""

    object_id: str
    scene_id: int
    image_id: int
    instance_id: int
    modes: tuple[PoseMode, ...]
    clustering_config: PoseModeClusteringConfig
    source_hypothesis_count: int
    uncertainty_method: str = "symmetry_quotient_weighted_se3_mixture"

    def __post_init__(self) -> None:
        if not self.object_id or not self.modes:
            raise ValueError("pose mixture requires an object ID and at least one mode")
        if min(self.scene_id, self.image_id, self.instance_id) < 0:
            raise ValueError("scene, image and instance IDs must be non-negative")
        if self.source_hypothesis_count < len(self.modes):
            raise ValueError("source hypothesis count cannot be below mode count")
        weights = [mode.weight for mode in self.modes]
        if any(first < second for first, second in zip(weights, weights[1:])):
            raise ValueError("modes must be ordered by decreasing weight")
        if not np.isclose(sum(weights), 1.0, atol=1e-8):
            raise ValueError("mode weights must sum to one")


def symmetry_aware_pose_distance(
    first: np.ndarray,
    second: np.ndarray,
    cad_object: CADObject,
    continuous_symmetry_samples: int = 360,
) -> tuple[float, float, np.ndarray, str]:
    """Return translation/rotation distance after symmetry alignment.

    ``second`` is mapped to the symmetry-equivalent representation nearest to
    ``first``.  Translation distance is in metres and rotation distance is in
    degrees.
    """

    reference = _transform(first, "first")
    candidate = _transform(second, "second")
    equivalent, symmetry_class = _closest_equivalent(
        candidate,
        reference,
        cad_object,
        continuous_samples=continuous_symmetry_samples,
    )
    translation_distance = float(
        np.linalg.norm(equivalent[:3, 3] - reference[:3, 3])
    )
    rotation_distance_deg = float(
        np.degrees(
            np.linalg.norm(
                _rotation_vector(reference[:3, :3].T @ equivalent[:3, :3])
            )
        )
    )
    return (
        translation_distance,
        rotation_distance_deg,
        equivalent,
        symmetry_class,
    )


def _weighted_rotation(rotations: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    matrix = sum(
        weight * rotation for weight, rotation in zip(weights, rotations)
    )
    left, _, right = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def _mode_statistics(
    members: list[PoseHypothesis],
    global_weights: dict[int, float],
    cad_object: CADObject,
    continuous_symmetry_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[float, str]]]:
    representative = max(members, key=lambda item: item.confidence)
    aligned: list[np.ndarray] = []
    classes: list[str] = []
    for member in members:
        _, _, equivalent, symmetry_class = symmetry_aware_pose_distance(
            representative.transform_camera_object,
            member.transform_camera_object,
            cad_object,
            continuous_symmetry_samples,
        )
        aligned.append(equivalent)
        classes.append(symmetry_class)

    raw_weights = np.asarray(
        [global_weights[item.hypothesis_id] for item in members],
        dtype=np.float64,
    )
    local_weights = raw_weights / np.sum(raw_weights)
    mean = np.eye(4)
    mean[:3, 3] = np.sum(
        local_weights[:, None]
        * np.asarray([transform[:3, 3] for transform in aligned]),
        axis=0,
    )
    mean[:3, :3] = _weighted_rotation(
        [transform[:3, :3] for transform in aligned], local_weights
    )
    residuals = []
    for transform in aligned:
        translation = transform[:3, 3] - mean[:3, 3]
        rotation = _rotation_vector(mean[:3, :3].T @ transform[:3, :3])
        residuals.append(np.concatenate((translation, rotation)))
    residual_array = np.asarray(residuals, dtype=np.float64)
    covariance = (
        residual_array * local_weights[:, None]
    ).T @ residual_array
    covariance = 0.5 * (covariance + covariance.T)
    metadata = {
        member.hypothesis_id: (float(weight), symmetry_class)
        for member, weight, symmetry_class in zip(members, local_weights, classes)
    }
    return mean, covariance, metadata


def cluster_pose_hypotheses(
    hypothesis_set: PoseHypothesisSet,
    cad_object: CADObject,
    config: PoseModeClusteringConfig | None = None,
) -> PoseMixture:
    """Cluster Top-K poses into deterministic non-equivalent pose modes."""

    settings = config or PoseModeClusteringConfig()
    hypotheses = sorted(
        hypothesis_set.hypotheses,
        key=lambda item: (-item.confidence, item.hypothesis_id),
    )
    confidences = np.asarray(
        [max(item.confidence, 0.0) for item in hypotheses], dtype=np.float64
    )
    if np.sum(confidences) <= 1e-12:
        confidences = np.ones_like(confidences)
    confidences /= np.sum(confidences)
    global_weights = {
        hypothesis.hypothesis_id: float(weight)
        for hypothesis, weight in zip(hypotheses, confidences)
    }

    clusters: list[list[PoseHypothesis]] = []
    for hypothesis in hypotheses:
        best_cluster = None
        best_distance = np.inf
        for index, cluster in enumerate(clusters):
            representative = max(cluster, key=lambda item: item.confidence)
            translation, rotation, _, _ = symmetry_aware_pose_distance(
                representative.transform_camera_object,
                hypothesis.transform_camera_object,
                cad_object,
                settings.continuous_symmetry_samples,
            )
            if (
                translation <= settings.maximum_translation_distance_m
                and rotation <= settings.maximum_rotation_distance_deg
            ):
                normalized = (
                    translation / settings.maximum_translation_distance_m
                    + rotation / settings.maximum_rotation_distance_deg
                )
                if normalized < best_distance:
                    best_distance = normalized
                    best_cluster = index
        if best_cluster is None:
            clusters.append([hypothesis])
        else:
            clusters[best_cluster].append(hypothesis)

    unsorted_modes = []
    for cluster in clusters:
        representative = max(cluster, key=lambda item: item.confidence)
        mean, covariance, metadata = _mode_statistics(
            cluster,
            global_weights,
            cad_object,
            settings.continuous_symmetry_samples,
        )
        mode_weight = float(
            sum(global_weights[item.hypothesis_id] for item in cluster)
        )
        unsorted_modes.append(
            (
                mode_weight,
                representative,
                tuple(sorted(cluster, key=lambda item: item.hypothesis_id)),
                mean,
                covariance,
                metadata,
            )
        )
    unsorted_modes.sort(key=lambda item: (-item[0], item[1].hypothesis_id))
    total_mode_weight = sum(item[0] for item in unsorted_modes)
    modes = []
    for mode_id, (
        mode_weight,
        representative,
        members,
        mean,
        covariance,
        metadata,
    ) in enumerate(unsorted_modes):
        modes.append(
            PoseMode(
                mode_id=mode_id,
                weight=mode_weight / total_mode_weight,
                transform_camera_object=mean,
                covariance_se3=covariance,
                representative_hypothesis_id=representative.hypothesis_id,
                member_hypothesis_ids=tuple(
                    item.hypothesis_id for item in members
                ),
                member_weights=tuple(
                    metadata[item.hypothesis_id][0] for item in members
                ),
                symmetry_classes=tuple(
                    metadata[item.hypothesis_id][1] for item in members
                ),
            )
        )
    return PoseMixture(
        object_id=hypothesis_set.object_id,
        scene_id=hypothesis_set.scene_id,
        image_id=hypothesis_set.image_id,
        instance_id=hypothesis_set.instance_id,
        modes=tuple(modes),
        clustering_config=settings,
        source_hypothesis_count=len(hypotheses),
    )


def pose_mixtures_to_jsonable(
    mixtures: list[PoseMixture],
) -> list[dict[str, Any]]:
    """Serialize pose mixtures without losing mode/member provenance."""

    return [
        {
            "object_id": mixture.object_id,
            "scene_id": mixture.scene_id,
            "image_id": mixture.image_id,
            "instance_id": mixture.instance_id,
            "uncertainty_method": mixture.uncertainty_method,
            "source_hypothesis_count": mixture.source_hypothesis_count,
            "mode_count": len(mixture.modes),
            "clustering_config": mixture.clustering_config.to_dict(),
            "modes": [
                {
                    "mode_id": mode.mode_id,
                    "weight": mode.weight,
                    "representative_hypothesis_id": (
                        mode.representative_hypothesis_id
                    ),
                    "member_hypothesis_ids": list(mode.member_hypothesis_ids),
                    "member_weights": list(mode.member_weights),
                    "symmetry_classes": list(mode.symmetry_classes),
                    "transform_camera_object": (
                        mode.transform_camera_object.tolist()
                    ),
                    "covariance_se3": mode.covariance_se3.tolist(),
                }
                for mode in mixture.modes
            ],
        }
        for mixture in mixtures
    ]
