"""FoundPose-to-project Top-K conversion with symmetry-aware uncertainty."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import CADObject, ContinuousSymmetry, PoseHypothesis, PoseHypothesisSet


def _axis_angle_transform(symmetry: ContinuousSymmetry, angle: float) -> np.ndarray:
    axis = symmetry.axis_xyz
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = symmetry.offset_m - rotation @ symmetry.offset_m
    return transform


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return 0.5 * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
    if np.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            axis *= -1.0
        return axis * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def _symmetry_variants(
    transform: np.ndarray,
    cad_object: CADObject,
    continuous_samples: int,
) -> list[tuple[np.ndarray, str]]:
    discrete = [(np.eye(4), "identity")]
    discrete.extend(
        (symmetry, f"discrete_{index}")
        for index, symmetry in enumerate(cad_object.symmetry_transforms)
    )
    variants = [(transform @ symmetry, label) for symmetry, label in discrete]
    for symmetry_index, symmetry in enumerate(cad_object.continuous_symmetries):
        for sample in range(1, continuous_samples):
            angle = 2.0 * np.pi * sample / continuous_samples
            continuous = _axis_angle_transform(symmetry, angle)
            for discrete_transform, discrete_label in discrete:
                variants.append(
                    (
                        transform @ discrete_transform @ continuous,
                        (
                            f"{discrete_label}+continuous_{symmetry_index}:"
                            f"{angle:.6f}"
                        ),
                    )
                )
    return variants


def _closest_equivalent(
    transform: np.ndarray,
    reference: np.ndarray,
    cad_object: CADObject,
    continuous_samples: int = 360,
) -> tuple[np.ndarray, str]:
    best_transform = transform
    best_label = "identity"
    best_angle = np.inf
    for variant, label in _symmetry_variants(
        transform, cad_object, continuous_samples
    ):
        relative = reference[:3, :3].T @ variant[:3, :3]
        angle = np.linalg.norm(_rotation_vector(relative))
        if angle < best_angle:
            best_transform = variant
            best_label = label
            best_angle = angle
    return best_transform, best_label


def symmetry_aware_covariance(
    hypotheses: tuple[PoseHypothesis, ...],
    cad_object: CADObject,
) -> np.ndarray | None:
    """Estimate an SE(3) covariance after mapping poses to symmetry equivalents."""

    if len(hypotheses) < 2:
        return None
    reference = hypotheses[0].transform_camera_object
    residuals = []
    scores = []
    for hypothesis in hypotheses:
        equivalent, _ = _closest_equivalent(
            hypothesis.transform_camera_object, reference, cad_object
        )
        translation = equivalent[:3, 3] - reference[:3, 3]
        rotation = _rotation_vector(reference[:3, :3].T @ equivalent[:3, :3])
        residuals.append(np.concatenate((translation, rotation)))
        scores.append(hypothesis.confidence)
    residual_array = np.asarray(residuals)
    weights = np.asarray(scores, dtype=np.float64)
    if np.sum(weights) <= 1e-12:
        weights = np.ones_like(weights)
    weights /= np.sum(weights)
    mean = np.sum(weights[:, None] * residual_array, axis=0)
    centered = residual_array - mean
    covariance = (centered * weights[:, None]).T @ centered
    return 0.5 * (covariance + covariance.T)


def _record_transform(record: dict[str, Any], units_to_m: float) -> np.ndarray:
    rotation = np.asarray(record["R"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(record["t"], dtype=np.float64).reshape(3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation * units_to_m
    return transform


def foundpose_records_to_top_k(
    records: list[dict[str, Any]],
    cad_object: CADObject,
    top_k: int = 5,
    translation_units_to_m: float = 0.001,
    expand_symmetry: bool = False,
) -> list[PoseHypothesisSet]:
    """Group FoundPose JSON records and emit project-native Top-K sets.

    By default only genuinely estimated PnP candidates are retained. Symmetry
    equivalents are used when covariance is canonicalized, but are not
    fabricated as independent uncertainty samples. ``expand_symmetry`` remains
    available for visualization-only callers and is labelled accordingly.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for record in records:
        configured_suffix = cad_object.object_id.rsplit("_", 1)[-1]
        configured_object_id = (
            int(configured_suffix) if configured_suffix.isdigit() else None
        )
        record_object_id = (
            int(record["obj_id"]) if "obj_id" in record else None
        )
        if (
            configured_object_id is not None
            and record_object_id is not None
            and record_object_id != configured_object_id
        ):
            raise ValueError(
                f"result obj_id={record_object_id} does not match "
                f"configured object {cad_object.object_id}"
            )
        key = (
            int(record["scene_id"]),
            int(record["img_id"]),
            int(record.get("inst_id", 0)),
        )
        grouped.setdefault(key, []).append(record)

    result = []
    for (scene_id, image_id, instance_id), group in sorted(grouped.items()):
        group.sort(key=lambda record: float(record.get("score", 0.0)), reverse=True)
        hypotheses = [
            PoseHypothesis(
                transform_camera_object=_record_transform(
                    record, translation_units_to_m
                ),
                confidence=float(np.clip(float(record.get("score", 0.0)), 0.0, 1.0)),
                symmetry_class="identity",
                source="foundpose_coarse",
                hypothesis_id=int(record.get("hypothesis_id", index)),
            )
            for index, record in enumerate(group[:top_k])
        ]
        if hypotheses:
            reference = hypotheses[0].transform_camera_object
            hypotheses = [
                replace(
                    hypothesis,
                    transform_camera_object=equivalent,
                    symmetry_class=label,
                )
                for hypothesis in hypotheses
                for equivalent, label in [
                    _closest_equivalent(
                        hypothesis.transform_camera_object,
                        reference,
                        cad_object,
                    )
                ]
            ]
        if expand_symmetry and hypotheses and len(hypotheses) < top_k:
            base = hypotheses[0]
            variants = _symmetry_variants(
                base.transform_camera_object,
                cad_object,
                continuous_samples=max(top_k, 2),
            )
            for transform, label in variants:
                if label == "identity":
                    continue
                hypotheses.append(
                    PoseHypothesis(
                        transform_camera_object=transform,
                        confidence=base.confidence,
                        symmetry_class=label,
                        source="foundpose_symmetry_equivalent",
                        hypothesis_id=len(hypotheses),
                    )
                )
                if len(hypotheses) == top_k:
                    break
        hypotheses = hypotheses[:top_k]
        hypotheses.sort(key=lambda item: item.confidence, reverse=True)
        hypothesis_tuple = tuple(hypotheses)
        covariance = symmetry_aware_covariance(hypothesis_tuple, cad_object)
        if covariance is not None:
            hypothesis_tuple = tuple(
                replace(hypothesis, covariance_se3=covariance)
                for hypothesis in hypothesis_tuple
            )
        result.append(
            PoseHypothesisSet(
                object_id=cad_object.object_id,
                scene_id=scene_id,
                image_id=image_id,
                instance_id=instance_id,
                hypotheses=hypothesis_tuple,
                covariance_se3=covariance,
                uncertainty_method=(
                    "weighted_top_k_symmetry_canonicalized"
                    if len(group) >= 2 and covariance is not None
                    else (
                        "symmetry_equivalents_only_no_statistical_uncertainty"
                        if expand_symmetry and covariance is not None
                        else "unavailable_single_hypothesis"
                    )
                ),
                symmetry_discretization_deg=(
                    1.0 if cad_object.continuous_symmetries else None
                ),
            )
        )
    return result


def load_foundpose_top_k(
    path: Path | str,
    cad_object: CADObject,
    top_k: int = 5,
) -> list[PoseHypothesisSet]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("FoundPose result must be a JSON list")
    return foundpose_records_to_top_k(records, cad_object, top_k=top_k)


def hypothesis_sets_to_jsonable(
    hypothesis_sets: list[PoseHypothesisSet],
) -> list[dict[str, Any]]:
    payload = []
    for item in hypothesis_sets:
        confidences = np.asarray(
            [hypothesis.confidence for hypothesis in item.hypotheses],
            dtype=np.float64,
        )
        if float(np.sum(confidences)) <= 1e-12:
            confidences = np.ones_like(confidences)
        weights = confidences / np.sum(confidences)
        payload.append(
            {
                "object_id": item.object_id,
                "scene_id": item.scene_id,
                "image_id": item.image_id,
                "instance_id": item.instance_id,
                "uncertainty_method": item.uncertainty_method,
                "symmetry_discretization_deg": item.symmetry_discretization_deg,
                "covariance_se3": (
                    None if item.covariance_se3 is None else item.covariance_se3.tolist()
                ),
                "hypotheses": [
                    {
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "confidence": hypothesis.confidence,
                        "weight": float(weight),
                        "symmetry_class": hypothesis.symmetry_class,
                        "source": hypothesis.source,
                        "transform_camera_object": (
                            hypothesis.transform_camera_object.tolist()
                        ),
                    }
                    for hypothesis, weight in zip(item.hypotheses, weights)
                ],
            }
        )
    return payload
