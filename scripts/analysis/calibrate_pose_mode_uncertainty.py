#!/usr/bin/env python3
"""Audit and calibrate predictive uncertainty for symmetry-aware pose modes.

This is an offline, ground-truth-dependent analysis.  It deliberately separates
the candidate dispersion already stored in ``covariance_se3`` from predictive
pose error learned from BOP ground-truth residuals.  The GT-associated mode is
selected only for conditional error calibration; this selection is not an
online oracle and does not measure whether the mode-ranking weights are right.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _bootstrap import activate_project_packages  # noqa: E402

activate_project_packages(PROJECT_ROOT)

from lzt_common.contracts import CADObject  # noqa: E402
from lzt_common.object_config import load_object_config  # noqa: E402
from lzt_perception.mask_evaluation import decode_coco_rle  # noqa: E402
from lzt_perception.pose_uncertainty import (  # noqa: E402
    PredictivePoseCovarianceModel,
    calibrate_covariance_scale,
    chi_square_quantile,
    combine_predictive_and_dispersion,
    conformal_upper_quantile,
    covariance_log_pseudodeterminant,
    empirical_coverage_statistics,
    fit_predictive_covariance,
    squared_mahalanobis,
    symmetry_aware_pose_residual,
    symmetry_observable_projector,
)


@dataclass(frozen=True)
class ModeResidualSample:
    object_id: str
    scene_id: int
    image_id: int
    instance_id: int
    selected_mode_id: int
    selected_mode_weight: float
    selected_member_count: int
    residual_se3: np.ndarray
    dispersion_covariance_se3: np.ndarray
    surface_displacement_proxy_m: float
    normalized_surface_displacement_proxy: float
    detection_gt_mask_iou: float


def _object_number(object_id: str) -> int:
    suffix = object_id.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"cannot derive BOP obj_id from {object_id!r}")
    return int(suffix)


def _bop_transform(record: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4)
    recorded_rotation = np.asarray(
        record["cam_R_m2c"], dtype=np.float64
    ).reshape(3, 3)
    # BOP JSON stores decimal-rounded rotations.  Project that representation
    # to the nearest proper rotation instead of weakening pose validation.
    left, _, right = np.linalg.svd(recorded_rotation)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    transform[:3, :3] = left @ correction @ right
    transform[:3, 3] = (
        np.asarray(record["cam_t_m2c"], dtype=np.float64).reshape(3) * 0.001
    )
    return transform


class GroundTruthStore:
    """Cache BOP ground-truth poses and visible masks."""

    def __init__(self, test_root: Path) -> None:
        self.test_root = test_root
        self._scenes: dict[int, dict[str, Any]] = {}

    def image_records(self, scene_id: int, image_id: int) -> list[dict[str, Any]]:
        if scene_id not in self._scenes:
            path = self.test_root / f"{scene_id:06d}" / "scene_gt.json"
            self._scenes[scene_id] = json.loads(path.read_text(encoding="utf-8"))
        return self._scenes[scene_id][str(image_id)]

    def pose(self, scene_id: int, image_id: int, gt_index: int) -> np.ndarray:
        records = self.image_records(scene_id, image_id)
        if not 0 <= gt_index < len(records):
            raise IndexError(
                f"GT index {gt_index} is out of range for {scene_id}/{image_id}"
            )
        return _bop_transform(records[gt_index])


def _align_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    source_height, source_width = mask.shape
    height, width = shape
    if source_height < height or source_width < width:
        raise ValueError(f"mask {mask.shape} is smaller than GT mask {shape}")
    offset_y = (source_height - height) // 2
    offset_x = (source_width - width) // 2
    result = mask[offset_y : offset_y + height, offset_x : offset_x + width]
    if result.shape != shape:
        raise ValueError(f"cannot align mask {mask.shape} to GT mask {shape}")
    return result


class DetectionGroundTruthMatcher:
    """Map FoundPose detection ranks to GT with one-to-one visible-mask IoU."""

    def __init__(
        self,
        detections_path: Path,
        test_root: Path,
        gt_store: GroundTruthStore,
    ) -> None:
        self.test_root = test_root
        self.gt_store = gt_store
        self._detections: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for record in json.loads(detections_path.read_text(encoding="utf-8")):
            key = (
                int(record["scene_id"]),
                int(record["image_id"]),
                int(record["category_id"]),
            )
            self._detections.setdefault(key, []).append(record)
        for group in self._detections.values():
            group.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        self._matches: dict[
            tuple[int, int, int], dict[int, tuple[int, float]]
        ] = {}

    def _match_group(
        self, scene_id: int, image_id: int, object_number: int
    ) -> dict[int, tuple[int, float]]:
        key = (scene_id, image_id, object_number)
        if key in self._matches:
            return self._matches[key]
        detections = self._detections.get(key, [])
        gt_entries = [
            (index, item)
            for index, item in enumerate(
                self.gt_store.image_records(scene_id, image_id)
            )
            if int(item["obj_id"]) == object_number
        ]
        gt_masks = {}
        for gt_index, _ in gt_entries:
            path = (
                self.test_root
                / f"{scene_id:06d}"
                / "mask_visib"
                / f"{image_id:06d}_{gt_index:06d}.png"
            )
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            gt_masks[gt_index] = image > 0

        unmatched_gt = set(gt_masks)
        result: dict[int, tuple[int, float]] = {}
        for instance_id, detection in enumerate(detections):
            if not unmatched_gt:
                break
            predicted_mask = decode_coco_rle(detection["segmentation"])
            best_gt_index = None
            best_iou = -1.0
            for gt_index in unmatched_gt:
                gt_mask = gt_masks[gt_index]
                aligned = _align_mask(predicted_mask, gt_mask.shape)
                intersection = int(np.count_nonzero(aligned & gt_mask))
                union = int(np.count_nonzero(aligned | gt_mask))
                iou = intersection / union if union else 0.0
                if iou > best_iou:
                    best_gt_index = gt_index
                    best_iou = iou
            if best_gt_index is not None:
                unmatched_gt.remove(best_gt_index)
                result[instance_id] = (best_gt_index, float(best_iou))
        self._matches[key] = result
        return result

    def match(
        self,
        scene_id: int,
        image_id: int,
        object_number: int,
        instance_id: int,
    ) -> tuple[np.ndarray, int, float]:
        group = self._match_group(scene_id, image_id, object_number)
        if instance_id not in group:
            raise KeyError(
                "detection rank has no one-to-one GT match: "
                f"scene={scene_id}, image={image_id}, obj={object_number}, "
                f"inst={instance_id}"
            )
        gt_index, iou = group[instance_id]
        return self.gt_store.pose(scene_id, image_id, gt_index), gt_index, iou


def _percentiles(values: list[float], unit: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "unit": unit,
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _maximum_std(covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    return float(np.sqrt(max(float(np.max(eigenvalues)), 0.0)))


def _load_case(
    object_config_path: Path,
    mode_path: Path,
    model_info: dict[str, Any],
    gt_matcher: DetectionGroundTruthMatcher,
    continuous_symmetry_samples: int,
) -> tuple[CADObject, list[ModeResidualSample]]:
    object_config = load_object_config(object_config_path, require_mesh=False)
    cad_object = object_config.cad_object
    object_number = _object_number(cad_object.object_id)
    diameter_m = float(model_info[str(object_number)]["diameter"]) * 0.001
    groups = json.loads(mode_path.read_text(encoding="utf-8"))
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"mode file must be a non-empty JSON list: {mode_path}")

    samples: list[ModeResidualSample] = []
    for group in groups:
        if str(group["object_id"]) != cad_object.object_id:
            raise ValueError(
                f"{mode_path} contains {group['object_id']}, expected "
                f"{cad_object.object_id}"
            )
        scene_id = int(group["scene_id"])
        image_id = int(group["image_id"])
        instance_id = int(group["instance_id"])
        target, _, match_iou = gt_matcher.match(
            scene_id,
            image_id,
            object_number,
            instance_id,
        )
        candidates = []
        for mode in group["modes"]:
            predicted = np.asarray(
                mode["transform_camera_object"], dtype=np.float64
            )
            residual, _, _ = symmetry_aware_pose_residual(
                predicted,
                target,
                cad_object,
                continuous_symmetry_samples=continuous_symmetry_samples,
            )
            translation_error = float(np.linalg.norm(residual[:3]))
            rotation_error = float(np.linalg.norm(residual[3:]))
            # Small-angle surface displacement: translation plus rotation at
            # half the CAD diameter.  It avoids reusing clustering thresholds
            # as an uncertainty scale.
            surface_proxy = float(
                np.hypot(
                    translation_error,
                    0.5 * diameter_m * rotation_error,
                )
            )
            candidates.append((surface_proxy, mode, residual))
        _, selected, residual = min(
            candidates,
            key=lambda item: (item[0], int(item[1]["mode_id"])),
        )
        samples.append(
            ModeResidualSample(
                object_id=cad_object.object_id,
                scene_id=scene_id,
                image_id=image_id,
                instance_id=instance_id,
                selected_mode_id=int(selected["mode_id"]),
                selected_mode_weight=float(selected["weight"]),
                selected_member_count=len(selected["member_hypothesis_ids"]),
                residual_se3=residual,
                dispersion_covariance_se3=np.asarray(
                    selected.get(
                        "dispersion_covariance_se3",
                        selected["covariance_se3"],
                    ),
                    dtype=np.float64,
                ),
                surface_displacement_proxy_m=min(candidates, key=lambda x: x[0])[0],
                normalized_surface_displacement_proxy=(
                    min(candidates, key=lambda x: x[0])[0] / diameter_m
                ),
                detection_gt_mask_iou=match_iou,
            )
        )
    return cad_object, samples


def _sample_summary(
    samples: list[ModeResidualSample],
    correct_threshold: float,
    minimum_match_iou: float,
) -> dict[str, Any]:
    translation_mm = [
        1000.0 * float(np.linalg.norm(item.residual_se3[:3]))
        for item in samples
    ]
    rotation_deg = [
        float(np.degrees(np.linalg.norm(item.residual_se3[3:])))
        for item in samples
    ]
    surface_mm = [
        1000.0 * item.surface_displacement_proxy_m for item in samples
    ]
    singleton_count = sum(item.selected_member_count == 1 for item in samples)
    zero_dispersion_count = sum(
        float(np.trace(item.dispersion_covariance_se3)) <= 1e-15
        for item in samples
    )
    matched = [
        item
        for item in samples
        if item.detection_gt_mask_iou >= minimum_match_iou
    ]
    correct = [
        item
        for item in matched
        if item.normalized_surface_displacement_proxy <= correct_threshold
    ]
    return {
        "sample_count": len(samples),
        "reliable_gt_match_count": len(matched),
        "excluded_low_iou_gt_match_share": 1.0 - len(matched) / len(samples),
        "nearest_mode_is_top_weight_share": float(
            np.mean([item.selected_mode_id == 0 for item in matched])
        ),
        "top_k_contains_correct_mode_share": len(correct) / len(matched),
        "explicit_null_hypothesis_share": 1.0 - len(correct) / len(matched),
        "correct_mode_sample_count_for_covariance": len(correct),
        "correct_mode_is_top_weight_share": (
            float(np.mean([item.selected_mode_id == 0 for item in correct]))
            if correct
            else None
        ),
        "nearest_mode_singleton_share": singleton_count / len(samples),
        "zero_dispersion_covariance_share": zero_dispersion_count / len(samples),
        "detection_gt_mask_iou": _percentiles(
            [item.detection_gt_mask_iou for item in samples], "iou"
        ),
        "correct_mode_detection_gt_mask_iou": _percentiles(
            [item.detection_gt_mask_iou for item in correct], "iou"
        ),
        "selected_mode_weight": _percentiles(
            [item.selected_mode_weight for item in samples], "probability_like_weight"
        ),
        "translation_error": _percentiles(translation_mm, "mm"),
        "observable_rotation_error": _percentiles(rotation_deg, "deg"),
        "surface_displacement_proxy": _percentiles(surface_mm, "mm"),
        "correct_mode_conditional_error": {
            "translation": _percentiles(
                [
                    1000.0 * float(np.linalg.norm(item.residual_se3[:3]))
                    for item in correct
                ],
                "mm",
            ),
            "observable_rotation": _percentiles(
                [
                    float(np.degrees(np.linalg.norm(item.residual_se3[3:])))
                    for item in correct
                ],
                "deg",
            ),
        },
    }


def _empirical_calibration(
    train_samples: list[ModeResidualSample],
    test_samples: list[ModeResidualSample],
    predictive_covariance: np.ndarray,
    projector: np.ndarray,
    include_dispersion: bool,
) -> dict[str, Any]:
    def distances(items: list[ModeResidualSample]) -> np.ndarray:
        result = []
        for item in items:
            covariance = predictive_covariance
            if include_dispersion:
                covariance = combine_predictive_and_dispersion(
                    predictive_covariance,
                    item.dispersion_covariance_se3,
                )
            result.append(
                squared_mahalanobis(item.residual_se3, covariance, projector)
            )
        return np.asarray(result, dtype=np.float64)

    train_distance = distances(train_samples)
    test_distance = distances(test_samples)
    coverages = {}
    for nominal in (0.5, 0.6827, 0.9, 0.95):
        threshold = float(np.quantile(train_distance, nominal, method="higher"))
        observed = float(np.mean(test_distance <= threshold))
        coverages[f"{nominal:.4f}"] = {
            "train_empirical_threshold_d2": threshold,
            "held_out_observed_coverage": observed,
            "coverage_error": observed - nominal,
        }
    return {
        "uses_candidate_dispersion": include_dispersion,
        "train_d2": _percentiles(train_distance.tolist(), "squared_mahalanobis"),
        "held_out_d2": _percentiles(test_distance.tolist(), "squared_mahalanobis"),
        "coverage": coverages,
    }


def _covariance_summary(covariance: np.ndarray) -> dict[str, Any]:
    return {
        "translation_max_std_mm": 1000.0 * _maximum_std(covariance[:3, :3]),
        "observable_rotation_max_std_deg": float(
            np.degrees(_maximum_std(covariance[3:, 3:]))
        ),
        "covariance_se3": covariance.tolist(),
    }


def _default_cases() -> list[tuple[Path, Path]]:
    base = PROJECT_ROOT / "outputs/innovation1_pose_modes/tless_objects2_4_v1"
    return [
        (
            PROJECT_ROOT
            / f"lzt_description/config/objects/tless_{number:06d}.yaml",
            base / f"obj{number}/refinement/pose-modes-refined.json",
        )
        for number in (2, 3, 4)
    ]


def _full_tless_cases() -> list[tuple[Path, Path]]:
    base = (
        PROJECT_ROOT
        / "outputs/innovation1_v3_full/tless_objects1_30_fixed_10mm_5deg_v1"
        / "mode_aware_10mm_5deg"
    )
    return [
        (
            PROJECT_ROOT
            / f"lzt_description/config/objects/tless_{number:06d}.yaml",
            base / f"obj{number}/pose-modes-refined.json",
        )
        for number in range(1, 31)
    ]


def _mode_samples_for_ids(
    samples_by_object: dict[str, list[ModeResidualSample]],
    object_numbers: list[int],
    correct_threshold: float,
    minimum_match_iou: float,
) -> list[ModeResidualSample]:
    requested = {f"tless_{number:06d}" for number in object_numbers}
    result = [
        sample
        for object_id, samples in samples_by_object.items()
        if object_id in requested
        for sample in samples
        if sample.detection_gt_mask_iou >= minimum_match_iou
        and sample.normalized_surface_displacement_proxy <= correct_threshold
    ]
    if not result:
        raise ValueError(f"no correct-mode samples for objects {sorted(requested)}")
    return result


def _uncertainty_distances(
    samples: list[ModeResidualSample],
    predictive_covariance: np.ndarray,
    projector: np.ndarray,
    semantics: str,
) -> np.ndarray:
    result = []
    for sample in samples:
        covariance = predictive_covariance
        if semantics == "predictive_plus_candidate_dispersion":
            covariance = combine_predictive_and_dispersion(
                predictive_covariance,
                sample.dispersion_covariance_se3,
            )
        elif semantics != "predictive_only":
            raise ValueError(f"unsupported covariance semantics {semantics}")
        result.append(
            squared_mahalanobis(sample.residual_se3, covariance, projector)
        )
    return np.asarray(result, dtype=np.float64)


def _uncertainty_sharpness(
    samples: list[ModeResidualSample],
    predictive_covariance: np.ndarray,
    projector: np.ndarray,
    semantics: str,
) -> dict[str, float]:
    log_determinants = []
    nll_values = []
    rank = int(np.linalg.matrix_rank(projector, tol=1e-10))
    for sample in samples:
        covariance = predictive_covariance
        if semantics == "predictive_plus_candidate_dispersion":
            covariance = combine_predictive_and_dispersion(
                predictive_covariance,
                sample.dispersion_covariance_se3,
            )
        log_pdet = covariance_log_pseudodeterminant(covariance, projector)
        distance = squared_mahalanobis(sample.residual_se3, covariance, projector)
        log_determinants.append(log_pdet)
        nll_values.append(0.5 * (distance + log_pdet + rank * np.log(2.0 * np.pi)))
    return {
        "mean_log_pseudodeterminant": float(np.mean(log_determinants)),
        "mean_gaussian_nll": float(np.mean(nll_values)),
    }


def _coverage_at_levels(
    distances: np.ndarray,
    thresholds: dict[str, float],
    levels: list[float],
) -> dict[str, Any]:
    return {
        f"{level:.4f}": empirical_coverage_statistics(
            distances,
            thresholds[f"{level:.4f}"],
            level,
        )
        for level in levels
    }


def _run_frozen_protocol(
    *,
    args: argparse.Namespace,
    protocol: dict[str, Any],
    samples_by_object: dict[str, list[ModeResidualSample]],
    projectors: dict[str, np.ndarray],
    resolved_cases: list[dict[str, Any]],
) -> None:
    all_object_numbers = sorted(_object_number(item) for item in samples_by_object)
    assigned = []
    for stratum in protocol["strata"].values():
        assigned.extend(stratum["train"])
        assigned.extend(stratum["calibration"])
        assigned.extend(stratum["test"])
    if len(assigned) != len(set(assigned)) or sorted(assigned) != all_object_numbers:
        raise ValueError("protocol strata must assign every supplied object exactly once")

    levels = [float(value) for value in protocol["nominal_coverage_levels"]]
    selection_level = float(protocol["model_selection_nominal_coverage"])
    minimum_calibration_coverage = float(protocol["minimum_calibration_coverage"])
    minimum_test_coverage = float(protocol["minimum_test_coverage"])
    minimum_object_coverage = float(protocol["minimum_per_object_test_coverage"])
    minimum_object_samples = int(protocol["minimum_per_object_test_samples"])
    max_translation_std = float(protocol["maximum_translation_std_m"])
    max_rotation_std_deg = float(
        protocol["maximum_observable_rotation_std_deg"]
    )
    deployable_semantics = str(
        protocol["selection_policy"]["deployable_semantics"]
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 2,
        "analysis_type": "frozen_split_pose_mode_predictive_uncertainty",
        "protocol_config": str(args.protocol_config.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "detections": str(args.detections.resolve()),
        "cases": resolved_cases,
        "ground_truth_usage": (
            "GT selects the corresponding correct mode and forms residuals only "
            "for offline fit, calibration and test; GT is absent online."
        ),
        "strata": {},
        "limitations": list(protocol.get("limitations", [])),
    }
    for stratum_name, split in protocol["strata"].items():
        split_ids = {
            name: [int(value) for value in split[name]]
            for name in ("train", "calibration", "test")
        }
        object_ids = [
            f"tless_{number:06d}"
            for name in ("train", "calibration", "test")
            for number in split_ids[name]
        ]
        stratum_projectors = [projectors[object_id] for object_id in object_ids]
        projector = stratum_projectors[0]
        if any(
            not np.allclose(item, projector, atol=1e-10)
            for item in stratum_projectors[1:]
        ):
            raise ValueError(f"stratum {stratum_name} mixes observable projectors")
        rank = int(np.linalg.matrix_rank(projector, tol=1e-10))
        if rank != int(split["observable_rank"]):
            raise ValueError(f"stratum {stratum_name} observable rank mismatch")
        samples = {
            name: _mode_samples_for_ids(
                samples_by_object,
                split_ids[name],
                args.correct_mode_normalized_surface_threshold,
                args.minimum_gt_match_iou,
            )
            for name in ("train", "calibration", "test")
        }
        train_residuals = np.asarray(
            [item.residual_se3 for item in samples["train"]], dtype=np.float64
        )
        calibration_residuals = np.asarray(
            [item.residual_se3 for item in samples["calibration"]],
            dtype=np.float64,
        )
        calibration_dispersions = np.asarray(
            [item.dispersion_covariance_se3 for item in samples["calibration"]],
            dtype=np.float64,
        )
        candidates = []
        fitted = []
        for shrinkage in protocol["shrinkage_grid"]:
            base_covariance = fit_predictive_covariance(
                train_residuals,
                projector,
                shrinkage=float(shrinkage),
            )
            for semantics in protocol["semantics_grid"]:
                use_dispersion = semantics == "predictive_plus_candidate_dispersion"
                scale, theoretical_coverage = calibrate_covariance_scale(
                    calibration_residuals,
                    base_covariance,
                    projector,
                    dispersion_covariances=(
                        calibration_dispersions if use_dispersion else None
                    ),
                    nominal_coverage=selection_level,
                    minimum_coverage=minimum_calibration_coverage,
                )
                predictive_covariance = scale * base_covariance
                calibration_distances = _uncertainty_distances(
                    samples["calibration"],
                    predictive_covariance,
                    projector,
                    str(semantics),
                )
                thresholds = {
                    f"{level:.4f}": conformal_upper_quantile(
                        calibration_distances, level
                    )
                    for level in levels
                }
                calibration_coverage = _coverage_at_levels(
                    calibration_distances, thresholds, levels
                )
                sharpness = _uncertainty_sharpness(
                    samples["calibration"],
                    predictive_covariance,
                    projector,
                    str(semantics),
                )
                translation_std = _maximum_std(predictive_covariance[:3, :3])
                rotation_std_deg = float(
                    np.degrees(_maximum_std(predictive_covariance[3:, 3:]))
                )
                candidate = {
                    "shrinkage": float(shrinkage),
                    "semantics": str(semantics),
                    "covariance_scale": float(scale),
                    "theoretical_threshold_d2": chi_square_quantile(
                        selection_level, rank
                    ),
                    "theoretical_calibration_coverage": theoretical_coverage,
                    "empirical_mahalanobis_thresholds": thresholds,
                    "empirical_calibration_coverage": calibration_coverage,
                    "sharpness": sharpness,
                    "translation_max_std_m": translation_std,
                    "observable_rotation_max_std_deg": rotation_std_deg,
                    "passes_calibration": bool(
                        theoretical_coverage["observed_coverage"]
                        >= minimum_calibration_coverage
                        and translation_std <= max_translation_std
                        and rotation_std_deg <= max_rotation_std_deg
                    ),
                }
                candidates.append(candidate)
                if (
                    candidate["passes_calibration"]
                    and semantics == deployable_semantics
                ):
                    key = (
                        sharpness["mean_log_pseudodeterminant"],
                        translation_std,
                        rotation_std_deg,
                        float(shrinkage),
                    )
                    fitted.append((key, base_covariance, candidate))
        if not fitted:
            raise ValueError(f"no deployable covariance candidate for {stratum_name}")
        fitted.sort(key=lambda item: item[0])
        _, base_covariance, selected = fitted[0]
        for candidate in candidates:
            candidate["selected"] = candidate is selected
        predictive_covariance = selected["covariance_scale"] * base_covariance
        thresholds = selected["empirical_mahalanobis_thresholds"]
        split_evaluation = {}
        for name in ("train", "calibration", "test"):
            distances = _uncertainty_distances(
                samples[name], predictive_covariance, projector, deployable_semantics
            )
            split_evaluation[name] = {
                "sample_count": len(samples[name]),
                "coverage": _coverage_at_levels(distances, thresholds, levels),
                "sharpness": _uncertainty_sharpness(
                    samples[name],
                    predictive_covariance,
                    projector,
                    deployable_semantics,
                ),
            }
        per_object_test = {}
        test_level_key = f"{selection_level:.4f}"
        approved_object_ids = []
        rejected_object_ids = []
        for number in split_ids["test"]:
            object_id = f"tless_{number:06d}"
            object_samples = _mode_samples_for_ids(
                samples_by_object,
                [number],
                args.correct_mode_normalized_surface_threshold,
                args.minimum_gt_match_iou,
            )
            distances = _uncertainty_distances(
                object_samples,
                predictive_covariance,
                projector,
                deployable_semantics,
            )
            coverage = _coverage_at_levels(distances, thresholds, levels)
            object_passed = bool(
                len(object_samples) >= minimum_object_samples
                and coverage[test_level_key]["observed_coverage"]
                >= minimum_object_coverage
            )
            per_object_test[object_id] = {
                "sample_count": len(object_samples),
                "coverage": coverage,
                "passed": object_passed,
                "failure_reasons": [
                    reason
                    for condition, reason in (
                        (
                            len(object_samples) < minimum_object_samples,
                            "insufficient_test_samples",
                        ),
                        (
                            coverage[test_level_key]["observed_coverage"]
                            < minimum_object_coverage,
                            "test_undercoverage",
                        ),
                    )
                    if condition
                ],
            }
            (approved_object_ids if object_passed else rejected_object_ids).append(
                object_id
            )
        overall_test_coverage = split_evaluation["test"]["coverage"][test_level_key]
        stratum_passed = bool(
            overall_test_coverage["observed_coverage"] >= minimum_test_coverage
            and selected["translation_max_std_m"] <= max_translation_std
            and selected["observable_rotation_max_std_deg"] <= max_rotation_std_deg
            and approved_object_ids
        )
        if not stratum_passed:
            rejected_object_ids = sorted(set(rejected_object_ids + approved_object_ids))
            approved_object_ids = []
        model = PredictivePoseCovarianceModel(
            object_ids=tuple(object_ids),
            approved_object_ids=tuple(approved_object_ids),
            rejected_object_ids=tuple(rejected_object_ids),
            covariance_se3=base_covariance,
            observable_projector_se3=projector,
            covariance_scale=float(selected["covariance_scale"]),
            nominal_coverage_levels=tuple(levels),
            empirical_mahalanobis_thresholds=thresholds,
            calibration_semantics=deployable_semantics,
            model_approved=stratum_passed,
            method="frozen_split_scaled_predictive_covariance",
            conditional_on=(
                "GT-associated refined mode is correct under diameter-normalized "
                "observable surface-displacement proxy threshold"
            ),
            training={
                "dataset": "T-LESS test_primesense",
                "stratum": stratum_name,
                "train_object_ids": split_ids["train"],
                "calibration_object_ids": split_ids["calibration"],
                "test_object_ids": split_ids["test"],
                "train_sample_count": len(samples["train"]),
                "calibration_sample_count": len(samples["calibration"]),
                "test_sample_count": len(samples["test"]),
                "selected_shrinkage": selected["shrinkage"],
                "test_used_for_model_selection": False,
            },
            schema_version=2,
            acceptance_policy={
                key: protocol[key]
                for key in (
                    "minimum_calibration_coverage",
                    "minimum_test_coverage",
                    "minimum_per_object_test_coverage",
                    "minimum_per_object_test_samples",
                    "maximum_translation_std_m",
                    "maximum_observable_rotation_std_deg",
                )
            },
            validation={
                "passed": stratum_passed,
                "split_evaluation": split_evaluation,
                "per_object_test": per_object_test,
                "candidate_configuration_count": len(candidates),
                "selected_candidate": selected,
                "test_used_for_model_selection": False,
            },
            limitations=tuple(protocol.get("limitations", [])),
        )
        model_path = output_dir / f"{stratum_name}_predictive_covariance_v4.json"
        model.save(model_path)
        report["strata"][stratum_name] = {
            "split": split_ids,
            "observable_rank": rank,
            "candidate_search": candidates,
            "selected_candidate": selected,
            "evaluation": split_evaluation,
            "per_object_test": per_object_test,
            "model_approved": stratum_passed,
            "approved_object_ids": approved_object_ids,
            "rejected_object_ids": rejected_object_ids,
            "model_file": str(model_path),
        }
    report_path = output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate GT residual covariance for pose modes"
    )
    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        metavar=("OBJECT_CONFIG", "MODE_JSON"),
        help="repeat once per object; defaults to T-LESS objects 2, 3 and 4",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT.parents[1] / "datasets/bop/tless",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=PROJECT_ROOT.parents[1]
        / "datasets/bop/detections/cnos-fastsam/cnos-fastsam_tless-test.json",
    )
    parser.add_argument("--continuous-symmetry-samples", type=int, default=360)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument(
        "--correct-mode-normalized-surface-threshold",
        type=float,
        default=0.1,
        help=(
            "maximum surface-displacement proxy divided by CAD diameter for "
            "conditional covariance fitting"
        ),
    )
    parser.add_argument(
        "--minimum-gt-match-iou",
        type=float,
        default=0.1,
        help="exclude ambiguous detection-to-GT associations below this IoU",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/innovation1_pose_modes/tless_objects2_4_v1"
        / "pose-mode-uncertainty-calibration-refined.json",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        help="frozen train/calibration/test covariance protocol (schema v2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new output directory required with --protocol-config",
    )
    args = parser.parse_args()

    if args.continuous_symmetry_samples < 2:
        raise SystemExit("--continuous-symmetry-samples must be at least 2")
    if args.correct_mode_normalized_surface_threshold <= 0.0:
        raise SystemExit(
            "--correct-mode-normalized-surface-threshold must be positive"
        )
    if not 0.0 <= args.minimum_gt_match_iou <= 1.0:
        raise SystemExit("--minimum-gt-match-iou must lie in [0, 1]")
    if (args.protocol_config is None) != (args.output_dir is None):
        raise SystemExit("--protocol-config and --output-dir must be provided together")
    if args.protocol_config is not None and not args.protocol_config.is_file():
        raise SystemExit(f"protocol config is missing: {args.protocol_config}")
    cases = (
        [(Path(first), Path(second)) for first, second in args.case]
        if args.case
        else (_full_tless_cases() if args.protocol_config else _default_cases())
    )
    model_info_path = args.dataset_root / "models_cad/models_info.json"
    model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
    gt_store = GroundTruthStore(args.dataset_root / "test_primesense")
    gt_matcher = DetectionGroundTruthMatcher(
        args.detections.resolve(),
        args.dataset_root / "test_primesense",
        gt_store,
    )

    samples_by_object: dict[str, list[ModeResidualSample]] = {}
    projectors: dict[str, np.ndarray] = {}
    resolved_cases = []
    for object_config_path, mode_path in cases:
        cad_object, samples = _load_case(
            object_config_path.resolve(),
            mode_path.resolve(),
            model_info,
            gt_matcher,
            args.continuous_symmetry_samples,
        )
        if cad_object.object_id in samples_by_object:
            raise SystemExit(f"duplicate case for {cad_object.object_id}")
        samples_by_object[cad_object.object_id] = samples
        projectors[cad_object.object_id] = symmetry_observable_projector(cad_object)
        resolved_cases.append(
            {
                "object_id": cad_object.object_id,
                "object_config": str(object_config_path.resolve()),
                "mode_json": str(mode_path.resolve()),
            }
        )

    if len(samples_by_object) < 2:
        raise SystemExit("at least two objects are required for held-out validation")
    if args.protocol_config is not None:
        protocol = json.loads(args.protocol_config.read_text(encoding="utf-8"))
        _run_frozen_protocol(
            args=args,
            protocol=protocol,
            samples_by_object=samples_by_object,
            projectors=projectors,
            resolved_cases=resolved_cases,
        )
        return
    first_projector = next(iter(projectors.values()))
    incompatible = [
        object_id
        for object_id, projector in projectors.items()
        if not np.allclose(projector, first_projector, atol=1e-10)
    ]
    if incompatible:
        raise SystemExit(
            "leave-one-object-out pooling requires the same observable subspace; "
            f"incompatible objects: {incompatible}"
        )

    calibration_samples_by_object = {
        object_id: [
            item
            for item in samples
            if item.detection_gt_mask_iou >= args.minimum_gt_match_iou
            and item.normalized_surface_displacement_proxy
            <= args.correct_mode_normalized_surface_threshold
        ]
        for object_id, samples in samples_by_object.items()
    }
    if any(len(samples) < 2 for samples in calibration_samples_by_object.values()):
        raise SystemExit("each object needs at least two correct-mode samples")

    leave_one_object_out = []
    for held_out_id, test_samples in calibration_samples_by_object.items():
        train_samples = [
            item
            for object_id, items in calibration_samples_by_object.items()
            if object_id != held_out_id
            for item in items
        ]
        train_residuals = np.asarray(
            [item.residual_se3 for item in train_samples], dtype=np.float64
        )
        covariance = fit_predictive_covariance(
            train_residuals,
            first_projector,
            shrinkage=args.shrinkage,
        )
        leave_one_object_out.append(
            {
                "held_out_object_id": held_out_id,
                "train_object_ids": [
                    object_id
                    for object_id in samples_by_object
                    if object_id != held_out_id
                ],
                "train_sample_count": len(train_samples),
                "held_out_sample_count": len(test_samples),
                "predictive_covariance": _covariance_summary(covariance),
                "predictive_only": _empirical_calibration(
                    train_samples,
                    test_samples,
                    covariance,
                    first_projector,
                    include_dispersion=False,
                ),
                "predictive_plus_candidate_dispersion": _empirical_calibration(
                    train_samples,
                    test_samples,
                    covariance,
                    first_projector,
                    include_dispersion=True,
                ),
            }
        )

    all_samples = [
        item for samples in calibration_samples_by_object.values() for item in samples
    ]
    full_covariance = fit_predictive_covariance(
        np.asarray([item.residual_se3 for item in all_samples]),
        first_projector,
        shrinkage=args.shrinkage,
    )
    report = {
        "schema_version": 1,
        "analysis_type": "conditional_pose_mode_predictive_uncertainty",
        "cases": resolved_cases,
        "dataset_root": str(args.dataset_root.resolve()),
        "detections": str(args.detections.resolve()),
        "method": {
            "ground_truth_instance_mapping": (
                "inst_id is a score-ranked detection index; detections are "
                "greedily matched one-to-one to visible BOP GT masks by IoU"
            ),
            "mode_association": (
                "offline oracle: minimum CAD-diameter-scaled observable SE(3) "
                "surface-displacement proxy"
            ),
            "correct_mode_definition": (
                "surface-displacement proxy / CAD diameter <= "
                f"{args.correct_mode_normalized_surface_threshold}"
            ),
            "minimum_detection_gt_match_iou": args.minimum_gt_match_iou,
            "incorrect_mode_handling": (
                "groups without a correct mode contribute to the explicit null "
                "hypothesis, not to local predictive covariance"
            ),
            "residual_convention": (
                "camera-frame translation plus predicted-object-frame rotation; "
                "continuous-symmetry rotation axes projected out"
            ),
            "candidate_covariance_semantics": "within-mode candidate dispersion",
            "predictive_covariance_semantics": (
                "uncentred GT residual second moment with unit-safe shrinkage"
            ),
            "combined_covariance": "predictive covariance + candidate dispersion",
            "continuous_symmetry_samples": args.continuous_symmetry_samples,
            "shrinkage": args.shrinkage,
            "validation": "leave-one-object-out over the supplied objects",
        },
        "limitations": [
            "The calibration is conditional on the GT-associated mode and "
            "does not calibrate mode weights.",
            "The correctness proxy is diameter-normalized but is not exact BOP MSSD.",
            "Detection-to-GT associations below the configured mask IoU are "
            "excluded from null-rate and covariance estimates.",
            "Objects 2-4 come from the same T-LESS distribution; this is not "
            "cross-CAD generalization evidence.",
            "Empirical coverage is offline evidence and is not yet a complete "
            "pick-place success experiment.",
        ],
        "per_object": {
            object_id: _sample_summary(
                samples,
                args.correct_mode_normalized_surface_threshold,
                args.minimum_gt_match_iou,
            )
            for object_id, samples in samples_by_object.items()
        },
        "leave_one_object_out": leave_one_object_out,
        "full_fit_model": {
            "training_object_ids": list(samples_by_object),
            "sample_count": len(all_samples),
            "observable_rank": int(np.linalg.matrix_rank(first_projector)),
            "observable_projector_se3": first_projector.tolist(),
            "predictive_covariance": _covariance_summary(full_covariance),
            "use": (
                "calibrated conditional error prior; add a mode's dispersion "
                "covariance before downstream risk propagation"
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "output": str(args.output.resolve()),
        "per_object": report["per_object"],
        "leave_one_object_out_90pct_coverage": {
            item["held_out_object_id"]: {
                "predictive_only": item["predictive_only"]["coverage"]["0.9000"][
                    "held_out_observed_coverage"
                ],
                "predictive_plus_dispersion": item[
                    "predictive_plus_candidate_dispersion"
                ]["coverage"]["0.9000"]["held_out_observed_coverage"],
            }
            for item in leave_one_object_out
        },
        "full_fit_predictive_covariance": report["full_fit_model"][
            "predictive_covariance"
        ],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
