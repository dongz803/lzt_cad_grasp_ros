#!/usr/bin/env python3
"""Compare a pose-mode mixture with its highest-weight single Gaussian.

This is an offline, GT-dependent evaluation on the frozen covariance test
objects only.  It does not fit, select or recalibrate either covariance model.
The Best-1 baseline uses mode 0 and its calibrated predictive-plus-dispersion
covariance.  The multimode method uses the union of all component ellipsoids
for coverage and the reliability-weighted Gaussian mixture for diagnostic NLL.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/analysis"))

from _bootstrap import activate_project_packages  # noqa: E402

activate_project_packages(PROJECT_ROOT)

from calibrate_pose_mode_uncertainty import (  # noqa: E402
    DetectionGroundTruthMatcher,
    GroundTruthStore,
)
from lzt_common.object_config import load_object_config  # noqa: E402
from lzt_perception.pose_uncertainty import (  # noqa: E402
    PredictivePoseCovarianceModel,
    combine_predictive_and_dispersion,
    covariance_log_pseudodeterminant,
    squared_mahalanobis,
    symmetry_aware_pose_residual,
    symmetry_observable_projector,
    wilson_coverage_interval,
)


DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "lzt_perception/config/uncertainty/"
    "tless_objects1_30_covariance_v4_protocol.json"
)
DEFAULT_MODE_ROOT = (
    PROJECT_ROOT
    / "outputs/innovation1_v3_full/"
    "tless_objects1_30_fixed_10mm_5deg_v1/"
    "mode_aware_10mm_5deg"
)
DEFAULT_MODEL_ROOT = (
    PROJECT_ROOT / "outputs/uncertainty/tless_objects1_30_covariance_v4_frozen_v1"
)
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "datasets/bop/tless"
DEFAULT_DETECTIONS = (
    WORKSPACE_ROOT
    / "datasets/bop/detections/cnos-fastsam/cnos-fastsam_tless-test.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/uncertainty/tless_multimode_vs_best1_v1"
)


@dataclass(frozen=True)
class ComponentEvaluation:
    mode_id: int
    weight: float
    squared_mahalanobis: float
    gaussian_log_density: float
    log_pseudodeterminant: float
    normalized_surface_displacement_proxy: float


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _object_id(number: int) -> str:
    return f"tless_{number:06d}"


def _logsumexp(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("log-sum-exp values must be non-empty and finite")
    maximum = float(np.max(array))
    return float(maximum + np.log(np.sum(np.exp(array - maximum))))


def ellipsoid_log_volume(
    covariance: np.ndarray,
    projector: np.ndarray,
    threshold: float,
) -> float:
    """Return the log volume in the observable covariance subspace."""

    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("ellipsoid threshold must be positive")
    rank = int(np.linalg.matrix_rank(projector, tol=1e-10))
    if rank < 1:
        raise ValueError("observable projector must have positive rank")
    unit_ball = 0.5 * rank * math.log(math.pi) - math.lgamma(0.5 * rank + 1.0)
    return float(
        unit_ball
        + 0.5 * rank * math.log(threshold)
        + 0.5 * covariance_log_pseudodeterminant(covariance, projector)
    )


def evaluate_components(
    *,
    modes: list[Mapping[str, Any]],
    target: np.ndarray,
    cad_object: Any,
    diameter_m: float,
    predictive_covariance: np.ndarray,
    projector: np.ndarray,
    continuous_symmetry_samples: int,
) -> tuple[list[ComponentEvaluation], list[np.ndarray]]:
    """Evaluate all mixture components against one offline GT pose."""

    if not modes:
        raise ValueError("pose mixture must contain at least one mode")
    ordered = sorted(modes, key=lambda item: int(item["mode_id"]))
    raw_weights = np.asarray([float(item["weight"]) for item in ordered])
    if np.any(raw_weights < 0.0) or not np.all(np.isfinite(raw_weights)):
        raise ValueError("mode weights must be finite and non-negative")
    if float(np.sum(raw_weights)) <= 0.0:
        raise ValueError("at least one mode weight must be positive")
    weights = raw_weights / np.sum(raw_weights)
    rank = int(np.linalg.matrix_rank(projector, tol=1e-10))
    evaluations: list[ComponentEvaluation] = []
    covariances: list[np.ndarray] = []
    for mode, weight in zip(ordered, weights):
        predicted = np.asarray(mode["transform_camera_object"], dtype=np.float64)
        residual, _, _ = symmetry_aware_pose_residual(
            predicted,
            target,
            cad_object,
            continuous_symmetry_samples=continuous_symmetry_samples,
        )
        dispersion = np.asarray(
            mode.get("dispersion_covariance_se3", mode["covariance_se3"]),
            dtype=np.float64,
        )
        total_covariance = combine_predictive_and_dispersion(
            predictive_covariance,
            dispersion,
        )
        distance = squared_mahalanobis(residual, total_covariance, projector)
        log_pdet = covariance_log_pseudodeterminant(total_covariance, projector)
        log_density = -0.5 * (
            distance + log_pdet + rank * math.log(2.0 * math.pi)
        )
        translation_error = float(np.linalg.norm(residual[:3]))
        rotation_error = float(np.linalg.norm(residual[3:]))
        surface_proxy = float(
            np.hypot(translation_error, 0.5 * diameter_m * rotation_error)
        )
        evaluations.append(
            ComponentEvaluation(
                mode_id=int(mode["mode_id"]),
                weight=float(weight),
                squared_mahalanobis=distance,
                gaussian_log_density=log_density,
                log_pseudodeterminant=log_pdet,
                normalized_surface_displacement_proxy=surface_proxy / diameter_m,
            )
        )
        covariances.append(total_covariance)
    if evaluations[0].mode_id != 0:
        raise ValueError("Best-1 baseline requires highest-weight mode_id 0")
    return evaluations, covariances


def sample_record(
    *,
    object_id: str,
    stratum: str,
    group: Mapping[str, Any],
    gt_match_iou: float,
    evaluations: list[ComponentEvaluation],
    covariances: list[np.ndarray],
    thresholds: Mapping[str, float],
    correct_mode_threshold: float,
    projector: np.ndarray,
    online_approved: bool,
) -> dict[str, Any]:
    """Construct a compact per-detection comparison record."""

    best = evaluations[0]
    nearest = min(
        evaluations,
        key=lambda item: (
            item.normalized_surface_displacement_proxy,
            item.mode_id,
        ),
    )
    mixture_log_density = _logsumexp(
        math.log(max(item.weight, np.finfo(float).tiny))
        + item.gaussian_log_density
        for item in evaluations
    )
    coverage = {}
    volume = {}
    for key, threshold in thresholds.items():
        component_log_volumes = [
            ellipsoid_log_volume(covariance, projector, float(threshold))
            for covariance in covariances
        ]
        coverage[key] = {
            "best1": best.squared_mahalanobis <= float(threshold),
            "multimode_union": any(
                item.squared_mahalanobis <= float(threshold)
                for item in evaluations
            ),
        }
        volume[key] = {
            "best1_log_volume": component_log_volumes[0],
            "multimode_log_sum_component_volumes": _logsumexp(
                component_log_volumes
            ),
        }
    return {
        "object_id": object_id,
        "stratum": stratum,
        "scene_id": int(group["scene_id"]),
        "image_id": int(group["image_id"]),
        "instance_id": int(group["instance_id"]),
        "gt_match_iou": float(gt_match_iou),
        "online_model_approved_for_object": bool(online_approved),
        "mode_count": len(evaluations),
        "nearest_mode_id": nearest.mode_id,
        "contains_correct_mode": (
            nearest.normalized_surface_displacement_proxy <= correct_mode_threshold
        ),
        "best1_is_correct": (
            best.normalized_surface_displacement_proxy <= correct_mode_threshold
        ),
        "best1_squared_mahalanobis": best.squared_mahalanobis,
        "multimode_min_squared_mahalanobis": min(
            item.squared_mahalanobis for item in evaluations
        ),
        "best1_nll": -best.gaussian_log_density,
        "multimode_mixture_nll": -mixture_log_density,
        "coverage": coverage,
        "volume_proxy": volume,
    }


def summarize_records(
    records: list[Mapping[str, Any]],
    levels: list[float],
) -> dict[str, Any]:
    if not records:
        return {"sample_count": 0}
    result: dict[str, Any] = {
        "sample_count": len(records),
        "object_ids": sorted({str(item["object_id"]) for item in records}),
        "mean_mode_count": float(np.mean([item["mode_count"] for item in records])),
        "multimode_group_share": float(
            np.mean([int(item["mode_count"]) > 1 for item in records])
        ),
        "contains_correct_mode_share": float(
            np.mean([bool(item["contains_correct_mode"]) for item in records])
        ),
        "best1_is_correct_share": float(
            np.mean([bool(item["best1_is_correct"]) for item in records])
        ),
        "nearest_mode_is_best1_share": float(
            np.mean([int(item["nearest_mode_id"]) == 0 for item in records])
        ),
        "mean_best1_nll": float(np.mean([item["best1_nll"] for item in records])),
        "mean_multimode_mixture_nll": float(
            np.mean([item["multimode_mixture_nll"] for item in records])
        ),
        "mean_nll_improvement_best1_minus_multimode": float(
            np.mean(
                [
                    item["best1_nll"] - item["multimode_mixture_nll"]
                    for item in records
                ]
            )
        ),
        "coverage": {},
    }
    for level in levels:
        key = f"{level:.4f}"
        # Thresholds can differ by stratum.  Boolean counts below are therefore
        # authoritative; Wilson intervals are computed from those same counts.
        best_flags = np.asarray(
            [bool(item["coverage"][key]["best1"]) for item in records]
        )
        mixture_flags = np.asarray(
            [bool(item["coverage"][key]["multimode_union"]) for item in records]
        )
        def boolean_stats(flags: np.ndarray) -> dict[str, Any]:
            successes = int(np.sum(flags))
            observed = successes / len(flags)
            lower, upper = wilson_coverage_interval(successes, len(flags))
            return {
                "sample_count": len(flags),
                "covered_count": successes,
                "nominal_coverage": level,
                "observed_coverage": observed,
                "coverage_error": observed - level,
                "undercoverage": max(level - observed, 0.0),
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "threshold_semantics": (
                    "frozen empirical d2 threshold from each sample's stratum"
                ),
            }
        best_stats = boolean_stats(best_flags)
        mixture_stats = boolean_stats(mixture_flags)
        rescued = int(np.sum((~best_flags) & mixture_flags))
        lost = int(np.sum(best_flags & (~mixture_flags)))
        result["coverage"][key] = {
            "best1": best_stats,
            "multimode_union": mixture_stats,
            "rescued_by_additional_modes": rescued,
            "lost_by_multimode": lost,
            "absolute_coverage_gain": float(
                np.mean(mixture_flags) - np.mean(best_flags)
            ),
            "mean_best1_log_volume": float(
                np.mean(
                    [item["volume_proxy"][key]["best1_log_volume"] for item in records]
                )
            ),
            "mean_multimode_log_sum_component_volumes": float(
                np.mean(
                    [
                        item["volume_proxy"][key][
                            "multimode_log_sum_component_volumes"
                        ]
                        for item in records
                    ]
                )
            ),
        }
    return result


def _model_paths(model_root: Path) -> dict[str, Path]:
    return {
        "continuous_axis_rank5": (
            model_root / "continuous_axis_rank5_predictive_covariance_v4.json"
        ),
        "full_se3_rank6": (
            model_root / "full_se3_rank6_predictive_covariance_v4.json"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multimode uncertainty with a Best-1 single Gaussian"
    )
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-gt-match-iou", type=float, default=0.1)
    parser.add_argument("--correct-mode-threshold", type=float, default=0.1)
    parser.add_argument("--continuous-symmetry-samples", type=int, default=360)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    if not 0.0 <= args.minimum_gt_match_iou <= 1.0:
        raise SystemExit("--minimum-gt-match-iou must lie in [0, 1]")
    if args.correct_mode_threshold <= 0.0:
        raise SystemExit("--correct-mode-threshold must be positive")

    protocol = _load_json(args.protocol_config)
    levels = [float(value) for value in protocol["nominal_coverage_levels"]]
    model_paths = _model_paths(args.model_root)
    models = {
        name: PredictivePoseCovarianceModel.load(path)
        for name, path in model_paths.items()
    }
    test_assignment: dict[int, str] = {}
    non_test_ids: set[int] = set()
    for stratum, split in protocol["strata"].items():
        for number in split["test"]:
            number = int(number)
            if number in test_assignment:
                raise SystemExit(f"duplicate frozen test object {number}")
            test_assignment[number] = stratum
        non_test_ids.update(int(value) for value in split["train"])
        non_test_ids.update(int(value) for value in split["calibration"])
    if set(test_assignment) & non_test_ids:
        raise SystemExit("frozen test objects overlap train/calibration objects")

    model_info = _load_json(args.dataset_root / "models_cad/models_info.json")
    test_root = args.dataset_root / "test_primesense"
    gt_store = GroundTruthStore(test_root)
    matcher = DetectionGroundTruthMatcher(args.detections, test_root, gt_store)

    records: list[dict[str, Any]] = []
    exclusions = {"no_one_to_one_gt_match": 0, "gt_mask_iou_below_threshold": 0}
    input_group_count = 0
    per_object_input_groups: dict[str, int] = {}
    for number, stratum in sorted(test_assignment.items()):
        object_id = _object_id(number)
        config_path = (
            PROJECT_ROOT / f"lzt_description/config/objects/tless_{number:06d}.yaml"
        )
        mode_path = args.mode_root / f"obj{number}/pose-modes-refined.json"
        cad_object = load_object_config(config_path, require_mesh=False).cad_object
        projector = symmetry_observable_projector(cad_object)
        model = models[stratum]
        if not np.allclose(projector, model.observable_projector_se3, atol=1e-9):
            raise SystemExit(
                f"observable projector mismatch for {object_id} in {stratum}"
            )
        predictive = model.covariance_for(
            object_id,
            calibration_semantics="predictive_plus_candidate_dispersion",
            allow_unapproved=True,
        )
        thresholds = {
            f"{level:.4f}": model.empirical_threshold(level) for level in levels
        }
        diameter_m = float(model_info[str(number)]["diameter"]) * 0.001
        groups = _load_json(mode_path)
        per_object_input_groups[object_id] = len(groups)
        input_group_count += len(groups)
        for group in groups:
            try:
                target, _, iou = matcher.match(
                    int(group["scene_id"]),
                    int(group["image_id"]),
                    number,
                    int(group["instance_id"]),
                )
            except KeyError:
                exclusions["no_one_to_one_gt_match"] += 1
                continue
            if iou < args.minimum_gt_match_iou:
                exclusions["gt_mask_iou_below_threshold"] += 1
                continue
            evaluations, covariances = evaluate_components(
                modes=list(group["modes"]),
                target=target,
                cad_object=cad_object,
                diameter_m=diameter_m,
                predictive_covariance=predictive,
                projector=projector,
                continuous_symmetry_samples=args.continuous_symmetry_samples,
            )
            record = sample_record(
                object_id=object_id,
                stratum=stratum,
                group=group,
                gt_match_iou=iou,
                evaluations=evaluations,
                covariances=covariances,
                thresholds=thresholds,
                correct_mode_threshold=args.correct_mode_threshold,
                projector=projector,
                online_approved=(
                    model.model_approved and object_id in model.approved_object_ids
                ),
            )
            # Store the actual per-stratum thresholds beside the booleans.
            for key, threshold in thresholds.items():
                record["coverage"][key]["threshold_d2"] = threshold
            records.append(record)

    reliable = records
    approved = [item for item in records if item["online_model_approved_for_object"]]
    conditional = [item for item in records if item["contains_correct_mode"]]
    approved_conditional = [
        item
        for item in records
        if item["online_model_approved_for_object"] and item["contains_correct_mode"]
    ]
    report = {
        "schema_version": 1,
        "experiment": "multimode_mixture_vs_best1_single_gaussian",
        "protocol": {
            "frozen_covariance_protocol": str(args.protocol_config.resolve()),
            "test_object_assignment": {
                stratum: [int(value) for value in split["test"]]
                for stratum, split in protocol["strata"].items()
            },
            "train_or_calibration_objects_used_for_evaluation": [],
            "mode_root": str(args.mode_root.resolve()),
            "model_files": {
                key: str(value.resolve()) for key, value in model_paths.items()
            },
            "covariance_semantics": "predictive_plus_candidate_dispersion",
            "best1_definition": "highest-weight mode_id 0 with one Gaussian",
            "multimode_coverage_definition": (
                "union of all component ellipsoids at each frozen empirical threshold"
            ),
            "multimode_nll_definition": (
                "reliability-weighted Gaussian mixture; diagnostic because mode "
                "weights are normalized candidate quality, not calibrated mode probabilities"
            ),
            "gt_usage": (
                "offline one-to-one detection matching, residual evaluation and null-mode audit only"
            ),
            "test_used_for_model_selection_or_recalibration": False,
            "out_of_ellipsoid_samples_removed": False,
            "minimum_gt_match_iou": args.minimum_gt_match_iou,
            "correct_mode_normalized_surface_threshold": args.correct_mode_threshold,
        },
        "accounting": {
            "input_group_count": input_group_count,
            "per_object_input_group_count": per_object_input_groups,
            "evaluated_group_count": len(records),
            "exclusions": exclusions,
        },
        "results": {
            "all_frozen_test_objects": summarize_records(reliable, levels),
            "approved_online_objects": summarize_records(approved, levels),
            "conditional_on_correct_mode_existing": summarize_records(
                conditional, levels
            ),
            "approved_and_correct_mode_existing": summarize_records(
                approved_conditional, levels
            ),
            "by_stratum": {
                stratum: summarize_records(
                    [item for item in records if item["stratum"] == stratum],
                    levels,
                )
                for stratum in protocol["strata"]
            },
            "by_object": {
                _object_id(number): summarize_records(
                    [
                        item
                        for item in records
                        if item["object_id"] == _object_id(number)
                    ],
                    levels,
                )
                for number in sorted(test_assignment)
            },
        },
        "interpretation_limits": [
            "Coverage is conditional evidence for the frozen T-LESS test objects, not grasp success.",
            "Union coverage can increase by adding components, so it must be read with the summed component-volume proxy and mode count.",
            "The summed component-volume proxy ignores overlap and is not the exact union volume.",
            "Objects 19 and 27 are retained for frozen offline testing but are not approved for online covariance loading.",
            "No covariance, threshold, ranking weight or clustering parameter was tuned on these results.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "samples.json", records)
    _write_json(args.output_dir / "summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
