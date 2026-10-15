#!/usr/bin/env python3
"""Train and cross-CAD evaluate a calibrated Top-K pose reliability model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)

from lzt_grasp_planning.cad_grasp import load_ascii_ply_geometry  # noqa: E402
from lzt_common.object_config import load_object_config  # noqa: E402
from lzt_perception.pose_reliability import (  # noqa: E402
    FEATURE_NAMES,
    PoseReliabilityModel,
    apply_affine_logit_calibration,
    binary_probability_metrics,
    fit_affine_logit_calibrator,
    fit_pose_reliability_model,
    pose_reliability_features,
)


def _group_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["scene_id"]),
        int(record["image_id"]),
        int(record["object_id"]),
        int(record.get("instance_id", 0)),
    )


def _cad_scale_m(object_id: int, dataset_root: Path) -> float:
    os.environ.setdefault("BOP_PATH", str(dataset_root.resolve().parent))
    config_path = PROJECT_ROOT / "lzt_description/config/objects" / f"tless_{object_id:06d}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"missing object config needed for CAD normalization: {config_path}"
        )
    config = load_object_config(config_path)
    geometry = load_ascii_ply_geometry(
        config.cad_object.mesh_path,
        config.cad_object.units_to_m,
    )
    return float(
        np.linalg.norm(
            np.max(geometry.vertices_m, axis=0)
            - np.min(geometry.vertices_m, axis=0)
        )
    )


def _load_samples(
    audit_paths: list[Path],
    label_threshold: float,
) -> tuple[list[dict[str, Any]], dict[int, int], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    denominators: dict[int, int] = {}
    sources: list[dict[str, Any]] = []
    for path in audit_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        records = payload["candidates"]
        object_id = int(summary["object_id"])
        dataset_root = Path(summary["dataset_root"])
        scale = _cad_scale_m(object_id, dataset_root)
        denominators[object_id] = int(summary["raw_group_count"])
        sources.append(
            {
                "audit": str(path.resolve()),
                "object_id": object_id,
                "raw_group_count": denominators[object_id],
                "cad_scale_m": scale,
            }
        )
        groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[_group_key(record)].append(record)
        for group_key, group in groups.items():
            for record in group:
                evaluation = record.get("evaluation_only", {})
                error = evaluation.get("mssd_normalized")
                if not record.get("accepted", False) or error is None:
                    continue
                samples.append(
                    {
                        "object_id": object_id,
                        "group_key": (object_id, *group_key),
                        "hypothesis_id": int(record.get("hypothesis_id", 0)),
                        "features": pose_reliability_features(record, scale, group),
                        "label": float(error) <= label_threshold,
                        "mssd_normalized": float(error),
                        "mssd_mm": record.get("evaluation_only", {}).get("mssd_mm"),
                        "original_score": float(record.get("original_score", 0.0)),
                        "robust_score": float(record.get("robust_score", 0.0)),
                    }
                )
    if not samples:
        raise ValueError("no accepted candidates with evaluation labels were found")
    return samples, denominators, sources


def _target_denominators(
    targets_json: Path, object_ids: list[int]
) -> dict[int, int]:
    targets = json.loads(targets_json.read_text(encoding="utf-8"))
    requested = set(object_ids)
    denominators = {object_id: 0 for object_id in object_ids}
    for target in targets:
        object_id = int(target["obj_id"])
        if object_id in requested:
            denominators[object_id] += int(target["inst_count"])
    missing = [object_id for object_id, count in denominators.items() if count <= 0]
    if missing:
        raise ValueError(f"targets JSON has no instances for objects: {missing}")
    return denominators


def _selection_summary(
    samples: list[dict[str, Any]],
    score_name: str,
    denominators: dict[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample["group_key"]].append(sample)
    selected = [
        max(group, key=lambda item: (float(item[score_name]), -item["hypothesis_id"]))
        for group in groups.values()
    ]
    denominator = int(sum(denominators.values()))
    correct = int(sum(bool(item["label"]) for item in selected))
    errors = [float(item["mssd_normalized"]) for item in selected]
    summary = {
        "fixed_target_denominator": denominator,
        "selected_groups": len(selected),
        "coverage": len(selected) / max(denominator, 1),
        "correct_selected": correct,
        "fixed_denominator_success_rate": correct / max(denominator, 1),
        "conditional_success_rate": correct / max(len(selected), 1),
        "conditional_median_mssd_normalized": (
            None if not errors else float(np.median(errors))
        ),
    }
    return summary, selected


def _risk_coverage(
    selected: list[dict[str, Any]],
    score_name: str,
    denominator: int,
) -> list[dict[str, float]]:
    ordered = sorted(selected, key=lambda item: float(item[score_name]), reverse=True)
    if not ordered:
        return []
    counts = sorted(
        {
            max(1, int(round(len(ordered) * fraction)))
            for fraction in np.linspace(0.1, 1.0, 10)
        }
    )
    curve = []
    for count in counts:
        retained = ordered[:count]
        correct = sum(bool(item["label"]) for item in retained)
        curve.append(
            {
                "retained": float(count),
                "fixed_denominator_coverage": count / max(denominator, 1),
                "conditional_risk": 1.0 - correct / count,
                "fixed_denominator_success_rate": correct / max(denominator, 1),
                "minimum_confidence": float(retained[-1][score_name]),
            }
        )
    return curve


def _load_frozen_split(
    path: Path,
    available_objects: list[int],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_split = payload.get("split")
    if not isinstance(raw_split, dict):
        raise ValueError("split config must contain a split object")
    split: dict[str, list[int]] = {}
    for name in ("train", "calibration", "test"):
        values = raw_split.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split config requires a non-empty {name} list")
        object_ids = [int(value) for value in values]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError(f"split {name} contains duplicate object IDs")
        split[name] = sorted(object_ids)
    memberships = [object_id for values in split.values() for object_id in values]
    if len(set(memberships)) != len(memberships):
        raise ValueError("train/calibration/test object IDs must be disjoint")
    available = set(available_objects)
    requested = set(memberships)
    if requested != available:
        raise ValueError(
            "split object IDs must exactly match audit objects; "
            f"missing={sorted(available - requested)}, "
            f"unknown={sorted(requested - available)}"
        )
    return split, payload


def _samples_for_objects(
    samples: list[dict[str, Any]], object_ids: list[int]
) -> list[dict[str, Any]]:
    requested = set(object_ids)
    selected = [sample for sample in samples if int(sample["object_id"]) in requested]
    if not selected:
        raise ValueError(f"no reliability samples for objects {sorted(requested)}")
    return selected


def _sample_arrays(
    samples: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack([sample["features"] for sample in samples]),
        np.asarray([sample["label"] for sample in samples], dtype=np.float64),
    )


def _calibration_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> list[dict[str, float | int | None]]:
    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for index in range(bins):
        if index + 1 == bins:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        count = int(np.sum(mask))
        result.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": (
                    None if count == 0 else float(np.mean(scores[mask]))
                ),
                "empirical_positive_fraction": (
                    None if count == 0 else float(np.mean(targets[mask]))
                ),
            }
        )
    return result


def _inventory(
    samples: list[dict[str, Any]],
    denominators: dict[int, int],
    sources: list[dict[str, Any]],
    split: dict[str, list[int]],
) -> list[dict[str, Any]]:
    source_by_object = {int(item["object_id"]): item for item in sources}
    membership = {
        object_id: name for name, object_ids in split.items() for object_id in object_ids
    }
    rows = []
    for object_id in sorted(source_by_object):
        object_samples = [
            sample for sample in samples if int(sample["object_id"]) == object_id
        ]
        positives = int(sum(bool(sample["label"]) for sample in object_samples))
        rows.append(
            {
                "object_id": object_id,
                "split": membership[object_id],
                "raw_pose_groups": int(source_by_object[object_id]["raw_group_count"]),
                "fixed_bop_target_instances": int(denominators[object_id]),
                "accepted_labeled_candidates": len(object_samples),
                "positive_candidates": positives,
                "positive_fraction": positives / max(len(object_samples), 1),
            }
        )
    return rows


def _evaluate_split(
    split_samples: list[dict[str, Any]],
    model: PoseReliabilityModel,
    denominators: dict[int, int],
    calibration_bins: int,
) -> tuple[dict[str, Any], np.ndarray]:
    features, labels = _sample_arrays(split_samples)
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    annotated = []
    for sample, probability in zip(split_samples, probabilities):
        row = dict(sample)
        row["calibrated_probability"] = float(probability)
        annotated.append(row)
    model_selection, selected = _selection_summary(
        annotated, "calibrated_probability", denominators
    )
    baselines = {}
    for score_name in ("original_score", "robust_score"):
        selection, _ = _selection_summary(annotated, score_name, denominators)
        baselines[score_name] = {
            "candidate_probability_metrics": binary_probability_metrics(
                labels,
                np.asarray([sample[score_name] for sample in annotated]),
                calibration_bins=calibration_bins,
            ),
            "group_selection": selection,
        }
    denominator = int(sum(denominators.values()))
    report = {
        "candidate_probability_metrics": binary_probability_metrics(
            labels, probabilities, calibration_bins=calibration_bins
        ),
        "calibration_curve": _calibration_curve(
            labels, probabilities, calibration_bins
        ),
        "group_selection": model_selection,
        "baselines": baselines,
        "risk_coverage": _risk_coverage(
            selected, "calibrated_probability", denominator
        ),
    }
    return report, probabilities


def _per_object_evaluation(
    split_samples: list[dict[str, Any]],
    model: PoseReliabilityModel,
    denominators: dict[int, int],
    calibration_bins: int,
) -> dict[str, Any]:
    reports = {}
    for object_id in sorted(denominators):
        object_samples = [
            sample
            for sample in split_samples
            if int(sample["object_id"]) == object_id
        ]
        features, labels = _sample_arrays(object_samples)
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        annotated = []
        for sample, probability in zip(object_samples, probabilities):
            row = dict(sample)
            row["calibrated_probability"] = float(probability)
            annotated.append(row)
        selection, _ = _selection_summary(
            annotated,
            "calibrated_probability",
            {object_id: denominators[object_id]},
        )
        reports[str(object_id)] = {
            "candidate_probability_metrics": binary_probability_metrics(
                labels, probabilities, calibration_bins=calibration_bins
            ),
            "group_selection": selection,
        }
    return reports


def _run_frozen_split_protocol(
    *,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    raw_group_denominators: dict[int, int],
    sources: list[dict[str, Any]],
    denominators: dict[int, int],
    objects: list[int],
) -> None:
    split, split_payload = _load_frozen_split(args.split_config.resolve(), objects)
    split_samples = {
        name: _samples_for_objects(samples, object_ids)
        for name, object_ids in split.items()
    }
    train_features, train_labels = _sample_arrays(split_samples["train"])
    calibration_features, calibration_labels = _sample_arrays(
        split_samples["calibration"]
    )
    test_features, test_labels = _sample_arrays(split_samples["test"])
    del test_features, test_labels  # Test labels are first consumed after selection.

    l2_grid = sorted(set(float(value) for value in args.l2_grid))
    if not l2_grid or any(not np.isfinite(value) or value < 0.0 for value in l2_grid):
        raise ValueError("--l2-grid values must be finite and non-negative")
    balance_options = tuple(dict.fromkeys(args.class_weight_options))
    candidates = []
    fitted: list[tuple[tuple[float, ...], PoseReliabilityModel, dict[str, Any]]] = []
    for l2_regularization in l2_grid:
        for class_weight in balance_options:
            balance_classes = class_weight == "balanced"
            base_model = fit_pose_reliability_model(
                train_features,
                train_labels,
                label_mssd_normalized_threshold=args.label_mssd_normalized,
                l2_regularization=l2_regularization,
                maximum_iterations=args.maximum_iterations,
                balance_classes=balance_classes,
            )
            raw_probabilities = np.asarray(
                base_model.predict_proba(calibration_features), dtype=np.float64
            )
            logit_scale, logit_bias = fit_affine_logit_calibrator(
                calibration_labels,
                raw_probabilities,
                maximum_iterations=args.maximum_iterations,
            )
            calibrated_model = base_model.with_logit_calibration(
                logit_scale, logit_bias
            )
            calibrated_probabilities = apply_affine_logit_calibration(
                raw_probabilities, logit_scale, logit_bias
            )
            raw_metrics = binary_probability_metrics(
                calibration_labels,
                raw_probabilities,
                calibration_bins=args.calibration_bins,
            )
            calibrated_metrics = binary_probability_metrics(
                calibration_labels,
                calibrated_probabilities,
                calibration_bins=args.calibration_bins,
            )
            candidate = {
                "l2_regularization": l2_regularization,
                "class_weight": class_weight,
                "logit_scale": logit_scale,
                "logit_bias": logit_bias,
                "raw_calibration_metrics": raw_metrics,
                "calibrated_metrics": calibrated_metrics,
            }
            # Calibration quality is primary.  AUC is not in the key because
            # positive affine-logit calibration preserves ranking.
            key = (
                float(calibrated_metrics["brier_score"]),
                float(calibrated_metrics["log_loss"]),
                float(calibrated_metrics["expected_calibration_error"]),
                float(l2_regularization),
                1.0 if balance_classes else 0.0,
            )
            fitted.append((key, calibrated_model, candidate))
            candidates.append(candidate)
    fitted.sort(key=lambda item: item[0])
    _, final_model, selected_candidate = fitted[0]
    selected_index = candidates.index(selected_candidate)
    for index, candidate in enumerate(candidates):
        candidate["selected"] = index == selected_index

    model_path = args.output_dir / "pose_reliability_model.json"
    model_payload = final_model.to_dict()
    model_payload["schema_version"] = 2
    model_payload["deployment"] = {
        "status": "research_candidate",
        "approved_for_online": False,
        "reason": (
            "offline train/calibration/test artifact; promotion requires an "
            "explicit post-evaluation decision"
        ),
    }
    model_payload["calibration"] = {
        "method": "affine_logit_platt_scaling",
        "folded_into_runtime_coefficients": True,
        "logit_scale": selected_candidate["logit_scale"],
        "logit_bias": selected_candidate["logit_bias"],
        "object_ids": split["calibration"],
    }
    model_payload["training"] = {
        "protocol": "frozen_object_level_train_calibration_test",
        "split_config": str(args.split_config.resolve()),
        "train_object_ids": split["train"],
        "calibration_object_ids": split["calibration"],
        "test_object_ids": split["test"],
        "train_sample_count": len(split_samples["train"]),
        "calibration_sample_count": len(split_samples["calibration"]),
        "test_sample_count": len(split_samples["test"]),
        "train_positive_fraction": float(np.mean(train_labels)),
        "selected_l2_regularization": selected_candidate["l2_regularization"],
        "selected_class_weight": selected_candidate["class_weight"],
        "candidate_configuration_count": len(candidates),
        "test_used_for_model_selection": False,
        "source_audits": sources,
        "limitations": split_payload.get("limitations", []),
    }
    model_path.write_text(
        json.dumps(model_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    evaluation = {}
    predictions_by_split = {}
    for name in ("train", "calibration", "test"):
        split_denominators = {
            object_id: denominators[object_id] for object_id in split[name]
        }
        evaluation[name], predictions_by_split[name] = _evaluate_split(
            split_samples[name],
            final_model,
            split_denominators,
            args.calibration_bins,
        )
        evaluation[name]["per_object"] = _per_object_evaluation(
            split_samples[name],
            final_model,
            split_denominators,
            args.calibration_bins,
        )

    historical_baseline = None
    if args.baseline_model is not None:
        baseline_model = PoseReliabilityModel.load(args.baseline_model.resolve())
        historical_baseline = {
            "model_file": str(args.baseline_model.resolve()),
            "role": "read_only_same_split_comparison_not_used_for_v4_selection",
            "evaluation": {},
        }
        for name in ("train", "calibration", "test"):
            split_denominators = {
                object_id: denominators[object_id] for object_id in split[name]
            }
            baseline_report, _ = _evaluate_split(
                split_samples[name],
                baseline_model,
                split_denominators,
                args.calibration_bins,
            )
            baseline_report["per_object"] = _per_object_evaluation(
                split_samples[name],
                baseline_model,
                split_denominators,
                args.calibration_bins,
            )
            historical_baseline["evaluation"][name] = baseline_report

    inventory = _inventory(samples, denominators, sources, split)
    (args.output_dir / "sample_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "model_selection.json").write_text(
        json.dumps(
            {
                "selection_split": "calibration",
                "selection_key": ["brier_score", "log_loss", "ece", "l2", "class_weight"],
                "candidate_configuration_count": len(candidates),
                "candidates": candidates,
                "test_labels_used_for_selection": False,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 2,
        "method": "frozen_split_calibrated_topk_pose_reliability",
        "ground_truth_usage": (
            "MSSD labels are used only for offline training, calibration and "
            "evaluation; runtime features and predictions do not contain ground truth."
        ),
        "label_mssd_normalized_threshold": args.label_mssd_normalized,
        "feature_names": list(FEATURE_NAMES),
        "split": split,
        "split_policy": split_payload.get("selection_policy", {}),
        "prior_exposure": split_payload.get("prior_exposure", {}),
        "sources": sources,
        "observed_raw_pose_groups": int(sum(raw_group_denominators.values())),
        "fixed_target_denominator": int(sum(denominators.values())),
        "targets_json": str(args.targets_json.resolve()),
        "selected_candidate": selected_candidate,
        "evaluation": evaluation,
        "historical_baseline": historical_baseline,
        "test_used_for_model_selection": False,
        "model_file": str(model_path.resolve()),
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "split_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "split",
            "object_id",
            "scene_id",
            "image_id",
            "instance_id",
            "hypothesis_id",
            "label",
            "mssd_normalized",
            "original_score",
            "robust_score",
            "calibrated_probability",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name in ("train", "calibration", "test"):
            for sample, probability in zip(
                split_samples[name], predictions_by_split[name]
            ):
                _, scene_id, image_id, _, instance_id = sample["group_key"]
                writer.writerow(
                    {
                        "split": name,
                        "object_id": sample["object_id"],
                        "scene_id": scene_id,
                        "image_id": image_id,
                        "instance_id": instance_id,
                        "hypothesis_id": sample["hypothesis_id"],
                        "label": int(sample["label"]),
                        "mssd_normalized": sample["mssd_normalized"],
                        "original_score": sample["original_score"],
                        "robust_score": sample["robust_score"],
                        "calibrated_probability": float(probability),
                    }
                )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a leakage-free soft reliability model from gate metrics and "
            "evaluate it by leaving each CAD object out in turn."
        )
    )
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--targets-json",
        type=Path,
        default=WORKSPACE_ROOT / "datasets/bop/tless/test_targets_bop19.json",
        help="authoritative BOP target list used for fixed-denominator metrics",
    )
    parser.add_argument("--label-mssd-normalized", type=float, default=0.1)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument(
        "--split-config",
        type=Path,
        help=(
            "frozen object-level train/calibration/test protocol; when set, "
            "the test split is excluded from all model selection"
        ),
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        help="optional frozen model evaluated after V4 selection for comparison only",
    )
    parser.add_argument(
        "--l2-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 1.0, 10.0, 100.0],
        help="L2 candidates selected only on the calibration split",
    )
    parser.add_argument(
        "--class-weight-options",
        choices=("unbalanced", "balanced"),
        nargs="+",
        default=["unbalanced", "balanced"],
        help="class weighting candidates selected only on the calibration split",
    )
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--maximum-iterations", type=int, default=100)
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="optimize minority ranking instead of calibrated probabilities",
    )
    args = parser.parse_args()
    if args.label_mssd_normalized <= 0.0:
        raise SystemExit("--label-mssd-normalized must be positive")
    if args.calibration_bins < 2:
        raise SystemExit("--calibration-bins must be at least 2")
    if args.split_config is not None and not args.split_config.is_file():
        raise SystemExit(f"split config is missing: {args.split_config}")
    if args.baseline_model is not None and not args.baseline_model.is_file():
        raise SystemExit(f"baseline model is missing: {args.baseline_model}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, raw_group_denominators, sources = _load_samples(
        [path.resolve() for path in args.audits],
        args.label_mssd_normalized,
    )
    objects = sorted({int(sample["object_id"]) for sample in samples})
    if len(objects) < 2:
        raise SystemExit("cross-CAD evaluation requires at least two objects")
    denominators = _target_denominators(args.targets_json.resolve(), objects)
    for source in sources:
        source["bop_target_instance_count"] = denominators[source["object_id"]]
    if args.split_config is not None:
        _run_frozen_split_protocol(
            args=args,
            samples=samples,
            raw_group_denominators=raw_group_denominators,
            sources=sources,
            denominators=denominators,
            objects=objects,
        )
        return
    features = np.vstack([sample["features"] for sample in samples])
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.float64)
    cross_validated = np.zeros(len(samples), dtype=np.float64)
    fold_reports = []
    for held_out in objects:
        train_mask = np.asarray(
            [sample["object_id"] != held_out for sample in samples], dtype=bool
        )
        test_mask = ~train_mask
        model = fit_pose_reliability_model(
            features[train_mask],
            labels[train_mask],
            label_mssd_normalized_threshold=args.label_mssd_normalized,
            l2_regularization=args.l2_regularization,
            maximum_iterations=args.maximum_iterations,
            balance_classes=args.balance_classes,
        )
        predictions = model.predict_proba(features[test_mask])
        cross_validated[test_mask] = predictions
        fold_reports.append(
            {
                "held_out_object_id": held_out,
                "train_samples": int(np.sum(train_mask)),
                "test_samples": int(np.sum(test_mask)),
                "metrics": binary_probability_metrics(labels[test_mask], predictions),
            }
        )
    for sample, probability in zip(samples, cross_validated):
        sample["calibrated_probability"] = float(probability)

    final_model = fit_pose_reliability_model(
        features,
        labels,
        label_mssd_normalized_threshold=args.label_mssd_normalized,
        l2_regularization=args.l2_regularization,
        maximum_iterations=args.maximum_iterations,
        balance_classes=args.balance_classes,
    )
    model_path = args.output_dir / "pose_reliability_model.json"
    model_payload = final_model.to_dict()
    model_payload["training"] = {
        "protocol": "leave_one_cad_object_out_then_refit_all",
        "source_audits": sources,
        "sample_count": len(samples),
        "object_ids": objects,
        "positive_fraction": float(np.mean(labels)),
        "l2_regularization": args.l2_regularization,
        "balance_classes": args.balance_classes,
    }
    model_path.write_text(
        json.dumps(model_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    method_reports: dict[str, Any] = {}
    selected_by_method = {}
    for method in ("original_score", "robust_score", "calibrated_probability"):
        probabilities = np.asarray([sample[method] for sample in samples])
        selection, selected = _selection_summary(samples, method, denominators)
        method_reports[method] = {
            "candidate_probability_metrics": binary_probability_metrics(
                labels, probabilities
            ),
            "group_selection": selection,
        }
        selected_by_method[method] = selected
    total_denominator = int(sum(denominators.values()))
    report = {
        "schema_version": 1,
        "method": "cross_cad_calibrated_topk_pose_reliability",
        "ground_truth_usage": (
            "MSSD labels are used for offline fitting/evaluation only; runtime "
            "features and predictions do not contain ground truth."
        ),
        "label_mssd_normalized_threshold": args.label_mssd_normalized,
        "feature_names": list(FEATURE_NAMES),
        "sources": sources,
        "accepted_candidate_samples": len(samples),
        "observed_raw_pose_groups": int(sum(raw_group_denominators.values())),
        "fixed_target_denominator": total_denominator,
        "targets_json": str(args.targets_json.resolve()),
        "leave_one_object_out_folds": fold_reports,
        "cross_validated_metrics": binary_probability_metrics(
            labels, cross_validated
        ),
        "methods": method_reports,
        "risk_coverage": _risk_coverage(
            selected_by_method["calibrated_probability"],
            "calibrated_probability",
            total_denominator,
        ),
        "model_file": str(model_path.resolve()),
    }
    report_path = args.output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "cross_validated_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "object_id",
            "scene_id",
            "image_id",
            "instance_id",
            "hypothesis_id",
            "label",
            "mssd_normalized",
            "original_score",
            "robust_score",
            "calibrated_probability",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            _, scene_id, image_id, _, instance_id = sample["group_key"]
            writer.writerow(
                {
                    "object_id": sample["object_id"],
                    "scene_id": scene_id,
                    "image_id": image_id,
                    "instance_id": instance_id,
                    "hypothesis_id": sample["hypothesis_id"],
                    "label": int(sample["label"]),
                    "mssd_normalized": sample["mssd_normalized"],
                    "original_score": sample["original_score"],
                    "robust_score": sample["robust_score"],
                    "calibrated_probability": sample["calibrated_probability"],
                }
            )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
