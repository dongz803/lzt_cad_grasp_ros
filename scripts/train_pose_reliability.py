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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.cad_grasp import load_ascii_ply_geometry  # noqa: E402
from lzt_cad_grasp_core.object_config import load_object_config  # noqa: E402
from lzt_cad_grasp_core.pose_reliability import (  # noqa: E402
    FEATURE_NAMES,
    PoseReliabilityModel,
    binary_probability_metrics,
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
    config_path = PROJECT_ROOT / "config/objects" / f"tless_{object_id:06d}.yaml"
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
    parser.add_argument("--maximum-iterations", type=int, default=100)
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="optimize minority ranking instead of calibrated probabilities",
    )
    args = parser.parse_args()
    if args.label_mssd_normalized <= 0.0:
        raise SystemExit("--label-mssd-normalized must be positive")
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
