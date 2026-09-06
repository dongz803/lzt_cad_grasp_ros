"""Unified BOP mask and downstream pose evaluation.

The evaluator deliberately uses one BOP19 target list for every mask source.
Ground-truth visible masks form an oracle source, while detection sources use
the same top-``inst_count`` selection policy as the FoundPose inference code.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


TargetKey = tuple[int, int, int]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_compressed_counts(value: str | bytes) -> list[int]:
    """Decode the compact COCO RLE count representation."""

    encoded = value.decode("ascii") if isinstance(value, bytes) else value
    counts: list[int] = []
    position = 0
    while position < len(encoded):
        number = 0
        shift_groups = 0
        more = True
        while more:
            code = ord(encoded[position]) - 48
            position += 1
            if code < 0 or code > 0x3F:
                raise ValueError("invalid character in compressed COCO RLE")
            number |= (code & 0x1F) << (5 * shift_groups)
            more = bool(code & 0x20)
            shift_groups += 1
            if not more and (code & 0x10):
                number |= -1 << (5 * shift_groups)
        if len(counts) > 2:
            number += counts[-2]
        if number < 0:
            raise ValueError("negative run length in compressed COCO RLE")
        counts.append(number)
    return counts


def decode_coco_rle(segmentation: Mapping[str, Any]) -> np.ndarray:
    """Decode compressed or uncompressed COCO RLE into a boolean mask."""

    if "size" not in segmentation or "counts" not in segmentation:
        raise ValueError("COCO segmentation requires size and counts")
    size = tuple(int(value) for value in segmentation["size"])
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("COCO segmentation size must be [height, width]")
    counts_value = segmentation["counts"]
    if isinstance(counts_value, (str, bytes)):
        counts = _decode_compressed_counts(counts_value)
    elif isinstance(counts_value, list):
        counts = [int(value) for value in counts_value]
    else:
        raise ValueError("unsupported COCO RLE counts type")
    if any(value < 0 for value in counts):
        raise ValueError("COCO RLE run lengths must be non-negative")
    expected_size = size[0] * size[1]
    if sum(counts) != expected_size:
        raise ValueError(
            f"COCO RLE covers {sum(counts)} pixels, expected {expected_size}"
        )
    values = np.arange(len(counts), dtype=np.uint8) & 1
    flattened = np.repeat(values, np.asarray(counts, dtype=np.int64))
    return flattened.reshape(size, order="F").astype(bool, copy=False)


def _align_mask_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Center-crop detector padding in the same manner as FoundPose."""

    if mask.shape == shape:
        return mask
    if mask.ndim != 2 or mask.shape[0] < shape[0] or mask.shape[1] < shape[1]:
        raise ValueError(f"cannot align mask shape {mask.shape} to {shape}")
    difference_y = mask.shape[0] - shape[0]
    difference_x = mask.shape[1] - shape[1]
    if difference_y % 2 or difference_x % 2:
        raise ValueError(f"mask padding is not symmetric: {mask.shape} vs {shape}")
    offset_y = difference_y // 2
    offset_x = difference_x // 2
    return mask[offset_y:offset_y + shape[0], offset_x:offset_x + shape[1]]


def _morphological_open(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size == 0:
        return mask.astype(bool, copy=False)
    if kernel_size < 0 or kernel_size % 2 == 0:
        raise ValueError("opening kernel must be zero or a positive odd number")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened.astype(bool)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return mask & ~eroded.astype(bool)


def boundary_f1(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    tolerance_px: int = 2,
) -> float:
    """Return a symmetric boundary F1 score with a pixel tolerance."""

    if tolerance_px < 0:
        raise ValueError("boundary tolerance cannot be negative")
    prediction_boundary = _mask_boundary(prediction)
    ground_truth_boundary = _mask_boundary(ground_truth)
    prediction_count = int(np.count_nonzero(prediction_boundary))
    ground_truth_count = int(np.count_nonzero(ground_truth_boundary))
    if prediction_count == 0 and ground_truth_count == 0:
        return 1.0
    if prediction_count == 0 or ground_truth_count == 0:
        return 0.0
    if tolerance_px:
        width = 2 * tolerance_px + 1
        kernel = np.ones((width, width), dtype=np.uint8)
        expanded_prediction = cv2.dilate(
            prediction_boundary.astype(np.uint8), kernel
        ).astype(bool)
        expanded_ground_truth = cv2.dilate(
            ground_truth_boundary.astype(np.uint8), kernel
        ).astype(bool)
    else:
        expanded_prediction = prediction_boundary
        expanded_ground_truth = ground_truth_boundary
    precision = float(
        np.count_nonzero(prediction_boundary & expanded_ground_truth)
    ) / prediction_count
    recall = float(
        np.count_nonzero(ground_truth_boundary & expanded_prediction)
    ) / ground_truth_count
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def mask_pair_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    boundary_tolerance_px: int = 2,
) -> dict[str, float | None]:
    """Calculate instance-mask overlap, boundary, size, and center metrics."""

    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    if prediction.shape != ground_truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and ground-truth masks need equal 2D shapes")
    intersection = int(np.count_nonzero(prediction & ground_truth))
    union = int(np.count_nonzero(prediction | ground_truth))
    prediction_area = int(np.count_nonzero(prediction))
    ground_truth_area = int(np.count_nonzero(ground_truth))
    iou = float(intersection / union) if union else 1.0
    precision = (
        float(intersection / prediction_area)
        if prediction_area
        else (1.0 if ground_truth_area == 0 else 0.0)
    )
    recall = (
        float(intersection / ground_truth_area)
        if ground_truth_area
        else (1.0 if prediction_area == 0 else 0.0)
    )
    center_error = None
    if prediction_area and ground_truth_area:
        prediction_points = np.argwhere(prediction)
        ground_truth_points = np.argwhere(ground_truth)
        center_error = float(
            np.linalg.norm(
                prediction_points.mean(axis=0) - ground_truth_points.mean(axis=0)
            )
        )
    area_ratio = (
        float(prediction_area / ground_truth_area)
        if ground_truth_area
        else None
    )
    return {
        "iou": iou,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "boundary_f1": boundary_f1(
            prediction, ground_truth, tolerance_px=boundary_tolerance_px
        ),
        "center_error_px": center_error,
        "area_ratio": area_ratio,
    }


def _empty_statistics(iou_thresholds: Sequence[float]) -> dict[str, Any]:
    return {
        "target_group_count": 0,
        "target_instance_count": 0,
        "eligible_gt_instance_count": 0,
        "available_prediction_count": 0,
        "selected_prediction_count": 0,
        "discarded_prediction_count": 0,
        "missing_prediction_slot_count": 0,
        "groups_without_prediction": 0,
        "metric_sums": {
            "iou": 0.0,
            "pixel_precision": 0.0,
            "pixel_recall": 0.0,
            "boundary_f1": 0.0,
        },
        "center_errors": [],
        "area_ratios": [],
        "threshold_counts": {
            f"{threshold:.2f}": {"tp": 0, "fp": 0, "fn": 0}
            for threshold in iou_thresholds
        },
    }


def _update_detection_counts(
    statistics: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
    gt_masks: Sequence[np.ndarray],
    expected_count: int,
    iou_thresholds: Sequence[float],
) -> None:
    for threshold in iou_thresholds:
        unmatched_gt = set(range(len(gt_masks)))
        true_positives = 0
        for prediction in predictions:
            if true_positives >= expected_count or not unmatched_gt:
                continue
            overlaps = [
                (
                    float(
                        np.count_nonzero(prediction["mask"] & gt_masks[index])
                        / max(
                            1,
                            np.count_nonzero(prediction["mask"] | gt_masks[index]),
                        )
                    ),
                    index,
                )
                for index in unmatched_gt
            ]
            overlap, gt_index = max(overlaps, default=(0.0, -1))
            if overlap >= threshold:
                true_positives += 1
                unmatched_gt.remove(gt_index)
        false_positives = len(predictions) - true_positives
        false_negatives = expected_count - true_positives
        counts = statistics["threshold_counts"][f"{threshold:.2f}"]
        counts["tp"] += true_positives
        counts["fp"] += false_positives
        counts["fn"] += false_negatives


def _finalize_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    target_count = statistics["target_instance_count"]
    detail_keys = {
        "metric_sums",
        "center_errors",
        "area_ratios",
        "threshold_counts",
    }
    report = {
        key: value
        for key, value in statistics.items()
        if key not in detail_keys
    }
    for name, value in statistics["metric_sums"].items():
        report[f"mean_{name}_per_target"] = (
            float(value / target_count) if target_count else None
        )
    report["mean_center_error_px_on_predictions"] = (
        float(np.mean(statistics["center_errors"]))
        if statistics["center_errors"]
        else None
    )
    report["mean_area_ratio_on_predictions"] = (
        float(np.mean(statistics["area_ratios"]))
        if statistics["area_ratios"]
        else None
    )
    report["at_iou"] = {}
    for threshold, counts in statistics["threshold_counts"].items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report["at_iou"][threshold] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return report


def _load_detection_groups(
    path: Path,
    target_keys: set[TargetKey],
) -> tuple[dict[TargetKey, list[dict[str, Any]]], dict[str, int]]:
    records = _load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"detection file must contain a list: {path}")
    groups: defaultdict[TargetKey, list[dict[str, Any]]] = defaultdict(list)
    target_record_count = 0
    for index, record in enumerate(records):
        key = (
            int(record["scene_id"]),
            int(record["image_id"]),
            int(record["category_id"]),
        )
        if key not in target_keys:
            continue
        if "segmentation" not in record:
            raise ValueError(f"detection {index} has no segmentation: {path}")
        groups[key].append({**record, "source_index": index})
        target_record_count += 1
    return dict(groups), {
        "file_record_count": len(records),
        "target_record_count": target_record_count,
        "ignored_non_target_record_count": len(records) - target_record_count,
    }


def _load_gt_group(
    dataset_root: Path,
    split: str,
    scene_id: int,
    image_id: int,
    object_id: int,
    minimum_visibility: float,
    scene_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    if scene_id not in scene_cache:
        scene_root = dataset_root / split / f"{scene_id:06d}"
        scene_cache[scene_id] = (
            _load_json(scene_root / "scene_gt.json"),
            _load_json(scene_root / "scene_gt_info.json"),
        )
    scene_gt, scene_info = scene_cache[scene_id]
    image_key = str(image_id)
    gt_entries = scene_gt[image_key]
    info_entries = scene_info[image_key]
    candidates: list[dict[str, Any]] = []
    for gt_index, (gt, info) in enumerate(zip(gt_entries, info_entries)):
        visibility = float(info.get("visib_fract", float("nan")))
        if int(gt["obj_id"]) != object_id or not np.isfinite(visibility):
            continue
        if visibility <= minimum_visibility:
            continue
        mask_path = (
            dataset_root
            / split
            / f"{scene_id:06d}"
            / "mask_visib"
            / f"{image_id:06d}_{gt_index:06d}.png"
        )
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise FileNotFoundError(mask_path)
        candidates.append(
            {
                "gt_index": gt_index,
                "visibility": visibility,
                "mask": mask_image > 0,
            }
        )
    return candidates


def _greedy_detailed_matches(
    predictions: Sequence[dict[str, Any]],
    gt_instances: Sequence[dict[str, Any]],
    boundary_tolerance_px: int,
) -> list[dict[str, Any]]:
    unmatched_gt = set(range(len(gt_instances)))
    matches: list[dict[str, Any]] = []
    for prediction_rank, prediction in enumerate(predictions):
        best_index = None
        best_metrics = None
        for candidate_index in unmatched_gt:
            metrics = mask_pair_metrics(
                prediction["mask"],
                gt_instances[candidate_index]["mask"],
                boundary_tolerance_px=boundary_tolerance_px,
            )
            if best_metrics is None or metrics["iou"] > best_metrics["iou"]:
                best_index = candidate_index
                best_metrics = metrics
        if best_index is not None:
            unmatched_gt.remove(best_index)
            gt_instance = gt_instances[best_index]
        else:
            gt_instance = None
        matches.append(
            {
                "prediction_rank": prediction_rank,
                "prediction_index": prediction["source_index"],
                "prediction_score": float(prediction.get("score", 1.0)),
                "matched_gt_index": (
                    gt_instance["gt_index"] if gt_instance is not None else None
                ),
                "gt_visibility": (
                    gt_instance["visibility"] if gt_instance is not None else None
                ),
                "status": "matched" if gt_instance is not None else "extra_prediction",
                **(
                    best_metrics
                    if best_metrics is not None
                    else {
                        "iou": 0.0,
                        "pixel_precision": 0.0,
                        "pixel_recall": 0.0,
                        "boundary_f1": 0.0,
                        "center_error_px": None,
                        "area_ratio": None,
                    }
                ),
            }
        )
    return matches


def evaluate_bop_mask_sources(
    dataset_root: Path | str,
    targets_path: Path | str,
    split: str,
    detection_sources: Mapping[str, Path | str],
    object_ids: Iterable[int] | None = None,
    minimum_visibility: float = 0.1,
    prediction_factor: float = 1.0,
    opening_kernels: Mapping[str, int] | None = None,
    iou_thresholds: Sequence[float] = (0.5, 0.75),
    boundary_tolerance_px: int = 2,
    limit_target_groups: int | None = None,
    dataset_name: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate GT-oracle and detection masks under a common BOP19 protocol."""

    dataset_root = Path(dataset_root).resolve()
    targets_path = Path(targets_path).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if prediction_factor <= 0.0:
        raise ValueError("prediction_factor must be positive")
    if not 0.0 <= minimum_visibility <= 1.0:
        raise ValueError("minimum_visibility must lie in [0, 1]")
    thresholds = tuple(float(value) for value in iou_thresholds)
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("IoU thresholds must lie in (0, 1]")
    selected_objects = (
        None if object_ids is None else {int(value) for value in object_ids}
    )
    targets = _load_json(targets_path)
    filtered_targets = [
        {
            "scene_id": int(target["scene_id"]),
            "image_id": int(target["im_id"]),
            "object_id": int(target["obj_id"]),
            "instance_count": int(target["inst_count"]),
        }
        for target in targets
        if selected_objects is None or int(target["obj_id"]) in selected_objects
    ]
    filtered_targets.sort(
        key=lambda target: (
            target["scene_id"], target["image_id"], target["object_id"]
        )
    )
    if limit_target_groups is not None:
        if limit_target_groups <= 0:
            raise ValueError("limit_target_groups must be positive")
        filtered_targets = filtered_targets[:limit_target_groups]
    if not filtered_targets:
        raise ValueError("no BOP targets selected")
    target_keys = {
        (target["scene_id"], target["image_id"], target["object_id"])
        for target in filtered_targets
    }
    if len(target_keys) != len(filtered_targets):
        raise ValueError("duplicate scene/image/object entries in target list")

    source_names = ["gt", *detection_sources.keys()]
    if len(set(source_names)) != len(source_names):
        raise ValueError("mask source names must be unique and cannot replace gt")
    opening_kernels = dict(opening_kernels or {})
    for name in source_names:
        kernel = int(opening_kernels.get(name, 0))
        if kernel < 0 or (kernel and kernel % 2 == 0):
            raise ValueError(f"invalid opening kernel for {name}: {kernel}")

    detection_groups = {}
    detection_metadata = {}
    for name, path_value in detection_sources.items():
        path = Path(path_value).resolve()
        groups, metadata = _load_detection_groups(path, target_keys)
        detection_groups[name] = groups
        detection_metadata[name] = {"path": str(path), **metadata}

    source_statistics = {
        name: _empty_statistics(thresholds) for name in source_names
    }
    per_object_statistics: dict[str, dict[int, dict[str, Any]]] = {
        name: defaultdict(lambda: _empty_statistics(thresholds))
        for name in source_names
    }
    rows: list[dict[str, Any]] = []
    scene_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    target_mismatch_groups = []

    for target in filtered_targets:
        scene_id = target["scene_id"]
        image_id = target["image_id"]
        object_id = target["object_id"]
        expected_count = target["instance_count"]
        key = (scene_id, image_id, object_id)
        gt_instances = _load_gt_group(
            dataset_root,
            split,
            scene_id,
            image_id,
            object_id,
            minimum_visibility,
            scene_cache,
        )
        if len(gt_instances) < expected_count:
            raise ValueError(
                f"target {key} requests {expected_count} instances but only "
                f"{len(gt_instances)} GT masks pass visibility"
            )
        if len(gt_instances) != expected_count:
            target_mismatch_groups.append(
                {
                    "scene_id": scene_id,
                    "image_id": image_id,
                    "object_id": object_id,
                    "target_instances": expected_count,
                    "eligible_gt_instances": len(gt_instances),
                }
            )
        gt_masks = [instance["mask"] for instance in gt_instances]
        gt_shape = gt_masks[0].shape

        for source_name in source_names:
            if source_name == "gt":
                oracle_instances = sorted(
                    gt_instances,
                    key=lambda instance: (
                        -instance["visibility"], instance["gt_index"]
                    ),
                )[:expected_count]
                available = [
                    {
                        "mask": instance["mask"],
                        "score": 1.0,
                        "source_index": instance["gt_index"],
                    }
                    for instance in oracle_instances
                ]
                selected = available
            else:
                available_records = detection_groups[source_name].get(key, [])
                available = sorted(
                    available_records,
                    key=lambda record: (
                        -float(record.get("score", 0.0)),
                        int(record["source_index"]),
                    ),
                )
                maximum_predictions = int(prediction_factor * expected_count)
                selected_records = available[:maximum_predictions]
                selected = []
                kernel_size = int(opening_kernels.get(source_name, 0))
                for record in selected_records:
                    mask = decode_coco_rle(record["segmentation"])
                    mask = _align_mask_shape(mask, gt_shape)
                    mask = _morphological_open(mask, kernel_size)
                    selected.append(
                        {
                            "mask": mask,
                            "score": float(record.get("score", 0.0)),
                            "source_index": int(record["source_index"]),
                        }
                    )

            statistics_targets = [
                source_statistics[source_name],
                per_object_statistics[source_name][object_id],
            ]
            for statistics in statistics_targets:
                statistics["target_group_count"] += 1
                statistics["target_instance_count"] += expected_count
                statistics["eligible_gt_instance_count"] += len(gt_instances)
                statistics["available_prediction_count"] += len(available)
                statistics["selected_prediction_count"] += len(selected)
                statistics["discarded_prediction_count"] += max(
                    0, len(available) - len(selected)
                )
                statistics["missing_prediction_slot_count"] += max(
                    0, expected_count - len(selected)
                )
                if not selected:
                    statistics["groups_without_prediction"] += 1
                _update_detection_counts(
                    statistics,
                    selected,
                    gt_masks,
                    expected_count,
                    thresholds,
                )

            detailed_matches = _greedy_detailed_matches(
                selected,
                gt_instances,
                boundary_tolerance_px,
            )
            target_slot_matches = detailed_matches[:expected_count]
            for match in target_slot_matches:
                for statistics in statistics_targets:
                    for metric_name in (
                        "iou",
                        "pixel_precision",
                        "pixel_recall",
                        "boundary_f1",
                    ):
                        statistics["metric_sums"][metric_name] += float(
                            match[metric_name]
                        )
                    if match["center_error_px"] is not None:
                        statistics["center_errors"].append(
                            float(match["center_error_px"])
                        )
                    if match["area_ratio"] is not None:
                        statistics["area_ratios"].append(
                            float(match["area_ratio"])
                        )
            common_row = {
                "source": source_name,
                "scene_id": scene_id,
                "image_id": image_id,
                "object_id": object_id,
                "expected_instances": expected_count,
                "eligible_gt_instances": len(gt_instances),
                "source_available_predictions": len(available),
                "source_selected_predictions": len(selected),
            }
            rows.extend({**common_row, **match} for match in detailed_matches)
            for missing_rank in range(len(selected), expected_count):
                rows.append(
                    {
                        **common_row,
                        "prediction_rank": missing_rank,
                        "prediction_index": None,
                        "prediction_score": None,
                        "matched_gt_index": None,
                        "gt_visibility": None,
                        "status": "missing_prediction",
                        "iou": 0.0,
                        "pixel_precision": 0.0,
                        "pixel_recall": 0.0,
                        "boundary_f1": 0.0,
                        "center_error_px": None,
                        "area_ratio": None,
                    }
                )

    source_reports = {}
    for source_name in source_names:
        source_reports[source_name] = {
            "kind": "gt_oracle" if source_name == "gt" else "bop_detection",
            "opening_kernel": int(opening_kernels.get(source_name, 0)),
            "input": detection_metadata.get(source_name),
            "overall": _finalize_statistics(source_statistics[source_name]),
            "per_object": {
                str(object_id): _finalize_statistics(statistics)
                for object_id, statistics in sorted(
                    per_object_statistics[source_name].items()
                )
            },
        }

    comparison_metrics = (
        "mean_iou_per_target",
        "mean_pixel_precision_per_target",
        "mean_pixel_recall_per_target",
        "mean_boundary_f1_per_target",
    )
    comparisons = {}
    gt_overall = source_reports["gt"]["overall"]
    for source_name in source_names[1:]:
        source_overall = source_reports[source_name]["overall"]
        comparisons[f"{source_name}_minus_gt"] = {
            name: source_overall[name] - gt_overall[name]
            for name in comparison_metrics
        }
    if "cnos" in source_reports:
        cnos_overall = source_reports["cnos"]["overall"]
        for source_name in source_names[1:]:
            if source_name == "cnos":
                continue
            source_overall = source_reports[source_name]["overall"]
            comparisons[f"{source_name}_minus_cnos"] = {
                name: source_overall[name] - cnos_overall[name]
                for name in comparison_metrics
            }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name or dataset_root.name,
        "dataset_root": str(dataset_root),
        "split": split,
        "targets_path": str(targets_path),
        "protocol": {
            "target_definition": "BOP19 scene/image/object/inst_count",
            "gt_reference": "visible instance mask",
            "minimum_visibility_exclusive": minimum_visibility,
            "prediction_selection": (
                "score-descending top int(prediction_factor * inst_count)"
            ),
            "prediction_factor": prediction_factor,
            "cnos_compatibility": "3x3 opening when opening_kernel=3",
            "matching": "score-ordered one-to-one IoU matching",
            "iou_thresholds": list(thresholds),
            "boundary_tolerance_px": boundary_tolerance_px,
            "unmatched_target_metrics": "zero",
        },
        "selected_object_ids": sorted(
            {target["object_id"] for target in filtered_targets}
        ),
        "targets": filtered_targets,
        "target_group_count": len(filtered_targets),
        "target_instance_count": sum(
            target["instance_count"] for target in filtered_targets
        ),
        "target_gt_count_mismatch_groups": target_mismatch_groups,
        "sources": source_reports,
        "comparisons": comparisons,
    }
    return report, rows


def _average_recall(
    errors: Sequence[float], denominator: int, thresholds: np.ndarray
) -> float | None:
    if denominator <= 0:
        return None
    values = np.asarray(errors, dtype=np.float64)
    return float(
        np.mean(
            [
                float(np.count_nonzero(values < threshold)) / denominator
                for threshold in thresholds
            ]
        )
    )


def summarize_foundpose_results(
    result_root: Path | str,
    targets: Sequence[Mapping[str, Any]],
    object_ids: Iterable[int],
    image_width: int = 720,
) -> dict[str, Any]:
    """Summarize optional FoundPose outputs using the same target denominator."""

    root = Path(result_root).resolve()
    selected_objects = sorted({int(value) for value in object_ids})
    target_keys = {
        (int(target["scene_id"]), int(target["image_id"]), int(target["object_id"]))
        for target in targets
    }
    target_counts = defaultdict(int)
    for target in targets:
        target_counts[int(target["object_id"])] += int(target["instance_count"])
    records = []
    missing_object_ids = []
    if root.is_file():
        loaded = _load_json(root)
        if not isinstance(loaded, list):
            raise ValueError(f"pose result must contain a list: {root}")
        records.extend(loaded)
    else:
        for object_id in selected_objects:
            path = root / str(object_id) / "estimated-poses.json"
            if not path.is_file():
                missing_object_ids.append(object_id)
                continue
            loaded = _load_json(path)
            if not isinstance(loaded, list):
                raise ValueError(f"pose result must contain a list: {path}")
            records.extend(loaded)

    best_by_instance: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for record in records:
        scene_id = int(record["scene_id"])
        image_id = int(record.get("img_id", record.get("image_id")))
        object_id = int(record["obj_id"])
        if (scene_id, image_id, object_id) not in target_keys:
            continue
        key = (scene_id, image_id, object_id, int(record.get("inst_id", 0)))
        current = best_by_instance.get(key)
        if current is None or float(record.get("score", 0.0)) > float(
            current.get("score", 0.0)
        ):
            best_by_instance[key] = record

    mssd_thresholds = np.arange(0.05, 0.51, 0.05)
    mspd_thresholds = np.arange(5.0, 51.0, 5.0) * (image_width / 640.0)
    per_object = {}
    for object_id in selected_objects:
        object_records = [
            record
            for key, record in best_by_instance.items()
            if key[2] == object_id
        ]
        denominator = target_counts[object_id]
        mssd = [float(record["mssd"]) for record in object_records if "mssd" in record]
        mssd_n = [
            float(record["mssd_n"])
            for record in object_records
            if "mssd_n" in record
        ]
        mspd = [float(record["mspd"]) for record in object_records if "mspd" in record]
        per_object[str(object_id)] = {
            "target_instances": denominator,
            "predicted_instances": len(object_records),
            "prediction_coverage": (
                float(len(object_records) / denominator) if denominator else None
            ),
            "ar_mssd": _average_recall(mssd_n, denominator, mssd_thresholds),
            "ar_mspd": _average_recall(mspd, denominator, mspd_thresholds),
            "mean_mssd_mm_on_predictions": float(np.mean(mssd)) if mssd else None,
            "median_mssd_mm_on_predictions": (
                float(np.median(mssd)) if mssd else None
            ),
            "mean_mspd_px_on_predictions": float(np.mean(mspd)) if mspd else None,
            "median_mspd_px_on_predictions": (
                float(np.median(mspd)) if mspd else None
            ),
        }
    available = [item for item in per_object.values() if item["target_instances"]]
    return {
        "input": str(root),
        "protocol": "BOP19 target-denominator preliminary MSSD/MSPD",
        "warning": "VSD and official BOP matching are not included",
        "missing_object_ids": missing_object_ids,
        "target_instances": sum(item["target_instances"] for item in available),
        "predicted_instances": sum(item["predicted_instances"] for item in available),
        "prediction_coverage": (
            sum(item["predicted_instances"] for item in available)
            / sum(item["target_instances"] for item in available)
            if available
            else None
        ),
        "mean_object_ar_mssd": (
            float(np.mean([item["ar_mssd"] for item in available]))
            if available
            else None
        ),
        "mean_object_ar_mspd": (
            float(np.mean([item["ar_mspd"] for item in available]))
            if available
            else None
        ),
        "per_object": per_object,
    }
