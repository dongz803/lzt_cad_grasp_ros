"""Coverage-aware evaluation helpers for selective Top-K pose policies.

The gate is allowed to abstain, so metrics computed only on surviving poses are
not comparable to an always-predicting baseline.  This module keeps the BOP
target-instance denominator fixed and reports selective risk as a function of
coverage.  Record-local MSSD/MSPD values are useful for fast diagnostics; an
official BOP evaluation is still required for one-to-one GT matching and VSD.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TargetKey = tuple[int, int, int]
InstanceKey = tuple[int, int, int, int]
ImageKey = tuple[int, int]


def load_target_counts(
    targets: Iterable[Mapping[str, Any]],
    object_ids: set[int] | None = None,
) -> dict[TargetKey, int]:
    """Return BOP target counts keyed by scene, image and object."""

    counts: dict[TargetKey, int] = {}
    for target in targets:
        object_id = int(target["obj_id"])
        if object_ids is not None and object_id not in object_ids:
            continue
        key = (
            int(target["scene_id"]),
            int(target.get("im_id", target.get("img_id"))),
            object_id,
        )
        count = int(target["inst_count"])
        if count < 0:
            raise ValueError(f"negative target count for {key}")
        counts[key] = counts.get(key, 0) + count
    return counts


def load_detection_counts(
    detections: Iterable[Mapping[str, Any]],
    object_ids: set[int] | None = None,
) -> dict[TargetKey, int]:
    """Count detector outputs by BOP scene, image and object.

    CNOS exports use ``image_id``/``category_id`` while BOP pose records use
    ``im_id``/``obj_id``.  Supporting both forms keeps the evaluation helper
    independent of the detector implementation.
    """

    counts: dict[TargetKey, int] = defaultdict(int)
    for detection in detections:
        object_id = int(detection.get("obj_id", detection.get("category_id")))
        if object_ids is not None and object_id not in object_ids:
            continue
        key = (
            int(detection["scene_id"]),
            int(
                detection.get(
                    "im_id", detection.get("img_id", detection.get("image_id"))
                )
            ),
            object_id,
        )
        counts[key] += 1
    return dict(counts)


def _record_target_key(record: Mapping[str, Any]) -> TargetKey:
    return (
        int(record["scene_id"]),
        int(record.get("img_id", record.get("im_id"))),
        int(record["obj_id"]),
    )


def _record_instance_key(record: Mapping[str, Any]) -> InstanceKey:
    return (*_record_target_key(record), int(record.get("inst_id", 0)))


def _score(record: Mapping[str, Any], score_key: str) -> float:
    try:
        value = float(record.get(score_key, 0.0))
    except (TypeError, ValueError):
        return -np.inf
    return value if np.isfinite(value) else -np.inf


def _error(record: Mapping[str, Any], field: str) -> float:
    try:
        value = float(record.get(field, np.inf))
    except (TypeError, ValueError):
        return np.inf
    return value if np.isfinite(value) else np.inf


def _group_candidates(
    records: Sequence[dict[str, Any]],
    target_counts: Mapping[TargetKey, int],
) -> dict[InstanceKey, list[dict[str, Any]]]:
    groups: dict[InstanceKey, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if _record_target_key(record) in target_counts:
            groups[_record_instance_key(record)].append(record)
    return groups


def _capped_candidate_groups(
    records: Sequence[dict[str, Any]],
    target_counts: Mapping[TargetKey, int],
    score_key: str,
) -> dict[InstanceKey, list[dict[str, Any]]]:
    """Keep the highest-confidence detection groups within each target budget."""

    grouped = _group_candidates(records, target_counts)
    by_target: dict[
        TargetKey, list[tuple[InstanceKey, list[dict[str, Any]]]]
    ] = defaultdict(list)
    for instance_key, candidates in grouped.items():
        by_target[instance_key[:3]].append((instance_key, candidates))

    capped: dict[InstanceKey, list[dict[str, Any]]] = {}
    for target_key, groups in by_target.items():
        groups.sort(
            key=lambda item: (
                max(_score(record, score_key) for record in item[1]),
                -item[0][3],
            ),
            reverse=True,
        )
        capped.update(groups[: target_counts[target_key]])
    return capped


def recall_at_k(
    records: Sequence[dict[str, Any]],
    target_counts: Mapping[TargetKey, int],
    field: str,
    threshold: float,
    ks: Sequence[int] = (1, 3, 5),
    score_key: str = "score",
) -> dict[str, Any]:
    """Compute fixed-denominator candidate Recall@K.

    A target-associated detection is successful at K when at least one of its
    K highest-scoring hypotheses is below ``threshold``.  Missing detections,
    missing poses, invalid errors and abstentions remain failures through the
    fixed BOP target-instance denominator.
    """

    denominator = sum(target_counts.values())
    if denominator <= 0:
        raise ValueError("no target instances selected")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")
    normalized_ks = sorted({int(k) for k in ks})
    if not normalized_ks or normalized_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")

    groups = _capped_candidate_groups(records, target_counts, score_key)
    successes = {k: 0 for k in normalized_ks}
    groups_with_valid_error = 0
    for candidates in groups.values():
        ranked = sorted(
            candidates,
            key=lambda item: (
                _score(item, score_key),
                -int(item.get("hypothesis_id", 0)),
            ),
            reverse=True,
        )
        if any(np.isfinite(_error(item, field)) for item in ranked):
            groups_with_valid_error += 1
        for k in normalized_ks:
            if any(
                _error(item, field) < threshold
                for item in ranked[:k]
            ):
                successes[k] += 1

    predicted_groups = len(groups)
    return {
        "field": field,
        "success_threshold_exclusive": float(threshold),
        "target_instances": denominator,
        "predicted_instance_groups": predicted_groups,
        "groups_with_valid_error": groups_with_valid_error,
        "missing_or_abstained_instances": denominator - predicted_groups,
        "values": {
            str(k): {
                "successful_target_instances": successes[k],
                "recall_fixed_target_denominator": successes[k] / denominator,
                "recall_on_predicted_groups": (
                    successes[k] / predicted_groups if predicted_groups else 0.0
                ),
            }
            for k in normalized_ks
        },
    }


def _failure_waterfall_counts(
    raw_records: Sequence[dict[str, Any]],
    policy_records: Sequence[dict[str, Any]],
    detection_counts: Mapping[TargetKey, int],
    target_counts: Mapping[TargetKey, int],
    field: str,
    threshold: float,
    raw_score_key: str,
    policy_score_key: str,
) -> tuple[dict[str, int], dict[str, int]]:
    raw_groups = _capped_candidate_groups(raw_records, target_counts, raw_score_key)
    policy_groups = _group_candidates(policy_records, target_counts)
    stages = {
        "missing_detection": 0,
        "detection_without_pose": 0,
        "all_hypotheses_hard_rejected": 0,
        "topk_has_no_correct_candidate": 0,
        "correct_candidate_but_wrong_top1": 0,
        "selected_top1_correct": 0,
    }
    diagnostics = {
        "detections_over_target_budget": 0,
        "raw_pose_groups_over_detected_budget": 0,
        "policy_groups_not_in_raw_budget": 0,
    }

    raw_by_target: dict[TargetKey, list[InstanceKey]] = defaultdict(list)
    for key in raw_groups:
        raw_by_target[key[:3]].append(key)
    policy_by_target: dict[TargetKey, set[InstanceKey]] = defaultdict(set)
    for key in policy_groups:
        policy_by_target[key[:3]].add(key)

    for target_key, target_count in target_counts.items():
        detected_total = int(detection_counts.get(target_key, 0))
        detected = min(target_count, detected_total)
        diagnostics["detections_over_target_budget"] += max(
            0, detected_total - target_count
        )
        stages["missing_detection"] += target_count - detected

        target_raw_keys = raw_by_target.get(target_key, [])
        raw_count = min(len(target_raw_keys), detected)
        diagnostics["raw_pose_groups_over_detected_budget"] += max(
            0, len(target_raw_keys) - detected
        )
        stages["detection_without_pose"] += detected - raw_count
        target_raw_keys = target_raw_keys[:raw_count]

        surviving_keys = [
            key for key in target_raw_keys if key in policy_by_target[target_key]
        ]
        stages["all_hypotheses_hard_rejected"] += (
            raw_count - len(surviving_keys)
        )
        diagnostics["policy_groups_not_in_raw_budget"] += len(
            policy_by_target[target_key] - set(target_raw_keys)
        )

        for key in surviving_keys:
            ranked = sorted(
                policy_groups[key],
                key=lambda item: (
                    _score(item, policy_score_key),
                    -int(item.get("hypothesis_id", 0)),
                ),
                reverse=True,
            )
            correct = [
                _error(item, field) < threshold
                for item in ranked
            ]
            if not any(correct):
                stages["topk_has_no_correct_candidate"] += 1
            elif correct[0]:
                stages["selected_top1_correct"] += 1
            else:
                stages["correct_candidate_but_wrong_top1"] += 1
    return stages, diagnostics


def failure_waterfall(
    raw_records: Sequence[dict[str, Any]],
    policy_records: Sequence[dict[str, Any]],
    detection_counts: Mapping[TargetKey, int],
    target_counts: Mapping[TargetKey, int],
    field: str = "mssd_n",
    threshold: float = 0.1,
    raw_score_key: str = "score",
    policy_score_key: str = "score",
) -> dict[str, Any]:
    """Attribute every target instance to one mutually exclusive outcome."""

    denominator = sum(target_counts.values())
    if denominator <= 0:
        raise ValueError("no target instances selected")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")

    def summarize(subset: Mapping[TargetKey, int]) -> dict[str, Any]:
        counts, diagnostics = _failure_waterfall_counts(
            raw_records,
            policy_records,
            detection_counts,
            subset,
            field,
            threshold,
            raw_score_key,
            policy_score_key,
        )
        total = sum(subset.values())
        attributed = sum(counts.values())
        if attributed != total:
            raise RuntimeError(
                f"failure waterfall is not exhaustive: {attributed} != {total}"
            )
        flow_counts = {
            "target_instances": total,
            "after_object_detection": total - counts["missing_detection"],
            "after_pose_generation": (
                total
                - counts["missing_detection"]
                - counts["detection_without_pose"]
            ),
            "after_hard_gate": (
                total
                - counts["missing_detection"]
                - counts["detection_without_pose"]
                - counts["all_hypotheses_hard_rejected"]
            ),
            "topk_contains_correct_candidate": (
                counts["correct_candidate_but_wrong_top1"]
                + counts["selected_top1_correct"]
            ),
            "selected_top1_correct": counts["selected_top1_correct"],
        }
        return {
            "target_instances": total,
            "flow": [
                {
                    "name": name,
                    "remaining_count": count,
                    "fraction_of_targets": count / total,
                }
                for name, count in flow_counts.items()
            ],
            "stages": [
                {
                    "name": name,
                    "count": count,
                    "fraction_of_targets": count / total,
                }
                for name, count in counts.items()
            ],
            "successful_target_instances": counts["selected_top1_correct"],
            "fixed_denominator_success_rate": (
                counts["selected_top1_correct"] / total
            ),
            "attributed_target_instances": attributed,
            "input_consistency": diagnostics,
        }

    result = {
        "correctness_field": field,
        "success_threshold_exclusive": float(threshold),
        "detection_attribution": (
            "object-class detection slots per scene/image, capped by BOP "
            "inst_count; this is pipeline-stage accounting, not mask-to-GT "
            "IoU matching"
        ),
        "stage_semantics": (
            "mutually exclusive first-failure attribution; all stages sum to "
            "the fixed BOP target-instance denominator"
        ),
        "aggregate": summarize(target_counts),
        "per_object": {},
    }
    for object_id in sorted({key[2] for key in target_counts}):
        subset = {
            key: count
            for key, count in target_counts.items()
            if key[2] == object_id
        }
        result["per_object"][str(object_id)] = summarize(subset)
    return result


def select_policy_predictions(
    records: Sequence[dict[str, Any]],
    target_counts: Mapping[TargetKey, int],
    score_key: str = "score",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select one pose per detected instance and respect each BOP target budget.

    FoundPose ``inst_id`` identifies a ranked detection, not a ground-truth
    instance.  The official evaluator performs final one-to-one GT matching.
    Here we first select the policy's Top-1 pose for each detection and cap the
    number of detections for every image/object pair to ``inst_count``.
    """

    grouped: dict[InstanceKey, list[dict[str, Any]]] = defaultdict(list)
    in_protocol_candidates = 0
    for record in records:
        target_key = _record_target_key(record)
        if target_key not in target_counts:
            continue
        grouped[_record_instance_key(record)].append(record)
        in_protocol_candidates += 1

    selected_by_target: dict[TargetKey, list[dict[str, Any]]] = defaultdict(list)
    for group in grouped.values():
        selected = max(
            group,
            key=lambda item: (
                _score(item, score_key),
                -int(item.get("hypothesis_id", 0)),
            ),
        )
        selected_by_target[_record_target_key(selected)].append(selected)

    selected_records: list[dict[str, Any]] = []
    over_budget = 0
    for target_key, predictions in selected_by_target.items():
        predictions.sort(
            key=lambda item: (
                _score(item, score_key),
                -int(item.get("inst_id", 0)),
            ),
            reverse=True,
        )
        target_count = target_counts[target_key]
        selected_records.extend(predictions[:target_count])
        over_budget += max(0, len(predictions) - target_count)

    selected_records.sort(
        key=lambda item: (
            int(item["obj_id"]),
            int(item["scene_id"]),
            int(item.get("img_id", item.get("im_id"))),
            int(item.get("inst_id", 0)),
        )
    )
    diagnostics = {
        "input_candidate_count": len(records),
        "in_protocol_candidate_count": in_protocol_candidates,
        "out_of_protocol_candidate_count": len(records) - in_protocol_candidates,
        "predicted_instance_groups_before_cap": len(grouped),
        "over_target_budget_group_count": over_budget,
        "selected_prediction_count": len(selected_records),
    }
    return selected_records, diagnostics


def _finite_values(
    records: Sequence[Mapping[str, Any]], field: str
) -> np.ndarray:
    values = []
    for record in records:
        if field not in record:
            continue
        value = float(record[field])
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def fixed_denominator_metric(
    records: Sequence[Mapping[str, Any]],
    field: str,
    thresholds: Sequence[float],
    denominator: int,
) -> dict[str, Any]:
    """Summarize threshold recall without dropping missing predictions."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    if threshold_array.ndim != 1 or threshold_array.size == 0:
        raise ValueError("thresholds must be a non-empty sequence")
    values = _finite_values(records, field)
    successes = np.asarray(
        [np.count_nonzero(values < threshold) for threshold in threshold_array],
        dtype=np.float64,
    )
    recalls = successes / float(denominator)
    conditional_recalls = (
        successes / float(len(records))
        if records
        else np.zeros_like(successes)
    )
    return {
        "field": field,
        "thresholds": threshold_array.tolist(),
        "prediction_count": len(records),
        "evaluated_prediction_count": int(values.size),
        "missing_or_invalid_error_count": len(records) - int(values.size),
        "recalls_fixed_target_denominator": recalls.tolist(),
        "average_recall_fixed_target_denominator": float(np.mean(recalls)),
        "average_recall_on_predictions_only": float(
            np.mean(conditional_recalls)
        ),
        "median_error_on_predictions": (
            None if values.size == 0 else float(np.median(values))
        ),
        "mean_error_on_predictions": (
            None if values.size == 0 else float(np.mean(values))
        ),
    }


def risk_coverage_curve(
    records: Sequence[dict[str, Any]],
    denominator: int,
    score_key: str,
    mssd_thresholds: Sequence[float],
    mspd_thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Return prefix-selective risk while retaining the fixed target coverage."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    ranked = sorted(records, key=lambda item: _score(item, score_key), reverse=True)
    mssd_thresholds_array = np.asarray(mssd_thresholds, dtype=np.float64)
    mspd_thresholds_array = np.asarray(mspd_thresholds, dtype=np.float64)
    mssd_successes = np.zeros(mssd_thresholds_array.size, dtype=np.int64)
    mspd_successes = np.zeros(mspd_thresholds_array.size, dtype=np.int64)
    points = []
    for rank, record in enumerate(ranked, start=1):
        mssd = float(record.get("mssd_n", np.inf))
        mspd = float(record.get("mspd", np.inf))
        if np.isfinite(mssd):
            mssd_successes += mssd < mssd_thresholds_array
        if np.isfinite(mspd):
            mspd_successes += mspd < mspd_thresholds_array
        conditional_ar_mssd = float(np.mean(mssd_successes / rank))
        conditional_ar_mspd = float(np.mean(mspd_successes / rank))
        points.append(
            {
                "accepted_predictions": rank,
                "coverage": rank / denominator,
                "minimum_confidence": _score(record, score_key),
                "conditional_ar_mssd": conditional_ar_mssd,
                "conditional_ar_mspd": conditional_ar_mspd,
                "selective_risk_mssd": 1.0 - conditional_ar_mssd,
                "selective_risk_mspd": 1.0 - conditional_ar_mspd,
                "fixed_denominator_ar_mssd": float(
                    np.mean(mssd_successes / denominator)
                ),
                "fixed_denominator_ar_mspd": float(
                    np.mean(mspd_successes / denominator)
                ),
            }
        )
    return points


def summarize_selected_predictions(
    records: Sequence[dict[str, Any]],
    target_counts: Mapping[TargetKey, int],
    score_key: str = "score",
    mssd_thresholds: Sequence[float] = tuple(np.arange(0.05, 0.51, 0.05)),
    mspd_thresholds: Sequence[float] = tuple(
        np.arange(5.0, 51.0, 5.0) * (720.0 / 640.0)
    ),
) -> dict[str, Any]:
    """Build end-to-end and per-object summaries for a selected policy."""

    denominator = sum(target_counts.values())
    if denominator <= 0:
        raise ValueError("no target instances selected")
    object_ids = sorted({key[2] for key in target_counts})

    def summarize_subset(
        subset_records: Sequence[dict[str, Any]], subset_denominator: int
    ) -> dict[str, Any]:
        return {
            "target_instances": subset_denominator,
            "selected_predictions": len(subset_records),
            "coverage": len(subset_records) / subset_denominator,
            "missing_or_abstained_instances": subset_denominator - len(subset_records),
            "mssd": fixed_denominator_metric(
                subset_records,
                "mssd_n",
                mssd_thresholds,
                subset_denominator,
            ),
            "mspd": fixed_denominator_metric(
                subset_records,
                "mspd",
                mspd_thresholds,
                subset_denominator,
            ),
        }

    aggregate = summarize_subset(records, denominator)
    per_object = {}
    for object_id in object_ids:
        object_records = [
            record for record in records if int(record["obj_id"]) == object_id
        ]
        object_denominator = sum(
            count for key, count in target_counts.items() if key[2] == object_id
        )
        per_object[str(object_id)] = summarize_subset(
            object_records, object_denominator
        )
    aggregate["per_object"] = per_object
    aggregate["risk_coverage"] = risk_coverage_curve(
        records,
        denominator,
        score_key,
        mssd_thresholds,
        mspd_thresholds,
    )
    return aggregate


def oracle_topk_summary(
    records: Sequence[dict[str, Any]],
    target_counts: Mapping[TargetKey, int],
    field: str,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Report evaluation-only Top-K headroom using the lowest GT error."""

    grouped: dict[InstanceKey, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if _record_target_key(record) in target_counts:
            grouped[_record_instance_key(record)].append(record)
    oracle_records = []
    for group in grouped.values():
        valid = [item for item in group if np.isfinite(float(item.get(field, np.inf)))]
        if valid:
            oracle_records.append(min(valid, key=lambda item: float(item[field])))

    # If an input violates the target budget, keep the lowest-error oracle
    # detections. This is an evaluation ceiling and is never executable.
    by_target: dict[TargetKey, list[dict[str, Any]]] = defaultdict(list)
    for record in oracle_records:
        by_target[_record_target_key(record)].append(record)
    capped = []
    for key, predictions in by_target.items():
        predictions.sort(key=lambda item: float(item[field]))
        capped.extend(predictions[: target_counts[key]])
    summary = fixed_denominator_metric(
        capped, field, thresholds, sum(target_counts.values())
    )
    summary["warning"] = "GT oracle for analysis only; not an executable policy"
    return summary


def common_coverage_comparison(
    methods: Mapping[str, tuple[Sequence[dict[str, Any]], str]],
    denominator: int,
    mssd_thresholds: Sequence[float],
    mspd_thresholds: Sequence[float],
    fractions: Sequence[float] = tuple(np.arange(0.1, 1.01, 0.1)),
) -> dict[str, Any]:
    """Compare methods at identical accepted-prediction counts."""

    if not methods:
        return {"common_prediction_count": 0, "points": []}
    common_count = min(len(records) for records, _ in methods.values())
    counts = sorted(
        {
            max(1, min(common_count, int(round(common_count * fraction))))
            for fraction in fractions
        }
    ) if common_count else []
    points = []
    for count in counts:
        point: dict[str, Any] = {
            "accepted_predictions": count,
            "coverage": count / denominator,
            "methods": {},
        }
        for name, (records, score_key) in methods.items():
            ranked = sorted(
                records, key=lambda item: _score(item, score_key), reverse=True
            )[:count]
            mssd = fixed_denominator_metric(
                ranked, "mssd_n", mssd_thresholds, count
            )
            mspd = fixed_denominator_metric(
                ranked, "mspd", mspd_thresholds, count
            )
            point["methods"][name] = {
                "conditional_ar_mssd": mssd[
                    "average_recall_fixed_target_denominator"
                ],
                "conditional_ar_mspd": mspd[
                    "average_recall_fixed_target_denominator"
                ],
                "selective_risk_mssd": 1.0
                - mssd["average_recall_fixed_target_denominator"],
                "selective_risk_mspd": 1.0
                - mspd["average_recall_fixed_target_denominator"],
            }
        points.append(point)
    return {
        "common_prediction_count": common_count,
        "common_max_coverage": common_count / denominator,
        "points": points,
    }


def aggregate_bop_image_runtimes(
    records: Sequence[Mapping[str, Any]],
) -> dict[ImageKey, float]:
    """Aggregate FoundPose runtime once per instance for each BOP image.

    FoundPose stores the complete instance-level runtime dictionary on every
    Top-K hypothesis.  Summing every record would therefore multiply the
    runtime by the number of hypotheses.  Different detected instances in the
    same image are still additive, while the shared detection time is counted
    only once per image.
    """

    detection_time: dict[ImageKey, float] = {}
    instance_time: dict[InstanceKey, float] = {}
    for record in records:
        image_key = (
            int(record["scene_id"]),
            int(record.get("img_id", record.get("im_id"))),
        )
        if "inst_id" not in record:
            raise ValueError(
                "FoundPose records require inst_id for Top-K runtime aggregation"
            )
        instance_key = (*image_key, int(record["obj_id"]), int(record["inst_id"]))
        detection_time[image_key] = max(
            detection_time.get(image_key, 0.0),
            float(record.get("cnos_time", 0.0)),
        )
        runtime = sum(float(value) for value in record.get("time", {}).values())
        previous = instance_time.get(instance_key)
        if previous is not None and not np.isclose(
            runtime, previous, rtol=1e-9, atol=1e-12
        ):
            raise ValueError(
                "inconsistent runtime across hypotheses for instance "
                f"{instance_key}: {previous} != {runtime}"
            )
        instance_time[instance_key] = runtime

    image_runtime: dict[ImageKey, float] = defaultdict(float)
    for instance_key, runtime in instance_time.items():
        image_runtime[instance_key[:2]] += runtime
    for image_key, runtime in detection_time.items():
        image_runtime[image_key] += runtime
    return dict(image_runtime)


def write_bop_csv(records: Sequence[dict[str, Any]], path: Path) -> None:
    """Export selected policy outputs for the official BOP evaluator."""

    image_runtime = aggregate_bop_image_runtimes(records)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for record in records:
            image_key = (
                int(record["scene_id"]),
                int(record.get("img_id", record.get("im_id"))),
            )
            rotation = np.asarray(record["R"], dtype=np.float64).reshape(-1)
            translation = np.asarray(record["t"], dtype=np.float64).reshape(-1)
            writer.writerow(
                [
                    image_key[0],
                    image_key[1],
                    int(record["obj_id"]),
                    float(record.get("score", 0.0)),
                    " ".join(map(str, rotation.tolist())),
                    " ".join(map(str, translation.tolist())),
                    image_runtime[image_key],
                ]
            )
