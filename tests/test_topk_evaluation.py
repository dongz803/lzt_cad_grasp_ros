from pathlib import Path
import csv
import importlib.util
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_benchmark.topk_evaluation import (  # noqa: E402
    aggregate_bop_image_runtimes,
    common_coverage_comparison,
    failure_waterfall,
    load_detection_counts,
    load_target_counts,
    oracle_topk_summary,
    recall_at_k,
    select_policy_predictions,
    summarize_selected_predictions,
    write_bop_csv,
)


def _load_evaluation_script():
    path = PROJECT_ROOT / "scripts/evaluation/evaluate_topk_coverage.py"
    spec = importlib.util.spec_from_file_location("evaluate_topk_coverage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bop_method_token_removes_filename_separator() -> None:
    script = _load_evaluation_script()

    assert script._bop_method_token("per_candidate") == "per-candidate"
    assert script._bop_method_token("mode aware") == "mode-aware"


def pose(
    image_id: int,
    instance_id: int,
    hypothesis_id: int,
    score: float,
    mssd_n: float,
    mspd: float,
) -> dict:
    return {
        "scene_id": 1,
        "img_id": image_id,
        "obj_id": 2,
        "inst_id": instance_id,
        "hypothesis_id": hypothesis_id,
        "score": score,
        "mssd_n": mssd_n,
        "mssd": 10.0 * mssd_n,
        "mspd": mspd,
        "R": np.eye(3).tolist(),
        "t": [[0.0], [0.0], [500.0]],
    }


def targets() -> list[dict]:
    return [
        {"scene_id": 1, "im_id": 1, "obj_id": 2, "inst_count": 2},
        {"scene_id": 1, "im_id": 2, "obj_id": 2, "inst_count": 1},
    ]


def test_fixed_denominator_keeps_missing_target_as_failure() -> None:
    counts = load_target_counts(targets(), {2})
    records = [
        pose(1, 0, 0, 0.9, 0.20, 20.0),
        pose(1, 0, 1, 0.8, 0.05, 5.0),
        pose(1, 1, 0, 0.6, 0.08, 8.0),
    ]
    selected, diagnostics = select_policy_predictions(records, counts)
    summary = summarize_selected_predictions(
        selected,
        counts,
        mssd_thresholds=(0.1, 0.2),
        mspd_thresholds=(10.0, 20.0),
    )

    assert diagnostics["selected_prediction_count"] == 2
    assert summary["target_instances"] == 3
    assert summary["coverage"] == 2 / 3
    assert summary["missing_or_abstained_instances"] == 1
    assert summary["mssd"]["average_recall_fixed_target_denominator"] == 1 / 3


def test_topk_oracle_reports_headroom_without_changing_policy_top1() -> None:
    counts = load_target_counts(targets(), {2})
    records = [
        pose(1, 0, 0, 0.9, 0.20, 20.0),
        pose(1, 0, 1, 0.8, 0.05, 5.0),
        pose(1, 1, 0, 0.6, 0.08, 8.0),
    ]
    selected, _ = select_policy_predictions(records, counts)
    policy = summarize_selected_predictions(
        selected,
        counts,
        mssd_thresholds=(0.1,),
        mspd_thresholds=(10.0,),
    )
    oracle = oracle_topk_summary(records, counts, "mssd_n", (0.1,))

    assert policy["mssd"]["average_recall_fixed_target_denominator"] == 1 / 3
    assert oracle["average_recall_fixed_target_denominator"] == 2 / 3
    assert "not an executable policy" in oracle["warning"]


def test_recall_at_k_uses_fixed_target_denominator() -> None:
    counts = load_target_counts(targets(), {2})
    records = [
        pose(1, 0, 0, 0.9, 0.20, 20.0),
        pose(1, 0, 1, 0.8, 0.05, 5.0),
        pose(1, 1, 0, 0.7, 0.08, 8.0),
    ]

    summary = recall_at_k(
        records, counts, "mssd_n", 0.1, ks=(1, 2), score_key="score"
    )

    assert summary["predicted_instance_groups"] == 2
    assert summary["values"]["1"]["successful_target_instances"] == 1
    assert summary["values"]["1"]["recall_fixed_target_denominator"] == 1 / 3
    assert summary["values"]["2"]["successful_target_instances"] == 2
    assert summary["values"]["2"]["recall_fixed_target_denominator"] == 2 / 3


def test_failure_waterfall_is_exhaustive_and_mutually_exclusive() -> None:
    target_payload = [
        {"scene_id": 1, "im_id": image_id, "obj_id": 2, "inst_count": 1}
        for image_id in range(1, 7)
    ]
    counts = load_target_counts(target_payload, {2})
    detections = [
        {"scene_id": 1, "image_id": image_id, "category_id": 2}
        for image_id in range(2, 7)
    ]
    detection_counts = load_detection_counts(detections, {2})
    raw = [
        pose(3, 0, 0, 0.9, 0.20, 20.0),
        pose(4, 0, 0, 0.9, 0.20, 20.0),
        pose(5, 0, 0, 0.9, 0.20, 20.0),
        pose(5, 0, 1, 0.8, 0.05, 5.0),
        pose(6, 0, 0, 0.9, 0.05, 5.0),
    ]
    policy = [record for record in raw if record["img_id"] != 3]

    report = failure_waterfall(
        raw,
        policy,
        detection_counts,
        counts,
        field="mssd_n",
        threshold=0.1,
    )

    stage_counts = {
        item["name"]: item["count"]
        for item in report["aggregate"]["stages"]
    }
    assert stage_counts == {
        "missing_detection": 1,
        "detection_without_pose": 1,
        "all_hypotheses_hard_rejected": 1,
        "topk_has_no_correct_candidate": 1,
        "correct_candidate_but_wrong_top1": 1,
        "selected_top1_correct": 1,
    }
    assert report["aggregate"]["attributed_target_instances"] == 6
    assert report["aggregate"]["fixed_denominator_success_rate"] == 1 / 6
    flow = {
        item["name"]: item["remaining_count"]
        for item in report["aggregate"]["flow"]
    }
    assert flow == {
        "target_instances": 6,
        "after_object_detection": 5,
        "after_pose_generation": 4,
        "after_hard_gate": 3,
        "topk_contains_correct_candidate": 2,
        "selected_top1_correct": 1,
    }


def test_target_budget_removes_extra_detection_groups() -> None:
    counts = load_target_counts(
        [{"scene_id": 1, "im_id": 1, "obj_id": 2, "inst_count": 1}],
        {2},
    )
    records = [
        pose(1, 0, 0, 0.3, 0.02, 2.0),
        pose(1, 1, 0, 0.8, 0.30, 30.0),
    ]
    selected, diagnostics = select_policy_predictions(records, counts)

    assert len(selected) == 1
    assert selected[0]["inst_id"] == 1
    assert diagnostics["over_target_budget_group_count"] == 1


def test_equal_coverage_uses_same_number_of_predictions() -> None:
    raw = [
        pose(1, 0, 0, 0.9, 0.3, 30.0),
        pose(1, 1, 0, 0.8, 0.2, 20.0),
    ]
    gated = [pose(1, 0, 0, 0.7, 0.05, 5.0)]
    comparison = common_coverage_comparison(
        {"raw": (raw, "score"), "gated": (gated, "score")},
        denominator=3,
        mssd_thresholds=(0.1,),
        mspd_thresholds=(10.0,),
        fractions=(1.0,),
    )

    assert comparison["common_prediction_count"] == 1
    assert comparison["points"][0]["accepted_predictions"] == 1
    assert comparison["points"][0]["coverage"] == 1 / 3
    assert comparison["points"][0]["methods"]["gated"][
        "selective_risk_mssd"
    ] == 0.0


def timed_pose(
    image_id: int,
    instance_id: int,
    hypothesis_id: int,
    runtime: dict[str, float],
    detection_time: float,
) -> dict:
    record = pose(image_id, instance_id, hypothesis_id, 0.9, 0.1, 10.0)
    record["time"] = runtime
    record["cnos_time"] = detection_time
    return record


def test_bop_runtime_counts_topk_instance_time_once() -> None:
    records = [
        timed_pose(1, 0, 0, {"prep": 1.0, "pnp": 0.5}, 0.25),
        timed_pose(1, 0, 1, {"prep": 1.0, "pnp": 0.5}, 0.25),
        timed_pose(1, 0, 2, {"prep": 1.0, "pnp": 0.5}, 0.25),
        timed_pose(1, 1, 0, {"prep": 2.0}, 0.25),
        timed_pose(2, 0, 0, {"prep": 0.75}, 0.10),
    ]

    runtimes = aggregate_bop_image_runtimes(records)

    assert runtimes == {(1, 1): 3.75, (1, 2): 0.85}


def test_write_bop_csv_uses_same_deduplicated_runtime_per_image(
    tmp_path: Path,
) -> None:
    records = [
        timed_pose(1, 0, 0, {"prep": 1.0}, 0.25),
        timed_pose(1, 0, 1, {"prep": 1.0}, 0.25),
        timed_pose(1, 1, 0, {"prep": 2.0}, 0.25),
    ]
    output = tmp_path / "topk.csv"

    write_bop_csv(records, output)

    with output.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert [float(row["time"]) for row in rows] == [3.25, 3.25, 3.25]


def test_bop_runtime_rejects_inconsistent_hypothesis_times() -> None:
    records = [
        timed_pose(1, 0, 0, {"prep": 1.0}, 0.25),
        timed_pose(1, 0, 1, {"prep": 2.0}, 0.25),
    ]

    with np.testing.assert_raises_regex(ValueError, "inconsistent runtime"):
        aggregate_bop_image_runtimes(records)
