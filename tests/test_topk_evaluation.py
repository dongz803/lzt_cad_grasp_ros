from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.topk_evaluation import (  # noqa: E402
    common_coverage_comparison,
    load_target_counts,
    oracle_topk_summary,
    select_policy_predictions,
    summarize_selected_predictions,
)


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
