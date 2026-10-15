import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = PROJECT_ROOT / "scripts/evaluation/run_candidate_ranking_ablation.py"
    spec = importlib.util.spec_from_file_location("candidate_ranking_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_enrich_gated_record_exposes_single_cue_scores_without_mutating() -> None:
    script = _load_script()
    record = {
        "original_score": 0.4,
        "pose_gate_metrics": {
            "pnp_score": 0.4,
            "pnp_inliers": 17,
            "silhouette_iou": 0.6,
        },
        "reliability_features": {"depth_error_over_cad_scale": 0.08},
    }

    enriched = script.enrich_gated_record(record)

    assert enriched is not record
    assert "ablation_pnp_score" not in record
    assert enriched["ablation_pnp_score"] == pytest.approx(0.4)
    assert enriched["ablation_pnp_inliers"] == pytest.approx(17)
    assert enriched["ablation_silhouette_iou"] == pytest.approx(0.6)
    assert enriched["ablation_negative_depth_error_over_cad_scale"] == pytest.approx(
        -0.08
    )


def test_roc_auc_handles_ties() -> None:
    script = _load_script()

    assert script._roc_auc([False, True], [0.0, 1.0]) == pytest.approx(1.0)
    assert script._roc_auc([False, True], [1.0, 0.0]) == pytest.approx(0.0)
    assert script._roc_auc([False, True], [0.5, 0.5]) == pytest.approx(0.5)


def test_evaluation_command_uses_one_frozen_pool_for_gate_methods(tmp_path: Path) -> None:
    script = _load_script()
    raw = tmp_path / "raw.json"
    gated = tmp_path / "gated.json"

    command = script._evaluation_command(
        raw,
        gated,
        tmp_path / "evaluation",
        tmp_path / "detections.json",
        tmp_path / "targets.json",
        [1, 2],
        True,
        3,
    )

    methods = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--method"
    ]
    assert methods[0] == f"raw={raw}"
    assert all(value.endswith(f"={gated}") for value in methods[1:])
    assert "--run-official" in command
    assert command[command.index("--official-workers") + 1] == "3"


def test_named_official_cache_rejects_unknown_and_duplicate_methods() -> None:
    script = _load_script()

    assert script._named_paths(["raw=/tmp/result"], "--official-cache") == {
        "raw": Path("/tmp/result")
    }
    with pytest.raises(ValueError, match="known METHOD"):
        script._named_paths(["unknown=/tmp/result"], "--official-cache")
    with pytest.raises(ValueError, match="duplicate"):
        script._named_paths(
            ["raw=/tmp/first", "raw=/tmp/second"], "--official-cache"
        )


def test_paired_top1_reports_discordant_exact_test() -> None:
    script = _load_script()
    records = []
    for group_id, first_correct, second_correct in (
        (1, True, False),
        (2, True, False),
        (3, False, True),
        (4, True, True),
    ):
        records.extend(
            [
                {
                    "scene_id": 1,
                    "img_id": group_id,
                    "obj_id": 1,
                    "inst_id": 0,
                    "hypothesis_id": 0,
                    "mssd_n": 0.01,
                    "first": 1.0 if first_correct else 0.0,
                    "second": 1.0 if second_correct else 0.0,
                },
                {
                    "scene_id": 1,
                    "img_id": group_id,
                    "obj_id": 1,
                    "inst_id": 0,
                    "hypothesis_id": 1,
                    "mssd_n": 0.2,
                    "first": 0.0 if first_correct else 1.0,
                    "second": 0.0 if second_correct else 1.0,
                },
            ]
        )

    paired = script._paired_top1(records, "first", "second")

    assert paired["both_correct"] == 1
    assert paired["first_only_correct"] == 2
    assert paired["second_only_correct"] == 1
    assert paired["both_wrong"] == 0
    assert paired["mcnemar_exact_two_sided_p_value"] == pytest.approx(1.0)
