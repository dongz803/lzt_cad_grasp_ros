from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATHS = {
    "run_industrial_cad_pick_place.py": "scripts/pipeline/run_industrial_cad_pick_place.py",
    "run_tless_obj4_pipeline.py": "scripts/pipeline/run_tless_obj4_pipeline.py",
    "run_innovation1_grasp_ablation.py": "scripts/evaluation/run_innovation1_grasp_ablation.py",
    "train_pose_reliability.py": "scripts/perception/train_pose_reliability.py",
}


def _load_script(name: str):
    path = PROJECT_ROOT / SCRIPT_PATHS[name]
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pick_place_run_directory_separates_mode_and_seed(tmp_path: Path) -> None:
    script = _load_script("run_industrial_cad_pick_place.py")
    script.PROJECT_ROOT = tmp_path
    args = Namespace(
        output=None,
        run_id="repeat_01",
        seed=23,
        check_only=True,
        pose_hypotheses=Path("prediction.json"),
    )

    output = script._prepare_output_directory(args, "part_a")

    assert output == (
        tmp_path
        / "outputs/industrial_cad_pick_place/part_a/predicted_check/repeat_01_seed23"
    )
    assert output.is_dir()


def test_explicit_nonempty_output_is_never_overwritten(tmp_path: Path) -> None:
    script = _load_script("run_industrial_cad_pick_place.py")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "result.json").write_text("{}\n", encoding="utf-8")
    args = Namespace(
        output=output,
        run_id=None,
        seed=0,
        check_only=False,
        pose_hypotheses=None,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        script._prepare_output_directory(args, "part_a")


def test_pipeline_run_id_rejects_path_components() -> None:
    script = _load_script("run_tless_obj4_pipeline.py")

    assert script._safe_run_id("baseline-v1.0") == "baseline-v1.0"
    with pytest.raises(ValueError, match="run-id"):
        script._safe_run_id("../shared-result")


def test_mode_aware_pipeline_wires_predictive_covariance_and_mode_output(
    tmp_path: Path,
) -> None:
    script = _load_script("run_tless_obj4_pipeline.py")
    model = tmp_path / "reliability.json"
    covariance = tmp_path / "covariance.json"
    command = script._build_refine_command(
        python_executable="python3",
        input_path=tmp_path / "gated-poses.json",
        object_config=tmp_path / "object.yaml",
        detections=tmp_path / "detections.json",
        dataset_root=tmp_path / "tless",
        output_dir=tmp_path / "refinement",
        reliability_model=model,
        gate_audit=tmp_path / "audit.json",
        mode_aware=True,
        predictive_covariance_model=covariance,
        mode_translation_threshold_m=0.03,
        mode_rotation_threshold_deg=15.0,
        mode_continuous_symmetry_samples=360,
    )

    assert "--mode-aware" in command
    assert command[command.index("--reliability-model") + 1] == str(model)
    assert command[command.index("--predictive-covariance-model") + 1] == str(
        covariance
    )
    assert script._execution_pose_path(
        tmp_path / "refinement", True
    ).name == "pose-modes-refined.json"


def test_predictive_covariance_requires_mode_aware_pipeline(tmp_path: Path) -> None:
    script = _load_script("run_tless_obj4_pipeline.py")

    with pytest.raises(ValueError, match="requires mode-aware"):
        script._build_refine_command(
            python_executable="python3",
            input_path=tmp_path / "gated-poses.json",
            object_config=tmp_path / "object.yaml",
            detections=tmp_path / "detections.json",
            dataset_root=tmp_path / "tless",
            output_dir=tmp_path / "refinement",
            reliability_model=None,
            gate_audit=None,
            mode_aware=False,
            predictive_covariance_model=tmp_path / "covariance.json",
            mode_translation_threshold_m=0.03,
            mode_rotation_threshold_deg=15.0,
            mode_continuous_symmetry_samples=360,
        )


def test_legacy_reuse_manifest_error_is_actionable(tmp_path: Path) -> None:
    script = _load_script("run_tless_obj4_pipeline.py")
    manifest = {"run_id": "legacy", "random_seed": 0}

    with pytest.raises(ValueError, match="predates initial XY/yaw tracking"):
        script._validate_reused_initial_condition(
            manifest, (-0.2, -0.12), 0.0
        )


def test_grasp_focus_output_isolated_from_precise_placement(tmp_path: Path) -> None:
    script = _load_script("run_industrial_cad_pick_place.py")
    script.PROJECT_ROOT = tmp_path
    args = Namespace(
        output=None,
        run_id="region_01",
        seed=7,
        check_only=False,
        pose_hypotheses=None,
        acceptance_mode="grasp_focus",
    )

    output = script._prepare_output_directory(args, "part_a")

    assert output == (
        tmp_path
        / "outputs/industrial_cad_pick_place/part_a/gt_dynamic_grasp_focus/region_01_seed7"
    )


def test_rectangular_region_requires_complete_cad_bounds() -> None:
    script = _load_script("run_industrial_cad_pick_place.py")
    transform = np.eye(4)
    local_min = np.array([-0.02, -0.01, -0.03])
    local_max = np.array([0.02, 0.01, 0.03])

    transform[:2, 3] = [-0.20, 0.28]
    assert script._object_inside_xy_region(
        transform,
        local_min,
        local_max,
        np.array([-0.38, 0.09]),
        np.array([-0.02, 0.47]),
    )

    transform[0, 3] = -0.37
    assert not script._object_inside_xy_region(
        transform,
        local_min,
        local_max,
        np.array([-0.38, 0.09]),
        np.array([-0.02, 0.47]),
    )


def test_paired_ablation_uses_fixed_denominator_and_discordant_pairs() -> None:
    script = _load_script("run_innovation1_grasp_ablation.py")
    runs = [
        {"trial": 0, "method": "baseline", "execution_result": {"grasp_success": True}},
        {"trial": 0, "method": "innovation1", "execution_result": {"grasp_success": True}},
        {"trial": 1, "method": "baseline", "execution_result": {"grasp_success": False}},
        {"trial": 1, "method": "innovation1", "execution_result": {"grasp_success": True}},
        {"trial": 2, "method": "baseline", "execution_result": None},
        {"trial": 2, "method": "innovation1", "execution_result": None},
    ]

    paired = script._paired_summary(runs, requested_trials=3)

    assert paired["both_success"] == 1
    assert paired["innovation1_only_success"] == 1
    assert paired["both_failed_or_missing"] == 1
    assert paired["fixed_denominator_success_rate_difference"] == pytest.approx(1 / 3)
    assert paired["mcnemar_exact_two_sided_p_value"] == 1.0


def test_active_risk_rejection_reports_each_failed_threshold() -> None:
    script = _load_script("run_industrial_cad_pick_place.py")

    reasons = script._risk_rejection_reasons(
        robust_score=0.2,
        pose_set_reliability=0.4,
        cvar_failure_risk=0.8,
        minimum_grasp_score=0.3,
        minimum_pose_set_reliability=0.5,
        maximum_cvar_failure_risk=0.7,
    )

    assert reasons == [
        "low_robust_grasp_score",
        "low_pose_set_reliability",
        "high_cvar_failure_risk",
    ]


def test_pick_place_loader_accepts_pose_modes_with_per_mode_covariance() -> None:
    script = _load_script("run_industrial_cad_pick_place.py")
    low = np.eye(6) * 1e-6
    high = np.eye(6) * 1e-3
    raw = {
        "object_id": "part_a",
        "scene_id": 1,
        "image_id": 2,
        "instance_id": 0,
        "reliability": 0.75,
        "uncertainty_method": "predictive_plus_dispersion",
        "modes": [
            {
                "mode_id": 0,
                "weight": 0.7,
                "transform_camera_object": np.eye(4).tolist(),
                "covariance_se3": low.tolist(),
            },
            {
                "mode_id": 1,
                "weight": 0.3,
                "transform_camera_object": np.eye(4).tolist(),
                "covariance_se3": high.tolist(),
            },
        ],
    }

    pose_set = script._pose_set_from_dict(raw)

    assert pose_set.reliability == pytest.approx(0.75)
    assert len(pose_set.hypotheses) == 2
    np.testing.assert_allclose(pose_set.hypotheses[0].covariance_se3, low)
    np.testing.assert_allclose(pose_set.hypotheses[1].covariance_se3, high)


def test_reliability_evaluation_counts_missing_bop_targets(tmp_path: Path) -> None:
    script = _load_script("train_pose_reliability.py")
    targets = tmp_path / "test_targets_bop19.json"
    targets.write_text(
        """[
  {"scene_id": 1, "im_id": 1, "obj_id": 2, "inst_count": 3},
  {"scene_id": 1, "im_id": 2, "obj_id": 2, "inst_count": 2},
  {"scene_id": 1, "im_id": 1, "obj_id": 3, "inst_count": 4}
]\n""",
        encoding="utf-8",
    )

    denominators = script._target_denominators(targets, [2, 3])

    assert denominators == {2: 5, 3: 4}


def test_reliability_frozen_split_must_be_disjoint_and_complete(tmp_path: Path) -> None:
    script = _load_script("train_pose_reliability.py")
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "split": {
                    "train": [1, 2],
                    "calibration": [3],
                    "test": [4],
                }
            }
        ),
        encoding="utf-8",
    )

    split, _ = script._load_frozen_split(valid, [1, 2, 3, 4])

    assert split == {"train": [1, 2], "calibration": [3], "test": [4]}

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "split": {
                    "train": [1, 2],
                    "calibration": [2, 3],
                    "test": [4],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        script._load_frozen_split(invalid, [1, 2, 3, 4])
