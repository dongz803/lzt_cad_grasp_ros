import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = PROJECT_ROOT / "scripts/evaluation/run_innovation1_pose_ablation.py"
    spec = importlib.util.spec_from_file_location("pose_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_named_object_paths_reject_duplicates_and_bad_ids() -> None:
    script = _load_script()

    assert script._named_object_paths(["2=obj2.json"], "--gated") == {
        2: Path("obj2.json")
    }
    with pytest.raises(ValueError, match="duplicate object ID"):
        script._named_object_paths(
            ["2=first.json", "2=second.json"], "--gated"
        )
    with pytest.raises(ValueError, match="must be an integer"):
        script._named_object_paths(["obj2=result.json"], "--gated")


def test_pose_ablation_uses_identical_inputs_and_isolated_outputs(tmp_path: Path) -> None:
    script = _load_script()
    gated = {2: tmp_path / "gating/gated-poses.json"}
    evaluation_dir = tmp_path / "experiment/evaluation"

    command = script._evaluation_command(
        python_executable="python3",
        gated_paths=gated,
        output_dir=evaluation_dir,
        detections=tmp_path / "detections.json",
        run_official=True,
        official_workers=1,
    )

    method_values = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--method"
    ]
    assert method_values == [
        f"gated={gated[2]}",
        (
            "per_candidate="
            f"{tmp_path / 'experiment/per_candidate/obj2/refined-poses.json'}"
        ),
        (
            "mode_aware="
            f"{tmp_path / 'experiment/mode_aware/obj2/refined-poses.json'}"
        ),
    ]
    assert "--run-official" in command


def test_mode_aware_command_changes_only_mode_configuration(tmp_path: Path) -> None:
    script = _load_script()
    common = dict(
        python_executable="python3",
        input_path=tmp_path / "gated.json",
        object_config=tmp_path / "object.yaml",
        detections=tmp_path / "detections.json",
        dataset_root=tmp_path / "tless",
        gate_audit=tmp_path / "audit.json",
        reliability_model=tmp_path / "model.json",
        output_dir=tmp_path / "output",
        top_k=5,
        progress_every=50,
        render_point_count=5000,
        preview_limit=0,
        limit_groups=None,
        mode_translation_threshold_m=0.03,
        mode_rotation_threshold_deg=15.0,
        mode_continuous_symmetry_samples=360,
    )

    per_candidate = script._refinement_command(mode_aware=False, **common)
    mode_aware = script._refinement_command(mode_aware=True, **common)

    assert "--mode-aware" not in per_candidate
    assert "--mode-aware" in mode_aware
    assert "--reliability-model" in per_candidate
    assert "--gate-audit" in per_candidate
