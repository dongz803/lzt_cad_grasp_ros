from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
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
