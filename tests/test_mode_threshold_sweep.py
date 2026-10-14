import argparse
import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = PROJECT_ROOT / "scripts/evaluation/run_mode_threshold_sweep.py"
    spec = importlib.util.spec_from_file_location("mode_threshold_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_parser_rejects_invalid_values() -> None:
    script = _load_script()

    assert script._threshold("0.03,15") == (0.03, 15.0)
    with pytest.raises(argparse.ArgumentTypeError):
        script._threshold("0,15")
    with pytest.raises(argparse.ArgumentTypeError):
        script._threshold("bad")


def test_sweep_changes_only_mode_threshold_and_output(tmp_path: Path) -> None:
    script = _load_script()
    args = argparse.Namespace(
        gated=["2=gated.json"],
        gate_audit=["2=audit.json"],
        object_config=["2=object.yaml"],
        detections=Path("detections.json"),
        reliability_model=Path("reliability.json"),
        predictive_covariance_model=Path("covariance.json"),
        dataset_root=Path("tless"),
        top_k=5,
        render_point_count=5000,
        preview_limit=0,
        progress_every=50,
        official_workers=1,
        limit_groups=None,
        run_official=True,
    )

    first = script._command(args, (0.02, 10.0), tmp_path / "first")
    second = script._command(args, (0.04, 20.0), tmp_path / "second")

    assert "--run-official" in first
    assert first[first.index("--mode-translation-threshold-m") + 1] == "0.02"
    assert second[second.index("--mode-rotation-threshold-deg") + 1] == "20.0"
    assert first[first.index("--output-dir") + 1] != second[
        second.index("--output-dir") + 1
    ]
