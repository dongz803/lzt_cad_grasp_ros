import importlib.util
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENTRY_POINTS = (
    "scripts/evaluation/run_innovation1_pose_ablation.py",
    "scripts/evaluation/run_mode_threshold_sweep.py",
    "scripts/evaluation/evaluate_topk_coverage.py",
    "scripts/evaluation/evaluate_bop19_official.py",
)


def _load_module(name: str, relative: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_guard():
    return _load_module(
        "official_eval_guard", "scripts/official_eval_guard.py"
    )


def test_probe_treats_unusable_interpreter_as_missing_everything(
    tmp_path: Path,
) -> None:
    guard = _load_guard()
    broken = tmp_path / "broken-python"
    broken.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    broken.chmod(0o755)

    assert guard.find_missing_modules(str(broken)) == list(guard.REQUIRED_MODULES)


def test_probe_parses_child_report(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = _load_guard()
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout='["vispy"]\n', stderr=""
    )
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: completed)

    assert guard.find_missing_modules("/usr/bin/python3") == ["vispy"]


def test_probe_rejects_garbage_child_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _load_guard()
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="not json\n", stderr=""
    )
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: completed)

    assert guard.find_missing_modules("/usr/bin/python3") == list(
        guard.REQUIRED_MODULES
    )


def test_guard_returns_interpreter_when_dependencies_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _load_guard()
    monkeypatch.setattr(guard, "find_missing_modules", lambda *a, **k: [])

    assert (
        guard.require_official_eval_dependencies("/some/python", PROJECT_ROOT)
        == "/some/python"
    )


def test_guard_fails_fast_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _load_guard()
    monkeypatch.setattr(
        guard, "find_missing_modules", lambda *a, **k: ["imageio"]
    )

    with pytest.raises(SystemExit) as excinfo:
        guard.require_official_eval_dependencies(
            "/usr/bin/python3", PROJECT_ROOT
        )

    message = str(excinfo.value)
    assert "/usr/bin/python3" in message
    assert "imageio" in message
    assert str(guard.recommended_interpreter(PROJECT_ROOT)) in message
    assert "EGL" in message


def test_recommended_interpreter_points_at_project_environment() -> None:
    guard = _load_guard()

    assert guard.recommended_interpreter(PROJECT_ROOT) == (
        PROJECT_ROOT.parents[1] / ".envs/foundpose_gpu/bin/python"
    )


@pytest.mark.parametrize("relative", ENTRY_POINTS)
def test_entry_points_use_the_guard(relative: str) -> None:
    source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")

    assert "require_official_eval_dependencies" in source


def test_ablation_script_exposes_the_guard() -> None:
    script = _load_module(
        "pose_ablation_guard_wiring",
        "scripts/evaluation/run_innovation1_pose_ablation.py",
    )

    assert callable(script.require_official_eval_dependencies)


def test_sweep_script_exposes_the_guard() -> None:
    script = _load_module(
        "mode_threshold_sweep_guard_wiring",
        "scripts/evaluation/run_mode_threshold_sweep.py",
    )

    assert callable(script.require_official_eval_dependencies)
