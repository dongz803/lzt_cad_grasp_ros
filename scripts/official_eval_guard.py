#!/usr/bin/env python3
"""Fail-fast environment checks for the official BOP19 evaluator.

The official evaluator lives in ``third_party/bop_toolkit_eval`` and imports
``imageio`` while rendering with ``vispy``/EGL.  Those packages are installed
in the project's dedicated environment, not necessarily in whichever
interpreter launches an experiment.  Because the evaluator always runs as a
subprocess of its launcher (``sys.executable``), starting an ablation with the
wrong interpreter used to fail only after every refinement stage had already
finished -- for example with ``ModuleNotFoundError: No module named
'imageio'`` after roughly an hour of work.

``require_official_eval_dependencies`` turns that into an immediate, actionable
error before any refinement, CSV writing or rendering starts.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


REQUIRED_MODULES = ("imageio", "vispy")
RECOMMENDED_ENV_PYTHON = Path(".envs/foundpose_gpu/bin/python")


def _probe_code(required: Iterable[str]) -> str:
    return (
        "import importlib.util, json\n"
        f"required = {list(required)!r}\n"
        "missing = []\n"
        "for name in required:\n"
        "    try:\n"
        "        found = importlib.util.find_spec(name) is not None\n"
        "    except Exception:\n"
        "        found = False\n"
        "    if not found:\n"
        "        missing.append(name)\n"
        "print(json.dumps(missing))\n"
    )


def recommended_interpreter(project_root: Path) -> Path:
    """Return the interpreter path that carries the official evaluator deps."""

    workspace = Path(project_root).resolve().parents[1]
    return workspace / RECOMMENDED_ENV_PYTHON


def find_missing_modules(
    interpreter: str | Path,
    required: Iterable[str] = REQUIRED_MODULES,
) -> list[str]:
    """Report which required modules ``interpreter`` cannot import."""

    required = tuple(required)
    completed = subprocess.run(
        [str(interpreter), "-c", _probe_code(required)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return list(required)
    payload = completed.stdout.strip().splitlines()
    if not payload:
        return list(required)
    try:
        missing = json.loads(payload[-1])
    except ValueError:
        return list(required)
    if not isinstance(missing, list):
        return list(required)
    return [str(name) for name in missing]


def failure_message(
    interpreter: str | Path,
    missing: Iterable[str],
    project_root: Path,
) -> str:
    """Build the operator-facing explanation for an unusable interpreter."""

    missing = list(missing)
    hint = recommended_interpreter(project_root)
    lines = [
        "official BOP19 evaluation cannot run with this interpreter:",
        f"  interpreter: {interpreter}",
        f"  missing modules: {', '.join(missing)}",
    ]
    if hint.is_file():
        lines.append("  re-run with the project environment interpreter, e.g.:")
        lines.append(f"    PYOPENGL_PLATFORM=egl {hint} <script> ... --run-official")
    else:
        lines.append(f"  expected environment interpreter is missing: {hint}")
        lines.append(
            "  install imageio and vispy into the launching interpreter first"
        )
    lines.append(
        "  official VSD rendering additionally needs a usable EGL device in the "
        "launching environment"
    )
    return "\n".join(lines)


def require_official_eval_dependencies(
    interpreter: str | Path | None = None,
    project_root: Path | None = None,
) -> str:
    """Raise ``SystemExit`` unless ``interpreter`` can run the official eval."""

    resolved_interpreter = str(interpreter or sys.executable)
    resolved_root = Path(
        project_root
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    missing = find_missing_modules(resolved_interpreter)
    if missing:
        raise SystemExit(
            failure_message(resolved_interpreter, missing, resolved_root)
        )
    return resolved_interpreter
