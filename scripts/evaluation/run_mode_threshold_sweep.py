#!/usr/bin/env python3
"""Run fixed-input mode-clustering threshold ablations.

Each threshold pair launches the same Innovation-1 pose ablation with an
isolated output directory.  The correctness criterion and BOP target
denominator remain inside the shared evaluator and are never derived from the
clustering thresholds.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABLATION_SCRIPT = (
    PROJECT_ROOT / "scripts/evaluation/run_innovation1_pose_ablation.py"
)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from official_eval_guard import (  # noqa: E402
    require_official_eval_dependencies,
)


def _threshold(value: str) -> tuple[float, float]:
    try:
        translation, rotation = (float(item) for item in value.split(",", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "threshold must be TRANSLATION_METRES,ROTATION_DEGREES"
        ) from error
    if translation <= 0.0 or not 0.0 < rotation <= 180.0:
        raise argparse.ArgumentTypeError("threshold values are outside valid ranges")
    return translation, rotation


def _run_name(translation_m: float, rotation_deg: float) -> str:
    return f"t{translation_m * 1000.0:05.1f}mm_r{rotation_deg:05.1f}deg".replace(
        ".", "p"
    )


def _command(
    args: argparse.Namespace,
    threshold: tuple[float, float],
    output_dir: Path,
) -> list[str]:
    translation_m, rotation_deg = threshold
    command = [sys.executable, str(ABLATION_SCRIPT)]
    for value in args.gated:
        command.extend(["--gated", value])
    for value in args.gate_audit:
        command.extend(["--gate-audit", value])
    for value in args.object_config:
        command.extend(["--object-config", value])
    command.extend(
        [
            "--detections",
            str(args.detections),
            "--reliability-model",
            str(args.reliability_model),
            "--predictive-covariance-model",
            str(args.predictive_covariance_model),
            "--dataset-root",
            str(args.dataset_root),
            "--output-dir",
            str(output_dir),
            "--top-k",
            str(args.top_k),
            "--render-point-count",
            str(args.render_point_count),
            "--preview-limit",
            str(args.preview_limit),
            "--progress-every",
            str(args.progress_every),
            "--mode-translation-threshold-m",
            str(translation_m),
            "--mode-rotation-threshold-deg",
            str(rotation_deg),
            "--official-workers",
            str(args.official_workers),
        ]
    )
    if args.limit_groups is not None:
        command.extend(["--limit-groups", str(args.limit_groups)])
    if args.run_official:
        command.append("--run-official")
    return command


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep mode thresholds while holding evaluation fixed"
    )
    parser.add_argument("--gated", action="append", required=True)
    parser.add_argument("--gate-audit", action="append", default=[])
    parser.add_argument("--object-config", action="append", default=[])
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--reliability-model", type=Path, required=True)
    parser.add_argument("--predictive-covariance-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=_threshold,
        action="append",
        default=[],
        help="repeat TRANSLATION_METRES,ROTATION_DEGREES",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--render-point-count", type=int, default=5000)
    parser.add_argument("--preview-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    thresholds = args.threshold or [
        (0.01, 5.0),
        (0.02, 10.0),
        (0.03, 15.0),
        (0.04, 20.0),
        (0.05, 30.0),
    ]
    if len(set(thresholds)) != len(thresholds):
        raise SystemExit("duplicate threshold pair")
    if args.run_official and not args.dry_run:
        # The official evaluator runs under this interpreter, so refuse to
        # start the sweep when it cannot import imageio/vispy.
        require_official_eval_dependencies(project_root=PROJECT_ROOT)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for threshold in thresholds:
        run_dir = output_dir / _run_name(*threshold)
        command = _command(args, threshold, run_dir)
        record = {
            "translation_threshold_m": threshold[0],
            "rotation_threshold_deg": threshold[1],
            "output_dir": str(run_dir),
            "command": command,
            "status": "planned" if args.dry_run else "running",
        }
        runs.append(record)
        _write_json(
            output_dir / "threshold_sweep_manifest.json",
            {
                "schema_version": 1,
                "created_at": datetime.now().astimezone().isoformat(),
                "fixed_correctness_criterion": True,
                "fixed_target_denominator": True,
                "runs": runs,
            },
        )
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            record["returncode"] = completed.returncode
            record["status"] = (
                "completed" if completed.returncode == 0 else "failed"
            )
            if completed.returncode != 0:
                _write_json(
                    output_dir / "threshold_sweep_manifest.json",
                    {
                        "schema_version": 1,
                        "created_at": datetime.now().astimezone().isoformat(),
                        "fixed_correctness_criterion": True,
                        "fixed_target_denominator": True,
                        "runs": runs,
                    },
                )
                raise SystemExit(completed.returncode)

    _write_json(
        output_dir / "threshold_sweep_manifest.json",
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "fixed_correctness_criterion": True,
            "fixed_target_denominator": True,
            "runs": runs,
        },
    )


if __name__ == "__main__":
    main()
