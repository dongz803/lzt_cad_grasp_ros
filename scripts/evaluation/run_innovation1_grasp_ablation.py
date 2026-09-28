#!/usr/bin/env python3
"""Run paired baseline/innovation-1 grasp trials on identical random scenes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = PROJECT_ROOT / "scripts/pipeline/run_tless_obj4_pipeline.py"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "lzt_perception/config/reliability/tless_objects2_4_logistic_v2.json"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _method_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in records if item.get("execution_result")]
    grasp_successes = sum(
        bool(item["execution_result"].get("grasp_success")) for item in completed
    )
    collision_free = sum(
        not item["execution_result"].get("runtime_forbidden_contacts")
        for item in completed
    )

    def mean(field: str) -> float | None:
        values = [
            float(item["execution_result"][field])
            for item in completed
            if item["execution_result"].get(field) is not None
        ]
        return None if not values else float(np.mean(values))

    requested = len(records)
    success_rate = grasp_successes / max(requested, 1)
    ci_low, ci_high = _wilson_interval(grasp_successes, requested)
    return {
        "requested_trials": len(records),
        "completed_execution_results": len(completed),
        "execution_completion_rate": len(completed) / max(requested, 1),
        "grasp_successes": grasp_successes,
        "fixed_denominator_grasp_success_rate": success_rate,
        "grasp_success_rate_wilson_95_ci": [ci_low, ci_high],
        "collision_free_trials": collision_free,
        "mean_grasp_target_position_error_m": mean(
            "grasp_target_position_error_m"
        ),
        "mean_selected_robust_score": mean("selected_robust_score"),
        "mean_selected_pose_set_reliability": mean(
            "selected_pose_set_reliability"
        ),
        "mean_expected_task_success": mean("selected_expected_task_success"),
        "mean_tail_task_success": mean("selected_tail_task_success"),
        "mean_selected_cvar_failure_risk": mean(
            "selected_cvar_failure_risk"
        ),
    }


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _mcnemar_exact_p_value(baseline_only: int, innovation_only: int) -> float:
    discordant = baseline_only + innovation_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, innovation_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2.0**discordant))


def _paired_summary(
    runs: list[dict[str, Any]], requested_trials: int
) -> dict[str, Any]:
    success_by_trial: dict[int, dict[str, bool]] = {
        trial: {"baseline": False, "innovation1": False}
        for trial in range(requested_trials)
    }
    for item in runs:
        result = item.get("execution_result") or {}
        success_by_trial[int(item["trial"])][str(item["method"])] = bool(
            result.get("grasp_success", False)
        )
    both_success = sum(
        pair["baseline"] and pair["innovation1"]
        for pair in success_by_trial.values()
    )
    baseline_only = sum(
        pair["baseline"] and not pair["innovation1"]
        for pair in success_by_trial.values()
    )
    innovation_only = sum(
        not pair["baseline"] and pair["innovation1"]
        for pair in success_by_trial.values()
    )
    both_failed = requested_trials - both_success - baseline_only - innovation_only
    return {
        "requested_pairs": requested_trials,
        "both_success": both_success,
        "baseline_only_success": baseline_only,
        "innovation1_only_success": innovation_only,
        "both_failed_or_missing": both_failed,
        "fixed_denominator_success_rate_difference": (
            innovation_only - baseline_only
        )
        / max(requested_trials, 1),
        "mcnemar_exact_two_sided_p_value": _mcnemar_exact_p_value(
            baseline_only, innovation_only
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the heuristic baseline and calibrated-probability/CVaR "
            "method with the same random object poses and fixed trial denominator."
        )
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--reliability-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/grasp_ablation/innovation1_smoke_v1",
    )
    parser.add_argument("--cvar-tail-probability", type=float, default=0.25)
    parser.add_argument("--cvar-risk-weight", type=float, default=0.5)
    parser.add_argument(
        "--reuse-perception-experiment",
        type=Path,
        help=(
            "reuse trial baseline perception runs from an earlier experiment; "
            "the pipeline validates the random XY/yaw conditions"
        ),
    )
    args = parser.parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    if not args.reliability_model.is_file():
        raise SystemExit(f"reliability model is missing: {args.reliability_model}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = output_dir / "runs"
    rng = np.random.default_rng(args.seed)
    conditions = [
        {
            "trial": index,
            "seed": args.seed + index,
            "initial_xy_m": [
                float(rng.uniform(-0.235, -0.165)),
                float(rng.uniform(-0.155, -0.085)),
            ],
            "initial_yaw_deg": float(rng.uniform(-180.0, 180.0)),
        }
        for index in range(args.trials)
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "innovation1_paired_grasp_ablation",
        "created_at": datetime.now().astimezone().isoformat(),
        "fixed_trial_denominator": args.trials,
        "random_seed": args.seed,
        "reliability_model": str(args.reliability_model.resolve()),
        "reused_perception_experiment": (
            None
            if args.reuse_perception_experiment is None
            else str(args.reuse_perception_experiment.resolve())
        ),
        "conditions": conditions,
        "runs": [],
        "status": "running",
    }
    manifest_path = output_dir / "experiment_manifest.json"
    _write_json(manifest_path, manifest)
    begin = time.perf_counter()
    for condition in conditions:
        original_perception_run = (
            None
            if args.reuse_perception_experiment is None
            else args.reuse_perception_experiment.resolve()
            / "runs"
            / (
                f"trial_{condition['trial']:03d}_baseline_"
                f"seed{condition['seed']}"
            )
        )
        for method in ("baseline", "innovation1"):
            run_id = f"trial_{condition['trial']:03d}_{method}"
            command = [
                sys.executable,
                str(PIPELINE),
                "--run-id",
                run_id,
                "--seed",
                str(condition["seed"]),
                "--output-root",
                str(run_root),
                "--grasp-focus",
                "--initial-xy-m",
                *(str(value) for value in condition["initial_xy_m"]),
                "--initial-yaw-deg",
                str(condition["initial_yaw_deg"]),
                "--grasp-ranking",
                "baseline" if method == "baseline" else "cvar",
            ]
            if method == "innovation1":
                command.extend(
                    [
                        "--reliability-model",
                        str(args.reliability_model.resolve()),
                        "--cvar-tail-probability",
                        str(args.cvar_tail_probability),
                        "--cvar-risk-weight",
                        str(args.cvar_risk_weight),
                        "--reuse-perception-run",
                        str(
                            original_perception_run
                            if original_perception_run is not None
                            else run_root
                            / (
                                f"trial_{condition['trial']:03d}_baseline_"
                                f"seed{condition['seed']}"
                            )
                        ),
                    ]
                )
            elif args.reuse_perception_experiment is not None:
                command.extend(
                    [
                        "--reuse-perception-run",
                        str(
                            original_perception_run
                        ),
                    ]
                )
            print(
                f"\n=== trial={condition['trial']} method={method} ===",
                flush=True,
            )
            completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            pipeline_root = run_root / f"{run_id}_seed{condition['seed']}"
            result_path = pipeline_root / "execution/result.json"
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file()
                else None
            )
            manifest["runs"].append(
                {
                    "trial": condition["trial"],
                    "method": method,
                    "command": command,
                    "return_code": completed.returncode,
                    "run_directory": str(pipeline_root),
                    "result_file": str(result_path) if result is not None else None,
                    "execution_result": result,
                }
            )
            _write_json(manifest_path, manifest)
    summaries = {
        method: _method_summary(
            [item for item in manifest["runs"] if item["method"] == method]
        )
        for method in ("baseline", "innovation1")
    }
    # A deliberate risk rejection makes the task result unsuccessful and the
    # standalone pipeline exits non-zero, but it is still a completed trial
    # for fixed-denominator evaluation when a structured result was written.
    all_completed = all(
        item["execution_result"] is not None for item in manifest["runs"]
    )
    paired = _paired_summary(manifest["runs"], args.trials)
    manifest["status"] = "completed" if all_completed else "completed_with_failures"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["runtime_s"] = time.perf_counter() - begin
    manifest["summary"] = summaries
    manifest["paired_comparison"] = paired
    _write_json(manifest_path, manifest)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "experiment": manifest["experiment"],
            "fixed_trial_denominator": args.trials,
            "summary": summaries,
            "paired_comparison": paired,
            "manifest": str(manifest_path),
        },
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    if not all_completed:
        raise SystemExit("one or more paired runs failed; inspect experiment_manifest.json")


if __name__ == "__main__":
    main()
