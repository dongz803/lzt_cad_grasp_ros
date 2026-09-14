#!/usr/bin/env python3
"""Run the reproducible T-LESS object-4 system baseline end to end."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FOUNDPOSE_PYTHON = WORKSPACE_ROOT / ".envs/foundpose_gpu/bin/python"


def _safe_run_id(raw: str | None) -> str:
    if raw is None:
        return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    if not raw or re.fullmatch(r"[A-Za-z0-9_.-]+", raw) is None:
        raise ValueError(
            "run-id may contain only letters, numbers, dot, dash and underscore"
        )
    return raw


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git_snapshot() -> dict[str, str]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_short": git("status", "--short"),
    }


def _run_stage(
    name: str,
    command: list[str],
    *,
    env: dict[str, str],
    manifest: dict[str, Any],
    manifest_path: Path,
    allow_nonzero: bool = False,
) -> None:
    started = datetime.now().astimezone()
    stage = {
        "name": name,
        "command": command,
        "started_at": started.isoformat(),
        "status": "running",
    }
    manifest["stages"].append(stage)
    _write_json(manifest_path, manifest)
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    begin = time.perf_counter()
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    except subprocess.CalledProcessError as error:
        if allow_nonzero:
            stage["status"] = "completed_with_nonzero_result"
            stage["return_code"] = error.returncode
            stage["duration_seconds"] = time.perf_counter() - begin
            stage["completed_at"] = datetime.now().astimezone().isoformat()
            _write_json(manifest_path, manifest)
            return
        stage["status"] = "failed"
        stage["return_code"] = error.returncode
        stage["duration_seconds"] = time.perf_counter() - begin
        stage["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["status"] = "failed"
        manifest["failed_stage"] = name
        _write_json(manifest_path, manifest)
        raise
    stage["status"] = "completed"
    stage["return_code"] = 0
    stage["duration_seconds"] = time.perf_counter() - begin
    stage["completed_at"] = datetime.now().astimezone().isoformat()
    _write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run capture -> FoundPose -> RGB-D gate -> pose refinement -> "
            "MuJoCo pick/place for T-LESS object 4."
        )
    )
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="run root; defaults to baseline or innovation1 according to options",
    )
    parser.add_argument(
        "--foundpose-python", type=Path, default=DEFAULT_FOUNDPOSE_PYTHON
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--reliability-model",
        type=Path,
        help="enable innovation-1 calibrated Top-K reliability",
    )
    parser.add_argument("--cvar-tail-probability", type=float, default=0.25)
    parser.add_argument("--cvar-risk-weight", type=float, default=0.5)
    parser.add_argument("--minimum-grasp-score", type=float, default=0.0)
    parser.add_argument("--minimum-pose-set-reliability", type=float, default=0.0)
    parser.add_argument("--maximum-cvar-failure-risk", type=float, default=1.0)
    parser.add_argument(
        "--grasp-ranking", choices=("baseline", "cvar"), default=None
    )
    parser.add_argument("--grasp-focus", action="store_true")
    parser.add_argument(
        "--initial-xy-m", type=float, nargs=2, default=(-0.2, -0.12)
    )
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--reuse-perception-run",
        type=Path,
        help="reuse another run's BOP observation and raw FoundPose Top-K",
    )
    args = parser.parse_args()

    run_id = _safe_run_id(args.run_id)
    innovation1_enabled = args.reliability_model is not None
    grasp_ranking = args.grasp_ranking or (
        "cvar" if innovation1_enabled else "baseline"
    )
    grasp_focus = args.grasp_focus or innovation1_enabled
    output_root = (
        args.output_root
        if args.output_root is not None
        else PROJECT_ROOT
        / (
            "outputs/system_experiments/innovation1/tless_000004"
            if innovation1_enabled or grasp_ranking == "cvar"
            else "outputs/system_baselines/tless_000004"
        )
    )
    run_root = output_root.resolve() / f"{run_id}_seed{args.seed}"
    if run_root.exists() and any(run_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty run directory: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    if not args.foundpose_python.is_file():
        raise SystemExit(f"FoundPose Python is missing: {args.foundpose_python}")
    if innovation1_enabled and not args.reliability_model.is_file():
        raise SystemExit(f"reliability model is missing: {args.reliability_model}")
    reuse_root = (
        None
        if args.reuse_perception_run is None
        else args.reuse_perception_run.resolve()
    )
    if reuse_root is not None:
        reused_manifest_path = reuse_root / "pipeline_manifest.json"
        reused_raw_poses = reuse_root / "foundpose/estimated-poses.json"
        reused_bop_root = reuse_root / "bop"
        for required in (
            reused_manifest_path,
            reused_raw_poses,
            reused_bop_root / "gt-mask-detections.json",
        ):
            if not required.exists():
                raise SystemExit(f"reused perception input is missing: {required}")
        reused_manifest = json.loads(
            reused_manifest_path.read_text(encoding="utf-8")
        )
        if not np.allclose(
            reused_manifest.get("initial_xy_m"), args.initial_xy_m, atol=1e-12
        ) or not np.isclose(
            float(reused_manifest.get("initial_yaw_deg")),
            args.initial_yaw_deg,
            atol=1e-12,
        ):
            raise SystemExit(
                "reused perception run does not match initial XY/yaw condition"
            )

    object_config = PROJECT_ROOT / "config/objects/tless_000004.yaml"
    grasp_candidates = PROJECT_ROOT / "outputs/cad_grasps/tless_000004.json"
    base_infer_config = (
        PROJECT_ROOT / "config/foundpose_bop19/tless_mujoco_obj4_infer.json"
    )
    for required in (object_config, grasp_candidates, base_infer_config):
        if not required.is_file():
            raise SystemExit(f"required baseline input is missing: {required}")

    snapshot_dir = run_root / "config_snapshot"
    snapshot_dir.mkdir(parents=True)
    shutil.copy2(object_config, snapshot_dir / object_config.name)
    shutil.copy2(grasp_candidates, snapshot_dir / grasp_candidates.name)
    if innovation1_enabled:
        shutil.copy2(
            args.reliability_model,
            snapshot_dir / "pose_reliability_model.json",
        )
    infer_payload = json.loads(base_infer_config.read_text(encoding="utf-8"))
    infer_version = f"mujoco_obj4_{run_id}_seed{args.seed}"
    infer_payload["infer_opts"]["version"] = infer_version
    infer_config = snapshot_dir / "foundpose_infer.json"
    _write_json(infer_config, infer_payload)

    manifest_path = run_root / "pipeline_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": (
            "tless_obj4_innovation1_calibrated_cvar"
            if innovation1_enabled
            else "tless_obj4_system_baseline"
        ),
        "run_id": run_id,
        "random_seed": args.seed,
        "initial_xy_m": list(args.initial_xy_m),
        "initial_yaw_deg": args.initial_yaw_deg,
        "grasp_focus": grasp_focus,
        "grasp_ranking": grasp_ranking,
        "risk_configuration": {
            "cvar_tail_probability": args.cvar_tail_probability,
            "cvar_risk_weight": args.cvar_risk_weight,
            "minimum_grasp_score": args.minimum_grasp_score,
            "minimum_pose_set_reliability": args.minimum_pose_set_reliability,
            "maximum_cvar_failure_risk": args.maximum_cvar_failure_risk,
            "confidence_is_calibrated_probability": innovation1_enabled,
        },
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "run_directory": str(run_root),
        "git": _git_snapshot(),
        "method_boundary": {
            "segmentation_input": "MuJoCo ground-truth instance mask",
            "foundpose_reads_ground_truth_pose": False,
            "gate_or_refiner_reads_ground_truth_pose": False,
            "reliability_model_reads_ground_truth_at_runtime": False,
            "ground_truth_pose_use": "post-hoc execution audit only",
        },
        "inputs": {
            "object_config": str(object_config),
            "grasp_candidates": str(grasp_candidates),
            "foundpose_config_snapshot": str(infer_config),
            "pose_reliability_model": (
                None
                if not innovation1_enabled
                else str(snapshot_dir / "pose_reliability_model.json")
            ),
            "reused_perception_run": (
                None if reuse_root is None else str(reuse_root)
            ),
        },
        "stages": [],
    }
    _write_json(manifest_path, manifest)

    base_env = os.environ.copy()
    base_env["BOP_PATH"] = str(WORKSPACE_ROOT / "datasets/bop")
    base_env["MUJOCO_GL"] = "egl"
    base_env["PYTHONHASHSEED"] = str(args.seed)
    base_env["LZT_RANDOM_SEED"] = str(args.seed)
    capture_dir = run_root / "capture"
    bop_root = run_root / "bop"
    gate_dir = run_root / "gating"
    refinement_dir = run_root / "refinement"
    execution_dir = run_root / "execution"

    if reuse_root is None:
        capture_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_industrial_cad_pick_place.py"),
            "--object-config", str(object_config),
            "--grasp-candidates", str(grasp_candidates),
            "--output", str(capture_dir),
            "--seed", str(args.seed),
            "--image-width", "720",
            "--image-height", "540",
            "--initial-xy-m", str(args.initial_xy_m[0]), str(args.initial_xy_m[1]),
            "--initial-yaw-deg", str(args.initial_yaw_deg),
            "--export-bop-root", str(bop_root),
            "--check-only",
        ]
        _run_stage(
            "capture", capture_command, env=base_env,
            manifest=manifest, manifest_path=manifest_path,
        )

        foundpose_output = PROJECT_ROOT / "outputs/foundpose_bop19"
        foundpose_env = base_env.copy()
        foundpose_env.update(
            {
                "BOP_PATH": str(bop_root),
                "BOP_OUTPUT_PATH": str(foundpose_output),
                "PYOPENGL_PLATFORM": "egl",
                "XFORMERS_DISABLED": "1",
                "MPLCONFIGDIR": str(run_root / "cache/matplotlib"),
            }
        )
        foundpose_command = [
            str(args.foundpose_python),
            str(PROJECT_ROOT / "scripts/run_foundpose.py"),
            "infer", "--opts-path", str(infer_config),
        ]
        _run_stage(
            "foundpose", foundpose_command, env=foundpose_env,
            manifest=manifest, manifest_path=manifest_path,
        )
        raw_pose_source = (
            foundpose_output / "inference" / f"tless_{infer_version}"
            / "4/estimated-poses.json"
        )
        if not raw_pose_source.is_file():
            raise FileNotFoundError(f"FoundPose output is missing: {raw_pose_source}")
        foundpose_run_dir = run_root / "foundpose"
        foundpose_run_dir.mkdir()
        raw_poses = foundpose_run_dir / "estimated-poses.json"
        shutil.copy2(raw_pose_source, raw_poses)
    else:
        capture_dir = reuse_root / "capture"
        bop_root = reused_bop_root
        raw_poses = reused_raw_poses
        manifest["stages"].append(
            {
                "name": "reuse_perception",
                "status": "completed",
                "source_run": str(reuse_root),
                "raw_poses": str(raw_poses),
                "bop_root": str(bop_root),
            }
        )
        _write_json(manifest_path, manifest)

    detections = bop_root / "gt-mask-detections.json"
    dataset_root = bop_root / "tless"
    gate_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/gate_foundpose_topk.py"),
        "--input", str(raw_poses),
        "--object-config", str(object_config),
        "--detections", str(detections),
        "--dataset-root", str(dataset_root),
        "--mask-opening-kernel", "1",
        "--output-dir", str(gate_dir),
    ]
    if innovation1_enabled:
        gate_command.extend(
            [
                "--reliability-model",
                str(snapshot_dir / "pose_reliability_model.json"),
            ]
        )
    _run_stage(
        "gate", gate_command, env=base_env,
        manifest=manifest, manifest_path=manifest_path,
    )

    refine_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/refine_foundpose_poses.py"),
        "--input", str(gate_dir / "gated-poses.json"),
        "--object-config", str(object_config),
        "--detections", str(detections),
        "--dataset-root", str(dataset_root),
        "--output-dir", str(refinement_dir),
    ]
    if not innovation1_enabled:
        refine_command.append("--reweight-score")
    _run_stage(
        "refine", refine_command, env=base_env,
        manifest=manifest, manifest_path=manifest_path,
    )

    pose_hypotheses = refinement_dir / "pose-hypotheses-refined.json"
    convert_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/convert_foundpose_topk.py"),
        "--input", str(refinement_dir / "refined-poses.json"),
        "--object-config", str(object_config),
        "--output", str(pose_hypotheses),
    ]
    _run_stage(
        "convert", convert_command, env=base_env,
        manifest=manifest, manifest_path=manifest_path,
    )

    execute_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_industrial_cad_pick_place.py"),
        "--object-config", str(object_config),
        "--grasp-candidates", str(grasp_candidates),
        "--pose-hypotheses", str(pose_hypotheses),
        "--output", str(execution_dir),
        "--seed", str(args.seed),
        "--image-width", "720",
        "--image-height", "540",
        "--initial-xy-m", str(args.initial_xy_m[0]), str(args.initial_xy_m[1]),
        "--initial-yaw-deg", str(args.initial_yaw_deg),
        "--grasp-ranking", grasp_ranking,
    ]
    if grasp_focus:
        execute_command.extend(
            [
                "--scene-name",
                "industrial_cad_grasp_focus.xml",
                "--slot",
                "place_region_center",
                "--acceptance-mode",
                "grasp_focus",
            ]
        )
    if grasp_ranking == "cvar":
        execute_command.extend(
            [
                "--cvar-tail-probability",
                str(args.cvar_tail_probability),
                "--cvar-risk-weight",
                str(args.cvar_risk_weight),
                "--minimum-grasp-score",
                str(args.minimum_grasp_score),
            ]
        )
        if innovation1_enabled:
            execute_command.extend(
                [
                    "--confidence-is-calibrated-probability",
                    "--minimum-pose-set-reliability",
                    str(args.minimum_pose_set_reliability),
                    "--maximum-cvar-failure-risk",
                    str(args.maximum_cvar_failure_risk),
                ]
            )
    if args.viewer:
        execute_command.append("--viewer")
    _run_stage(
        "pick_place", execute_command, env=base_env,
        manifest=manifest, manifest_path=manifest_path,
        allow_nonzero=True,
    )

    execution_result_path = execution_dir / "result.json"
    execution_result = json.loads(execution_result_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["success"] = bool(execution_result.get("success", False))
    manifest["outputs"] = {
        "capture": str(capture_dir),
        "foundpose_raw": str(raw_poses),
        "gating": str(gate_dir),
        "refinement": str(refinement_dir),
        "pose_hypotheses": str(pose_hypotheses),
        "execution": str(execution_dir),
        "execution_result": str(execution_result_path),
    }
    _write_json(manifest_path, manifest)
    _write_json(
        run_root / "pipeline_result.json",
        {
            "success": manifest["success"],
            "run_directory": str(run_root),
            "pipeline_manifest": str(manifest_path),
            "execution_result": execution_result,
        },
    )
    print(f"\npipeline_success={manifest['success']}", flush=True)
    print(f"pipeline_output={run_root}", flush=True)
    if not manifest["success"]:
        raise SystemExit("pipeline completed but pick/place acceptance failed")


if __name__ == "__main__":
    main()
