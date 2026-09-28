#!/usr/bin/env python3
"""Run a fixed-input ablation for innovation-1 pose refinement.
commands in ``experiment_manifest.json``.
消融实验保持门控的Top-K候选、检测、可靠性模型和BOP目标分母不变。它比较：
1. gated: 没有RGB-D姿态优化；
2. per_candidate: 独立地优化每个保留的假设；
3. mode_aware: 聚类对称等效假设，并在重新校准和重新分组之前优化每个非等效模式的一个中心。

"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
REFINE_SCRIPT = PROJECT_ROOT / "scripts/refine_foundpose_poses.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "scripts/evaluate_topk_coverage.py"
DEFAULT_DETECTIONS = (
    PROJECT_ROOT
    / "outputs/mask_refinement/tless_objects2_4_guarded/"
    "detections_rgbd_refined.json"
)
DEFAULT_RELIABILITY_MODEL = (
    PROJECT_ROOT
    / "lzt_perception/config/reliability/"
    "tless_objects2_4_logistic_v2.json"
)
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "datasets/bop/tless"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _named_object_paths(values: list[str], argument: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{argument} expects OBJECT_ID=PATH, got: {value}")
        raw_object_id, raw_path = value.split("=", 1)
        try:
            object_id = int(raw_object_id)
        except ValueError as exc:
            raise ValueError(
                f"{argument} object ID must be an integer, got: {raw_object_id}"
            ) from exc
        if object_id <= 0 or not raw_path:
            raise ValueError(f"{argument} expects positive OBJECT_ID=PATH: {value}")
        if object_id in result:
            raise ValueError(f"duplicate object ID for {argument}: {object_id}")
        result[object_id] = Path(raw_path)
    return result


def _default_object_config(object_id: int) -> Path:
    return (
        PROJECT_ROOT
        / "lzt_description/config/objects"
        / f"tless_{object_id:06d}.yaml"
    )


def _refinement_command(
    *,
    python_executable: str,
    input_path: Path,
    object_config: Path,
    detections: Path,
    dataset_root: Path,
    gate_audit: Path,
    reliability_model: Path,
    output_dir: Path,
    top_k: int,
    progress_every: int,
    render_point_count: int,
    preview_limit: int,
    limit_groups: int | None,
    mode_aware: bool,
    mode_translation_threshold_m: float,
    mode_rotation_threshold_deg: float,
    mode_continuous_symmetry_samples: int,
) -> list[str]:
    command = [
        python_executable,
        str(REFINE_SCRIPT),
        "--input",
        str(input_path),
        "--object-config",
        str(object_config),
        "--detections",
        str(detections),
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(output_dir),
        "--top-k",
        str(top_k),
        "--render-point-count",
        str(render_point_count),
        "--preview-limit",
        str(preview_limit),
        "--progress-every",
        str(progress_every),
        "--reliability-model",
        str(reliability_model),
        "--gate-audit",
        str(gate_audit),
    ]
    if limit_groups is not None:
        command.extend(["--limit-groups", str(limit_groups)])
    if mode_aware:
        command.extend(
            [
                "--mode-aware",
                "--mode-translation-threshold-m",
                str(mode_translation_threshold_m),
                "--mode-rotation-threshold-deg",
                str(mode_rotation_threshold_deg),
                "--mode-continuous-symmetry-samples",
                str(mode_continuous_symmetry_samples),
            ]
        )
    return command


def _evaluation_command(
    *,
    python_executable: str,
    gated_paths: dict[int, Path],
    output_dir: Path,
    detections: Path,
    run_official: bool,
    official_workers: int,
) -> list[str]:
    command = [python_executable, str(EVALUATE_SCRIPT)]
    for method in ("gated", "per_candidate", "mode_aware"):
        for object_id in sorted(gated_paths):
            path = (
                gated_paths[object_id]
                if method == "gated"
                else output_dir.parent
                / method
                / f"obj{object_id}"
                / "refined-poses.json"
            )
            command.extend(["--method", f"{method}={path}"])
    for object_id in sorted(gated_paths):
        command.extend(["--object-id", str(object_id)])
    command.extend(
        [
            "--raw-method",
            "gated",
            "--detections",
            str(detections),
            "--output-dir",
            str(output_dir),
            "--official-workers",
            str(official_workers),
        ]
    )
    if run_official:
        command.append("--run-official")
    return command


def _require_files(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare no refinement, independent per-candidate refinement and "
            "symmetry-aware mode-level refinement on identical gated inputs."
        )
    )
    parser.add_argument(
        "--gated",
        action="append",
        required=True,
        help="repeatable OBJECT_ID=PATH to a gated-poses.json file",
    )
    parser.add_argument(
        "--gate-audit",
        action="append",
        default=[],
        help=(
            "optional repeatable OBJECT_ID=PATH; defaults to audit.json next "
            "to each gated input"
        ),
    )
    parser.add_argument(
        "--object-config",
        action="append",
        default=[],
        help=(
            "optional repeatable OBJECT_ID=PATH; defaults to the matching "
            "T-LESS config under lzt_description"
        ),
    )
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument(
        "--reliability-model", type=Path, default=DEFAULT_RELIABILITY_MODEL
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--render-point-count", type=int, default=5000)
    parser.add_argument("--preview-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--mode-translation-threshold-m", type=float, default=0.03)
    parser.add_argument("--mode-rotation-threshold-deg", type=float, default=15.0)
    parser.add_argument(
        "--mode-continuous-symmetry-samples", type=int, default=360
    )
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-workers", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and write the command manifest without execution",
    )
    args = parser.parse_args()

    try:
        gated_paths = _named_object_paths(args.gated, "--gated")
        gate_audits = _named_object_paths(args.gate_audit, "--gate-audit")
        object_configs = _named_object_paths(args.object_config, "--object-config")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown_audits = set(gate_audits) - set(gated_paths)
    unknown_configs = set(object_configs) - set(gated_paths)
    if unknown_audits or unknown_configs:
        raise SystemExit(
            "--gate-audit/--object-config IDs must also appear in --gated"
        )
    if args.top_k <= 0 or args.official_workers <= 0:
        raise SystemExit("--top-k and --official-workers must be positive")
    if args.limit_groups is not None and args.limit_groups <= 0:
        raise SystemExit("--limit-groups must be positive")
    if args.progress_every < 0 or args.preview_limit < 0:
        raise SystemExit("--progress-every and --preview-limit must be non-negative")

    for object_id, path in gated_paths.items():
        gate_audits.setdefault(object_id, path.parent / "audit.json")
        object_configs.setdefault(object_id, _default_object_config(object_id))

    required_files: dict[str, Path] = {
        "detections": args.detections,
        "reliability_model": args.reliability_model,
        "refinement_script": REFINE_SCRIPT,
        "evaluation_script": EVALUATE_SCRIPT,
    }
    for object_id in sorted(gated_paths):
        required_files[f"gated_obj{object_id}"] = gated_paths[object_id]
        required_files[f"gate_audit_obj{object_id}"] = gate_audits[object_id]
        required_files[f"object_config_obj{object_id}"] = object_configs[object_id]
    _require_files(required_files)
    if not args.dataset_root.is_dir():
        raise SystemExit(f"dataset root is missing: {args.dataset_root}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, Any]] = []
    for method, mode_aware in (("per_candidate", False), ("mode_aware", True)):
        for object_id in sorted(gated_paths):
            run_output = output_dir / method / f"obj{object_id}"
            command = _refinement_command(
                python_executable=sys.executable,
                input_path=gated_paths[object_id].resolve(),
                object_config=object_configs[object_id].resolve(),
                detections=args.detections.resolve(),
                dataset_root=args.dataset_root.resolve(),
                gate_audit=gate_audits[object_id].resolve(),
                reliability_model=args.reliability_model.resolve(),
                output_dir=run_output,
                top_k=args.top_k,
                progress_every=args.progress_every,
                render_point_count=args.render_point_count,
                preview_limit=args.preview_limit,
                limit_groups=args.limit_groups,
                mode_aware=mode_aware,
                mode_translation_threshold_m=args.mode_translation_threshold_m,
                mode_rotation_threshold_deg=args.mode_rotation_threshold_deg,
                mode_continuous_symmetry_samples=(
                    args.mode_continuous_symmetry_samples
                ),
            )
            commands.append(
                {
                    "stage": "refinement",
                    "method": method,
                    "object_id": object_id,
                    "command": command,
                    "status": "pending",
                }
            )
    commands.append(
        {
            "stage": "evaluation",
            "method": "all",
            "object_id": None,
            "command": _evaluation_command(
                python_executable=sys.executable,
                gated_paths={
                    object_id: path.resolve()
                    for object_id, path in gated_paths.items()
                },
                output_dir=output_dir / "evaluation",
                detections=args.detections.resolve(),
                run_official=args.run_official,
                official_workers=args.official_workers,
            ),
            "status": "pending",
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "innovation1_pose_refinement_ablation",
        "created_at": datetime.now().astimezone().isoformat(),
        "methods": {
            "gated": "fixed input control without RGB-D refinement",
            "per_candidate": "independent RGB-D refinement of every hypothesis",
            "mode_aware": (
                "symmetry-aware clustering, one refinement per mode, correction "
                "propagation, recalibration and regrouping"
            ),
        },
        "object_ids": sorted(gated_paths),
        "fixed_inputs": {
            "gated": {
                str(object_id): str(path.resolve())
                for object_id, path in gated_paths.items()
            },
            "gate_audits": {
                str(object_id): str(path.resolve())
                for object_id, path in gate_audits.items()
            },
            "detections": str(args.detections.resolve()),
            "reliability_model": str(args.reliability_model.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
        },
        "mode_clustering": {
            "translation_threshold_m": args.mode_translation_threshold_m,
            "rotation_threshold_deg": args.mode_rotation_threshold_deg,
            "continuous_symmetry_samples": args.mode_continuous_symmetry_samples,
        },
        "official_bop19_requested": args.run_official,
        "dry_run": args.dry_run,
        "commands": commands,
        "status": "dry_run" if args.dry_run else "running",
    }
    manifest_path = output_dir / "experiment_manifest.json"
    _write_json(manifest_path, manifest)
    if args.dry_run:
        print(f"dry-run manifest: {manifest_path}")
        return

    experiment_begin = time.perf_counter()
    for entry in commands:
        print(
            f"\n=== stage={entry['stage']} method={entry['method']} "
            f"object={entry['object_id']} ===",
            flush=True,
        )
        begin = time.perf_counter()
        completed = subprocess.run(
            entry["command"], cwd=PROJECT_ROOT, check=False
        )
        entry["duration_s"] = time.perf_counter() - begin
        entry["returncode"] = completed.returncode
        entry["status"] = "completed" if completed.returncode == 0 else "failed"
        _write_json(manifest_path, manifest)
        if completed.returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_stage"] = entry["stage"]
            manifest["failed_method"] = entry["method"]
            manifest["failed_object_id"] = entry["object_id"]
            manifest["duration_s"] = time.perf_counter() - experiment_begin
            _write_json(manifest_path, manifest)
            raise SystemExit(completed.returncode)

    manifest["status"] = "completed"
    manifest["duration_s"] = time.perf_counter() - experiment_begin
    manifest["evaluation_summary"] = str(
        (output_dir / "evaluation/summary.json").resolve()
    )
    _write_json(manifest_path, manifest)
    print(f"completed pose ablation: {output_dir}")


if __name__ == "__main__":
    main()
