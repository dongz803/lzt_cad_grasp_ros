#!/usr/bin/env python3
"""Compare ordinary-SE(3) and symmetry-quotient clustering on fixed poses.

The experiment reuses the frozen per-candidate RGB-D refined hypotheses.  It
does not rerun FoundPose, gating, reliability fitting, CAD rendering or pose
refinement.  Both methods use the same 10 mm / 5 degree thresholds and differ
only in whether configured CAD symmetries are used when measuring distance and
forming the weighted mode centre.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)

from lzt_common.contracts import PoseHypothesis, PoseHypothesisSet  # noqa: E402
from lzt_common.object_config import load_object_config  # noqa: E402
from lzt_perception.pose_gating import foundpose_record_transform  # noqa: E402
from lzt_perception.pose_modes import (  # noqa: E402
    PoseMixture,
    PoseModeClusteringConfig,
    cluster_pose_hypotheses,
    pose_mixtures_to_jsonable,
)


EVALUATOR = PROJECT_ROOT / "scripts/evaluation/evaluate_topk_coverage.py"
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/innovation1_v3_full/"
    "tless_objects1_30_fixed_10mm_5deg_v1/per_candidate"
)
DEFAULT_DETECTIONS = (
    WORKSPACE_ROOT
    / "datasets/bop/detections/cnos-fastsam/cnos-fastsam_tless-test.json"
)
DEFAULT_TARGETS = WORKSPACE_ROOT / "datasets/bop/tless/test_targets_bop19.json"
METHODS = ("ordinary_se3", "symmetry_aware")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _record_key(record: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["scene_id"]),
        int(record.get("img_id", record.get("im_id"))),
        int(record["obj_id"]),
        int(record.get("inst_id", 0)),
    )


def records_to_hypothesis_set(
    records: list[dict[str, Any]], object_name: str
) -> PoseHypothesisSet:
    if not records:
        raise ValueError("cannot cluster an empty candidate group")
    ordered = sorted(
        records,
        key=lambda item: (
            -float(item.get("score", 0.0)),
            int(item.get("hypothesis_id", 0)),
        ),
    )
    first = ordered[0]
    hypotheses = tuple(
        PoseHypothesis(
            transform_camera_object=foundpose_record_transform(record),
            confidence=float(np.clip(float(record.get("score", 0.0)), 0.0, 1.0)),
            symmetry_class=str(record.get("symmetry_class", "identity")),
            source=str(record.get("source", "per_candidate_refined")),
            hypothesis_id=int(record.get("hypothesis_id", index)),
        )
        for index, record in enumerate(ordered)
    )
    return PoseHypothesisSet(
        object_id=object_name,
        scene_id=int(first["scene_id"]),
        image_id=int(first.get("img_id", first.get("im_id"))),
        instance_id=int(first.get("inst_id", 0)),
        hypotheses=hypotheses,
    )


def mode_candidate_records(
    mixture: PoseMixture,
    source_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_hypothesis = {
        int(record.get("hypothesis_id", index)): record
        for index, record in enumerate(source_records)
    }
    result = []
    for mode in mixture.modes:
        source = by_hypothesis[mode.representative_hypothesis_id]
        transform = mode.transform_camera_object
        result.append(
            {
                "scene_id": int(source["scene_id"]),
                "img_id": int(source.get("img_id", source.get("im_id"))),
                "obj_id": int(source["obj_id"]),
                "inst_id": int(source.get("inst_id", 0)),
                "hypothesis_id": mode.mode_id,
                "score": mode.weight,
                "R": transform[:3, :3].tolist(),
                "t": (transform[:3, 3] * 1000.0).reshape(3, 1).tolist(),
                "time": dict(source.get("time", {})),
                "cnos_time": float(source.get("cnos_time", 0.0)),
                "source": mixture.uncertainty_method,
                "source_hypothesis_count": mixture.source_hypothesis_count,
                "mode_member_count": len(mode.member_hypothesis_ids),
                "member_hypothesis_ids": list(mode.member_hypothesis_ids),
                "dispersion_covariance_se3": (
                    None
                    if mode.dispersion_covariance_se3 is None
                    else mode.dispersion_covariance_se3.tolist()
                ),
            }
        )
    return result


def _partition(mixture: PoseMixture) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(tuple(sorted(mode.member_hypothesis_ids)) for mode in mixture.modes)
    )


def summarize_mixtures(mixtures: Iterable[PoseMixture]) -> dict[str, Any]:
    items = list(mixtures)
    source_count = sum(item.source_hypothesis_count for item in items)
    modes = [mode for item in items for mode in item.modes]
    mode_count = len(modes)
    singleton_count = sum(len(mode.member_hypothesis_ids) == 1 for mode in modes)
    merged_groups = sum(len(item.modes) < item.source_hypothesis_count for item in items)
    return {
        "group_count": len(items),
        "source_hypothesis_count": source_count,
        "mode_count": mode_count,
        "collapsed_hypothesis_count": source_count - mode_count,
        "mode_reduction_fraction": (
            (source_count - mode_count) / source_count if source_count else 0.0
        ),
        "groups_with_any_merge": merged_groups,
        "groups_with_any_merge_fraction": (
            merged_groups / len(items) if items else 0.0
        ),
        "singleton_mode_count": singleton_count,
        "singleton_mode_fraction": (
            singleton_count / mode_count if mode_count else 0.0
        ),
        "mean_modes_per_group": mode_count / len(items) if items else 0.0,
    }


def _symmetry_type(cad_object: Any) -> str:
    if cad_object.continuous_symmetries:
        return "continuous"
    if cad_object.symmetry_transforms:
        return "discrete"
    return "none"


def _evaluation_command(
    method_paths: Mapping[str, list[Path]],
    output_dir: Path,
    detections: Path,
    targets: Path,
    object_ids: list[int],
    run_official: bool,
    official_workers: int,
) -> list[str]:
    command = [sys.executable, str(EVALUATOR)]
    for method in METHODS:
        for path in method_paths[method]:
            command.extend(["--method", f"{method}={path}"])
    for object_id in object_ids:
        command.extend(["--object-id", str(object_id)])
    command.extend(
        [
            "--targets",
            str(targets),
            "--detections",
            str(detections),
            "--raw-method",
            "ordinary_se3",
            "--output-dir",
            str(output_dir),
            "--official-workers",
            str(official_workers),
        ]
    )
    if run_official:
        command.append("--run-official")
    return command


def _official_subset_scores(
    result_root: Path, object_ids: set[int]
) -> dict[str, Any]:
    per_metric: dict[str, dict[int, list[float]]] = {
        "mssd": defaultdict(list),
        "mspd": defaultdict(list),
        "vsd": defaultdict(list),
    }
    for directory in result_root.glob("error=*"):
        metric = directory.name.split("_", 1)[0].split("=", 1)[-1]
        if metric not in per_metric:
            continue
        for score_path in directory.glob("scores_th=*.json"):
            recalls = _load_json(score_path).get("obj_recalls", {})
            for object_id in object_ids:
                if str(object_id) in recalls:
                    per_metric[metric][object_id].append(
                        float(recalls[str(object_id)])
                    )
    per_object: dict[str, dict[str, float | None]] = {}
    for object_id in sorted(object_ids):
        per_object[str(object_id)] = {
            metric: (
                None if not values[object_id] else float(np.mean(values[object_id]))
            )
            for metric, values in per_metric.items()
        }
        components = [
            value for value in per_object[str(object_id)].values() if value is not None
        ]
        per_object[str(object_id)]["ar"] = (
            None if not components else float(np.mean(components))
        )
    macro = {}
    for metric in ("mssd", "mspd", "vsd", "ar"):
        values = [item[metric] for item in per_object.values() if item[metric] is not None]
        macro[metric] = None if not values else float(np.mean(values))
    return {
        "protocol": "official_bop19_one_to_one_matching_subset_macro",
        "object_ids": sorted(object_ids),
        "macro_average": macro,
        "per_object": per_object,
    }


def _finalize_existing_official(
    output_dir: Path,
    object_ids: list[int],
    detections: Path,
    targets: Path,
    official_workers: int,
) -> None:
    method_paths = {
        method: [
            output_dir / "mode_candidates" / method / f"obj{object_id}.json"
            for object_id in object_ids
        ]
        for method in METHODS
    }
    missing = [
        path for paths in method_paths.values() for path in paths if not path.is_file()
    ]
    if missing:
        raise SystemExit(
            "cannot finalize because mode candidates are missing:\n"
            + "\n".join(map(str, missing))
        )
    recovery_evaluation_dir = output_dir / "evaluation_recovered"
    if recovery_evaluation_dir.exists() and any(recovery_evaluation_dir.iterdir()):
        raise SystemExit(
            "refusing to overwrite non-empty recovery evaluation: "
            f"{recovery_evaluation_dir}"
        )
    command = _evaluation_command(
        method_paths,
        recovery_evaluation_dir,
        detections,
        targets,
        object_ids,
        False,
        official_workers,
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    evaluation_path = recovery_evaluation_dir / "summary.json"
    evaluation = _load_json(evaluation_path)
    official_root = output_dir / "evaluation/official_eval"
    for method in METHODS:
        result_csv = Path(evaluation["methods"][method]["bop_result_csv"])
        original_stem = result_csv.stem
        result_root = official_root / method / original_stem
        score_file = result_root / "scores_bop19.json"
        if not score_file.is_file():
            raise SystemExit(f"official score file is missing: {score_file}")
        evaluation["methods"][method]["official_bop19"] = _official_subset_scores(
            result_root, set(object_ids)
        )
        evaluation["methods"][method]["official_bop19_scores"] = _load_json(
            score_file
        )
    _write_json(evaluation_path, evaluation)
    structure = _load_json(output_dir / "structure_summary.json")
    summary = {
        "schema_version": 1,
        "experiment": "ordinary_se3_vs_symmetry_aware_clustering",
        "structure": structure,
        "official_bop19": {
            method: evaluation["methods"][method]["official_bop19"]
            for method in METHODS
        },
        "official_bop19_scores": {
            method: evaluation["methods"][method]["official_bop19_scores"]
            for method in METHODS
        },
        "evaluation_summary": str(evaluation_path.resolve()),
        "recovery": (
            "official EGL evaluation was run outside the sandbox from the "
            "already-generated BOP CSV files after the sandbox EGL failure"
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    manifest_path = output_dir / "experiment_manifest.json"
    manifest = _load_json(manifest_path)
    manifest.update(
        {
            "status": "completed_after_official_recovery",
            "recovery_evaluation_command": command,
            "summary": str((output_dir / "summary.json").resolve()),
        }
    )
    _write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", type=int, action="append", default=[])
    parser.add_argument("--translation-threshold-m", type=float, default=0.01)
    parser.add_argument("--rotation-threshold-deg", type=float, default=5.0)
    parser.add_argument("--continuous-symmetry-samples", type=int, default=360)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--finalize-existing-official",
        action="store_true",
        help=(
            "skip clustering and combine already-generated official score "
            "directories after an EGL recovery run"
        ),
    )
    args = parser.parse_args()

    object_ids = sorted(set(args.object_id or range(1, 31)))
    if any(not 1 <= value <= 30 for value in object_ids):
        raise SystemExit("T-LESS object IDs must lie in [1, 30]")
    if args.translation_threshold_m <= 0.0:
        raise SystemExit("--translation-threshold-m must be positive")
    if not 0.0 < args.rotation_threshold_deg <= 180.0:
        raise SystemExit("--rotation-threshold-deg must lie in (0, 180]")
    if args.continuous_symmetry_samples < 2 or args.official_workers <= 0:
        raise SystemExit("symmetry samples and official workers must be positive")
    output_dir = args.output_dir.resolve()
    if args.finalize_existing_official:
        if not output_dir.is_dir():
            raise SystemExit(f"existing output is missing: {output_dir}")
        _finalize_existing_official(
            output_dir,
            object_ids,
            args.detections.resolve(),
            args.targets.resolve(),
            args.official_workers,
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output_dir}")

    inputs = {
        object_id: args.input_root / f"obj{object_id}/refined-poses.json"
        for object_id in object_ids
    }
    configs = {
        object_id: PROJECT_ROOT
        / "lzt_description/config/objects"
        / f"tless_{object_id:06d}.yaml"
        for object_id in object_ids
    }
    missing = [path for path in (*inputs.values(), *configs.values()) if not path.is_file()]
    missing.extend(path for path in (args.detections, args.targets) if not path.is_file())
    if missing:
        raise SystemExit("missing required input:\n" + "\n".join(map(str, missing)))

    command_preview = _evaluation_command(
        {method: [] for method in METHODS},
        output_dir / "evaluation",
        args.detections.resolve(),
        args.targets.resolve(),
        object_ids,
        args.run_official,
        args.official_workers,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "ordinary_se3_vs_symmetry_aware_clustering",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if args.dry_run else "running",
        "object_ids": object_ids,
        "fixed_input": "per-candidate RGB-D refined poses with frozen V3 reliability",
        "input_root": str(args.input_root.resolve()),
        "thresholds": {
            "translation_m": args.translation_threshold_m,
            "rotation_deg": args.rotation_threshold_deg,
            "continuous_symmetry_samples": args.continuous_symmetry_samples,
        },
        "methods": {
            "ordinary_se3": "raw SE(3) distance and raw weighted centre",
            "symmetry_aware": "configured CAD symmetry quotient distance and centre",
        },
        "official_bop19_requested": args.run_official,
        "evaluation_command_preview": command_preview,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "experiment_manifest.json"
    _write_json(manifest_path, manifest)
    if args.dry_run:
        print(manifest_path)
        return

    begin = time.perf_counter()
    mixtures: dict[str, list[PoseMixture]] = {method: [] for method in METHODS}
    method_paths: dict[str, list[Path]] = {method: [] for method in METHODS}
    object_types: dict[str, list[int]] = defaultdict(list)
    comparison = {
        "group_count": 0,
        "groups_with_changed_partition": 0,
        "groups_with_fewer_symmetry_modes": 0,
        "ordinary_minus_symmetry_mode_count": 0,
    }

    for object_id in object_ids:
        records = _load_json(inputs[object_id])
        if not isinstance(records, list):
            raise SystemExit(f"pose input must be a JSON list: {inputs[object_id]}")
        grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[_record_key(record)].append(record)
        object_config = load_object_config(configs[object_id], require_mesh=False)
        object_types[_symmetry_type(object_config.cad_object)].append(object_id)
        object_mixtures: dict[str, list[PoseMixture]] = {
            method: [] for method in METHODS
        }
        object_candidates: dict[str, list[dict[str, Any]]] = {
            method: [] for method in METHODS
        }
        for group in grouped.values():
            hypothesis_set = records_to_hypothesis_set(
                group, object_config.cad_object.object_id
            )
            ordinary = cluster_pose_hypotheses(
                hypothesis_set,
                object_config.cad_object,
                PoseModeClusteringConfig(
                    maximum_translation_distance_m=args.translation_threshold_m,
                    maximum_rotation_distance_deg=args.rotation_threshold_deg,
                    continuous_symmetry_samples=args.continuous_symmetry_samples,
                    use_symmetry_equivalence=False,
                ),
                confidence_is_calibrated_probability=True,
            )
            symmetry = cluster_pose_hypotheses(
                hypothesis_set,
                object_config.cad_object,
                PoseModeClusteringConfig(
                    maximum_translation_distance_m=args.translation_threshold_m,
                    maximum_rotation_distance_deg=args.rotation_threshold_deg,
                    continuous_symmetry_samples=args.continuous_symmetry_samples,
                    use_symmetry_equivalence=True,
                ),
                confidence_is_calibrated_probability=True,
            )
            for method, mixture in (
                ("ordinary_se3", ordinary),
                ("symmetry_aware", symmetry),
            ):
                object_mixtures[method].append(mixture)
                mixtures[method].append(mixture)
                object_candidates[method].extend(
                    mode_candidate_records(mixture, group)
                )
            comparison["group_count"] += 1
            changed = _partition(ordinary) != _partition(symmetry)
            comparison["groups_with_changed_partition"] += int(changed)
            fewer = len(symmetry.modes) < len(ordinary.modes)
            comparison["groups_with_fewer_symmetry_modes"] += int(fewer)
            comparison["ordinary_minus_symmetry_mode_count"] += (
                len(ordinary.modes) - len(symmetry.modes)
            )

        for method in METHODS:
            mode_path = output_dir / "modes" / method / f"obj{object_id}.json"
            candidate_path = (
                output_dir / "mode_candidates" / method / f"obj{object_id}.json"
            )
            _write_json(mode_path, pose_mixtures_to_jsonable(object_mixtures[method]))
            _write_json(candidate_path, object_candidates[method])
            method_paths[method].append(candidate_path)

    comparison["changed_partition_fraction"] = (
        comparison["groups_with_changed_partition"] / comparison["group_count"]
        if comparison["group_count"]
        else 0.0
    )
    comparison["fewer_symmetry_modes_fraction"] = (
        comparison["groups_with_fewer_symmetry_modes"] / comparison["group_count"]
        if comparison["group_count"]
        else 0.0
    )
    structure = {
        "methods": {
            method: summarize_mixtures(mixtures[method]) for method in METHODS
        },
        "comparison": comparison,
        "object_symmetry_types": dict(object_types),
    }
    _write_json(output_dir / "structure_summary.json", structure)

    evaluation_command = _evaluation_command(
        method_paths,
        output_dir / "evaluation",
        args.detections.resolve(),
        args.targets.resolve(),
        object_ids,
        args.run_official,
        args.official_workers,
    )
    manifest["evaluation_command"] = evaluation_command
    _write_json(manifest_path, manifest)
    completed = subprocess.run(evaluation_command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        manifest.update(
            {
                "status": "failed",
                "returncode": completed.returncode,
                "duration_s": time.perf_counter() - begin,
            }
        )
        _write_json(manifest_path, manifest)
        raise SystemExit(completed.returncode)

    evaluation_path = output_dir / "evaluation/summary.json"
    evaluation = _load_json(evaluation_path)
    summary = {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "structure": structure,
        "official_bop19": {
            method: evaluation["methods"][method].get("official_bop19")
            for method in METHODS
        },
        "evaluation_summary": str(evaluation_path.resolve()),
    }
    _write_json(output_dir / "summary.json", summary)
    manifest.update(
        {
            "status": "completed",
            "duration_s": time.perf_counter() - begin,
            "summary": str((output_dir / "summary.json").resolve()),
        }
    )
    _write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
