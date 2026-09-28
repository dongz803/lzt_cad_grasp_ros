#!/usr/bin/env python3
"""Compare the single-covariance caliber against the pose-mixture caliber.

The project has two ways to turn a Top-K set into an uncertainty statement:

  global  : ``pose_hypotheses.symmetry_aware_covariance``.  All candidates form
            one pose set with one 6x6 SE(3) covariance around a weighted mean.
            This is what ``task_grasp.rank_task_grasps`` consumes today.
  mixture : ``pose_modes.cluster_pose_hypotheses``.  Candidates are clustered in
            the symmetry quotient into non-equivalent modes; every mode keeps its
            own mean pose, its own *local* covariance and a mixture weight.

The two calibers do not report the same quantity, so they are compared at the
point where they meet the downstream consumer: the tolerated-uncertainty score
``exp(-0.5 (sigma * std / tolerance)^2)`` of ``lzt_grasp_planning.task_grasp``.

For the mixture caliber the score is aggregated over modes twice:

  expected : weight-averaged score, i.e. E_mode[success]
  worst    : minimum score over modes, a conservative bound

Ground-truth MSSD is never used to compute a score; it is only carried along so
the reported uncertainty can be audited afterwards.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _bootstrap import activate_project_packages  # noqa: E402

activate_project_packages(PROJECT_ROOT)

from lzt_common.contracts import (  # noqa: E402
    CADObject,
    ContinuousSymmetry,
    PoseHypothesis,
    PoseHypothesisSet,
)
from lzt_perception.pose_hypotheses import symmetry_aware_covariance  # noqa: E402
from lzt_perception.pose_modes import cluster_pose_hypotheses  # noqa: E402


def _maximum_std(covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    return float(np.sqrt(max(float(np.max(eigenvalues)), 0.0)))


def _tolerance_score(std_deg: float, tolerance_deg: float, risk_sigma: float) -> float:
    normalized = risk_sigma * std_deg / tolerance_deg
    return float(np.exp(-0.5 * normalized * normalized))


def _percentile(values: list[float], name: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {f"{name}_{key}": float("nan") for key in ("median", "mean", "p90", "max")}
    return {
        f"{name}_median": float(np.median(array)),
        f"{name}_mean": float(np.mean(array)),
        f"{name}_p90": float(np.percentile(array, 90)),
        f"{name}_max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose-hypotheses",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/pose_gating/tless_000004/pose-hypotheses-gated.json",
    )
    parser.add_argument("--object-id", default="tless_000004")
    parser.add_argument("--symmetry-axis", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--rotation-threshold-deg", type=float, default=15.0)
    parser.add_argument("--translation-threshold-m", type=float, default=0.03)
    parser.add_argument("--tolerance-deg", type=float, default=10.0)
    parser.add_argument("--risk-sigma", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cad_object = CADObject(
        object_id=args.object_id,
        mesh_path=Path("unused.ply"),
        units_to_m=1.0,
        continuous_symmetries=(
            ContinuousSymmetry(
                axis_xyz=np.asarray(args.symmetry_axis, dtype=np.float64),
                offset_m=np.zeros(3),
            ),
        ),
    )

    groups = json.loads(args.pose_hypotheses.read_text(encoding="utf-8"))
    if not isinstance(groups, list) or not groups:
        raise SystemExit("pose hypotheses must be a non-empty JSON list")
    if args.limit is not None:
        groups = groups[: args.limit]

    from lzt_perception.pose_modes import PoseModeClusteringConfig

    clustering = PoseModeClusteringConfig(
        maximum_translation_distance_m=args.translation_threshold_m,
        maximum_rotation_distance_deg=args.rotation_threshold_deg,
    )

    mode_counts: Counter[int] = Counter()
    member_counts: Counter[int] = Counter()
    global_std: list[float] = []
    mode_std_all: list[float] = []
    mode_std_multi: list[float] = []
    mode_std_single: list[float] = []
    global_score: list[float] = []
    expected_score: list[float] = []
    worst_score: list[float] = []
    best_mode_score: list[float] = []
    mode_weight_repr: list[float] = []
    singleton_weight: list[float] = []
    per_group: list[dict[str, Any]] = []

    for group in groups:
        hypotheses = sorted(
            group["hypotheses"], key=lambda item: -float(item["confidence"])
        )
        if len(hypotheses) < 2:
            continue
        pose_set = PoseHypothesisSet(
            object_id=group["object_id"],
            scene_id=int(group["scene_id"]),
            image_id=int(group["image_id"]),
            instance_id=int(group["instance_id"]),
            hypotheses=tuple(
                PoseHypothesis(
                    transform_camera_object=np.asarray(
                        item["transform_camera_object"], dtype=np.float64
                    ),
                    confidence=float(np.clip(float(item["confidence"]), 0.0, 1.0)),
                    symmetry_class=str(item.get("symmetry_class", "identity")),
                    source=str(item.get("source", "unknown")),
                    hypothesis_id=int(item["hypothesis_id"]),
                )
                for item in hypotheses
            ),
        )

        covariance = symmetry_aware_covariance(pose_set.hypotheses, cad_object)
        if covariance is None:
            continue
        global_rotation_std = float(
            np.degrees(_maximum_std(covariance[3:, 3:]))
        )

        mixture = cluster_pose_hypotheses(pose_set, cad_object, clustering)
        mode_counts[len(mixture.modes)] += 1
        scores_expected = 0.0
        scores_worst = 1.0
        for mode in mixture.modes:
            rotation_std = float(
                np.degrees(_maximum_std(mode.covariance_se3[3:, 3:]))
            )
            score = _tolerance_score(rotation_std, args.tolerance_deg, args.risk_sigma)
            member_count = len(mode.member_hypothesis_ids)
            member_counts[member_count] += 1
            mode_std_all.append(rotation_std)
            if member_count >= 2:
                mode_std_multi.append(rotation_std)
            else:
                mode_std_single.append(rotation_std)
                singleton_weight.append(float(mode.weight))
            mode_weight_repr.append(float(mode.weight))
            scores_expected += float(mode.weight) * score
            scores_worst = min(scores_worst, score)

        global_score.append(
            _tolerance_score(global_rotation_std, args.tolerance_deg, args.risk_sigma)
        )
        global_std.append(global_rotation_std)
        expected_score.append(scores_expected)
        worst_score.append(scores_worst)
        best_mode_score.append(max(scores_expected, 1e-12))
        per_group.append(
            {
                "scene_id": int(group["scene_id"]),
                "image_id": int(group["image_id"]),
                "instance_id": int(group["instance_id"]),
                "candidate_count": len(hypotheses),
                "mode_count": len(mixture.modes),
                "global_rotation_std_deg": global_rotation_std,
                "global_angle_score": global_score[-1],
                "mode_rotation_std_deg": [
                    float(np.degrees(_maximum_std(mode.covariance_se3[3:, 3:])))
                    for mode in mixture.modes
                ],
                "mode_weights": [float(mode.weight) for mode in mixture.modes],
                "mode_member_counts": [
                    len(mode.member_hypothesis_ids) for mode in mixture.modes
                ],
                "expected_angle_score": scores_expected,
                "worst_angle_score": scores_worst,
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "object_id": args.object_id,
        "pose_hypotheses": str(args.pose_hypotheses.resolve()),
        "clustering": {
            "maximum_translation_distance_m": args.translation_threshold_m,
            "maximum_rotation_distance_deg": args.rotation_threshold_deg,
        },
        "tolerance_deg": args.tolerance_deg,
        "risk_sigma": args.risk_sigma,
        "group_count": len(per_group),
        "clustering_structure": {
            "mode_count_histogram": dict(sorted(mode_counts.items())),
            "mean_mode_count": (
                float(np.mean([g["mode_count"] for g in per_group]))
                if per_group
                else None
            ),
            "member_count_histogram": dict(sorted(member_counts.items())),
            "singleton_mode_share": (
                float(member_counts.get(1, 0) / max(sum(member_counts.values()), 1))
            ),
            "mean_singleton_weight": (
                float(np.mean(singleton_weight)) if singleton_weight else None
            ),
        },
        "uncertainty": {
            "global_rotation_std_deg": _percentile(global_std, "value"),
            "mode_rotation_std_deg_all": _percentile(mode_std_all, "value"),
            "mode_rotation_std_deg_multi_member": _percentile(mode_std_multi, "value"),
            "mode_rotation_std_deg_singleton": _percentile(mode_std_single, "value"),
        },
        "downstream_angle_score": {
            "global_single_covariance": _percentile(global_score, "value"),
            "mixture_expected": _percentile(expected_score, "value"),
            "mixture_worst_mode": _percentile(worst_score, "value"),
        },
        "per_group": per_group,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "per_group"}, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
