#!/usr/bin/env python3
"""Quantify the unobservable-symmetry contamination in the current SE(3) covariance.

On the same Top-K FoundPose candidates for one symmetric CAD, three covariance
estimates are compared:

  naive   : no symmetry handling; residuals are taken as they are
  merged  : current project implementation (pose_hypotheses.symmetry_aware_covariance)
  layered : symmetry canonicalized, then the axial (unobservable) rotational
            degree of freedom is removed before the covariance is formed

The axial DOF is the rotation about the configured continuous symmetry axis.
Because every residual is expressed in the reference object frame, that axis is
the object-frame Z column, so the axial component is the third rotational entry
of the 6-vector ``[translation, rotation_vector]``.

The script also propagates each covariance to real CAD grasp points using the
same first-order Jacobian as ``lzt_grasp_planning.task_grasp``.  That propagation
is the quantity that actually enters the grasp ranking, so it is the fairest
place to compare the three calibers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _bootstrap import activate_project_packages  # noqa: E402

activate_project_packages(PROJECT_ROOT)

from lzt_perception.pose_hypotheses import _rotation_vector  # noqa: E402


MODES = ("naive", "merged", "layered")


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _maximum_std(covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    return float(np.sqrt(max(float(np.max(eigenvalues)), 0.0)))


def _covariance(residuals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64)
    scores = np.asarray(weights, dtype=np.float64)
    if float(np.sum(scores)) <= 1e-12:
        scores = np.ones_like(scores)
    scores = scores / np.sum(scores)
    mean = np.sum(scores[:, None] * values, axis=0)
    centered = values - mean
    covariance = (centered * scores[:, None]).T @ centered
    return 0.5 * (covariance + covariance.T)


def _raw_transform(record: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(record["R"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(record["t"], dtype=np.float64).reshape(3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation * 0.001
    return transform


def _residuals(transforms: list[np.ndarray], mode: str) -> np.ndarray:
    """Residuals of one candidate set against its highest-confidence member.

    ``naive`` and ``merged`` differ only in their input transforms: the naive
    set is the raw FoundPose output, while the merged set is the
    symmetry-canonicalized output of ``foundpose_records_to_top_k``.  ``layered``
    reuses the canonicalized set and additionally drops the axial (unobservable)
    rotational component.
    """

    reference = transforms[0]
    rows = []
    for transform in transforms:
        translation = transform[:3, 3] - reference[:3, 3]
        rotation = _rotation_vector(reference[:3, :3].T @ transform[:3, :3])
        if mode == "layered":
            rotation = rotation.copy()
            rotation[2] = 0.0
        rows.append(np.concatenate((translation, rotation)))
    return np.asarray(rows, dtype=np.float64)


def _percentile(values: np.ndarray, name: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{name}_median": float(np.median(values)),
        f"{name}_mean": float(np.mean(values)),
        f"{name}_p90": float(np.percentile(values, 90)),
        f"{name}_max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose-hypotheses",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/foundpose_bop19_topk/inference/"
        "tless_bop19_standard_topk_v1/4/pose-hypotheses-top5.json",
        help="symmetry-canonicalized Top-K (already merged by the project)",
    )
    parser.add_argument(
        "--raw-poses",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/foundpose_bop19_topk/inference/"
        "tless_bop19_standard_topk_v1/4/estimated-poses.json",
        help="raw FoundPose records, used as the no-symmetry-handling control",
    )
    parser.add_argument(
        "--grasp-candidates",
        type=Path,
        default=PROJECT_ROOT / "outputs/cad_grasps/tless_000004.json",
    )
    parser.add_argument("--object-id", default="tless_000004")
    parser.add_argument("--symmetry-axis", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    groups = json.loads(args.pose_hypotheses.read_text(encoding="utf-8"))
    if not isinstance(groups, list) or not groups:
        raise SystemExit("pose hypotheses must be a non-empty JSON list")
    if args.limit is not None:
        groups = groups[: args.limit]

    raw_records = json.loads(args.raw_poses.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise SystemExit("raw poses must be a JSON list")
    raw_index: dict[tuple[int, int, int], dict[int, dict[str, Any]]] = {}
    for record in raw_records:
        key = (
            int(record["scene_id"]),
            int(record["img_id"]),
            int(record.get("inst_id", 0)),
        )
        raw_index.setdefault(key, {})[int(record.get("hypothesis_id", 0))] = record

    grasp_payload = json.loads(args.grasp_candidates.read_text(encoding="utf-8"))
    grasp_points = [
        np.asarray(item["transform_object_grasp"], dtype=np.float64)[:3, 3]
        for item in grasp_payload["candidates"]
    ]
    lever_arms_mm = [
        float(np.linalg.norm(point)) * 1000.0 for point in grasp_points
    ]

    rot_std_deg: dict[str, list[float]] = {mode: [] for mode in MODES}
    trans_std_mm: dict[str, list[float]] = {mode: [] for mode in MODES}
    pick_std_mm: dict[str, list[float]] = {mode: [] for mode in MODES}
    axial_fraction: list[float] = []
    axial_deg: list[float] = []
    per_group: list[dict[str, Any]] = []

    axis_error_deg: list[float] = []
    axis = np.asarray(args.symmetry_axis, dtype=np.float64)
    for group in groups:
        hypotheses = group["hypotheses"]
        if len(hypotheses) < 2:
            continue
        key = (
            int(group["scene_id"]),
            int(group["image_id"]),
            int(group["instance_id"]),
        )
        raw_group = raw_index.get(key, {})
        order = [int(item["hypothesis_id"]) for item in hypotheses]
        if not all(identifier in raw_group for identifier in order):
            continue
        merged_transforms = [
            np.asarray(item["transform_camera_object"], dtype=np.float64)
            for item in hypotheses
        ]
        raw_transforms = [_raw_transform(raw_group[identifier]) for identifier in order]
        weights = np.asarray(
            [float(item.get("weight", 0.0)) for item in hypotheses], dtype=np.float64
        )
        reference_rotation = merged_transforms[0][:3, :3]
        axis_reference = raw_transforms[0][:3, :3] @ axis
        for transform in raw_transforms:
            direction = transform[:3, :3] @ axis
            axis_error_deg.append(
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(float(direction @ axis_reference), -1.0, 1.0)
                        )
                    )
                )
            )

        group_row: dict[str, Any] = {
            "scene_id": int(group["scene_id"]),
            "image_id": int(group["image_id"]),
            "instance_id": int(group["instance_id"]),
        }
        for mode in MODES:
            source = raw_transforms if mode == "naive" else merged_transforms
            residual = _residuals(source, mode)
            covariance = _covariance(residual, weights)
            rot = float(np.degrees(_maximum_std(covariance[3:, 3:])))
            trans = float(_maximum_std(covariance[:3, :3]) * 1000.0)
            rot_std_deg[mode].append(rot)
            trans_std_mm[mode].append(trans)
            group_row[f"{mode}_rot_std_deg"] = rot
            group_row[f"{mode}_trans_std_mm"] = trans
            for point in grasp_points:
                jacobian = np.hstack(
                    (np.eye(3), -reference_rotation @ _skew(point))
                )
                pick_covariance = jacobian @ covariance @ jacobian.T
                pick_std_mm[mode].append(
                    float(_maximum_std(pick_covariance) * 1000.0)
                )
            if mode == "merged":
                axial = np.abs(residual[:, 5])
                norm = np.linalg.norm(residual[:, 3:], axis=1)
                axial_fraction.extend(
                    float(value) for value in axial / np.maximum(norm, 1e-12)
                )
                axial_deg.extend(float(np.degrees(value)) for value in axial)
        per_group.append(group_row)

    report: dict[str, Any] = {
        "schema_version": 1,
        "object_id": args.object_id,
        "pose_hypotheses": str(args.pose_hypotheses.resolve()),
        "raw_poses": str(args.raw_poses.resolve()),
        "grasp_candidates": str(args.grasp_candidates.resolve()),
        "group_count": len(per_group),
        "candidate_count": len(per_group) * 5,
        "raw_axis_direction_error_deg": _percentile(
            np.asarray(axis_error_deg), "value"
        ),
        "grasp_point_count": len(grasp_points),
        "grasp_lever_arm_mm": {
            "min": float(np.min(lever_arms_mm)),
            "median": float(np.median(lever_arms_mm)),
            "max": float(np.max(lever_arms_mm)),
        },
        "symmetry_axis_object_frame": list(args.symmetry_axis),
        "metrics": {},
        "axial_component_after_merging": {
            **_percentile(np.asarray(axial_deg), "axial_deg"),
            **_percentile(np.asarray(axial_fraction), "axial_share"),
        },
        "per_group": per_group,
    }
    for mode in MODES:
        report["metrics"][mode] = {
            "rotation_std_deg": _percentile(np.asarray(rot_std_deg[mode]), "value"),
            "translation_std_mm": _percentile(np.asarray(trans_std_mm[mode]), "value"),
            "propagated_pick_position_std_mm": _percentile(
                np.asarray(pick_std_mm[mode]), "value"
            ),
        }

    baseline = report["metrics"]["merged"]
    report["relative_to_merged"] = {
        mode: {
            "rotation_std_ratio": (
                report["metrics"][mode]["rotation_std_deg"]["value_median"]
                / max(baseline["rotation_std_deg"]["value_median"], 1e-12)
            ),
            "pick_position_std_ratio": (
                report["metrics"][mode]["propagated_pick_position_std_mm"][
                    "value_median"
                ]
                / max(
                    baseline["propagated_pick_position_std_mm"]["value_median"],
                    1e-12,
                )
            ),
        }
        for mode in MODES
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
