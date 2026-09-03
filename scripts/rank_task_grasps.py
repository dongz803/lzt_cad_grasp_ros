#!/usr/bin/env python3
"""Rank CAD grasps with Top-K pose uncertainty and pick/place constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.contracts import (  # noqa: E402
    GraspCandidate,
    PickPlaceTask,
    PoseHypothesis,
    PoseHypothesisSet,
)
from lzt_cad_grasp_core.object_config import load_object_config  # noqa: E402
from lzt_cad_grasp_core.task_grasp import (  # noqa: E402
    RobustScoringConfig,
    rank_task_grasps,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pose_set(raw: dict) -> PoseHypothesisSet:
    hypotheses = tuple(
        PoseHypothesis(
            transform_camera_object=np.asarray(
                item["transform_camera_object"], dtype=np.float64
            ),
            confidence=float(item["confidence"]),
            symmetry_class=str(item.get("symmetry_class", "identity")),
            covariance_se3=(
                None
                if raw.get("covariance_se3") is None
                else np.asarray(raw["covariance_se3"], dtype=np.float64)
            ),
            source=str(item.get("source", "unknown")),
            hypothesis_id=int(item["hypothesis_id"]),
        )
        for item in raw["hypotheses"]
    )
    covariance = raw.get("covariance_se3")
    return PoseHypothesisSet(
        object_id=str(raw["object_id"]),
        scene_id=int(raw["scene_id"]),
        image_id=int(raw["image_id"]),
        instance_id=int(raw["instance_id"]),
        hypotheses=hypotheses,
        covariance_se3=(
            None if covariance is None else np.asarray(covariance, dtype=np.float64)
        ),
        uncertainty_method=str(raw.get("uncertainty_method", "unavailable")),
        symmetry_discretization_deg=raw.get("symmetry_discretization_deg"),
    )


def _candidates(raw: dict) -> list[GraspCandidate]:
    return [
        GraspCandidate(
            grasp_id=str(item["grasp_id"]),
            transform_object_grasp=np.asarray(
                item["transform_object_grasp"], dtype=np.float64
            ),
            jaw_opening_m=float(item["jaw_opening_m"]),
            analytic_quality=float(item["analytic_quality"]),
            approach_distance_m=float(item["approach_distance_m"]),
            antipodal_score=float(item["antipodal_score"]),
        )
        for item in raw["candidates"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-hypotheses", type=Path, required=True)
    parser.add_argument("--grasp-candidates", type=Path, required=True)
    parser.add_argument("--object-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set-index", type=int, default=0)
    parser.add_argument("--goal-xyz-m", nargs=3, type=float, default=[0.5, 0.0, 0.2])
    parser.add_argument("--pick-position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--pick-angle-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--risk-sigma", type=float, default=2.0)
    args = parser.parse_args()

    config = load_object_config(args.object_config)
    pose_payload = _load(args.pose_hypotheses)
    grasp_payload = _load(args.grasp_candidates)
    if not isinstance(pose_payload, list) or not pose_payload:
        raise SystemExit("pose-hypotheses JSON must be a non-empty list")
    if not 0 <= args.set_index < len(pose_payload):
        raise SystemExit("set-index is outside the pose-hypothesis list")
    pose_set = _pose_set(pose_payload[args.set_index])
    if pose_set.object_id != config.cad_object.object_id:
        raise SystemExit("pose set and object config refer to different objects")
    if grasp_payload.get("object_id") != pose_set.object_id:
        raise SystemExit("grasp candidates and pose set refer to different objects")

    goal = np.eye(4)
    goal[:3, 3] = args.goal_xyz_m
    task = PickPlaceTask(
        transform_world_object_goal=goal,
        position_tolerance_m=config.placement_position_tolerance_m,
        angle_tolerance_deg=config.placement_angle_tolerance_deg,
    )
    scoring = RobustScoringConfig(
        pick_position_tolerance_m=args.pick_position_tolerance_m,
        pick_angle_tolerance_deg=args.pick_angle_tolerance_deg,
        risk_sigma=args.risk_sigma,
    )
    ranked = rank_task_grasps(
        pose_set,
        _candidates(grasp_payload),
        task,
        transform_world_camera=np.eye(4),
        config=scoring,
    )
    payload = {
        "schema_version": 1,
        "object_id": pose_set.object_id,
        "scene_id": pose_set.scene_id,
        "image_id": pose_set.image_id,
        "instance_id": pose_set.instance_id,
        "uncertainty_method": pose_set.uncertainty_method,
        "risk_sigma": scoring.risk_sigma,
        "candidate_count": len(ranked),
        "ranked_candidates": [
            {
                "rank": index + 1,
                "grasp_id": item.candidate.grasp_id,
                "robust_score": item.robust_score,
                "analytic_quality": item.candidate.analytic_quality,
                "feasible_confidence_mass": item.feasible_confidence_mass,
                "uncertainty_success": item.uncertainty_success,
                "travel_score": item.travel_score,
                "pick_position_std_m": item.pick_position_std_m,
                "pick_angle_std_deg": item.pick_angle_std_deg,
                "place_position_std_m": item.place_position_std_m,
                "place_angle_std_deg": item.place_angle_std_deg,
                "uncertainty_available": item.uncertainty_available,
                "top_k_evaluated": len(item.hypothesis_evaluations),
                "transform_world_grasp_pick_best": (
                    item.hypothesis_evaluations[0]
                    .transform_world_grasp_pick.tolist()
                ),
                "transform_world_grasp_place": (
                    item.hypothesis_evaluations[0]
                    .transform_world_grasp_place.tolist()
                ),
            }
            for index, item in enumerate(ranked)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    best = ranked[0]
    print(
        f"ranked {len(ranked)} candidates; best={best.candidate.grasp_id} "
        f"score={best.robust_score:.6f} uncertainty={best.uncertainty_success:.6f}"
    )


if __name__ == "__main__":
    main()
