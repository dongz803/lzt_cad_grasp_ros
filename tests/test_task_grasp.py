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
from lzt_cad_grasp_core.task_grasp import (  # noqa: E402
    MotionConstraintResult,
    RobustScoringConfig,
    rank_task_grasps,
)


def transform(translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = translation
    return result


def candidate(grasp_id: str, point, quality=0.8) -> GraspCandidate:
    return GraspCandidate(
        grasp_id=grasp_id,
        transform_object_grasp=transform(point),
        jaw_opening_m=0.04,
        analytic_quality=quality,
    )


def pose_set(covariance: np.ndarray | None) -> PoseHypothesisSet:
    hypotheses = (
        PoseHypothesis(transform((0.0, 0.0, 0.5)), 0.8, hypothesis_id=0),
        PoseHypothesis(transform((0.01, 0.0, 0.5)), 0.2, hypothesis_id=1),
    )
    return PoseHypothesisSet(
        object_id="part",
        scene_id=1,
        image_id=2,
        instance_id=0,
        hypotheses=hypotheses,
        covariance_se3=covariance,
        uncertainty_method="test",
    )


def task() -> PickPlaceTask:
    return PickPlaceTask(transform((0.5, 0.0, 0.2)), 0.01, 5.0)


def test_rotation_uncertainty_penalizes_long_grasp_lever_arm() -> None:
    covariance = np.zeros((6, 6))
    covariance[5, 5] = np.radians(5.0) ** 2
    ranked = rank_task_grasps(
        pose_set(covariance),
        [candidate("far", (0.1, 0.0, 0.0)), candidate("center", (0.0, 0.0, 0.0))],
        task(),
        np.eye(4),
    )

    assert [item.candidate.grasp_id for item in ranked] == ["center", "far"]
    assert ranked[0].pick_position_std_m == 0.0
    assert np.isclose(ranked[1].pick_position_std_m, 0.1 * np.radians(5.0))
    assert ranked[1].uncertainty_success < ranked[0].uncertainty_success


def test_top_k_constraints_are_weighted_by_hypothesis_confidence() -> None:
    covariance = np.zeros((6, 6))

    def constraints(pick, place, grasp, hypothesis):
        del pick, place
        if grasp.grasp_id == "fragile" and hypothesis.hypothesis_id == 1:
            return MotionConstraintResult(pick_collision_free=False, reason="bin wall")
        return MotionConstraintResult()

    ranked = rank_task_grasps(
        pose_set(covariance),
        [candidate("fragile", (0.0, 0.0, 0.0), 0.9), candidate("robust", (0.0, 0.0, 0.0), 0.8)],
        task(),
        np.eye(4),
        constraint_evaluator=constraints,
    )

    by_name = {item.candidate.grasp_id: item for item in ranked}
    assert np.isclose(by_name["fragile"].feasible_confidence_mass, 0.8)
    assert np.isclose(by_name["robust"].feasible_confidence_mass, 1.0)
    assert ranked[0].candidate.grasp_id == "robust"


def test_missing_uncertainty_is_not_treated_as_zero_risk() -> None:
    config = RobustScoringConfig(missing_uncertainty_penalty=0.25)
    result = rank_task_grasps(
        pose_set(None),
        [candidate("only", (0.0, 0.0, 0.0))],
        task(),
        np.eye(4),
        config=config,
    )[0]

    assert result.uncertainty_available is False
    assert result.uncertainty_success == 0.25
    assert result.pick_position_std_m is None
