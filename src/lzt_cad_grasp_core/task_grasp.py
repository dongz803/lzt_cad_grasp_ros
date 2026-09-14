"""Uncertainty-aware joint ranking of CAD pick-and-place grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .contracts import (
    GraspCandidate,
    PickPlaceTask,
    PoseHypothesis,
    PoseHypothesisSet,
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _homogeneous(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation must have determinant +1")
    return transform.copy()


@dataclass(frozen=True)
class MotionConstraintResult:
    """Per-hypothesis robot and scene checks supplied by IK/collision backends."""

    pick_reachable: bool = True
    place_reachable: bool = True
    pick_collision_free: bool = True
    place_collision_free: bool = True
    normalized_margin: float = 1.0
    reason: str = ""
    pick_joint_positions_rad: tuple[float, ...] | None = None
    place_joint_positions_rad: tuple[float, ...] | None = None
    collision_pairs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.normalized_margin <= 1.0:
            raise ValueError("normalized_margin must lie in [0, 1]")
        for field_name in (
            "pick_joint_positions_rad",
            "place_joint_positions_rad",
        ):
            values = getattr(self, field_name)
            if values is not None and not all(np.isfinite(values)):
                raise ValueError(f"{field_name} must contain finite values")

    @property
    def feasible(self) -> bool:
        return (
            self.pick_reachable
            and self.place_reachable
            and self.pick_collision_free
            and self.place_collision_free
        )


ConstraintEvaluator = Callable[
    [np.ndarray, np.ndarray, GraspCandidate, PoseHypothesis],
    MotionConstraintResult,
]


@dataclass(frozen=True)
class RobustScoringConfig:
    """Risk and task trade-offs for robust candidate ranking."""

    pick_position_tolerance_m: float = 0.01
    pick_angle_tolerance_deg: float = 10.0
    risk_sigma: float = 2.0
    travel_reference_m: float = 1.0
    quality_weight: float = 0.8
    travel_weight: float = 0.2
    missing_uncertainty_penalty: float = 0.5
    cvar_tail_probability: float = 0.25
    cvar_risk_weight: float = 0.0
    include_place_uncertainty: bool = True
    confidence_is_calibrated_probability: bool = False

    def __post_init__(self) -> None:
        if self.pick_position_tolerance_m <= 0.0:
            raise ValueError("pick_position_tolerance_m must be positive")
        if self.pick_angle_tolerance_deg <= 0.0:
            raise ValueError("pick_angle_tolerance_deg must be positive")
        if self.risk_sigma <= 0.0 or self.travel_reference_m <= 0.0:
            raise ValueError("risk_sigma and travel_reference_m must be positive")
        weights = (
            self.quality_weight,
            self.travel_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("score weights must be non-negative")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("score weights must sum to 1")
        if not 0.0 <= self.missing_uncertainty_penalty <= 1.0:
            raise ValueError("missing_uncertainty_penalty must lie in [0, 1]")
        if not 0.0 < self.cvar_tail_probability <= 1.0:
            raise ValueError("cvar_tail_probability must lie in (0, 1]")
        if not 0.0 <= self.cvar_risk_weight <= 1.0:
            raise ValueError("cvar_risk_weight must lie in [0, 1]")


@dataclass(frozen=True)
class HypothesisTaskEvaluation:
    hypothesis_id: int
    confidence_weight: float
    transform_world_grasp_pick: np.ndarray
    transform_world_grasp_place: np.ndarray
    constraints: MotionConstraintResult


@dataclass(frozen=True)
class RankedTaskGrasp:
    """One CAD grasp with traceable task, uncertainty and feasibility terms."""

    candidate: GraspCandidate
    robust_score: float
    pose_set_reliability: float
    feasible_confidence_mass: float
    expected_task_success: float
    tail_task_success: float
    cvar_failure_risk: float
    uncertainty_success: float
    travel_score: float
    pick_position_std_m: float | None
    pick_angle_std_deg: float | None
    place_position_std_m: float | None
    place_angle_std_deg: float | None
    uncertainty_available: bool
    hypothesis_evaluations: tuple[HypothesisTaskEvaluation, ...]


def _maximum_std(covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    return float(np.sqrt(max(float(np.max(eigenvalues)), 0.0)))


def _tolerance_score(std: float, tolerance: float, risk_sigma: float) -> float:
    normalized = risk_sigma * std / tolerance
    return float(np.exp(-0.5 * normalized * normalized))


def weighted_cvar(
    losses: np.ndarray,
    weights: np.ndarray,
    tail_probability: float,
) -> float:
    """Return expected loss in the worst discrete probability tail."""

    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(values) == 0 or len(values) != len(probabilities):
        raise ValueError("losses and weights must be non-empty and aligned")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("losses and weights must be finite")
    if np.any(probabilities < 0.0) or np.sum(probabilities) <= 0.0:
        raise ValueError("weights must be non-negative with positive mass")
    if not 0.0 < tail_probability <= 1.0:
        raise ValueError("tail_probability must lie in (0, 1]")
    probabilities = probabilities / np.sum(probabilities)
    order = np.argsort(-values, kind="mergesort")
    remaining = tail_probability
    accumulated = 0.0
    for index in order:
        mass = min(float(probabilities[index]), remaining)
        accumulated += mass * float(values[index])
        remaining -= mass
        if remaining <= 1e-12:
            break
    return accumulated / tail_probability


def _propagated_uncertainty(
    covariance_se3: np.ndarray,
    transform_world_camera: np.ndarray,
    transform_camera_object: np.ndarray,
    grasp_point_object: np.ndarray,
) -> tuple[float, float, float, float]:
    """Propagate [camera translation, object-local rotation] covariance.

    The Top-K converter stores translation residuals in the camera frame and
    rotational residuals in the canonicalized object frame.  For grasp point
    ``r`` the first-order position Jacobian is ``[I, -R_co [r]x]``.  Camera
    and object rotations then express the position and angle covariances in
    the world frame.  Maximum directional standard deviation is used as a
    conservative scalar risk measure.
    """

    covariance = np.asarray(covariance_se3, dtype=np.float64)
    rotation_world_camera = transform_world_camera[:3, :3]
    rotation_camera_object = transform_camera_object[:3, :3]
    jacobian = np.hstack(
        (
            np.eye(3),
            -rotation_camera_object @ _skew(grasp_point_object),
        )
    )
    pick_position_camera = jacobian @ covariance @ jacobian.T
    pick_position_world = (
        rotation_world_camera
        @ pick_position_camera
        @ rotation_world_camera.T
    )
    object_position_world = (
        rotation_world_camera
        @ covariance[:3, :3]
        @ rotation_world_camera.T
    )
    rotation_world_object = rotation_world_camera @ rotation_camera_object
    angle_world = (
        rotation_world_object
        @ covariance[3:, 3:]
        @ rotation_world_object.T
    )
    angle_std_deg = float(np.degrees(_maximum_std(angle_world)))
    return (
        _maximum_std(pick_position_world),
        angle_std_deg,
        _maximum_std(object_position_world),
        angle_std_deg,
    )


def _default_constraint_evaluator(
    pick: np.ndarray,
    place: np.ndarray,
    candidate: GraspCandidate,
    hypothesis: PoseHypothesis,
) -> MotionConstraintResult:
    del pick, place, candidate, hypothesis
    return MotionConstraintResult()


def rank_task_grasps(
    pose_set: PoseHypothesisSet,
    grasp_candidates: list[GraspCandidate],
    task: PickPlaceTask,
    transform_world_camera: np.ndarray,
    constraint_evaluator: ConstraintEvaluator | None = None,
    config: RobustScoringConfig | None = None,
) -> list[RankedTaskGrasp]:
    """Rank CAD grasps over Top-K poses, uncertainty and pick/place constraints."""

    if not grasp_candidates:
        return []
    world_camera = _homogeneous(transform_world_camera, "transform_world_camera")
    scoring = config or RobustScoringConfig()
    evaluator = constraint_evaluator or _default_constraint_evaluator
    confidences = np.asarray(
        [hypothesis.confidence for hypothesis in pose_set.hypotheses],
        dtype=np.float64,
    )
    # Compute absolute reliability before the zero-sum fallback used only to
    # obtain relative scenario weights.
    pose_set_reliability = (
        float(np.max(confidences))
        if scoring.confidence_is_calibrated_probability
        else 1.0
    )
    if np.sum(confidences) <= 1e-12:
        confidences = np.ones_like(confidences)
    weights = confidences / np.sum(confidences)
    # Normalized Top-K weights express only relative preference.  With
    # calibrated correctness probabilities, preserve the absolute chance that
    # no retained pose is reliable.  ``max`` is a conservative group proxy for
    # strongly correlated Top-K pose events.
    covariance = pose_set.covariance_se3
    if covariance is None:
        covariance = pose_set.hypotheses[0].covariance_se3

    ranked = []
    for candidate in grasp_candidates:
        place = (
            task.transform_world_object_goal
            @ candidate.transform_object_grasp
        )
        hypothesis_evaluations = []
        feasibility_terms = []
        travel_terms = []
        for weight, hypothesis in zip(weights, pose_set.hypotheses):
            world_object = world_camera @ hypothesis.transform_camera_object
            pick = world_object @ candidate.transform_object_grasp
            constraints = evaluator(pick, place, candidate, hypothesis)
            feasibility_terms.append(
                weight
                * constraints.normalized_margin
                * float(constraints.feasible)
            )
            travel_distance = float(np.linalg.norm(place[:3, 3] - pick[:3, 3]))
            travel_terms.append(
                weight * np.exp(-travel_distance / scoring.travel_reference_m)
            )
            hypothesis_evaluations.append(
                HypothesisTaskEvaluation(
                    hypothesis_id=hypothesis.hypothesis_id,
                    confidence_weight=float(weight),
                    transform_world_grasp_pick=pick,
                    transform_world_grasp_place=place.copy(),
                    constraints=constraints,
                )
            )

        uncertainty_available = covariance is not None
        if covariance is None:
            uncertainty_success = scoring.missing_uncertainty_penalty
            pick_position_std = None
            pick_angle_std = None
            place_position_std = None
            place_angle_std = None
        else:
            (
                pick_position_std,
                pick_angle_std,
                place_position_std,
                place_angle_std,
            ) = _propagated_uncertainty(
                covariance,
                world_camera,
                pose_set.hypotheses[0].transform_camera_object,
                candidate.transform_object_grasp[:3, 3],
            )
            scores = [
                _tolerance_score(
                    pick_position_std,
                    scoring.pick_position_tolerance_m,
                    scoring.risk_sigma,
                ),
                _tolerance_score(
                    np.radians(pick_angle_std),
                    np.radians(scoring.pick_angle_tolerance_deg),
                    scoring.risk_sigma,
                ),
            ]
            if scoring.include_place_uncertainty:
                scores.extend(
                    [
                        _tolerance_score(
                            place_position_std,
                            task.position_tolerance_m,
                            scoring.risk_sigma,
                        ),
                        _tolerance_score(
                            np.radians(place_angle_std),
                            np.radians(task.angle_tolerance_deg),
                            scoring.risk_sigma,
                        ),
                    ]
                )
            uncertainty_success = float(np.prod(scores))

        feasible_mass = float(np.sum(feasibility_terms))
        scenario_success = np.asarray(
            [
                evaluation.constraints.normalized_margin
                * float(evaluation.constraints.feasible)
                for evaluation in hypothesis_evaluations
            ],
            dtype=np.float64,
        )
        scenario_weights = weights
        if scoring.confidence_is_calibrated_probability:
            scenario_success = np.r_[scenario_success, 0.0]
            scenario_weights = np.r_[
                weights * pose_set_reliability,
                1.0 - pose_set_reliability,
            ]
        expected_task_success = pose_set_reliability * feasible_mass
        cvar_failure_risk = weighted_cvar(
            1.0 - scenario_success,
            scenario_weights,
            scoring.cvar_tail_probability,
        )
        tail_task_success = float(np.clip(1.0 - cvar_failure_risk, 0.0, 1.0))
        risk_adjusted_success = (
            (1.0 - scoring.cvar_risk_weight) * expected_task_success
            + scoring.cvar_risk_weight * tail_task_success
        )
        travel_score = float(np.sum(travel_terms))
        preference_score = (
            scoring.quality_weight * candidate.analytic_quality
            + scoring.travel_weight * travel_score
        )
        robust_score = float(
            np.clip(
                risk_adjusted_success * uncertainty_success * preference_score,
                0.0,
                1.0,
            )
        )
        ranked.append(
            RankedTaskGrasp(
                candidate=candidate,
                robust_score=robust_score,
                pose_set_reliability=pose_set_reliability,
                feasible_confidence_mass=feasible_mass,
                expected_task_success=expected_task_success,
                tail_task_success=tail_task_success,
                cvar_failure_risk=cvar_failure_risk,
                uncertainty_success=uncertainty_success,
                travel_score=travel_score,
                pick_position_std_m=pick_position_std,
                pick_angle_std_deg=pick_angle_std,
                place_position_std_m=place_position_std,
                place_angle_std_deg=place_angle_std,
                uncertainty_available=uncertainty_available,
                hypothesis_evaluations=tuple(hypothesis_evaluations),
            )
        )
    ranked.sort(key=lambda item: (-item.robust_score, item.candidate.grasp_id))
    return ranked
