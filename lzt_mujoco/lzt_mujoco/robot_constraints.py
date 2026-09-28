"""UR5e Jacobian IK and MuJoCo path-collision constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from lzt_common.contracts import GraspCandidate, PoseHypothesis
from lzt_common.robot_frames import validate_transform
from lzt_grasp_planning.task_grasp import MotionConstraintResult
from lzt_kinematics import IKSolution
from lzt_motion_planning import CollisionCheckResult
from .simulation import GRIPPER_PINCH_SITE, UR5E_JOINT_NAMES


def _rotation_error_vector(rotation_error: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation_error) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-4:
        eigenvalues, eigenvectors = np.linalg.eig(rotation_error)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        return angle * axis
    skew = np.array(
        [
            rotation_error[2, 1] - rotation_error[1, 2],
            rotation_error[0, 2] - rotation_error[2, 0],
            rotation_error[1, 0] - rotation_error[0, 1],
        ]
    )
    return angle * skew / (2.0 * np.sin(angle))


def _joint_metadata(
    model: mujoco.MjModel, joint_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.asarray([model.joint(name).id for name in joint_names], dtype=int)
    return (
        joint_ids,
        model.jnt_dofadr[joint_ids].astype(int),
        model.jnt_qposadr[joint_ids].astype(int),
    )


def _normalized_joint_margin(
    model: mujoco.MjModel,
    joint_ids: np.ndarray,
    qpos_addresses: np.ndarray,
    qpos: np.ndarray,
) -> float:
    margins = []
    for joint_id, address in zip(joint_ids, qpos_addresses):
        if not model.jnt_limited[joint_id]:
            continue
        low, high = model.jnt_range[joint_id]
        span = float(high - low)
        if span <= 0.0:
            continue
        distance = min(float(qpos[address] - low), float(high - qpos[address]))
        margins.append(float(np.clip(2.0 * distance / span, 0.0, 1.0)))
    return min(margins, default=1.0)


def solve_site_ik(
    model: mujoco.MjModel,
    reference_data: mujoco.MjData,
    target_world_grasp: np.ndarray,
    *,
    site_name: str = GRIPPER_PINCH_SITE,
    joint_names: Sequence[str] = UR5E_JOINT_NAMES,
    initial_joint_positions_rad: Sequence[float] | None = None,
    max_iterations: int = 300,
    damping: float = 0.03,
    max_step_rad: float = 0.08,
    position_tolerance_m: float = 3e-5,
    rotation_tolerance_rad: float = 3e-4,
) -> IKSolution:
    """Solve a six-dimensional gripper-site target using the real model Jacobian."""

    target = validate_transform(target_world_grasp, "target_world_grasp")
    data = mujoco.MjData(model)
    data.qpos[:] = reference_data.qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = reference_data.ctrl
    joint_ids, dof_addresses, qpos_addresses = _joint_metadata(model, joint_names)
    if initial_joint_positions_rad is not None:
        initial = np.asarray(initial_joint_positions_rad, dtype=np.float64)
        if initial.shape != (len(joint_names),):
            raise ValueError("initial_joint_positions_rad has incorrect length")
        data.qpos[qpos_addresses] = initial
    mujoco.mj_forward(model, data)
    site_id = model.site(site_name).id

    position_error = np.full(3, np.inf)
    rotation_error = np.full(3, np.inf)
    for iteration in range(1, max_iterations + 1):
        position_error = target[:3, 3] - data.site_xpos[site_id]
        rotation_error = _rotation_error_vector(
            target[:3, :3] @ data.site_xmat[site_id].reshape(3, 3).T
        )
        if (
            np.linalg.norm(position_error) < position_tolerance_m
            and np.linalg.norm(rotation_error) < rotation_tolerance_rad
        ):
            joints = tuple(float(value) for value in data.qpos[qpos_addresses])
            return IKSolution(
                success=True,
                joint_positions_rad=joints,
                iterations=iteration,
                position_error_m=float(np.linalg.norm(position_error)),
                rotation_error_rad=float(np.linalg.norm(rotation_error)),
                normalized_joint_margin=_normalized_joint_margin(
                    model, joint_ids, qpos_addresses, data.qpos
                ),
            )

        jacobian_position = np.zeros((3, model.nv))
        jacobian_rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            site_id,
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, dof_addresses],
                jacobian_rotation[:, dof_addresses],
            )
        )
        error = np.concatenate((position_error, rotation_error))
        # Damped least squares remains stable near singular UR5e postures and
        # solves against the Jacobian of the actual MuJoCo kinematic chain.
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), error
        )
        largest = float(np.max(np.abs(delta)))
        if largest > max_step_rad:
            delta *= max_step_rad / largest
        data.qpos[qpos_addresses] += delta
        for joint_id, address in zip(joint_ids, qpos_addresses):
            if model.jnt_limited[joint_id]:
                low, high = model.jnt_range[joint_id]
                data.qpos[address] = np.clip(data.qpos[address], low, high)
        mujoco.mj_forward(model, data)

    joints = tuple(float(value) for value in data.qpos[qpos_addresses])
    return IKSolution(
        success=False,
        joint_positions_rad=joints,
        iterations=max_iterations,
        position_error_m=float(np.linalg.norm(position_error)),
        rotation_error_rad=float(np.linalg.norm(rotation_error)),
        normalized_joint_margin=_normalized_joint_margin(
            model, joint_ids, qpos_addresses, data.qpos
        ),
        reason="ik_did_not_converge",
    )


def _geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    name = model.geom(geom_id).name
    if name:
        return name
    body = model.body(int(model.geom_bodyid[geom_id])).name or "body"
    return f"{body}/geom_{geom_id}"


def _is_adjacent_body(model: mujoco.MjModel, first: int, second: int) -> bool:
    return (
        int(model.body_parentid[first]) == second
        or int(model.body_parentid[second]) == first
    )


def _disallowed_contact_pair(
    model: mujoco.MjModel,
    geom1: int,
    geom2: int,
    ignored_environment_bodies: set[str],
) -> bool:
    body1 = int(model.geom_bodyid[geom1])
    body2 = int(model.geom_bodyid[geom2])
    name1 = model.body(body1).name or ""
    name2 = model.body(body2).name or ""
    robot1 = name1.startswith("ur5e_")
    robot2 = name2.startswith("ur5e_")
    if not robot1 and not robot2:
        return False
    if robot1 and robot2:
        if body1 == body2 or _is_adjacent_body(model, body1, body2):
            return False
        if name1.startswith("ur5e_gripper_") and name2.startswith("ur5e_gripper_"):
            return False
        return True
    environment_name = name2 if robot1 else name1
    return environment_name not in ignored_environment_bodies


def check_joint_path_collisions(
    model: mujoco.MjModel,
    reference_data: mujoco.MjData,
    start_joint_positions_rad: Sequence[float],
    end_joint_positions_rad: Sequence[float],
    *,
    joint_names: Sequence[str] = UR5E_JOINT_NAMES,
    samples: int = 25,
    penetration_tolerance_m: float = -1e-6,
    ignored_environment_bodies: Sequence[str] = ("target",),
) -> CollisionCheckResult:
    """Interpolate arm joints and inspect real MuJoCo contact pairs."""

    if samples < 2:
        raise ValueError("samples must be at least 2")
    start = np.asarray(start_joint_positions_rad, dtype=np.float64)
    end = np.asarray(end_joint_positions_rad, dtype=np.float64)
    if start.shape != (len(joint_names),) or end.shape != (len(joint_names),):
        raise ValueError("joint path endpoints have incorrect length")
    data = mujoco.MjData(model)
    data.qpos[:] = reference_data.qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = reference_data.ctrl
    _, _, qpos_addresses = _joint_metadata(model, joint_names)
    ignored = set(ignored_environment_bodies)
    collision_pairs: set[str] = set()
    minimum_distance = None
    for phase in np.linspace(0.0, 1.0, samples):
        # Collision is evaluated along the interpolated path, not only at the
        # final IK solution where an intermediate platform hit could be missed.
        data.qpos[qpos_addresses] = (1.0 - phase) * start + phase * end
        mujoco.mj_forward(model, data)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            distance = float(contact.dist)
            if distance >= penetration_tolerance_m:
                continue
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if not _disallowed_contact_pair(model, geom1, geom2, ignored):
                continue
            collision_pairs.add(
                f"{_geom_label(model, geom1)} <-> {_geom_label(model, geom2)}"
            )
            if minimum_distance is None or distance < minimum_distance:
                minimum_distance = distance
    return CollisionCheckResult(
        collision_free=not collision_pairs,
        checked_configurations=samples,
        collision_pairs=tuple(sorted(collision_pairs)),
        minimum_contact_distance_m=minimum_distance,
    )


def current_disallowed_robot_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    penetration_tolerance_m: float = -1e-6,
    ignored_environment_bodies: Sequence[str] = ("target",),
) -> tuple[str, ...]:
    """Return forbidden robot contacts in the current dynamic state."""

    ignored = set(ignored_environment_bodies)
    pairs: set[str] = set()
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if float(contact.dist) >= penetration_tolerance_m:
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if not _disallowed_contact_pair(model, geom1, geom2, ignored):
            continue
        pairs.add(
            f"{_geom_label(model, geom1)} <-> {_geom_label(model, geom2)}"
        )
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class RobotConstraintConfig:
    """IK and path-collision settings for one grasp feasibility query."""

    site_name: str = GRIPPER_PINCH_SITE
    joint_names: tuple[str, ...] = UR5E_JOINT_NAMES
    path_samples: int = 25
    penetration_tolerance_m: float = -1e-6
    ignored_environment_bodies: tuple[str, ...] = ("target",)


class MujocoUR5eConstraintEvaluator:
    """Callable adapter consumed directly by ``rank_task_grasps``."""

    def __init__(
        self,
        model: mujoco.MjModel,
        reference_data: mujoco.MjData,
        transform_world_base: np.ndarray,
        config: RobotConstraintConfig | None = None,
    ) -> None:
        self.model = model
        self.reference_data = reference_data
        self.transform_world_base = validate_transform(
            transform_world_base, "transform_world_base"
        )
        self.config = config or RobotConstraintConfig()
        _, _, addresses = _joint_metadata(model, self.config.joint_names)
        self.reference_joints = reference_data.qpos[addresses].copy()
        self.audit_log: list[dict] = []

    def _evaluate_target(
        self,
        transform_base_grasp: np.ndarray,
        approach_distance_m: float,
        initial_joints: Sequence[float],
    ) -> tuple[IKSolution, IKSolution, CollisionCheckResult, CollisionCheckResult]:
        target_base = validate_transform(transform_base_grasp, "transform_base_grasp")
        # The grasp local +Z axis is the approach direction used by the CAD
        # candidate generator; pregrasp backs away along that same axis.
        pregrasp_base = target_base.copy()
        pregrasp_base[:3, 3] -= approach_distance_m * target_base[:3, 2]
        pregrasp_world = self.transform_world_base @ pregrasp_base
        target_world = self.transform_world_base @ target_base
        pregrasp_ik = solve_site_ik(
            self.model,
            self.reference_data,
            pregrasp_world,
            site_name=self.config.site_name,
            joint_names=self.config.joint_names,
            initial_joint_positions_rad=initial_joints,
        )
        if pregrasp_ik.success:
            target_ik = solve_site_ik(
                self.model,
                self.reference_data,
                target_world,
                site_name=self.config.site_name,
                joint_names=self.config.joint_names,
                initial_joint_positions_rad=pregrasp_ik.joint_positions_rad,
            )
        else:
            target_ik = pregrasp_ik
        empty = CollisionCheckResult(True, 0, (), None)
        if not pregrasp_ik.success:
            return pregrasp_ik, target_ik, empty, empty
        approach_path = check_joint_path_collisions(
            self.model,
            self.reference_data,
            initial_joints,
            pregrasp_ik.joint_positions_rad,
            joint_names=self.config.joint_names,
            samples=self.config.path_samples,
            penetration_tolerance_m=self.config.penetration_tolerance_m,
            ignored_environment_bodies=self.config.ignored_environment_bodies,
        )
        if not target_ik.success:
            return pregrasp_ik, target_ik, approach_path, empty
        grasp_path = check_joint_path_collisions(
            self.model,
            self.reference_data,
            pregrasp_ik.joint_positions_rad,
            target_ik.joint_positions_rad,
            joint_names=self.config.joint_names,
            samples=self.config.path_samples,
            penetration_tolerance_m=self.config.penetration_tolerance_m,
            ignored_environment_bodies=self.config.ignored_environment_bodies,
        )
        return pregrasp_ik, target_ik, approach_path, grasp_path

    def __call__(
        self,
        pick: np.ndarray,
        place: np.ndarray,
        candidate: GraspCandidate,
        hypothesis: PoseHypothesis,
    ) -> MotionConstraintResult:
        pick_results = self._evaluate_target(
            pick, candidate.approach_distance_m, self.reference_joints
        )
        pick_pregrasp, pick_grasp, pick_approach_path, pick_grasp_path = pick_results
        place_start = (
            pick_grasp.joint_positions_rad
            if pick_grasp.success
            else self.reference_joints
        )
        place_results = self._evaluate_target(
            place, candidate.approach_distance_m, place_start
        )
        (
            place_pregrasp,
            place_grasp,
            place_approach_path,
            place_grasp_path,
        ) = place_results
        pick_collision_free = (
            pick_approach_path.collision_free and pick_grasp_path.collision_free
        )
        place_collision_free = (
            place_approach_path.collision_free and place_grasp_path.collision_free
        )
        collision_pairs = tuple(
            sorted(
                set(
                    pick_approach_path.collision_pairs
                    + pick_grasp_path.collision_pairs
                    + place_approach_path.collision_pairs
                    + place_grasp_path.collision_pairs
                )
            )
        )
        reasons = []
        if not pick_grasp.success:
            reasons.append("pick_ik_failed")
        if not place_grasp.success:
            reasons.append("place_ik_failed")
        if not pick_collision_free:
            reasons.append("pick_collision")
        if not place_collision_free:
            reasons.append("place_collision")
        margins = [
            result.normalized_joint_margin
            for result in (pick_pregrasp, pick_grasp, place_pregrasp, place_grasp)
            if result.success
        ]
        normalized_margin = min(margins, default=0.0)
        audit = {
            "grasp_id": candidate.grasp_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "pick_pregrasp_ik": pick_pregrasp.to_dict(),
            "pick_grasp_ik": pick_grasp.to_dict(),
            "place_pregrasp_ik": place_pregrasp.to_dict(),
            "place_grasp_ik": place_grasp.to_dict(),
            "pick_approach_path": pick_approach_path.to_dict(),
            "pick_grasp_path": pick_grasp_path.to_dict(),
            "place_approach_path": place_approach_path.to_dict(),
            "place_grasp_path": place_grasp_path.to_dict(),
        }
        self.audit_log.append(audit)
        return MotionConstraintResult(
            pick_reachable=pick_grasp.success,
            place_reachable=place_grasp.success,
            pick_collision_free=pick_collision_free,
            place_collision_free=place_collision_free,
            normalized_margin=normalized_margin,
            reason=";".join(reasons),
            pick_joint_positions_rad=(
                pick_grasp.joint_positions_rad if pick_grasp.success else None
            ),
            place_joint_positions_rad=(
                place_grasp.joint_positions_rad if place_grasp.success else None
            ),
            collision_pairs=collision_pairs,
        )
