from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.contracts import GraspCandidate, PoseHypothesis  # noqa: E402
from lzt_cad_grasp_core.robot_constraints import (  # noqa: E402
    MujocoUR5eConstraintEvaluator,
    RobotConstraintConfig,
    check_joint_path_collisions,
    solve_site_ik,
)
from lzt_cad_grasp_core.robot_frames import (  # noqa: E402
    load_robot_camera_calibration,
    mujoco_base_camera_calibration,
    transform_camera_object_to_base,
)
from lzt_cad_grasp_core.simulation import (  # noqa: E402
    GRIPPER_PINCH_SITE,
    UR5E_BASE_BODY,
    build_workcell_model,
    reset_robot_home,
)


def homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def workcell():
    model = build_workcell_model()
    data = mujoco.MjData(model)
    reset_robot_home(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def test_simulated_camera_pose_composes_exactly_into_robot_base() -> None:
    model, data = workcell()
    camera_id = model.camera("vision_camera").id
    object_id = model.body("target").id
    base_id = model.body(UR5E_BASE_BODY).id
    world_camera_mj = homogeneous(
        data.cam_xmat[camera_id].reshape(3, 3), data.cam_xpos[camera_id]
    )
    world_camera_cv = world_camera_mj @ np.diag([1.0, -1.0, -1.0, 1.0])
    world_object = homogeneous(
        data.xmat[object_id].reshape(3, 3), data.xpos[object_id]
    )
    world_base = homogeneous(
        data.xmat[base_id].reshape(3, 3), data.xpos[base_id]
    )
    camera_object = np.linalg.inv(world_camera_cv) @ world_object

    actual = transform_camera_object_to_base(
        camera_object, mujoco_base_camera_calibration(model, data)
    )

    np.testing.assert_allclose(
        actual, np.linalg.inv(world_base) @ world_object, atol=1e-12
    )


def test_committed_mujoco_calibration_matches_live_scene() -> None:
    model, data = workcell()
    expected = mujoco_base_camera_calibration(model, data)
    actual = load_robot_camera_calibration(
        PROJECT_ROOT
        / "config/calibration/mujoco_vision_camera_to_ur5e_base.json"
    )

    assert actual.base_frame == "ur5e_base"
    np.testing.assert_allclose(
        actual.transform_base_camera_cv,
        expected.transform_base_camera_cv,
        atol=1e-12,
    )


def test_real_ur5e_jacobian_ik_reaches_requested_site_pose() -> None:
    model, data = workcell()
    site_id = model.site(GRIPPER_PINCH_SITE).id
    target = homogeneous(
        data.site_xmat[site_id].reshape(3, 3),
        data.site_xpos[site_id] + np.array([0.0, 0.0, 0.03]),
    )

    solution = solve_site_ik(model, data, target)

    assert solution.success
    assert solution.position_error_m < 3e-5
    assert solution.rotation_error_rad < 3e-4
    assert len(solution.joint_positions_rad) == 6


def test_collision_checker_detects_robot_environment_penetration() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="wall" type="box" pos="0.5 0 0" size="0.1 0.1 0.1"/>
            <body name="ur5e_test_link">
              <joint name="slide" type="slide" axis="1 0 0" range="0 0.5"/>
              <geom name="robot_sphere" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    result = check_joint_path_collisions(
        model,
        data,
        [0.0],
        [0.5],
        joint_names=("slide",),
        samples=11,
    )

    assert not result.collision_free
    assert any("wall" in pair and "robot_sphere" in pair for pair in result.collision_pairs)


def test_constraint_adapter_supplies_real_ik_and_collision_results() -> None:
    model, data = workcell()
    base_id = model.body(UR5E_BASE_BODY).id
    site_id = model.site(GRIPPER_PINCH_SITE).id
    world_base = homogeneous(
        data.xmat[base_id].reshape(3, 3), data.xpos[base_id]
    )
    world_grasp = homogeneous(
        data.site_xmat[site_id].reshape(3, 3), data.site_xpos[site_id]
    )
    base_grasp = np.linalg.inv(world_base) @ world_grasp
    base_place = base_grasp.copy()
    base_place[:3, 3] += [0.0, -0.03, 0.03]
    evaluator = MujocoUR5eConstraintEvaluator(
        model,
        data,
        world_base,
        RobotConstraintConfig(path_samples=10),
    )
    candidate = GraspCandidate(
        "test", np.eye(4), 0.04, approach_distance_m=0.03
    )

    result = evaluator(
        base_grasp,
        base_place,
        candidate,
        PoseHypothesis(np.eye(4), 1.0),
    )

    assert result.feasible
    assert result.pick_joint_positions_rad is not None
    assert result.place_joint_positions_rad is not None
    assert result.collision_pairs == ()
    assert len(evaluator.audit_log) == 1
