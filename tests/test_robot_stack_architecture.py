from lzt_control import RobotState
from lzt_kinematics import IKSolution
from lzt_motion_planning import CollisionCheckResult, JointPath
from lzt_trajectory import linear_time_parameterize


def test_geometric_path_becomes_timed_trajectory():
    path = JointPath(
        joint_names=("joint_1", "joint_2"),
        waypoints_rad=((0.0, 0.0), (0.2, -0.1), (0.3, 0.1)),
    )
    trajectory = linear_time_parameterize(
        path, maximum_velocity_rad_s=0.5, sample_period_s=0.05
    )
    assert trajectory.joint_names == path.joint_names
    assert trajectory.points[0].positions_rad == path.waypoints_rad[0]
    assert trajectory.points[-1].positions_rad == path.waypoints_rad[-1]
    assert trajectory.points[-1].time_from_start_s > 0.0


def test_robot_algorithm_result_contracts_remain_backend_independent():
    ik = IKSolution(True, (0.0,) * 6, 4, 1e-6, 1e-6, 0.5)
    collision = CollisionCheckResult(True, 25, (), None)
    state = RobotState(("j1",), (0.0,), (0.0,), (0.0,))
    assert ik.success and collision.collision_free
    assert state.joint_names == ("j1",)

