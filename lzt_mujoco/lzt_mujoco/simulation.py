"""Compose an independent copy of the baseline MuJoCo workcell."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco

from lzt_common.paths import asset_path, scene_path


UR5E_PREFIX = "ur5e_"
GRIPPER_PREFIX = "gripper_"
UR5E_HOME_KEY = f"{UR5E_PREFIX}home"
D435_CAMERA_NAME = f"{UR5E_PREFIX}{GRIPPER_PREFIX}d435_camera"
GRIPPER_ACTUATOR_NAME = f"{UR5E_PREFIX}{GRIPPER_PREFIX}fingers_actuator"
GRIPPER_PINCH_SITE = f"{UR5E_PREFIX}{GRIPPER_PREFIX}pinch"
UR5E_BASE_BODY = f"{UR5E_PREFIX}base"
UR5E_JOINT_NAMES = (
    f"{UR5E_PREFIX}shoulder_pan_joint",
    f"{UR5E_PREFIX}shoulder_lift_joint",
    f"{UR5E_PREFIX}elbow_joint",
    f"{UR5E_PREFIX}wrist_1_joint",
    f"{UR5E_PREFIX}wrist_2_joint",
    f"{UR5E_PREFIX}wrist_3_joint",
)
UR5E_ACTUATOR_NAMES = (
    f"{UR5E_PREFIX}shoulder_pan",
    f"{UR5E_PREFIX}shoulder_lift",
    f"{UR5E_PREFIX}elbow",
    f"{UR5E_PREFIX}wrist_1",
    f"{UR5E_PREFIX}wrist_2",
    f"{UR5E_PREFIX}wrist_3",
)


def build_workcell_spec(
    robot_position: Sequence[float] = (-0.75, 0.0, 0.1425),
    robot_quaternion_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    *,
    scene_name: str = "exploration_workcell.xml",
) -> mujoco.MjSpec:
    """Attach copied Tracer, UR5e and 2F-85 assets to a copied scene."""
    selected_scene = scene_path(scene_name)
    tracer_path = asset_path("robots/tracer/tracer.xml")
    arm_path = asset_path("robots/universal_robots_ur5e/ur5e.xml")
    gripper_path = asset_path("grippers/robotiq_2f85/2f85.xml")

    scene = mujoco.MjSpec.from_file(str(selected_scene))
    tracer = mujoco.MjSpec.from_file(str(tracer_path))
    arm = mujoco.MjSpec.from_file(str(arm_path))
    gripper = mujoco.MjSpec.from_file(str(gripper_path))

    arm.attach(
        gripper,
        site=arm.site("attachment_site"),
        prefix=GRIPPER_PREFIX,
    )
    arm.compile()
    tracer.attach(
        arm,
        site=tracer.site("arm_mount"),
        prefix=UR5E_PREFIX,
    )
    tracer.compile()
    mount = scene.worldbody.add_frame(
        name="mobile_robot_mount",
        pos=list(robot_position),
        quat=list(robot_quaternion_wxyz),
    )
    scene.attach(tracer, frame=mount, prefix="")
    return scene


def build_workcell_model(
    robot_position: Sequence[float] = (-0.75, 0.0, 0.1425),
    robot_quaternion_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    *,
    scene_name: str = "exploration_workcell.xml",
) -> mujoco.MjModel:
    """Compile the full copied mobile-manipulator workcell."""
    return build_workcell_spec(
        robot_position,
        robot_quaternion_wxyz,
        scene_name=scene_name,
    ).compile()


def reset_robot_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset the complete model to the copied UR5e home keyframe."""
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, UR5E_HOME_KEY)
    if home_id < 0:
        raise ValueError(f"missing home keyframe: {UR5E_HOME_KEY}")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
