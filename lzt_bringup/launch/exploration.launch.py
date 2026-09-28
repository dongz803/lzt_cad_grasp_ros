"""Launch the isolated CAD-grasp exploration node."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_file = PathJoinSubstitution(
        [FindPackageShare("lzt_bringup"), "config", "exploration.yaml"]
    )
    return LaunchDescription(
        [
            Node(
                package="lzt_bringup",
                executable="exploration_node",
                name="cad_grasp_exploration_node",
                output="screen",
                parameters=[config_file],
            )
        ]
    )
