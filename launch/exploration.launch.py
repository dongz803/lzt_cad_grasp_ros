"""Launch the isolated CAD-grasp exploration node."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_file = PathJoinSubstitution(
        [FindPackageShare("lzt_cad_grasp_ros"), "config", "exploration.yaml"]
    )
    return LaunchDescription(
        [
            Node(
                package="lzt_cad_grasp_ros",
                executable="cad_grasp_exploration_node",
                name="cad_grasp_exploration_node",
                output="screen",
                parameters=[config_file],
            )
        ]
    )
