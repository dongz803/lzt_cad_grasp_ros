"""ROS node that validates the isolated exploration configuration and scene."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from lzt_cad_grasp_core.paths import package_share, scene_path
from lzt_cad_grasp_core.simulation import build_workcell_model


class CADGraspExplorationNode(Node):
    """Expose package readiness without selecting a learning backend yet."""

    def __init__(self) -> None:
        super().__init__("cad_grasp_exploration_node")
        self.declare_parameter("scene_name", "exploration_workcell.xml")
        self.declare_parameter("object_config", "objects/copper_baseline.yaml")
        self.declare_parameter("dataset_format", "bop")
        self.declare_parameter("pose_backend", "unconfigured")
        self.declare_parameter("publish_rate_hz", 1.0)

        selected_scene = scene_path(str(self.get_parameter("scene_name").value))
        object_config = package_share() / "config" / str(
            self.get_parameter("object_config").value
        )
        if not object_config.is_file():
            raise FileNotFoundError(object_config)
        model = build_workcell_model(scene_name=selected_scene.name)
        self.status = {
            "route": "cad_driven_learned_pose_and_robust_pick_place",
            "scene": str(selected_scene),
            "object_config": str(object_config),
            "dataset_format": str(self.get_parameter("dataset_format").value),
            "pose_backend": str(self.get_parameter("pose_backend").value),
            "model": {
                "bodies": int(model.nbody),
                "joints": int(model.njnt),
                "actuators": int(model.nu),
            },
            "ready": True,
        }
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(
            String, "/cad_grasp/exploration/status", qos
        )
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.publish_status)
        self.get_logger().info(json.dumps(self.status, ensure_ascii=False))

    def publish_status(self) -> None:
        """Publish the current exploration route and selected resources."""
        message = String()
        message.data = json.dumps(self.status, ensure_ascii=False)
        self.publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    """Run the exploration readiness node."""
    rclpy.init(args=args)
    node = CADGraspExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
