"""Controller and robot-command ports independent of the execution backend."""

from dataclasses import dataclass
from typing import Protocol

from lzt_trajectory import JointTrajectory


@dataclass(frozen=True)
class RobotState:
    joint_names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    efforts_nm: tuple[float, ...]


class TrajectoryController(Protocol):
    def execute(self, trajectory: JointTrajectory) -> None:
        """Track a time-parameterized trajectory on one backend."""


class ForceController(Protocol):
    def set_target_wrench(self, wrench_xyz_rpy: tuple[float, ...]) -> None:
        """Set a Cartesian wrench target for compliant contact handling."""

