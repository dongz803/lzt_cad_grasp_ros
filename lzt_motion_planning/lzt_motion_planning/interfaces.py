"""Motion-planning ports independent of MuJoCo and MoveIt."""

from typing import Protocol

from .types import CollisionCheckResult, JointPath


class CollisionChecker(Protocol):
    def check(self, path: JointPath) -> CollisionCheckResult:
        """Audit all configurations along a geometric joint path."""


class MotionPlanner(Protocol):
    def plan(
        self,
        start_joint_positions_rad: tuple[float, ...],
        goal_joint_positions_rad: tuple[float, ...],
    ) -> JointPath:
        """Return a collision-aware geometric path without time stamps."""

