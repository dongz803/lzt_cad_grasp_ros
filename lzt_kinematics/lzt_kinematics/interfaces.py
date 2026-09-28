"""Kinematics ports implemented by MuJoCo, MoveIt or a real robot model."""

from typing import Protocol, Sequence

import numpy as np

from .types import IKSolution


class IKSolver(Protocol):
    def solve(
        self,
        target_base_tool: np.ndarray,
        initial_joint_positions_rad: Sequence[float] | None = None,
    ) -> IKSolution:
        """Solve a tool target without exposing the backend implementation."""

