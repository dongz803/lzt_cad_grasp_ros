"""Backend-independent kinematics result types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IKSolution:
    """Numerical IK result with residual and joint-limit evidence."""

    success: bool
    joint_positions_rad: tuple[float, ...]
    iterations: int
    position_error_m: float
    rotation_error_rad: float
    normalized_joint_margin: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "joint_positions_rad": list(self.joint_positions_rad),
            "iterations": self.iterations,
            "position_error_m": self.position_error_m,
            "rotation_error_rad": self.rotation_error_rad,
            "normalized_joint_margin": self.normalized_joint_margin,
            "reason": self.reason,
        }

