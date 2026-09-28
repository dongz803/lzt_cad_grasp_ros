"""Time-parameterized joint trajectory contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JointTrajectoryPoint:
    time_from_start_s: float
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    accelerations_rad_s2: tuple[float, ...]


@dataclass(frozen=True)
class JointTrajectory:
    joint_names: tuple[str, ...]
    points: tuple[JointTrajectoryPoint, ...]

    def __post_init__(self) -> None:
        if not self.joint_names or len(self.points) < 2:
            raise ValueError("a trajectory needs joint names and at least two points")
        previous_time = -1.0
        for point in self.points:
            if point.time_from_start_s <= previous_time:
                raise ValueError("trajectory times must be strictly increasing")
            previous_time = point.time_from_start_s
            vectors = (
                point.positions_rad,
                point.velocities_rad_s,
                point.accelerations_rad_s2,
            )
            if any(len(vector) != len(self.joint_names) for vector in vectors):
                raise ValueError("trajectory vectors must match the joint-name count")

