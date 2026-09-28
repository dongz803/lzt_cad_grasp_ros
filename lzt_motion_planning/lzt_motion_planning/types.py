"""Backend-independent path and collision result types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JointPath:
    joint_names: tuple[str, ...]
    waypoints_rad: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.joint_names or len(self.waypoints_rad) < 2:
            raise ValueError("a joint path needs joint names and at least two waypoints")
        if any(len(point) != len(self.joint_names) for point in self.waypoints_rad):
            raise ValueError("each waypoint must match the joint-name count")


@dataclass(frozen=True)
class CollisionCheckResult:
    collision_free: bool
    checked_configurations: int
    collision_pairs: tuple[str, ...]
    minimum_contact_distance_m: float | None

    def to_dict(self) -> dict:
        return {
            "collision_free": self.collision_free,
            "checked_configurations": self.checked_configurations,
            "collision_pairs": list(self.collision_pairs),
            "minimum_contact_distance_m": self.minimum_contact_distance_m,
        }

