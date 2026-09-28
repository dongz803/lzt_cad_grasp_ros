"""Deterministic baseline time parameterization for geometric joint paths."""

import numpy as np

from lzt_motion_planning import JointPath

from .types import JointTrajectory, JointTrajectoryPoint


def linear_time_parameterize(
    path: JointPath,
    *,
    maximum_velocity_rad_s: float = 0.5,
    sample_period_s: float = 0.02,
) -> JointTrajectory:
    if maximum_velocity_rad_s <= 0.0 or sample_period_s <= 0.0:
        raise ValueError("velocity and sample period must be positive")
    segments: list[JointTrajectoryPoint] = []
    elapsed = 0.0
    zeros = tuple(0.0 for _ in path.joint_names)
    for segment_index, (start_raw, end_raw) in enumerate(
        zip(path.waypoints_rad[:-1], path.waypoints_rad[1:])
    ):
        start = np.asarray(start_raw, dtype=np.float64)
        end = np.asarray(end_raw, dtype=np.float64)
        delta = end - start
        duration = max(float(np.max(np.abs(delta))) / maximum_velocity_rad_s, sample_period_s)
        sample_count = max(2, int(np.ceil(duration / sample_period_s)) + 1)
        velocity = tuple(float(value) for value in delta / duration)
        for sample_index, phase in enumerate(np.linspace(0.0, 1.0, sample_count)):
            if segment_index and sample_index == 0:
                continue
            positions = tuple(float(value) for value in start + phase * delta)
            segments.append(
                JointTrajectoryPoint(
                    time_from_start_s=elapsed + float(phase) * duration,
                    positions_rad=positions,
                    velocities_rad_s=velocity,
                    accelerations_rad_s2=zeros,
                )
            )
        elapsed += duration
    return JointTrajectory(path.joint_names, tuple(segments))

