"""Trajectory interpolation and time parameterization."""

from .linear import linear_time_parameterize
from .types import JointTrajectory, JointTrajectoryPoint

__all__ = ["JointTrajectory", "JointTrajectoryPoint", "linear_time_parameterize"]

