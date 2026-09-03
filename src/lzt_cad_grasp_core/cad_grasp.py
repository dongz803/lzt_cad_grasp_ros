"""Analytic parallel-jaw grasp candidates generated in the CAD object frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import GraspCandidate
from .object_config import ObjectConfig


@dataclass(frozen=True)
class MeshGeometry:
    vertices_m: np.ndarray
    vertex_normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError("vertices_m must have shape (N, 3) with N >= 4")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices_m must contain finite values")
        object.__setattr__(self, "vertices_m", vertices.copy())
        if self.vertex_normals is not None:
            normals = np.asarray(self.vertex_normals, dtype=np.float64)
            if normals.shape != vertices.shape or not np.all(np.isfinite(normals)):
                raise ValueError("vertex_normals must match vertices_m")
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.divide(
                normals,
                norms,
                out=np.zeros_like(normals),
                where=norms > 1e-12,
            )
            object.__setattr__(self, "vertex_normals", normals)


def load_ascii_ply_geometry(path: Path | str, units_to_m: float) -> MeshGeometry:
    """Load vertices and optional normals from an ASCII PLY CAD model."""

    mesh_path = Path(path)
    if units_to_m <= 0.0:
        raise ValueError("units_to_m must be positive")
    with mesh_path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("not a PLY file")
        vertex_count = None
        vertex_properties: list[str] = []
        in_vertex_element = False
        while True:
            line = stream.readline()
            if line == "":
                raise ValueError("PLY header has no end_header")
            words = line.strip().split()
            if words[:2] == ["format", "ascii"]:
                pass
            elif words and words[0] == "format":
                raise ValueError("only ASCII PLY is supported without trimesh")
            elif words[:2] == ["element", "vertex"]:
                vertex_count = int(words[2])
                in_vertex_element = True
            elif words and words[0] == "element":
                in_vertex_element = False
            elif in_vertex_element and words and words[0] == "property":
                vertex_properties.append(words[-1])
            elif words and words[0] == "end_header":
                break
        if vertex_count is None or vertex_count < 4:
            raise ValueError("PLY contains too few vertices")
        required = [vertex_properties.index(name) for name in ("x", "y", "z")]
        normal_indices = None
        if all(name in vertex_properties for name in ("nx", "ny", "nz")):
            normal_indices = [
                vertex_properties.index(name) for name in ("nx", "ny", "nz")
            ]
        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        normals = (
            np.empty((vertex_count, 3), dtype=np.float64)
            if normal_indices is not None
            else None
        )
        for index in range(vertex_count):
            values = [float(value) for value in stream.readline().split()]
            vertices[index] = [values[column] for column in required]
            if normals is not None and normal_indices is not None:
                normals[index] = [values[column] for column in normal_indices]
    return MeshGeometry(vertices * units_to_m, normals)


def _principal_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(vertices, axis=0)
    covariance = np.cov(vertices - mean, rowvar=False)
    _, axes = np.linalg.eigh(covariance)
    axes = axes[:, ::-1]
    for column in range(3):
        vector = axes[:, column]
        dominant = int(np.argmax(np.abs(vector)))
        if vector[dominant] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    coordinates = (vertices - mean) @ axes
    lower = np.min(coordinates, axis=0)
    upper = np.max(coordinates, axis=0)
    center_coordinates = 0.5 * (lower + upper)
    center = mean + axes @ center_coordinates
    return center, axes, np.vstack((lower - center_coordinates, upper - center_coordinates))


def _antipodal_score(
    coordinates: np.ndarray,
    normals: np.ndarray | None,
    closing_axis: np.ndarray,
    closing_axis_index: int,
    lower: float,
    upper: float,
    band_fraction: float,
) -> float:
    if normals is None:
        return 0.5
    band = max((upper - lower) * band_fraction, 1e-6)
    plus = coordinates[:, closing_axis_index] >= upper - band
    minus = coordinates[:, closing_axis_index] <= lower + band
    if not np.any(plus) or not np.any(minus):
        return 0.0
    plus_alignment = np.mean(np.clip(normals[plus] @ closing_axis, 0.0, 1.0))
    minus_alignment = np.mean(np.clip(normals[minus] @ -closing_axis, 0.0, 1.0))
    return float(np.sqrt(plus_alignment * minus_alignment))


def generate_parallel_jaw_candidates(
    config: ObjectConfig,
    geometry: MeshGeometry,
) -> list[GraspCandidate]:
    """Generate and rank PCA-frame antipodal candidates in the object frame.

    The grasp frame uses +X as the jaw closing axis and +Z as the direction from
    the pre-grasp pose towards the object. Collision and robot reachability are
    intentionally deferred to the task-level planner.
    """

    center, axes, bounds = _principal_frame(geometry.vertices_m)
    coordinates = (geometry.vertices_m - center) @ axes
    candidates: list[GraspCandidate] = []
    for closing_index in range(3):
        closing = axes[:, closing_index]
        span = float(bounds[1, closing_index] - bounds[0, closing_index])
        opening = span + 2.0 * config.gripper.finger_clearance_m
        if not (
            config.gripper.minimum_opening_m
            <= opening
            <= config.gripper.maximum_opening_m
        ):
            continue
        antipodal = _antipodal_score(
            coordinates=coordinates,
            normals=geometry.vertex_normals,
            closing_axis=closing,
            closing_axis_index=closing_index,
            lower=float(bounds[0, closing_index]),
            upper=float(bounds[1, closing_index]),
            band_fraction=config.grasp_generation.surface_band_fraction,
        )
        if antipodal < config.grasp_generation.minimum_antipodal_score:
            continue
        opening_margin = np.clip(
            (config.gripper.maximum_opening_m - opening)
            / (
                config.gripper.maximum_opening_m
                - config.gripper.minimum_opening_m
            ),
            0.0,
            1.0,
        )
        quality = float(np.clip(0.7 * antipodal + 0.3 * opening_margin, 0.0, 1.0))
        approach_indices = [index for index in range(3) if index != closing_index]
        for approach_index in approach_indices:
            for sign in (-1.0, 1.0):
                approach = sign * axes[:, approach_index]
                binormal = np.cross(approach, closing)
                binormal /= np.linalg.norm(binormal)
                closing_orthogonal = np.cross(binormal, approach)
                rotation = np.column_stack((closing_orthogonal, binormal, approach))
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = center
                candidates.append(
                    GraspCandidate(
                        grasp_id=(
                            f"pca_close_{closing_index}_approach_"
                            f"{approach_index}_{'pos' if sign > 0 else 'neg'}"
                        ),
                        transform_object_grasp=transform,
                        jaw_opening_m=opening,
                        analytic_quality=quality,
                        approach_distance_m=(
                            config.grasp_generation.approach_distance_m
                        ),
                        antipodal_score=antipodal,
                    )
                )
    candidates.sort(key=lambda item: (-item.analytic_quality, item.grasp_id))
    return candidates[: config.grasp_generation.maximum_candidates]
