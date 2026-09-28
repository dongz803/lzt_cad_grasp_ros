"""Geometry-auditable parallel-jaw candidates in the CAD object frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lzt_common.contracts import GraspCandidate
from lzt_common.object_config import ObjectConfig


@dataclass(frozen=True)
class MeshGeometry:
    """Triangle-mesh geometry represented in metres."""

    vertices_m: np.ndarray
    vertex_normals: np.ndarray | None = None
    faces: np.ndarray | None = None

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError("vertices_m must have shape (N, 3) with N >= 4")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices_m must contain finite values")
        object.__setattr__(self, "vertices_m", vertices.copy())
        if self.vertex_normals is not None:
            normals = np.asarray(self.vertex_normals, dtype=np.float64)
            if normals.shape != vertices.shape or not np.all(
                np.isfinite(normals)
            ):
                raise ValueError("vertex_normals must match vertices_m")
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.divide(
                normals,
                norms,
                out=np.zeros_like(normals),
                where=norms > 1e-12,
            )
            object.__setattr__(self, "vertex_normals", normals)
        if self.faces is not None:
            faces = np.asarray(self.faces, dtype=np.int64)
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError("faces must have shape (M, 3)")
            if (
                len(faces) == 0
                or np.min(faces) < 0
                or np.max(faces) >= len(vertices)
            ):
                raise ValueError("faces contain invalid vertex indices")
            object.__setattr__(self, "faces", faces.copy())


@dataclass(frozen=True)
class GraspGenerationDiagnostics:
    """Counts and geometry facts retained from one generation run."""

    evaluated_frames: int
    accepted_before_limit: int
    returned_candidates: int
    rejection_counts: dict[str, int]
    center_of_mass_object_m: np.ndarray
    mesh_triangle_count: int

    def __post_init__(self) -> None:
        center = np.asarray(self.center_of_mass_object_m, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center_of_mass_object_m must have shape (3,)")
        object.__setattr__(self, "center_of_mass_object_m", center.copy())
        object.__setattr__(self, "rejection_counts", dict(self.rejection_counts))


@dataclass(frozen=True)
class _ContactPatch:
    point_object_m: np.ndarray
    normal_object: np.ndarray
    normal_alignment: float
    edge_distance_m: float


def load_ascii_ply_geometry(
    path: Path | str, units_to_m: float
) -> MeshGeometry:
    """Load vertices, optional normals, and faces from an ASCII PLY."""
    mesh_path = Path(path)
    if units_to_m <= 0.0:
        raise ValueError("units_to_m must be positive")
    with mesh_path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("not a PLY file")
        vertex_count = None
        face_count = 0
        vertex_properties: list[str] = []
        current_element = ""
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
                current_element = "vertex"
            elif words[:2] == ["element", "face"]:
                face_count = int(words[2])
                current_element = "face"
            elif words and words[0] == "element":
                current_element = words[1]
            elif current_element == "vertex" and words and words[0] == "property":
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
        triangles: list[tuple[int, int, int]] = []
        for _ in range(face_count):
            words = stream.readline().split()
            if not words:
                raise ValueError("PLY face list ended early")
            count = int(words[0])
            indices = [int(value) for value in words[1:count + 1]]
            if len(indices) != count or count < 3:
                raise ValueError("PLY contains an invalid face")
            triangles.extend(
                (indices[0], indices[offset], indices[offset + 1])
                for offset in range(1, count - 1)
            )
    faces = np.asarray(triangles, dtype=np.int64) if triangles else None
    return MeshGeometry(vertices * units_to_m, normals, faces)


def _principal_frame(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return center, axes, np.vstack(
        (lower - center_coordinates, upper - center_coordinates)
    )


def _mesh_center_of_mass(geometry: MeshGeometry) -> np.ndarray:
    """Estimate the uniform-density volume centroid, with safe mesh fallbacks."""
    if geometry.faces is None:
        return np.mean(geometry.vertices_m, axis=0)
    triangles = geometry.vertices_m[geometry.faces]
    signed_six_volume = np.einsum(
        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    )
    absolute_volume = float(np.sum(np.abs(signed_six_volume)))
    signed_volume = float(np.sum(signed_six_volume))
    if absolute_volume > 1e-18 and abs(signed_volume) / absolute_volume > 0.5:
        centroid = np.sum(
            signed_six_volume[:, None] * np.sum(triangles, axis=1), axis=0
        ) / (4.0 * signed_volume)
        if np.all(np.isfinite(centroid)):
            return centroid
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if float(np.sum(areas)) > 1e-18:
        return np.average(np.mean(triangles, axis=1), axis=0, weights=areas)
    return np.mean(geometry.vertices_m, axis=0)


def _cross_2d(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    a = first - origin
    b = second - origin
    return float(a[0] * b[1] - a[1] * b[0])


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) <= 2:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]
    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def _distance_to_hull_edges(point: np.ndarray, hull: np.ndarray) -> float:
    if len(hull) < 3:
        return 0.0
    starts = hull
    ends = np.roll(hull, -1, axis=0)
    segments = ends - starts
    squared_lengths = np.sum(segments * segments, axis=1)
    fractions = np.divide(
        np.sum((point - starts) * segments, axis=1),
        squared_lengths,
        out=np.zeros_like(squared_lengths),
        where=squared_lengths > 1e-18,
    )
    projections = starts + np.clip(fractions, 0.0, 1.0)[:, None] * segments
    return float(np.min(np.linalg.norm(projections - point, axis=1)))


def _contact_patch(
    geometry: MeshGeometry,
    coordinates: np.ndarray,
    normal_coordinates: np.ndarray | None,
    target_lateral_coordinates: np.ndarray,
    axes: np.ndarray,
    bounds: np.ndarray,
    closing_index: int,
    approach_index: int,
    binormal_index: int,
    side: float,
    config: ObjectConfig,
) -> _ContactPatch | None:
    lower = float(bounds[0, closing_index])
    upper = float(bounds[1, closing_index])
    band = max(
        (upper - lower) * config.grasp_generation.surface_band_fraction,
        1e-6,
    )
    if side > 0.0:
        extreme = coordinates[:, closing_index] >= upper - band
    else:
        extreme = coordinates[:, closing_index] <= lower + band
    if normal_coordinates is None:
        alignments = np.full(len(coordinates), 0.5)
    else:
        alignments = side * normal_coordinates[:, closing_index]
        aligned_extreme = extreme & (alignments > 0.05)
        if np.any(aligned_extreme):
            extreme = aligned_extreme
    indices = np.flatnonzero(extreme)
    if len(indices) == 0:
        return None

    lateral_indices = [binormal_index, approach_index]
    projected = coordinates[indices][:, lateral_indices]
    target = np.asarray(target_lateral_coordinates, dtype=np.float64)
    half_extents = np.array(
        [config.gripper.finger_width_m, config.gripper.finger_depth_m]
    ) * 0.5
    in_footprint = np.all(np.abs(projected - target) <= half_extents, axis=1)
    selected = indices[in_footprint]
    if len(selected) < 3:
        normalized_distance = np.linalg.norm(
            (projected - target) / np.maximum(half_extents, 1e-6), axis=1
        )
        count = min(len(indices), max(3, int(np.sqrt(len(indices)))))
        nearest = np.argpartition(normalized_distance, count - 1)[:count]
        selected = indices[nearest]

    weights = np.clip(alignments[selected], 0.05, 1.0)
    point = np.average(geometry.vertices_m[selected], axis=0, weights=weights)
    if geometry.vertex_normals is None:
        normal = side * axes[:, closing_index]
        normal_alignment = 0.5
    else:
        normal = np.average(
            geometry.vertex_normals[selected], axis=0, weights=weights
        )
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            return None
        normal /= length
        normal_alignment = float(
            np.clip(side * normal @ axes[:, closing_index], 0.0, 1.0)
        )
    point_projected = point @ axes[:, lateral_indices]
    hull_object_projection = (
        geometry.vertices_m[indices] @ axes[:, lateral_indices]
    )
    edge_distance = _distance_to_hull_edges(
        point_projected, _convex_hull_2d(hull_object_projection)
    )
    return _ContactPatch(
        point_object_m=point,
        normal_object=normal,
        normal_alignment=normal_alignment,
        edge_distance_m=edge_distance,
    )


def _triangles_intersect_aabb(
    triangles: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    if len(triangles) == 0:
        return False
    overlaps_bounds = np.all(
        (np.max(triangles, axis=1) >= lower)
        & (np.min(triangles, axis=1) <= upper),
        axis=1,
    )
    triangles = triangles[overlaps_bounds]
    if len(triangles) == 0:
        return False
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    vertices = triangles - center
    separated = np.any(
        (np.max(vertices, axis=1) < -half)
        | (np.min(vertices, axis=1) > half),
        axis=1,
    )
    edges = (
        vertices[:, 1] - vertices[:, 0],
        vertices[:, 2] - vertices[:, 1],
        vertices[:, 0] - vertices[:, 2],
    )
    normals = np.cross(edges[0], edges[1])
    normal_projection = np.einsum("ij,ij->i", vertices[:, 0], normals)
    normal_radius = np.sum(np.abs(normals) * half, axis=1)
    separated |= np.abs(normal_projection) > normal_radius + 1e-12
    box_axes = np.eye(3)
    for edge in edges:
        for box_axis in box_axes:
            test_axes = np.cross(edge, box_axis)
            projections = np.einsum("mij,mj->mi", vertices, test_axes)
            radius = np.sum(np.abs(test_axes) * half, axis=1)
            separated |= (np.min(projections, axis=1) > radius + 1e-12) | (
                np.max(projections, axis=1) < -radius - 1e-12
            )
    return bool(np.any(~separated))


def _mesh_intersects_aabb(
    vertices: np.ndarray,
    faces: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    inside = np.all((vertices >= lower) & (vertices <= upper), axis=1)
    if np.any(inside):
        return True
    if faces is None:
        return False
    return _triangles_intersect_aabb(vertices[faces], lower, upper)


def _finger_collision_free(
    geometry: MeshGeometry,
    transform_object_grasp: np.ndarray,
    jaw_opening_m: float,
    config: ObjectConfig,
) -> bool:
    rotation = transform_object_grasp[:3, :3]
    center = transform_object_grasp[:3, 3]
    vertices = (geometry.vertices_m - center) @ rotation
    half_width = 0.5 * config.gripper.finger_width_m
    half_depth = 0.5 * config.gripper.finger_depth_m
    inner = 0.5 * jaw_opening_m
    outer = inner + config.gripper.finger_thickness_m
    boxes = (
        (
            np.array([inner, -half_width, -half_depth]),
            np.array([outer, half_width, half_depth]),
        ),
        (
            np.array([-outer, -half_width, -half_depth]),
            np.array([-inner, half_width, half_depth]),
        ),
    )
    return not any(
        _mesh_intersects_aabb(vertices, geometry.faces, lower, upper)
        for lower, upper in boxes
    )


def generate_parallel_jaw_candidates_with_diagnostics(
    config: ObjectConfig,
    geometry: MeshGeometry,
) -> tuple[list[GraspCandidate], GraspGenerationDiagnostics]:
    """Generate candidates and expose why geometrically unsafe frames fail."""
    center, axes, bounds = _principal_frame(geometry.vertices_m)
    coordinates = (geometry.vertices_m - center) @ axes
    normal_coordinates = None
    if geometry.vertex_normals is not None:
        normal_coordinates = geometry.vertex_normals @ axes
    center_of_mass = _mesh_center_of_mass(geometry)
    center_of_mass_coordinates = (center_of_mass - center) @ axes
    candidates: list[GraspCandidate] = []
    rejection_counts = {
        "contact_patch": 0,
        "jaw_opening": 0,
        "antipodal": 0,
        "center_of_mass": 0,
        "edge_distance": 0,
        "finger_collision": 0,
        "duplicate_candidate": 0,
    }
    evaluated_frames = 0

    for closing_index in range(3):
        closing = axes[:, closing_index]
        approach_indices = [
            index for index in range(3) if index != closing_index
        ]
        for approach_index in approach_indices:
            binormal_index = next(
                index
                for index in range(3)
                if index not in (closing_index, approach_index)
            )
            lateral_indices = [binormal_index, approach_index]
            lateral_span = (
                bounds[1, lateral_indices] - bounds[0, lateral_indices]
            )
            maximum_sample_offset = (
                0.5 * config.grasp_generation.maximum_center_of_mass_offset_m
            )
            offsets = np.minimum(
                config.grasp_generation.lateral_sample_fraction * lateral_span,
                maximum_sample_offset,
            )
            sample_offsets = [
                ("center", np.zeros(2)),
                ("bin_neg", np.array([-offsets[0], 0.0])),
                ("bin_pos", np.array([offsets[0], 0.0])),
                ("approach_neg", np.array([0.0, -offsets[1]])),
                ("approach_pos", np.array([0.0, offsets[1]])),
            ]
            for sample_name, sample_offset in sample_offsets:
                target = (
                    center_of_mass_coordinates[lateral_indices] + sample_offset
                )
                plus = _contact_patch(
                    geometry,
                    coordinates,
                    normal_coordinates,
                    target,
                    axes,
                    bounds,
                    closing_index,
                    approach_index,
                    binormal_index,
                    1.0,
                    config,
                )
                minus = _contact_patch(
                    geometry,
                    coordinates,
                    normal_coordinates,
                    target,
                    axes,
                    bounds,
                    closing_index,
                    approach_index,
                    binormal_index,
                    -1.0,
                    config,
                )
                for sign in (-1.0, 1.0):
                    evaluated_frames += 1
                    if plus is None or minus is None:
                        rejection_counts["contact_patch"] += 1
                        continue
                    delta = plus.point_object_m - minus.point_object_m
                    separation = float(abs(delta @ closing))
                    opening = (
                        separation + 2.0 * config.gripper.finger_clearance_m
                    )
                    opening_margin = min(
                        opening - config.gripper.minimum_opening_m,
                        config.gripper.maximum_opening_m - opening,
                    )
                    if opening_margin < 0.0:
                        rejection_counts["jaw_opening"] += 1
                        continue

                    delta_length = float(np.linalg.norm(delta))
                    line_alignment = (
                        abs(float(delta @ closing)) / delta_length
                        if delta_length > 1e-9
                        else 0.0
                    )
                    normal_opposition = float(
                        np.clip(
                            0.5
                            * (1.0 - plus.normal_object @ minus.normal_object),
                            0.0,
                            1.0,
                        )
                    )
                    antipodal = float(
                        np.sqrt(plus.normal_alignment * minus.normal_alignment)
                        * np.sqrt(normal_opposition)
                        * line_alignment
                    )
                    if (
                        antipodal
                        < config.grasp_generation.minimum_antipodal_score
                    ):
                        rejection_counts["antipodal"] += 1
                        continue

                    midpoint = 0.5 * (
                        plus.point_object_m + minus.point_object_m
                    )
                    center_of_mass_offset = float(
                        np.linalg.norm(midpoint - center_of_mass)
                    )
                    if (
                        center_of_mass_offset
                        > config.grasp_generation.maximum_center_of_mass_offset_m
                    ):
                        rejection_counts["center_of_mass"] += 1
                        continue
                    edge_distance = min(
                        plus.edge_distance_m, minus.edge_distance_m
                    )
                    if (
                        edge_distance
                        < config.grasp_generation.minimum_edge_distance_m
                    ):
                        rejection_counts["edge_distance"] += 1
                        continue

                    approach = sign * axes[:, approach_index]
                    binormal = np.cross(approach, closing)
                    binormal /= np.linalg.norm(binormal)
                    closing_orthogonal = np.cross(binormal, approach)
                    rotation = np.column_stack(
                        (closing_orthogonal, binormal, approach)
                    )
                    transform = np.eye(4)
                    transform[:3, :3] = rotation
                    transform[:3, 3] = midpoint
                    if not _finger_collision_free(
                        geometry, transform, opening, config
                    ):
                        rejection_counts["finger_collision"] += 1
                        continue

                    opening_score = float(
                        np.clip(
                            opening_margin
                            / max(
                                0.5
                                * (
                                    config.gripper.maximum_opening_m
                                    - config.gripper.minimum_opening_m
                                ),
                                1e-9,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    center_score = float(
                        np.clip(
                            1.0
                            - center_of_mass_offset
                            / (
                                config.grasp_generation.
                                maximum_center_of_mass_offset_m
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    edge_score = float(
                        np.clip(
                            edge_distance
                            / max(
                                0.5
                                * min(
                                    config.gripper.finger_width_m,
                                    config.gripper.finger_depth_m,
                                ),
                                1e-9,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    quality = float(
                        np.clip(
                            0.4 * antipodal
                            + 0.15 * opening_score
                            + 0.25 * center_score
                            + 0.2 * edge_score,
                            0.0,
                            1.0,
                        )
                    )
                    sample_label = (
                        "" if sample_name == "center" else f"_{sample_name}"
                    )
                    candidates.append(
                        GraspCandidate(
                            grasp_id=(
                                f"pca_close_{closing_index}_approach_"
                                f"{approach_index}{sample_label}_"
                                f"{'pos' if sign > 0 else 'neg'}"
                            ),
                            transform_object_grasp=transform,
                            jaw_opening_m=opening,
                            analytic_quality=quality,
                            approach_distance_m=(
                                config.grasp_generation.approach_distance_m
                            ),
                            antipodal_score=antipodal,
                            contact_points_object_m=np.vstack(
                                (minus.point_object_m, plus.point_object_m)
                            ),
                            contact_normals_object=np.vstack(
                                (minus.normal_object, plus.normal_object)
                            ),
                            center_of_mass_offset_m=center_of_mass_offset,
                            minimum_edge_distance_m=edge_distance,
                            opening_margin_m=opening_margin,
                            finger_collision_free=True,
                        )
                    )

    unique_candidates: dict[tuple[float, ...], GraspCandidate] = {}
    for candidate in candidates:
        key = tuple(
            np.round(candidate.transform_object_grasp.reshape(-1), decimals=8)
        )
        previous = unique_candidates.get(key)
        if (
            previous is None
            or candidate.analytic_quality > previous.analytic_quality
        ):
            unique_candidates[key] = candidate
    rejection_counts["duplicate_candidate"] = len(candidates) - len(
        unique_candidates
    )
    candidates = list(unique_candidates.values())
    candidates.sort(key=lambda item: (-item.analytic_quality, item.grasp_id))
    accepted_before_limit = len(candidates)
    candidates = candidates[: config.grasp_generation.maximum_candidates]
    diagnostics = GraspGenerationDiagnostics(
        evaluated_frames=evaluated_frames,
        accepted_before_limit=accepted_before_limit,
        returned_candidates=len(candidates),
        rejection_counts=rejection_counts,
        center_of_mass_object_m=center_of_mass,
        mesh_triangle_count=(
            0 if geometry.faces is None else len(geometry.faces)
        ),
    )
    return candidates, diagnostics


def generate_parallel_jaw_candidates(
    config: ObjectConfig,
    geometry: MeshGeometry,
) -> list[GraspCandidate]:
    """Generate ranked PCA-frame candidates after all CAD geometry checks."""
    candidates, _ = generate_parallel_jaw_candidates_with_diagnostics(
        config, geometry
    )
    return candidates
