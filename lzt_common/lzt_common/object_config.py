"""Validation and loading for CAD-backed industrial object configurations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .contracts import CADObject, ContinuousSymmetry


@dataclass(frozen=True)
class GripperSpec:
    minimum_opening_m: float
    maximum_opening_m: float
    finger_clearance_m: float = 0.002
    finger_thickness_m: float = 0.008
    finger_width_m: float = 0.022
    finger_depth_m: float = 0.0375

    def __post_init__(self) -> None:
        if self.minimum_opening_m < 0.0:
            raise ValueError("minimum_opening_m must be non-negative")
        if self.maximum_opening_m <= self.minimum_opening_m:
            raise ValueError("maximum_opening_m must exceed minimum_opening_m")
        if self.finger_clearance_m < 0.0:
            raise ValueError("finger_clearance_m must be non-negative")
        if self.finger_thickness_m <= 0.0:
            raise ValueError("finger_thickness_m must be positive")
        if self.finger_width_m <= 0.0:
            raise ValueError("finger_width_m must be positive")
        if self.finger_depth_m <= 0.0:
            raise ValueError("finger_depth_m must be positive")


@dataclass(frozen=True)
class GraspGenerationSpec:
    approach_distance_m: float = 0.08
    surface_band_fraction: float = 0.08
    minimum_antipodal_score: float = 0.3
    minimum_edge_distance_m: float = 0.001
    maximum_center_of_mass_offset_m: float = 0.02
    lateral_sample_fraction: float = 0.2
    maximum_candidates: int = 12

    def __post_init__(self) -> None:
        if self.approach_distance_m <= 0.0:
            raise ValueError("approach_distance_m must be positive")
        if not 0.0 < self.surface_band_fraction <= 0.5:
            raise ValueError("surface_band_fraction must lie in (0, 0.5]")
        if not 0.0 <= self.minimum_antipodal_score <= 1.0:
            raise ValueError("minimum_antipodal_score must lie in [0, 1]")
        if self.minimum_edge_distance_m < 0.0:
            raise ValueError("minimum_edge_distance_m must be non-negative")
        if self.maximum_center_of_mass_offset_m <= 0.0:
            raise ValueError("maximum_center_of_mass_offset_m must be positive")
        if not 0.0 <= self.lateral_sample_fraction <= 0.5:
            raise ValueError("lateral_sample_fraction must lie in [0, 0.5]")
        if self.maximum_candidates <= 0:
            raise ValueError("maximum_candidates must be positive")


@dataclass(frozen=True)
class ObjectConfig:
    schema_version: int
    display_name: str
    cad_object: CADObject
    object_frame: str
    gripper: GripperSpec
    grasp_generation: GraspGenerationSpec
    placement_position_tolerance_m: float
    placement_angle_tolerance_deg: float


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _mesh_path(raw_path: Any, config_path: Path, require_mesh: bool) -> Path:
    if raw_path is None:
        if require_mesh:
            raise ValueError("cad.mesh is required for CAD grasp generation")
        return Path()
    expanded = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    if not expanded.is_absolute():
        expanded = config_path.parent / expanded
    expanded = expanded.resolve()
    if require_mesh and not expanded.is_file():
        raise FileNotFoundError(expanded)
    if expanded.suffix.lower() not in {".ply", ".obj", ".stl"}:
        raise ValueError("cad.mesh must be a PLY, OBJ or STL file")
    return expanded


def _continuous_symmetries(raw: dict[str, Any]) -> tuple[ContinuousSymmetry, ...]:
    symmetry_type = str(raw.get("type", "none"))
    if symmetry_type in {"none", "discrete"}:
        return ()
    if symmetry_type != "continuous_axis":
        raise ValueError(f"unsupported symmetry.type: {symmetry_type}")
    return (
        ContinuousSymmetry(
            axis_xyz=np.asarray(raw.get("axis_xyz"), dtype=np.float64),
            offset_m=np.asarray(raw.get("offset_m", [0.0, 0.0, 0.0])),
        ),
    )


def _discrete_symmetries(raw: dict[str, Any]) -> tuple[np.ndarray, ...]:
    transforms = raw.get("transforms", [])
    if not isinstance(transforms, list):
        raise ValueError("symmetry.transforms must be a list")
    result = []
    for transform in transforms:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("each symmetry transform must have shape (4, 4)")
        result.append(matrix)
    return tuple(result)


def load_object_config(path: Path | str, require_mesh: bool = True) -> ObjectConfig:
    """Load one object YAML and reject incomplete or unsafe configurations."""

    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    if root.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    object_id = str(root.get("object_id", "")).strip()
    display_name = str(root.get("display_name", "")).strip()
    if not object_id or not display_name:
        raise ValueError("object_id and display_name are required")

    cad = _mapping(root.get("cad"), "cad")
    units_to_m = float(cad.get("units_to_m", 0.0))
    mesh_path = _mesh_path(cad.get("mesh"), config_path, require_mesh)
    object_frame = str(cad.get("object_frame", "")).strip()
    if not object_frame:
        raise ValueError("cad.object_frame is required")

    symmetry = _mapping(root.get("symmetry", {}), "symmetry")
    cad_object = CADObject(
        object_id=object_id,
        mesh_path=mesh_path,
        units_to_m=units_to_m,
        symmetry_transforms=_discrete_symmetries(symmetry),
        continuous_symmetries=_continuous_symmetries(symmetry),
    )

    gripper_raw = _mapping(root.get("gripper"), "gripper")
    if gripper_raw.get("type") != "parallel_jaw":
        raise ValueError("only gripper.type=parallel_jaw is currently supported")
    gripper = GripperSpec(
        minimum_opening_m=float(gripper_raw.get("minimum_opening_m")),
        maximum_opening_m=float(gripper_raw.get("maximum_opening_m")),
        finger_clearance_m=float(gripper_raw.get("finger_clearance_m", 0.002)),
        finger_thickness_m=float(gripper_raw.get("finger_thickness_m", 0.008)),
        finger_width_m=float(gripper_raw.get("finger_width_m", 0.022)),
        finger_depth_m=float(gripper_raw.get("finger_depth_m", 0.0375)),
    )

    generation_raw = _mapping(root.get("grasp_generation", {}), "grasp_generation")
    generation = GraspGenerationSpec(
        approach_distance_m=float(generation_raw.get("approach_distance_m", 0.08)),
        surface_band_fraction=float(generation_raw.get("surface_band_fraction", 0.08)),
        minimum_antipodal_score=float(
            generation_raw.get("minimum_antipodal_score", 0.3)
        ),
        minimum_edge_distance_m=float(
            generation_raw.get("minimum_edge_distance_m", 0.001)
        ),
        maximum_center_of_mass_offset_m=float(
            generation_raw.get("maximum_center_of_mass_offset_m", 0.02)
        ),
        lateral_sample_fraction=float(
            generation_raw.get("lateral_sample_fraction", 0.2)
        ),
        maximum_candidates=int(generation_raw.get("maximum_candidates", 12)),
    )

    task = _mapping(root.get("task"), "task")
    position_tolerance = float(task.get("placement_position_tolerance_m", 0.0))
    angle_tolerance = float(task.get("placement_angle_tolerance_deg", 0.0))
    if position_tolerance <= 0.0 or angle_tolerance <= 0.0:
        raise ValueError("task placement tolerances must be positive")

    return ObjectConfig(
        schema_version=1,
        display_name=display_name,
        cad_object=cad_object,
        object_frame=object_frame,
        gripper=gripper,
        grasp_generation=generation,
        placement_position_tolerance_m=position_tolerance,
        placement_angle_tolerance_deg=angle_tolerance,
    )
