"""Build a same-source MuJoCo workcell around an onboarded industrial CAD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from lzt_common.object_config import ObjectConfig
from lzt_common.robot_frames import validate_transform
from lzt_grasp_planning.cad_grasp import MeshGeometry, load_ascii_ply_geometry
from .simulation import build_workcell_spec


TARGET_BODY_NAME = "target"
TARGET_JOINT_NAME = "target_freejoint"
TARGET_GEOM_NAME = "target_geom"
FIXED_CAMERA_NAME = "vision_camera"


@dataclass(frozen=True)
class IndustrialWorkcellInfo:
    """Traceable geometry and initial placement used to compile a workcell."""

    object_id: str
    source_mesh_path: Path
    mujoco_mesh_path: Path
    local_bounds_min_m: np.ndarray
    local_bounds_max_m: np.ndarray
    platform_surface_z_m: float
    initial_transform_world_object: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_bounds_min_m",
            np.asarray(self.local_bounds_min_m, dtype=np.float64).copy(),
        )
        object.__setattr__(
            self,
            "local_bounds_max_m",
            np.asarray(self.local_bounds_max_m, dtype=np.float64).copy(),
        )
        object.__setattr__(
            self,
            "initial_transform_world_object",
            validate_transform(
                self.initial_transform_world_object,
                "initial_transform_world_object",
            ),
        )


def transform_from_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return a homogeneous transform from a 3-vector and 3x3 rotation."""

    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return validate_transform(result, "transform")


def body_transform(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Read one MuJoCo body transform in world coordinates."""

    return transform_from_pose(
        data.xpos[body_id], data.xmat[body_id].reshape(3, 3)
    )


def set_freejoint_transform(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    transform_world_body: np.ndarray,
) -> None:
    """Reset a free body pose without relying on an older keyframe.

    The base workcell keyframe is compiled before the industrial target is
    injected. MuJoCo pads that keyframe with zeros for the new free joint, so
    applying the robot-home keyframe alone would move the target to the world
    origin. Explicitly restoring the configured pose keeps robot and object
    resets independent.
    """

    transform = validate_transform(
        transform_world_body, "transform_world_body"
    )
    joint_id = model.joint(joint_name).id
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{joint_name} is not a free joint")
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, transform[:3, :3].reshape(9))
    data.qpos[qpos_address : qpos_address + 3] = transform[:3, 3]
    data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
    data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def export_geometry_as_obj(
    geometry: MeshGeometry,
    output_path: Path | str,
) -> Path:
    """Export metre-scaled triangle geometry to a MuJoCo-readable OBJ file."""

    if geometry.faces is None or len(geometry.faces) == 0:
        raise ValueError("triangle faces are required for MuJoCo mesh export")
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="ascii") as stream:
        stream.write("# generated from the configured CAD mesh; units: metres\n")
        for vertex in geometry.vertices_m:
            stream.write(
                f"v {vertex[0]:.10g} {vertex[1]:.10g} {vertex[2]:.10g}\n"
            )
        for face in geometry.faces:
            # OBJ indices are one-based.
            stream.write(
                f"f {int(face[0]) + 1} {int(face[1]) + 1} "
                f"{int(face[2]) + 1}\n"
            )
    temporary.replace(target)
    return target


def _platform_surface_z(spec: mujoco.MjSpec) -> float:
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_id = model.geom("platform_geom").id
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    normal = rotation[:, 2]
    surface = data.geom_xpos[geom_id] + normal * model.geom_size[geom_id, 2]
    if not np.allclose(normal, [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("industrial scene currently requires a horizontal platform")
    return float(surface[2])


def build_industrial_cad_workcell(
    config: ObjectConfig,
    mujoco_mesh_path: Path | str,
    *,
    scene_name: str = "industrial_cad_single.xml",
    initial_xy_m: tuple[float, float] = (-0.2, -0.12),
    initial_yaw_rad: float = 0.0,
    placement_clearance_m: float = 0.001,
    target_mass_kg: float = 0.2,
) -> tuple[mujoco.MjModel, IndustrialWorkcellInfo]:
    """Inject a configured CAD as the dynamic target in the copied workcell."""

    if config.cad_object.mesh_path.suffix.lower() != ".ply":
        raise ValueError("the dependency-free MuJoCo path currently accepts ASCII PLY")
    if placement_clearance_m < 0.0 or target_mass_kg <= 0.0:
        raise ValueError("clearance must be non-negative and mass must be positive")

    geometry = load_ascii_ply_geometry(
        config.cad_object.mesh_path,
        config.cad_object.units_to_m,
    )
    converted_mesh = export_geometry_as_obj(geometry, mujoco_mesh_path)
    lower = np.min(geometry.vertices_m, axis=0)
    upper = np.max(geometry.vertices_m, axis=0)

    spec = build_workcell_spec(scene_name=scene_name)
    surface_z = _platform_surface_z(spec)
    cosine = float(np.cos(initial_yaw_rad))
    sine = float(np.sin(initial_yaw_rad))
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    center = np.array(
        [
            float(initial_xy_m[0]),
            float(initial_xy_m[1]),
            surface_z - float(lower[2]) + placement_clearance_m,
        ]
    )
    initial_transform = transform_from_pose(center, rotation)

    spec.add_material(
        name="industrial_target_material",
        rgba=[0.72, 0.74, 0.78, 1.0],
        specular=0.55,
        shininess=0.45,
        reflectance=0.10,
    )
    spec.add_mesh(
        name="industrial_target_mesh",
        file=str(converted_mesh),
        maxhullvert=128,
        smoothnormal=1,
    )
    target = spec.worldbody.add_body(
        name=TARGET_BODY_NAME,
        pos=center.tolist(),
        quat=[
            float(np.cos(0.5 * initial_yaw_rad)),
            0.0,
            0.0,
            float(np.sin(0.5 * initial_yaw_rad)),
        ],
    )
    target.add_freejoint(name=TARGET_JOINT_NAME)
    target.add_geom(
        name=TARGET_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="industrial_target_mesh",
        material="industrial_target_material",
        mass=target_mass_kg,
        friction=[1.2, 0.03, 0.003],
        condim=4,
    )
    model = spec.compile()
    return model, IndustrialWorkcellInfo(
        object_id=config.cad_object.object_id,
        source_mesh_path=config.cad_object.mesh_path,
        mujoco_mesh_path=converted_mesh,
        local_bounds_min_m=lower,
        local_bounds_max_m=upper,
        platform_surface_z_m=surface_z,
        initial_transform_world_object=initial_transform,
    )


def target_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    """Return all geoms attached directly to the dynamic target body."""

    body_id = model.body(TARGET_BODY_NAME).id
    return np.asarray(
        [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
        ],
        dtype=np.int32,
    )


def camera_intrinsics(
    model: mujoco.MjModel,
    camera_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    """Derive a square-pixel pinhole matrix from MuJoCo's vertical FOV."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    camera_id = model.camera(camera_name).id
    fy = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[camera_id]) * 0.5)
    return np.array(
        [[fy, 0.0, 0.5 * width], [0.0, fy, 0.5 * height], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def camera_object_transform(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_name: str = FIXED_CAMERA_NAME,
    body_name: str = TARGET_BODY_NAME,
) -> np.ndarray:
    """Return exact ``T_camera_object`` using OpenCV optical axes."""

    camera_id = model.camera(camera_name).id
    world_camera_mujoco = transform_from_pose(
        data.cam_xpos[camera_id], data.cam_xmat[camera_id].reshape(3, 3)
    )
    world_camera_cv = world_camera_mujoco @ np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(world_camera_cv) @ body_transform(
        data, model.body(body_name).id
    )
