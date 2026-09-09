"""Explicit OpenCV-camera to robot-base transform contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def validate_transform(value: np.ndarray, name: str) -> np.ndarray:
    """Validate and copy an SE(3) homogeneous transform."""

    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation must have determinant +1")
    return transform.copy()


@dataclass(frozen=True)
class RobotCameraCalibration:
    """Calibrated transform from an OpenCV optical frame to robot base."""

    transform_base_camera_cv: np.ndarray
    base_frame: str
    camera_frame: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_base_camera_cv",
            validate_transform(
                self.transform_base_camera_cv, "transform_base_camera_cv"
            ),
        )
        if not self.base_frame or not self.camera_frame or not self.source:
            raise ValueError("calibration frame names and source must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "base_frame": self.base_frame,
            "camera_frame": self.camera_frame,
            "camera_convention": "opencv_x_right_y_down_z_forward",
            "source": self.source,
            "transform_base_camera_cv": self.transform_base_camera_cv.tolist(),
        }


def load_robot_camera_calibration(path: Path | str) -> RobotCameraCalibration:
    """Load an explicit robot-base-from-camera calibration JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("camera_convention") not in (
        None,
        "opencv_x_right_y_down_z_forward",
    ):
        raise ValueError("calibration camera convention must be OpenCV")
    transform = payload.get(
        "transform_base_camera_cv", payload.get("transform_robot_camera")
    )
    if transform is None:
        raise ValueError("calibration must contain transform_base_camera_cv")
    return RobotCameraCalibration(
        transform_base_camera_cv=np.asarray(transform, dtype=np.float64),
        base_frame=str(payload.get("base_frame", "robot_base")),
        camera_frame=str(payload.get("camera_frame", "camera_color_optical_frame")),
        source=str(payload.get("source", "unspecified")),
    )


def transform_camera_object_to_base(
    transform_camera_object: np.ndarray,
    calibration: RobotCameraCalibration,
) -> np.ndarray:
    """Return ``T_base_object = T_base_camera_cv @ T_camera_object``."""

    camera_object = validate_transform(
        transform_camera_object, "transform_camera_object"
    )
    return calibration.transform_base_camera_cv @ camera_object


def mujoco_base_camera_calibration(
    model: Any,
    data: Any,
    camera_name: str = "vision_camera",
    base_body_name: str = "ur5e_base",
) -> RobotCameraCalibration:
    """Derive exact simulated extrinsics with an OpenCV optical convention."""

    camera_id = model.camera(camera_name).id
    base_id = model.body(base_body_name).id
    world_base = np.eye(4)
    world_base[:3, :3] = data.xmat[base_id].reshape(3, 3)
    world_base[:3, 3] = data.xpos[base_id]
    world_camera_mujoco = np.eye(4)
    world_camera_mujoco[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3)
    world_camera_mujoco[:3, 3] = data.cam_xpos[camera_id]
    # MuJoCo camera: x right, y up, view along -z. OpenCV: x right,
    # y down, view along +z.
    camera_mujoco_from_camera_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    world_camera_cv = world_camera_mujoco @ camera_mujoco_from_camera_cv
    return RobotCameraCalibration(
        transform_base_camera_cv=np.linalg.inv(world_base) @ world_camera_cv,
        base_frame=base_body_name,
        camera_frame=f"{camera_name}_opencv",
        source="mujoco_exact_model_extrinsic",
    )


def mujoco_transform_base_world(
    model: Any,
    data: Any,
    base_body_name: str = "ur5e_base",
) -> np.ndarray:
    """Return the transform from MuJoCo world coordinates to robot base."""

    base_id = model.body(base_body_name).id
    world_base = np.eye(4)
    world_base[:3, :3] = data.xmat[base_id].reshape(3, 3)
    world_base[:3, 3] = data.xpos[base_id]
    return np.linalg.inv(world_base)
