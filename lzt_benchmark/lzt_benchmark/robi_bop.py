"""ROBI RealSense scene conversion into the standard BOP directory contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


ROBI_OBJECT_IDS = {
    "Zigzag": 1,
    "Chrome_screw": 2,
    "Gear": 3,
    "Eye_bolt": 4,
    "Tube_fitting": 5,
    "DIN_connector": 6,
    "DSub_connector": 7,
}


@dataclass(frozen=True)
class ROBICamera:
    width: int
    height: int
    intrinsics: np.ndarray
    raw_depth_unit_mm: float = 0.03125

    def __post_init__(self) -> None:
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3)")
        if self.raw_depth_unit_mm <= 0.0:
            raise ValueError("raw_depth_unit_mm must be positive")
        object.__setattr__(self, "intrinsics", intrinsics.copy())


REALSENSE_RGB = ROBICamera(
    width=1280,
    height=720,
    intrinsics=np.array(
        [
            [923.844666, 0.0, 644.324097],
            [0.0, 923.844666, 361.015259],
            [0.0, 0.0, 1.0],
        ]
    ),
)
REALSENSE_DEPTH = ROBICamera(
    width=1280,
    height=720,
    intrinsics=np.array(
        [
            [905.905579, 0.0, 631.507141],
            [0.0, 905.905579, 362.560059],
            [0.0, 0.0, 1.0],
        ]
    ),
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_robi_pose(path: Path | str) -> np.ndarray:
    """Load the official ``cam_pose`` matrix (camera pose in world frame)."""

    pose_path = Path(path)
    # Official ROBI camera metadata is JSON Lines: the first record contains
    # ``cam_pose`` and later records contain per-object visibility values.
    # Test fixtures and future exports may also be regular one-record JSON.
    pose_record = None
    for line in pose_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "cam_pose" in record:
            pose_record = record
            break
    if pose_record is None:
        raise ValueError(f"cam_pose record missing: {pose_path}")
    matrix = np.asarray(pose_record["cam_pose"], dtype=np.float64)
    matrix = matrix.reshape(4, 4)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"invalid homogeneous pose: {path}")
    return matrix


def reproject_depth_to_rgb(
    raw_depth: np.ndarray,
    transform_world_depth: np.ndarray,
    transform_world_rgb: np.ndarray,
    depth_camera: ROBICamera = REALSENSE_DEPTH,
    rgb_camera: ROBICamera = REALSENSE_RGB,
) -> np.ndarray:
    """Register official ROBI depth pixels to the RealSense RGB camera.

    Output is uint16 millimetres with zero for pixels that receive no sample.
    A z-buffer resolves collisions after reprojection.
    """

    depth_raw = np.asarray(raw_depth)
    if depth_raw.shape != (depth_camera.height, depth_camera.width):
        raise ValueError("raw depth shape does not match the depth camera")
    depth_mm = depth_raw.astype(np.float64) * depth_camera.raw_depth_unit_mm
    rows, columns = np.nonzero(depth_mm > 0.0)
    if len(rows) == 0:
        return np.zeros((rgb_camera.height, rgb_camera.width), dtype=np.uint16)
    z = depth_mm[rows, columns]
    kd = depth_camera.intrinsics
    points_depth = np.vstack(
        (
            (columns - kd[0, 2]) * z / kd[0, 0],
            (rows - kd[1, 2]) * z / kd[1, 1],
            z,
            np.ones_like(z),
        )
    )
    transform_rgb_world = np.linalg.inv(np.asarray(transform_world_rgb))
    points_rgb = (
        transform_rgb_world
        @ np.asarray(transform_world_depth, dtype=np.float64)
        @ points_depth
    )
    valid = points_rgb[2] > 0.0
    points_rgb = points_rgb[:, valid]
    if points_rgb.shape[1] == 0:
        return np.zeros((rgb_camera.height, rgb_camera.width), dtype=np.uint16)
    kr = rgb_camera.intrinsics
    projected_x = np.rint(
        kr[0, 0] * points_rgb[0] / points_rgb[2] + kr[0, 2]
    ).astype(np.int64)
    projected_y = np.rint(
        kr[1, 1] * points_rgb[1] / points_rgb[2] + kr[1, 2]
    ).astype(np.int64)
    inside = (
        (projected_x >= 0)
        & (projected_x < rgb_camera.width)
        & (projected_y >= 0)
        & (projected_y < rgb_camera.height)
    )
    projected_x = projected_x[inside]
    projected_y = projected_y[inside]
    projected_z = points_rgb[2, inside]
    flat = projected_y * rgb_camera.width + projected_x
    z_buffer = np.full(rgb_camera.width * rgb_camera.height, np.inf)
    np.minimum.at(z_buffer, flat, projected_z)
    z_buffer[~np.isfinite(z_buffer)] = 0.0
    return np.rint(np.clip(z_buffer, 0.0, 65535.0)).astype(np.uint16).reshape(
        rgb_camera.height, rgb_camera.width
    )


def _transform_to_bop(transform: np.ndarray, translation_scale: float) -> dict:
    return {
        "cam_R_m2c": transform[:3, :3].reshape(-1).tolist(),
        "cam_t_m2c": (transform[:3, 3] * translation_scale).tolist(),
    }


def convert_robi_realsense_to_bop(
    source_root: Path | str,
    output_root: Path | str,
    object_names: list[str],
    scene_ids: list[int] | None = None,
    maximum_views_per_scene: int | None = None,
    pose_units_to_mm: float = 1.0,
) -> dict:
    """Convert selected real ROBI scenes while preserving traceable ID maps."""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if pose_units_to_mm <= 0.0:
        raise ValueError("pose_units_to_mm must be positive")
    unknown = sorted(set(object_names) - set(ROBI_OBJECT_IDS))
    if unknown:
        raise ValueError(f"unknown ROBI object names: {unknown}")
    test_root = output / "test"
    test_root.mkdir(parents=True, exist_ok=True)
    targets = []
    scene_map = []
    bop_scene_id = 1
    converted_images = 0

    for object_name in object_names:
        object_root = source / object_name
        available_scenes = sorted(
            int(path.name.split("_", 1)[1])
            for path in object_root.glob("Scene_*")
            if path.is_dir() and path.name.split("_", 1)[1].isdigit()
        )
        selected_scenes = available_scenes if scene_ids is None else scene_ids
        for source_scene_id in selected_scenes:
            scene_root = object_root / f"Scene_{source_scene_id}"
            if not scene_root.is_dir():
                continue
            sensor_root = scene_root / "RealSense"
            views = (sensor_root / "list.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            views = [view.strip() for view in views if view.strip()]
            if maximum_views_per_scene is not None:
                views = views[:maximum_views_per_scene]
            world_objects_raw = _load_json(scene_root / "GT_world2obj.json")
            world_objects = [
                np.asarray(world_objects_raw[key], dtype=np.float64).reshape(4, 4)
                for key in sorted(world_objects_raw)
            ]
            destination = test_root / f"{bop_scene_id:06d}"
            rgb_dir = destination / "rgb"
            depth_dir = destination / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            scene_camera = {}
            scene_gt = {}
            scene_gt_info = {}
            image_map = {}

            for bop_image_id, view in enumerate(views):
                rgb_source = sensor_root / "Color" / f"RGB_{view}.bmp"
                rgb_pose_path = sensor_root / "Color" / f"RGB_{view}.json"
                depth_source = sensor_root / "Depth" / f"DEPTH_{view}.png"
                depth_pose_path = sensor_root / "Depth" / f"DEPTH_{view}.json"
                image = cv2.imread(str(rgb_source), cv2.IMREAD_COLOR)
                raw_depth = cv2.imread(str(depth_source), cv2.IMREAD_UNCHANGED)
                if image is None or raw_depth is None:
                    raise FileNotFoundError(f"missing ROBI view data: {view}")
                world_rgb = load_robi_pose(rgb_pose_path)
                world_depth = load_robi_pose(depth_pose_path)
                registered_depth = reproject_depth_to_rgb(
                    raw_depth,
                    world_depth,
                    world_rgb,
                )
                rgb_path = rgb_dir / f"{bop_image_id:06d}.png"
                depth_path = depth_dir / f"{bop_image_id:06d}.png"
                if not cv2.imwrite(str(rgb_path), image):
                    raise OSError(f"failed to write {rgb_path}")
                if not cv2.imwrite(str(depth_path), registered_depth):
                    raise OSError(f"failed to write {depth_path}")

                rgb_world = np.linalg.inv(world_rgb)
                gt = []
                for world_object in world_objects:
                    item = _transform_to_bop(
                        rgb_world @ world_object,
                        pose_units_to_mm,
                    )
                    item["obj_id"] = ROBI_OBJECT_IDS[object_name]
                    gt.append(item)
                scene_gt[str(bop_image_id)] = gt
                scene_gt_info[str(bop_image_id)] = [
                    {
                        "bbox_obj": [0, 0, REALSENSE_RGB.width, REALSENSE_RGB.height],
                        "bbox_visib": [0, 0, REALSENSE_RGB.width, REALSENSE_RGB.height],
                        "px_count_all": 0,
                        "px_count_valid": 0,
                        "px_count_visib": 0,
                        "visib_fract": 1.0,
                    }
                    for _ in gt
                ]
                world_rgb_scaled = world_rgb.copy()
                world_rgb_scaled[:3, 3] *= pose_units_to_mm
                scene_camera[str(bop_image_id)] = {
                    "cam_K": REALSENSE_RGB.intrinsics.reshape(-1).tolist(),
                    "depth_scale": 1.0,
                    "cam_R_w2c": world_rgb_scaled[:3, :3].reshape(-1).tolist(),
                    "cam_t_w2c": world_rgb_scaled[:3, 3].tolist(),
                }
                targets.append(
                    {
                        "im_id": bop_image_id,
                        "inst_count": len(gt),
                        "obj_id": ROBI_OBJECT_IDS[object_name],
                        "scene_id": bop_scene_id,
                    }
                )
                image_map[str(bop_image_id)] = view
                converted_images += 1

            for name, payload in (
                ("scene_camera.json", scene_camera),
                ("scene_gt.json", scene_gt),
                ("scene_gt_info.json", scene_gt_info),
                ("robi_image_map.json", image_map),
            ):
                (destination / name).write_text(
                    json.dumps(payload, indent=2) + "\n",
                    encoding="utf-8",
                )
            scene_map.append(
                {
                    "bop_scene_id": bop_scene_id,
                    "object_name": object_name,
                    "object_id": ROBI_OBJECT_IDS[object_name],
                    "robi_scene_id": source_scene_id,
                    "image_count": len(views),
                }
            )
            bop_scene_id += 1

    camera_payload = {
        "cx": float(REALSENSE_RGB.intrinsics[0, 2]),
        "cy": float(REALSENSE_RGB.intrinsics[1, 2]),
        "depth_scale": 1.0,
        "fx": float(REALSENSE_RGB.intrinsics[0, 0]),
        "fy": float(REALSENSE_RGB.intrinsics[1, 1]),
        "height": REALSENSE_RGB.height,
        "width": REALSENSE_RGB.width,
    }
    for name, payload in (
        ("camera.json", camera_payload),
        ("test_targets_bop19.json", targets),
        ("robi_scene_map.json", scene_map),
        ("robi_object_ids.json", ROBI_OBJECT_IDS),
    ):
        (output / name).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "object_names": object_names,
        "scene_count": len(scene_map),
        "image_count": converted_images,
        "target_count": len(targets),
        "output_root": str(output),
    }
