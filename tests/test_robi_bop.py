import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.robi_bop import (  # noqa: E402
    REALSENSE_DEPTH,
    convert_robi_realsense_to_bop,
    load_robi_pose,
    reproject_depth_to_rgb,
)


def write_pose(path: Path, translation=(0.0, 0.0, 0.0)) -> None:
    pose = np.eye(4)
    pose[:3, 3] = translation
    path.write_text(json.dumps({"cam_pose": pose.reshape(-1).tolist()}), encoding="utf-8")


def test_load_robi_pose_accepts_official_json_lines(tmp_path: Path) -> None:
    pose = np.eye(4)
    pose[:3, 3] = [21.7, -72.9, 391.4]
    path = tmp_path / "camera.json"
    path.write_text(
        json.dumps({"cam_pose": pose.reshape(-1).tolist()})
        + "\n"
        + json.dumps({"object_1_visibility": 0.7})
        + "\n",
        encoding="utf-8",
    )

    assert np.allclose(load_robi_pose(path), pose)


def test_identity_depth_registration_preserves_metric_depth() -> None:
    depth = np.zeros((REALSENSE_DEPTH.height, REALSENSE_DEPTH.width), dtype=np.uint16)
    depth[360, 640] = 16000
    registered = reproject_depth_to_rgb(depth, np.eye(4), np.eye(4))

    assert registered.dtype == np.uint16
    assert registered.max() == 500
    assert np.count_nonzero(registered) == 1


def test_robi_scene_becomes_traceable_bop_scene(tmp_path: Path) -> None:
    source = tmp_path / "source"
    scene = source / "Chrome_screw/Scene_1"
    color = scene / "RealSense/Color"
    depth = scene / "RealSense/Depth"
    color.mkdir(parents=True)
    depth.mkdir(parents=True)
    (scene / "RealSense/list.txt").write_text("view_0\n", encoding="utf-8")
    object_pose = np.eye(4)
    object_pose[:3, 3] = [10.0, 20.0, 500.0]
    (scene / "GT_world2obj.json").write_text(
        json.dumps({"Object_1": object_pose.reshape(-1).tolist()}),
        encoding="utf-8",
    )
    image = np.full((720, 1280, 3), 127, dtype=np.uint8)
    raw_depth = np.zeros((720, 1280), dtype=np.uint16)
    raw_depth[360, 640] = 16000
    assert cv2.imwrite(str(color / "RGB_view_0.bmp"), image)
    assert cv2.imwrite(str(depth / "DEPTH_view_0.png"), raw_depth)
    write_pose(color / "RGB_view_0.json")
    write_pose(depth / "DEPTH_view_0.json")

    output = tmp_path / "bop/robi"
    summary = convert_robi_realsense_to_bop(
        source,
        output,
        ["Chrome_screw"],
    )

    assert summary["image_count"] == 1
    assert (output / "test/000001/rgb/000000.png").is_file()
    converted_depth = cv2.imread(
        str(output / "test/000001/depth/000000.png"), cv2.IMREAD_UNCHANGED
    )
    assert converted_depth.max() == 500
    gt = json.loads((output / "test/000001/scene_gt.json").read_text())
    assert gt["0"][0]["obj_id"] == 2
    np.testing.assert_allclose(gt["0"][0]["cam_t_m2c"], [10.0, 20.0, 500.0])
    targets = json.loads((output / "test_targets_bop19.json").read_text())
    assert targets == [{"im_id": 0, "inst_count": 1, "obj_id": 2, "scene_id": 1}]
