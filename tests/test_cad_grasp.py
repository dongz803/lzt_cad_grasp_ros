from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.cad_grasp import (  # noqa: E402
    MeshGeometry,
    generate_parallel_jaw_candidates,
    load_ascii_ply_geometry,
)
from lzt_cad_grasp_core.object_config import load_object_config  # noqa: E402


def write_box_ply(path: Path) -> None:
    vertices = np.array(
        [
            [-10, -15, -20],
            [-10, -15, 20],
            [-10, 15, -20],
            [-10, 15, 20],
            [10, -15, -20],
            [10, -15, 20],
            [10, 15, -20],
            [10, 15, 20],
        ],
        dtype=float,
    )
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 8",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(" ".join(map(str, vertex)) for vertex in vertices)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_object_config_and_cad_candidates(tmp_path: Path, monkeypatch) -> None:
    mesh_path = tmp_path / "box.ply"
    write_box_ply(mesh_path)
    monkeypatch.setenv("TEST_CAD_ROOT", str(tmp_path))
    config_path = tmp_path / "box.yaml"
    config_path.write_text(
        """schema_version: 1
object_id: box
display_name: Test box
cad:
  mesh: ${TEST_CAD_ROOT}/box.ply
  units_to_m: 0.001
  object_frame: cad_origin
symmetry:
  type: none
gripper:
  type: parallel_jaw
  minimum_opening_m: 0.005
  maximum_opening_m: 0.085
  finger_clearance_m: 0.002
grasp_generation:
  maximum_candidates: 12
  minimum_antipodal_score: 0.3
task:
  placement_position_tolerance_m: 0.005
  placement_angle_tolerance_deg: 5.0
""",
        encoding="utf-8",
    )

    config = load_object_config(config_path)
    geometry = load_ascii_ply_geometry(mesh_path, units_to_m=0.001)
    candidates = generate_parallel_jaw_candidates(config, geometry)

    assert config.cad_object.object_id == "box"
    assert geometry.vertices_m.shape == (8, 3)
    assert len(candidates) == 12
    assert all(0.005 <= item.jaw_opening_m <= 0.085 for item in candidates)
    assert all(
        np.isclose(np.linalg.det(item.transform_object_grasp[:3, :3]), 1.0)
        for item in candidates
    )


def test_mesh_geometry_rejects_bad_normals() -> None:
    vertices = np.zeros((8, 3))
    try:
        MeshGeometry(vertices, np.zeros((7, 3)))
    except ValueError:
        return
    raise AssertionError("MeshGeometry accepted mismatched normals")
