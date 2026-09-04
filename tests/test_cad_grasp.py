from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.cad_grasp import (  # noqa: E402
    MeshGeometry,
    _mesh_intersects_aabb,
    _mesh_center_of_mass,
    generate_parallel_jaw_candidates,
    generate_parallel_jaw_candidates_with_diagnostics,
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
    candidates, diagnostics = generate_parallel_jaw_candidates_with_diagnostics(
        config, geometry
    )

    assert config.cad_object.object_id == "box"
    assert geometry.vertices_m.shape == (8, 3)
    assert len(candidates) == 12
    assert diagnostics.evaluated_frames == 60
    assert diagnostics.returned_candidates == 12
    assert diagnostics.accepted_before_limit >= diagnostics.returned_candidates
    assert set(diagnostics.rejection_counts) == {
        "contact_patch",
        "jaw_opening",
        "antipodal",
        "center_of_mass",
        "edge_distance",
        "finger_collision",
        "duplicate_candidate",
    }
    assert all(0.005 <= item.jaw_opening_m <= 0.085 for item in candidates)
    assert all(
        item.contact_points_object_m.shape == (2, 3) for item in candidates
    )
    assert all(
        item.contact_normals_object.shape == (2, 3) for item in candidates
    )
    assert all(item.center_of_mass_offset_m <= 0.02 for item in candidates)
    assert all(item.minimum_edge_distance_m >= 0.001 for item in candidates)
    assert all(item.finger_collision_free for item in candidates)
    assert all(
        np.isclose(np.linalg.det(item.transform_object_grasp[:3, :3]), 1.0)
        for item in candidates
    )
    regenerated = generate_parallel_jaw_candidates(config, geometry)
    assert [item.grasp_id for item in regenerated] == [
        item.grasp_id for item in candidates
    ]


def test_mesh_geometry_rejects_bad_normals() -> None:
    vertices = np.zeros((8, 3))
    try:
        MeshGeometry(vertices, np.zeros((7, 3)))
    except ValueError:
        return
    raise AssertionError("MeshGeometry accepted mismatched normals")


def test_mesh_triangle_collision_detects_face_crossing_finger_box() -> None:
    vertices = np.array(
        [
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    faces = np.array([[0, 1, 2]])
    lower = np.array([-0.1, -0.1, -0.1])
    upper = np.array([0.1, 0.1, 0.1])

    assert _mesh_intersects_aabb(vertices, faces, lower, upper)
    assert not _mesh_intersects_aabb(
        vertices + np.array([0.0, 0.0, 3.0]), faces, lower, upper
    )


def test_closed_mesh_center_of_mass_uses_uniform_volume() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ]
    )

    center = _mesh_center_of_mass(MeshGeometry(vertices, faces=faces))

    np.testing.assert_allclose(center, [0.25, 0.25, 0.25])


def test_ascii_ply_loader_triangulates_polygon_faces(tmp_path: Path) -> None:
    path = tmp_path / "tetra.ply"
    path.write_text(
        """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
1 1 0
0 1 0
4 0 1 2 3
""",
        encoding="ascii",
    )

    geometry = load_ascii_ply_geometry(path, units_to_m=0.001)

    assert geometry.faces.shape == (2, 3)
    np.testing.assert_array_equal(geometry.faces, [[0, 1, 2], [0, 2, 3]])
