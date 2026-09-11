from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.contracts import CADObject  # noqa: E402
from lzt_cad_grasp_core.industrial_workcell import (  # noqa: E402
    TARGET_BODY_NAME,
    TARGET_JOINT_NAME,
    body_transform,
    build_industrial_cad_workcell,
    set_freejoint_transform,
)
from lzt_cad_grasp_core.object_config import (  # noqa: E402
    GraspGenerationSpec,
    GripperSpec,
    ObjectConfig,
)
from lzt_cad_grasp_core.simulation import reset_robot_home  # noqa: E402


CUBE_PLY = """ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 12
property list uchar int vertex_indices
end_header
-10 -10 -10
10 -10 -10
10 10 -10
-10 10 -10
-10 -10 10
10 -10 10
10 10 10
-10 10 10
3 0 2 1
3 0 3 2
3 4 5 6
3 4 6 7
3 0 1 5
3 0 5 4
3 1 2 6
3 1 6 5
3 2 3 7
3 2 7 6
3 3 0 4
3 3 4 7
"""


def _object_config(mesh_path: Path) -> ObjectConfig:
    return ObjectConfig(
        schema_version=1,
        display_name="test cube",
        cad_object=CADObject("test_cube", mesh_path, units_to_m=0.001),
        object_frame="cad_origin",
        gripper=GripperSpec(0.005, 0.085),
        grasp_generation=GraspGenerationSpec(),
        placement_position_tolerance_m=0.005,
        placement_angle_tolerance_deg=5.0,
    )


def test_dynamic_cad_workcell_restores_target_after_robot_keyframe(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "cube.ply"
    mesh_path.write_text(CUBE_PLY, encoding="ascii")
    model, info = build_industrial_cad_workcell(
        _object_config(mesh_path),
        tmp_path / "cube.obj",
        initial_xy_m=(-0.2, -0.12),
        placement_clearance_m=0.0,
    )
    data = mujoco.MjData(model)

    reset_robot_home(model, data)
    set_freejoint_transform(
        model,
        data,
        TARGET_JOINT_NAME,
        info.initial_transform_world_object,
    )

    actual = body_transform(data, model.body(TARGET_BODY_NAME).id)
    np.testing.assert_allclose(
        actual, info.initial_transform_world_object, atol=1e-12
    )
    assert model.nu == 7
    assert model.njnt == 17
