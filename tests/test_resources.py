from pathlib import Path
import sys

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_common.paths import scene_path  # noqa: E402
from lzt_mujoco.simulation import build_workcell_model  # noqa: E402


def test_all_baseline_scenes_load_independently() -> None:
    for scene_name in (
        "baseline_vision_single.xml",
        "baseline_vision_ten.xml",
        "exploration_workcell.xml",
        "industrial_cad_single.xml",
        "industrial_cad_grasp_focus.xml",
    ):
        model = mujoco.MjModel.from_xml_path(str(scene_path(scene_name)))
        assert model.nbody > 1


def test_scene_path_rejects_directory_traversal() -> None:
    try:
        scene_path("../lzt_mujoco_ros/scenes/vision_scene.xml")
    except ValueError:
        return
    raise AssertionError("scene_path accepted directory traversal")


def test_complete_copied_workcell_compiles() -> None:
    model = build_workcell_model()

    assert model.nbody == 38
    assert model.njnt == 17
    assert model.nu == 7
