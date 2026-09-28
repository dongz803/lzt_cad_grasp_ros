"""Enforce the direction of the mechanical-arm algorithm stack."""

from xml.etree import ElementTree

from ament_index_python.packages import get_package_share_directory


PACKAGES = {
    "lzt_common",
    "lzt_interfaces",
    "lzt_description",
    "lzt_perception",
    "lzt_grasp_planning",
    "lzt_kinematics",
    "lzt_motion_planning",
    "lzt_trajectory",
    "lzt_control",
    "lzt_mujoco",
    "lzt_benchmark",
    "lzt_bringup",
    "lzt_examples",
    "lzt_system_tests",
    "lzt_cad_grasp_ros",
}
DEPENDENCY_TAGS = {"depend", "build_depend", "build_export_depend", "exec_depend"}


def _dependencies(name):
    root = ElementTree.parse(
        get_package_share_directory(name) + "/package.xml"
    ).getroot()
    return {
        element.text.strip()
        for element in root
        if element.tag in DEPENDENCY_TAGS
        and element.text
        and element.text.strip() in PACKAGES
    }


def test_algorithm_layers_do_not_depend_on_execution_backends():
    backend = {"lzt_mujoco", "lzt_bringup", "lzt_cad_grasp_ros"}
    for package in (
        "lzt_perception",
        "lzt_grasp_planning",
        "lzt_kinematics",
        "lzt_motion_planning",
        "lzt_trajectory",
        "lzt_control",
    ):
        assert _dependencies(package).isdisjoint(backend)


def test_perception_and_grasp_research_stay_independent_of_robot_execution():
    robot_stack = {
        "lzt_kinematics",
        "lzt_motion_planning",
        "lzt_trajectory",
        "lzt_control",
        "lzt_mujoco",
    }
    assert _dependencies("lzt_perception").isdisjoint(robot_stack)
    assert _dependencies("lzt_grasp_planning").isdisjoint(robot_stack)


def test_path_trajectory_control_order_is_one_way():
    assert "lzt_trajectory" not in _dependencies("lzt_motion_planning")
    assert "lzt_control" not in _dependencies("lzt_trajectory")
    assert "lzt_mujoco" not in _dependencies("lzt_control")
    assert "lzt_motion_planning" in _dependencies("lzt_trajectory")
    assert "lzt_trajectory" in _dependencies("lzt_control")


def test_project_dependency_graph_is_acyclic():
    graph = {name: _dependencies(name) for name in PACKAGES}
    visiting = set()
    visited = set()

    def visit(name):
        if name in visiting:
            raise AssertionError(f"dependency cycle at {name}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in graph:
        visit(name)

