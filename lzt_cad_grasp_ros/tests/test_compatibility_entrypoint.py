from lzt_cad_grasp_ros.exploration_node import CADGraspExplorationNode, main


def test_original_ros_entrypoint_is_preserved():
    assert CADGraspExplorationNode.__name__ == "CADGraspExplorationNode"
    assert callable(main)

