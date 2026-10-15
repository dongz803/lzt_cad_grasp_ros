import importlib.util
from pathlib import Path

import numpy as np

from lzt_common.contracts import PoseHypothesis, PoseHypothesisSet
from lzt_perception.pose_modes import PoseModeClusteringConfig, cluster_pose_hypotheses


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = PROJECT_ROOT / "scripts/evaluation/run_symmetry_clustering_ablation.py"
    spec = importlib.util.spec_from_file_location("symmetry_clustering_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mode_candidate_records_preserve_bop_units_and_mode_weight() -> None:
    script = _load_script()
    pose = np.eye(4)
    pose[:3, 3] = [0.01, -0.02, 0.6]
    hypothesis_set = PoseHypothesisSet(
        object_id="test",
        scene_id=1,
        image_id=2,
        instance_id=3,
        hypotheses=(PoseHypothesis(pose, 0.8, hypothesis_id=7),),
    )
    from lzt_common.contracts import CADObject

    mixture = cluster_pose_hypotheses(
        hypothesis_set,
        CADObject(object_id="test", mesh_path=Path("unused.ply")),
        PoseModeClusteringConfig(use_symmetry_equivalence=False),
    )
    records = script.mode_candidate_records(
        mixture,
        [
            {
                "scene_id": 1,
                "img_id": 2,
                "obj_id": 1,
                "inst_id": 3,
                "hypothesis_id": 7,
                "time": {"pose": 0.1},
                "cnos_time": 0.2,
            }
        ],
    )

    assert len(records) == 1
    assert records[0]["score"] == 1.0
    assert np.asarray(records[0]["t"]).reshape(3).tolist() == [10.0, -20.0, 600.0]
    assert records[0]["member_hypothesis_ids"] == [7]


def test_structure_summary_counts_mode_reduction() -> None:
    script = _load_script()
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.001
    hypothesis_set = PoseHypothesisSet(
        object_id="test",
        scene_id=1,
        image_id=2,
        instance_id=0,
        hypotheses=(
            PoseHypothesis(first, 0.6, hypothesis_id=0),
            PoseHypothesis(second, 0.4, hypothesis_id=1),
        ),
    )
    from lzt_common.contracts import CADObject

    mixture = cluster_pose_hypotheses(
        hypothesis_set,
        CADObject(object_id="test", mesh_path=Path("unused.ply")),
        PoseModeClusteringConfig(maximum_translation_distance_m=0.01),
    )
    summary = script.summarize_mixtures([mixture])

    assert summary["source_hypothesis_count"] == 2
    assert summary["mode_count"] == 1
    assert summary["collapsed_hypothesis_count"] == 1
    assert summary["groups_with_any_merge"] == 1
