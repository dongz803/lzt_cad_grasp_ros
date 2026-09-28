from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_common.contracts import CADObject, ContinuousSymmetry  # noqa: E402
from lzt_perception.pose_hypotheses import (  # noqa: E402
    foundpose_records_to_top_k,
    hypothesis_sets_to_jsonable,
)


def test_foundpose_record_becomes_symmetry_aware_top_k() -> None:
    cad = CADObject(
        object_id="axis_part",
        mesh_path=Path("part.ply"),
        units_to_m=0.001,
        continuous_symmetries=(
            ContinuousSymmetry(np.array([0.0, 0.0, 1.0]), np.zeros(3)),
        ),
    )
    records = [
        {
            "scene_id": "20",
            "img_id": "3",
            "obj_id": "1",
            "inst_id": "0",
            "hypothesis_id": "0",
            "score": "0.75",
            "R": np.eye(3).tolist(),
            "t": [[100.0], [0.0], [500.0]],
        }
    ]

    sets = foundpose_records_to_top_k(
        records,
        cad,
        top_k=5,
        expand_symmetry=True,
    )

    assert len(sets) == 1
    assert len(sets[0].hypotheses) == 5
    assert sets[0].hypotheses[0].source == "foundpose_coarse"
    assert sets[0].hypotheses[1].source == "foundpose_symmetry_equivalent"
    assert sets[0].hypotheses[1].symmetry_class != "identity+continuous_0:0.000000"
    np.testing.assert_allclose(
        sets[0].hypotheses[0].transform_camera_object[:3, 3],
        [0.1, 0.0, 0.5],
    )
    assert sets[0].covariance_se3 is not None
    assert sets[0].covariance_se3.shape == (6, 6)
    assert sets[0].symmetry_discretization_deg == 1.0
    jsonable = hypothesis_sets_to_jsonable(sets)[0]["hypotheses"]
    assert jsonable[0]["confidence"] == 0.75
    assert np.isclose(sum(item["weight"] for item in jsonable), 1.0)


def test_single_pose_is_not_fabricated_into_statistical_top_k() -> None:
    cad = CADObject(object_id="tless_000001", mesh_path=Path("part.ply"))
    records = [
        {
            "scene_id": 1,
            "img_id": 2,
            "obj_id": 1,
            "inst_id": 0,
            "score": 0.8,
            "R": np.eye(3).tolist(),
            "t": [[0.0], [0.0], [500.0]],
        }
    ]

    result = foundpose_records_to_top_k(records, cad, top_k=5)[0]

    assert len(result.hypotheses) == 1
    assert result.covariance_se3 is None
    assert result.uncertainty_method == "unavailable_single_hypothesis"


def test_two_pose_hypotheses_capture_translation_uncertainty() -> None:
    cad = CADObject(object_id="part", mesh_path=Path("part.ply"))
    records = []
    for hypothesis_id, x_mm in enumerate((0.0, 10.0)):
        records.append(
            {
                "scene_id": 1,
                "img_id": 2,
                "inst_id": 0,
                "hypothesis_id": hypothesis_id,
                "score": 0.5,
                "R": np.eye(3).tolist(),
                "t": [[x_mm], [0.0], [500.0]],
            }
        )
    result = foundpose_records_to_top_k(
        records,
        cad,
        top_k=2,
        expand_symmetry=False,
    )[0]

    assert result.covariance_se3 is not None
    assert np.isclose(result.covariance_se3[0, 0], 0.000025)


def test_json_weights_are_normalized_from_raw_confidence() -> None:
    cad = CADObject(object_id="part", mesh_path=Path("part.ply"))
    records = []
    for hypothesis_id, confidence in enumerate((0.6, 0.3, 0.1)):
        records.append(
            {
                "scene_id": 1,
                "img_id": 2,
                "inst_id": 0,
                "hypothesis_id": hypothesis_id,
                "score": confidence,
                "R": np.eye(3).tolist(),
                "t": [[float(hypothesis_id)], [0.0], [500.0]],
            }
        )

    result = foundpose_records_to_top_k(records, cad, top_k=3)
    hypotheses = hypothesis_sets_to_jsonable(result)[0]["hypotheses"]

    np.testing.assert_allclose(
        [item["weight"] for item in hypotheses], [0.6, 0.3, 0.1]
    )
