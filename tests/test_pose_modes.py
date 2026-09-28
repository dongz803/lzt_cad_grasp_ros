from pathlib import Path

import numpy as np
import pytest

from lzt_common.contracts import (
    CADObject,
    ContinuousSymmetry,
    PoseHypothesis,
    PoseHypothesisSet,
)
from lzt_perception.pose_modes import (
    PoseModeClusteringConfig,
    cluster_pose_hypotheses,
    pose_mixtures_to_jsonable,
)


def _rotation(axis: int, angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    if axis == 0:
        result[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 1:
        result[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return result


def _cad() -> CADObject:
    return CADObject(
        object_id="tless_000004",
        mesh_path=Path("unused.ply"),
        continuous_symmetries=(
            ContinuousSymmetry(np.array([0.0, 0.0, 1.0]), np.zeros(3)),
        ),
    )


def _set(*hypotheses: PoseHypothesis) -> PoseHypothesisSet:
    return PoseHypothesisSet(
        object_id="tless_000004",
        scene_id=1,
        image_id=2,
        instance_id=0,
        hypotheses=hypotheses,
    )


def test_continuous_symmetry_merges_yaw_but_preserves_tilt_mode() -> None:
    first = np.eye(4)
    first[:3, 3] = [0.0, 0.0, 0.6]
    yaw = first @ _rotation(2, 180.0)
    tilt = first @ _rotation(0, 35.0)
    mixture = cluster_pose_hypotheses(
        _set(
            PoseHypothesis(first, 0.5, hypothesis_id=0),
            PoseHypothesis(yaw, 0.3, hypothesis_id=1),
            PoseHypothesis(tilt, 0.2, hypothesis_id=2),
        ),
        _cad(),
    )

    assert len(mixture.modes) == 2
    assert mixture.modes[0].member_hypothesis_ids == (0, 1)
    assert mixture.modes[0].weight == pytest.approx(0.8)
    assert mixture.modes[1].member_hypothesis_ids == (2,)
    assert mixture.modes[1].weight == pytest.approx(0.2)


def test_translation_separates_modes_and_covariance_is_psd() -> None:
    first = np.eye(4)
    first[:3, 3] = [0.0, 0.0, 0.6]
    close = first.copy()
    close[0, 3] += 0.005
    far = first.copy()
    far[0, 3] += 0.08
    mixture = cluster_pose_hypotheses(
        _set(
            PoseHypothesis(first, 0.6, hypothesis_id=5),
            PoseHypothesis(close, 0.25, hypothesis_id=2),
            PoseHypothesis(far, 0.15, hypothesis_id=9),
        ),
        _cad(),
        PoseModeClusteringConfig(maximum_translation_distance_m=0.02),
    )

    assert len(mixture.modes) == 2
    assert mixture.modes[0].member_hypothesis_ids == (2, 5)
    assert sum(mixture.modes[0].member_weights) == pytest.approx(1.0)
    assert np.min(np.linalg.eigvalsh(mixture.modes[0].covariance_se3)) >= -1e-12
    payload = pose_mixtures_to_jsonable([mixture])[0]
    assert payload["mode_count"] == 2
    assert payload["source_hypothesis_count"] == 3
    assert payload["modes"][0]["representative_hypothesis_id"] == 5


def test_invalid_mode_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError):
        PoseModeClusteringConfig(maximum_translation_distance_m=0.0)
    with pytest.raises(ValueError):
        PoseModeClusteringConfig(maximum_rotation_distance_deg=181.0)

