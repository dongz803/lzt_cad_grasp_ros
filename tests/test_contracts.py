from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.contracts import (  # noqa: E402
    CameraObservation,
    GraspCandidate,
    PoseHypothesis,
)


def test_observation_accepts_calibrated_rgbd() -> None:
    observation = CameraObservation(
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        depth_m=np.ones((48, 64), dtype=np.float32),
        intrinsics=np.eye(3),
        mask=np.ones((48, 64), dtype=bool),
    )

    assert observation.rgb.shape == (48, 64, 3)


def test_pose_hypothesis_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        PoseHypothesis(np.eye(4), confidence=1.1)


def test_grasp_candidate_is_expressed_in_object_frame() -> None:
    grasp = GraspCandidate(
        grasp_id="parallel_001",
        transform_object_grasp=np.eye(4),
        jaw_opening_m=0.04,
    )

    np.testing.assert_allclose(grasp.transform_object_grasp, np.eye(4))
