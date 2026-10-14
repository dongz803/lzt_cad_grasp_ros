from pathlib import Path

import numpy as np
import pytest

from lzt_common.contracts import CADObject, ContinuousSymmetry
from lzt_perception.pose_uncertainty import (
    PredictivePoseCovarianceModel,
    combine_predictive_and_dispersion,
    fit_predictive_covariance,
    squared_mahalanobis,
    symmetry_aware_pose_residual,
    symmetry_observable_projector,
)


def _rotation_z(angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return transform


def _continuous_cad() -> CADObject:
    return CADObject(
        object_id="axis_symmetric",
        mesh_path=Path("unused.ply"),
        continuous_symmetries=(
            ContinuousSymmetry(np.array([0.0, 0.0, 1.0]), np.zeros(3)),
        ),
    )


def test_continuous_symmetry_axis_is_removed_from_residual() -> None:
    predicted = np.eye(4)
    target = _rotation_z(47.3)
    target[:3, 3] = [0.01, -0.02, 0.03]

    residual, _, _ = symmetry_aware_pose_residual(
        predicted,
        target,
        _continuous_cad(),
        continuous_symmetry_samples=36,
    )

    assert residual[:3] == pytest.approx([0.01, -0.02, 0.03])
    assert residual[3:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)
    projector = symmetry_observable_projector(_continuous_cad())
    assert np.linalg.matrix_rank(projector) == 5
    assert projector[5, 5] == pytest.approx(0.0)


def test_discrete_symmetry_equivalent_has_zero_rotation_residual() -> None:
    half_turn = _rotation_z(180.0)
    cad_object = CADObject(
        object_id="two_fold",
        mesh_path=Path("unused.ply"),
        symmetry_transforms=(half_turn,),
    )
    residual, _, symmetry_class = symmetry_aware_pose_residual(
        np.eye(4),
        half_turn,
        cad_object,
    )

    assert residual == pytest.approx(np.zeros(6), abs=1e-10)
    assert symmetry_class == "discrete_0"


def test_predictive_covariance_stays_nonzero_for_singleton_dispersion() -> None:
    residuals = np.array(
        [
            [0.010, 0.000, 0.002, 0.020, 0.000, 0.0],
            [-0.005, 0.003, -0.001, -0.010, 0.015, 0.0],
            [0.004, -0.002, 0.003, 0.005, -0.010, 0.0],
        ]
    )
    projector = symmetry_observable_projector(_continuous_cad())
    predictive = fit_predictive_covariance(residuals, projector)
    total = combine_predictive_and_dispersion(predictive, np.zeros((6, 6)))

    assert np.trace(total[:3, :3]) > 0.0
    assert np.trace(total[3:, 3:]) > 0.0
    assert total[5, 5] == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(total, total.T)
    assert np.min(np.linalg.eigvalsh(total)) >= -1e-12
    distance = squared_mahalanobis(residuals[0], total, projector)
    assert np.isfinite(distance)
    assert distance >= 0.0


def test_invalid_predictive_fit_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_predictive_covariance(np.zeros((1, 6)))
    with pytest.raises(ValueError):
        fit_predictive_covariance(np.zeros((2, 6)), shrinkage=1.1)


def test_frozen_predictive_model_rejects_uncalibrated_objects() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "lzt_perception/config/uncertainty/tless_objects2_4_predictive_v3.json"
    )
    model = PredictivePoseCovarianceModel.load(path)

    covariance = model.covariance_for("tless_000004")
    assert covariance.shape == (6, 6)
    assert np.trace(covariance) > 0.0
    with pytest.raises(ValueError, match="not calibrated"):
        model.covariance_for("tless_000005")
