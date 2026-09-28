import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_perception.pose_reliability import (  # noqa: E402
    FEATURE_NAMES,
    PoseReliabilityModel,
    binary_probability_metrics,
    fit_pose_reliability_model,
    pose_reliability_features,
)


def audit_record(score: float, depth_error: float, accepted: bool = True) -> dict:
    return {
        "original_score": score,
        "accepted": accepted,
        "geometric_quality": 1.0 - depth_error / 0.1,
        "robust_score": score * (1.0 - depth_error / 0.1),
        "metrics": {
            "pnp_score": score,
            "pnp_inliers": int(100 * score),
            "positive_model_fraction": 1.0,
            "projected_model_fraction": 1.0,
            "rendered_silhouette_pixels": 1000,
            "observed_mask_pixels": 1000,
            "silhouette_iou": 1.0 - depth_error,
            "silhouette_boundary_f1": 1.0 - depth_error,
            "silhouette_center_error_px": depth_error * 100,
            "reprojection_error_px": depth_error * 200,
            "depth_overlap_pixels": 900,
            "median_depth_error_m": depth_error,
            "depth_inlier_fraction": 1.0 - depth_error / 0.1,
        },
    }


def test_reliability_features_are_finite_and_group_relative() -> None:
    best = audit_record(0.8, 0.01)
    weak = audit_record(0.4, 0.04)
    group = [best, weak]

    best_features = pose_reliability_features(best, 0.1, group)
    weak_features = pose_reliability_features(weak, 0.1, group)

    assert best_features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(best_features))
    assert best_features[FEATURE_NAMES.index("relative_pnp_score")] == 1.0
    assert weak_features[FEATURE_NAMES.index("relative_pnp_score")] == 0.5


def test_logistic_model_learns_and_round_trips(tmp_path: Path) -> None:
    negative = np.tile(np.linspace(-2.0, -0.2, 20)[:, None], (1, len(FEATURE_NAMES)))
    positive = np.tile(np.linspace(0.2, 2.0, 20)[:, None], (1, len(FEATURE_NAMES)))
    features = np.vstack((negative, positive))
    labels = np.r_[np.zeros(20), np.ones(20)]

    model = fit_pose_reliability_model(features, labels, l2_regularization=1.0)
    probabilities = model.predict_proba(features)
    path = model.save(tmp_path / "model.json")
    restored = PoseReliabilityModel.load(path)

    assert np.mean(probabilities[:20]) < 0.2
    assert np.mean(probabilities[20:]) > 0.8
    np.testing.assert_allclose(restored.predict_proba(features), probabilities)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["label"]["ground_truth_used_at_runtime"] is False


def test_probability_metrics_reward_calibrated_separation() -> None:
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    good = binary_probability_metrics(labels, np.array([0.05, 0.2, 0.8, 0.95]))
    bad = binary_probability_metrics(labels, np.array([0.5, 0.5, 0.5, 0.5]))

    assert good["roc_auc"] == 1.0
    assert good["brier_score"] < bad["brier_score"]


def test_frozen_cross_cad_model_is_loadable_without_outputs() -> None:
    path = (
        PROJECT_ROOT
        / "lzt_perception/config/reliability/tless_objects2_4_logistic_v2.json"
    )
    model = PoseReliabilityModel.load(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert tuple(payload["feature_names"]) == FEATURE_NAMES
    assert payload["label"]["ground_truth_used_at_runtime"] is False
    assert payload["training"]["object_ids"] == [2, 3, 4]
    assert payload["training"]["fixed_bop_target_instances"] == 1419
    probability = float(model.predict_proba(np.zeros(len(FEATURE_NAMES))))
    assert 0.0 <= probability <= 1.0
