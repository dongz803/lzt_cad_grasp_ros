import importlib.util
from pathlib import Path
import sys

import numpy as np

from lzt_common.contracts import CADObject


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = PROJECT_ROOT / "scripts/analysis/compare_multimode_best1_uncertainty.py"
    spec = importlib.util.spec_from_file_location("multimode_best1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(best: bool, mixture: bool, mode_count: int = 2):
    coverage = {}
    volume = {}
    for key in ("0.5000", "0.6827", "0.9000", "0.9500"):
        coverage[key] = {"best1": best, "multimode_union": mixture}
        volume[key] = {
            "best1_log_volume": -10.0,
            "multimode_log_sum_component_volumes": -9.0,
        }
    return {
        "object_id": "tless_000013",
        "mode_count": mode_count,
        "contains_correct_mode": True,
        "best1_is_correct": best,
        "nearest_mode_id": 0 if best else 1,
        "best1_nll": 5.0,
        "multimode_mixture_nll": 4.0,
        "best1_squared_mahalanobis": 2.0,
        "multimode_min_squared_mahalanobis": 1.0,
        "coverage": coverage,
        "volume_proxy": volume,
    }


def test_multimode_union_counts_rescue_without_deleting_failure() -> None:
    script = _load_script()
    summary = script.summarize_records(
        [_record(False, True), _record(False, False)],
        [0.5, 0.6827, 0.9, 0.95],
    )

    result = summary["coverage"]["0.9000"]
    assert result["best1"]["sample_count"] == 2
    assert result["best1"]["covered_count"] == 0
    assert result["multimode_union"]["covered_count"] == 1
    assert result["rescued_by_additional_modes"] == 1
    assert result["lost_by_multimode"] == 0


def test_singleton_mixture_matches_best1_nll_and_coverage() -> None:
    script = _load_script()
    cad = CADObject(object_id="test", mesh_path=Path("unused.ply"))
    covariance = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2])
    modes = [
        {
            "mode_id": 0,
            "weight": 1.0,
            "transform_camera_object": np.eye(4).tolist(),
            "covariance_se3": np.zeros((6, 6)).tolist(),
            "dispersion_covariance_se3": np.zeros((6, 6)).tolist(),
        }
    ]
    evaluations, covariances = script.evaluate_components(
        modes=modes,
        target=np.eye(4),
        cad_object=cad,
        diameter_m=0.1,
        predictive_covariance=covariance,
        projector=np.eye(6),
        continuous_symmetry_samples=8,
    )
    record = script.sample_record(
        object_id="test",
        stratum="full_se3_rank6",
        group={"scene_id": 1, "image_id": 2, "instance_id": 0},
        gt_match_iou=1.0,
        evaluations=evaluations,
        covariances=covariances,
        thresholds={"0.9000": 10.0},
        correct_mode_threshold=0.1,
        projector=np.eye(6),
        online_approved=True,
    )

    assert record["best1_nll"] == record["multimode_mixture_nll"]
    assert record["coverage"]["0.9000"]["best1"]
    assert record["coverage"]["0.9000"]["multimode_union"]


def test_ellipsoid_volume_grows_with_threshold() -> None:
    script = _load_script()
    covariance = np.eye(6)
    assert script.ellipsoid_log_volume(covariance, np.eye(6), 10.0) > (
        script.ellipsoid_log_volume(covariance, np.eye(6), 5.0)
    )
