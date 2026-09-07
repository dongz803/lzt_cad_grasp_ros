from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.pose_gating import (  # noqa: E402
    PoseGateConfig,
    contour_reprojection_error_px,
    evaluate_pose_candidate,
    render_model_points,
    reweighted_foundpose_record,
)


def model_plane() -> np.ndarray:
    values = np.linspace(-0.05, 0.05, 31)
    xx, yy = np.meshgrid(values, values)
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))


def record(x_m: float = 0.0, z_m: float = 0.5) -> dict:
    return {
        "scene_id": 1,
        "img_id": 2,
        "obj_id": 1,
        "inst_id": 0,
        "hypothesis_id": 0,
        "score": 0.8,
        "inliers_est": 60,
        "R": np.eye(3).tolist(),
        "t": [[1000.0 * x_m], [0.0], [1000.0 * z_m]],
    }


def observation_for(source_record: dict):
    points = model_plane()
    intrinsics = np.array(
        [[200.0, 0.0, 60.0], [0.0, 200.0, 50.0], [0.0, 0.0, 1.0]]
    )
    transform = np.eye(4)
    transform[0, 3] = np.asarray(source_record["t"])[0, 0] / 1000.0
    transform[2, 3] = np.asarray(source_record["t"])[2, 0] / 1000.0
    mask, rendered_depth, _ = render_model_points(
        points, transform, intrinsics, (100, 120), splat_radius_px=2
    )
    depth = np.zeros(mask.shape, dtype=np.float64)
    finite = np.isfinite(rendered_depth)
    depth[finite] = rendered_depth[finite]
    return points, intrinsics, mask, depth


def test_consistent_pose_passes_all_physical_and_rgbd_gates() -> None:
    source = record()
    points, intrinsics, mask, depth = observation_for(source)
    settings = PoseGateConfig(
        minimum_silhouette_iou=0.5,
        maximum_reprojection_error_px=2.0,
        minimum_depth_overlap_pixels=10,
        minimum_depth_inlier_fraction=0.9,
        maximum_median_depth_error_m=0.005,
    )

    decision = evaluate_pose_candidate(
        source, points, intrinsics, mask, depth, settings
    )

    assert decision.accepted
    assert decision.rejection_reasons == ()
    assert decision.metrics["silhouette_iou"] == 1.0
    assert decision.metrics["median_depth_error_m"] == 0.0
    assert decision.robust_score > 0.0
    reweighted = reweighted_foundpose_record(source, decision)
    assert reweighted["original_score"] == 0.8
    assert reweighted["score"] == decision.robust_score
    assert reweighted["source"] == "foundpose_coarse_rgbd_gated"


def test_negative_and_out_of_workspace_pose_is_rejected() -> None:
    source = record(z_m=-0.5)
    points, intrinsics, mask, depth = observation_for(record())

    decision = evaluate_pose_candidate(
        source, points, intrinsics, mask, depth, PoseGateConfig()
    )

    assert not decision.accepted
    assert "nonpositive_depth" in decision.rejection_reasons
    assert "outside_camera_workspace" in decision.rejection_reasons
    assert "model_behind_camera" in decision.rejection_reasons
    assert decision.robust_score == 0.0


def test_geometrically_shifted_pose_fails_silhouette_and_reprojection() -> None:
    reference = record()
    points, intrinsics, mask, depth = observation_for(reference)
    shifted = record(x_m=0.10)

    decision = evaluate_pose_candidate(
        shifted,
        points,
        intrinsics,
        mask,
        depth,
        PoseGateConfig(
            minimum_silhouette_iou=0.2,
            maximum_reprojection_error_px=5.0,
        ),
    )

    assert not decision.accepted
    assert "low_silhouette_iou" in decision.rejection_reasons
    assert "high_reprojection_error" in decision.rejection_reasons


def test_robot_workspace_requires_calibrated_transform() -> None:
    source = record()
    points, intrinsics, mask, depth = observation_for(source)
    settings = PoseGateConfig(
        robot_min_xyz_m=(-0.2, -0.2, 0.0),
        robot_max_xyz_m=(0.2, 0.2, 1.0),
    )

    missing = evaluate_pose_candidate(
        source, points, intrinsics, mask, depth, settings
    )
    outside_transform = np.eye(4)
    outside_transform[0, 3] = 1.0
    outside = evaluate_pose_candidate(
        source,
        points,
        intrinsics,
        mask,
        depth,
        settings,
        transform_robot_camera=outside_transform,
    )

    assert "missing_robot_camera_transform" in missing.rejection_reasons
    assert "outside_robot_workspace" in outside.rejection_reasons


def test_contour_error_is_zero_for_equal_masks() -> None:
    mask = np.zeros((20, 25), dtype=bool)
    mask[4:15, 7:19] = True
    assert contour_reprojection_error_px(mask, mask) == 0.0
