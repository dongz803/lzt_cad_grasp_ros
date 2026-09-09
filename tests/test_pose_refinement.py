from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.pose_gating import render_model_points  # noqa: E402
from lzt_cad_grasp_core.pose_refinement import (  # noqa: E402
    PoseRefinementConfig,
    refine_pose_rgbd,
)


def box_surface_points() -> np.ndarray:
    points = []
    for x in np.linspace(-0.04, 0.04, 25):
        for y in np.linspace(-0.025, 0.025, 19):
            points.extend(([x, y, -0.015], [x, y, 0.015]))
    for x in np.linspace(-0.04, 0.04, 25):
        for z in np.linspace(-0.015, 0.015, 13):
            points.extend(([x, -0.025, z], [x, 0.025, z]))
    for y in np.linspace(-0.025, 0.025, 19):
        for z in np.linspace(-0.015, 0.015, 13):
            points.extend(([-0.04, y, z], [0.04, y, z]))
    return np.asarray(points)


def test_rgbd_feedback_reduces_pose_and_alignment_error() -> None:
    points = box_surface_points()
    intrinsics = np.array(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
    )
    truth = np.eye(4)
    truth[:3, 3] = [0.01, -0.02, 0.65]
    mask, rendered_depth, _ = render_model_points(
        points, truth, intrinsics, (240, 320), splat_radius_px=2
    )
    depth = np.zeros(mask.shape)
    finite = np.isfinite(rendered_depth)
    depth[finite] = rendered_depth[finite]
    initial = truth.copy()
    initial[:3, 3] += [0.012, -0.008, 0.012]

    result = refine_pose_rgbd(
        points,
        initial,
        intrinsics,
        mask,
        depth,
        PoseRefinementConfig(iterations_per_level=4),
    )

    assert result.accepted
    assert result.objective_improvement > 0.1
    assert result.refined_metrics.silhouette_iou > result.initial_metrics.silhouette_iou
    assert result.refined_metrics.median_depth_error_m < 0.002
    assert np.linalg.norm(result.refined_transform_camera_object[:3, 3] - truth[:3, 3]) < 0.002


def test_rgbd_feedback_can_correct_a_small_rotation_error() -> None:
    points = box_surface_points()
    intrinsics = np.array(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
    )
    truth = np.eye(4)
    truth[:3, 3] = [0.025, -0.015, 0.58]
    mask, rendered_depth, _ = render_model_points(
        points, truth, intrinsics, (240, 320), splat_radius_px=2
    )
    depth = np.zeros(mask.shape)
    finite = np.isfinite(rendered_depth)
    depth[finite] = rendered_depth[finite]
    angle = np.radians(4.0)
    initial = truth.copy()
    initial[:3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )

    result = refine_pose_rgbd(points, initial, intrinsics, mask, depth)

    initial_angle = np.degrees(
        np.arccos(np.clip((np.trace(initial[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    )
    refined_angle = np.degrees(
        np.arccos(
            np.clip(
                (np.trace(result.refined_transform_camera_object[:3, :3]) - 1.0)
                * 0.5,
                -1.0,
                1.0,
            )
        )
    )
    assert result.accepted
    assert refined_angle < initial_angle


def test_invalid_refinement_schedule_is_rejected() -> None:
    try:
        PoseRefinementConfig(
            translation_steps_m=(0.01,), rotation_steps_deg=(5.0, 2.0)
        )
    except ValueError:
        return
    raise AssertionError("mismatched refinement schedules were accepted")
