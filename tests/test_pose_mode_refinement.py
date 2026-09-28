from pathlib import Path

import numpy as np
import pytest

from lzt_common.contracts import CADObject, PoseHypothesis, PoseHypothesisSet
from lzt_perception.pose_gating import render_model_points
from lzt_perception.pose_mode_refinement import refine_pose_mixture_rgbd
from lzt_perception.pose_modes import cluster_pose_hypotheses
from lzt_perception.pose_refinement import PoseRefinementConfig


def _box_surface_points() -> np.ndarray:
    points = []
    for x in np.linspace(-0.04, 0.04, 21):
        for y in np.linspace(-0.025, 0.025, 15):
            points.extend(([x, y, -0.015], [x, y, 0.015]))
    for x in np.linspace(-0.04, 0.04, 21):
        for z in np.linspace(-0.015, 0.015, 11):
            points.extend(([x, -0.025, z], [x, 0.025, z]))
    return np.asarray(points)


def test_mode_refinement_runs_once_and_propagates_same_correction() -> None:
    points = _box_surface_points()
    intrinsics = np.array(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
    )
    truth = np.eye(4)
    truth[:3, 3] = [0.01, -0.02, 0.65]
    mask, rendered_depth, _ = render_model_points(
        points, truth, intrinsics, (240, 320), splat_radius_px=2
    )
    depth = np.zeros(mask.shape)
    depth[np.isfinite(rendered_depth)] = rendered_depth[np.isfinite(rendered_depth)]

    first = truth.copy()
    first[:3, 3] += [0.012, -0.008, 0.012]
    second = first.copy()
    second[0, 3] += 0.001
    hypothesis_set = PoseHypothesisSet(
        object_id="industrial_box",
        scene_id=1,
        image_id=2,
        instance_id=0,
        hypotheses=(
            PoseHypothesis(first, 0.7, hypothesis_id=0, source="coarse"),
            PoseHypothesis(second, 0.3, hypothesis_id=1, source="coarse"),
        ),
    )
    mixture = cluster_pose_hypotheses(
        hypothesis_set,
        CADObject("industrial_box", Path("unused.ply")),
    )

    result = refine_pose_mixture_rgbd(
        hypothesis_set,
        mixture,
        points,
        intrinsics,
        mask,
        depth,
        PoseRefinementConfig(iterations_per_level=4),
    )

    assert len(result.mode_refinements) == 1
    assert result.accepted_mode_count == 1
    before_delta = second[:3, 3] - first[:3, 3]
    after_delta = (
        result.hypothesis_set.hypotheses[1].transform_camera_object[:3, 3]
        - result.hypothesis_set.hypotheses[0].transform_camera_object[:3, 3]
    )
    assert after_delta == pytest.approx(before_delta)
    assert all(
        "mode_rgbd_refined" in item.source
        for item in result.hypothesis_set.hypotheses
    )


def test_mode_refinement_rejects_incomplete_partition() -> None:
    transform = np.eye(4)
    transform[2, 3] = 0.5
    hypothesis_set = PoseHypothesisSet(
        object_id="test",
        scene_id=0,
        image_id=0,
        instance_id=0,
        hypotheses=(PoseHypothesis(transform, 1.0, hypothesis_id=0),),
    )
    mixture = cluster_pose_hypotheses(
        hypothesis_set, CADObject("test", Path("unused.ply"))
    )
    mismatched_set = PoseHypothesisSet(
        object_id="test",
        scene_id=0,
        image_id=0,
        instance_id=1,
        hypotheses=hypothesis_set.hypotheses,
    )
    with pytest.raises(ValueError, match="identity"):
        refine_pose_mixture_rgbd(
            mismatched_set,
            mixture,
            np.zeros((1, 3)),
            np.eye(3),
            np.zeros((1, 1), dtype=bool),
            np.zeros((1, 1)),
        )
