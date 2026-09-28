from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_perception.mask_evaluation import decode_coco_rle  # noqa: E402
from lzt_perception.mask_refinement import (  # noqa: E402
    RGBDMaskRefinementConfig,
    binary_mask_to_rle,
    mask_bbox_xywh,
    refine_rgbd_mask,
)


def test_rgbd_refinement_grows_object_and_removes_depth_outlier() -> None:
    rgb = np.full((60, 80, 3), 30, dtype=np.uint8)
    ground_truth = np.zeros((60, 80), dtype=bool)
    ground_truth[20:42, 25:53] = True
    rgb[ground_truth] = [180, 90, 60]
    depth = np.ones((60, 80), dtype=np.float64)
    depth[ground_truth] = 0.5
    initial = np.zeros_like(ground_truth)
    initial[22:40, 27:51] = True
    initial[28:31, 35:39] = False
    initial[5:11, 5:11] = True
    rgb[5:11, 5:11] = [180, 90, 60]

    result = refine_rgbd_mask(
        rgb,
        depth,
        initial,
        RGBDMaskRefinementConfig(
            search_dilation_px=4,
            minimum_correction_fraction=0.0,
            maximum_correction_fraction=1.0,
            maximum_growth_fraction=1.0,
        ),
    )

    initial_iou = np.count_nonzero(initial & ground_truth) / np.count_nonzero(
        initial | ground_truth
    )
    refined_iou = np.count_nonzero(result.mask & ground_truth) / np.count_nonzero(
        result.mask | ground_truth
    )
    assert result.status == "refined"
    assert refined_iou > initial_iou
    assert not np.any(result.mask[5:11, 5:11])
    assert np.all(result.mask[28:31, 35:39])


def test_rgbd_refinement_falls_back_when_depth_is_missing() -> None:
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    depth = np.zeros((20, 30), dtype=np.float64)
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:20] = True

    result = refine_rgbd_mask(rgb, depth, mask)

    assert result.status == "insufficient_valid_depth"
    np.testing.assert_array_equal(result.mask, mask)


def test_refined_mask_rle_and_bbox_are_bop_compatible() -> None:
    mask = np.zeros((9, 13), dtype=bool)
    mask[2:7, 4:11] = True
    rle = binary_mask_to_rle(mask)
    np.testing.assert_array_equal(decode_coco_rle(rle), mask)
    assert mask_bbox_xywh(mask) == [4, 2, 7, 5]
