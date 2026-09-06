import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.mask_evaluation import (  # noqa: E402
    decode_coco_rle,
    evaluate_bop_mask_sources,
    mask_pair_metrics,
    summarize_foundpose_results,
)


def encode_uncompressed(mask: np.ndarray) -> dict:
    flattened = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts = []
    current_value = 0
    run_length = 0
    for value in flattened:
        if int(value) == current_value:
            run_length += 1
        else:
            counts.append(run_length)
            current_value = int(value)
            run_length = 1
    counts.append(run_length)
    return {"size": list(mask.shape), "counts": counts}


def encode_compressed_counts(counts: list[int]) -> str:
    characters = []
    for index, count in enumerate(counts):
        value = count - counts[index - 2] if index > 2 else count
        more = True
        while more:
            code = value & 0x1F
            value >>= 5
            more = (code & 0x10 and value != -1) or (
                not code & 0x10 and value != 0
            )
            if more:
                code |= 0x20
            characters.append(chr(code + 48))
    return "".join(characters)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def build_bop_fixture(root: Path) -> tuple[Path, Path]:
    scene = root / "test/000001"
    scene.mkdir(parents=True)
    first = np.zeros((8, 10), dtype=bool)
    first[1:4, 1:4] = True
    second = np.zeros((8, 10), dtype=bool)
    second[3:6, 5:8] = True
    other = np.zeros((8, 10), dtype=bool)
    other[4:7, 1:4] = True
    (scene / "scene_gt.json").write_text(
        json.dumps(
            {
                "0": [
                    {"obj_id": 1},
                    {"obj_id": 1},
                    {"obj_id": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    (scene / "scene_gt_info.json").write_text(
        json.dumps(
            {
                "0": [
                    {"visib_fract": 0.9},
                    {"visib_fract": 0.8},
                    {"visib_fract": 0.95},
                ]
            }
        ),
        encoding="utf-8",
    )
    for index, mask in enumerate((first, second, other)):
        write_mask(scene / f"mask_visib/000000_{index:06d}.png", mask)
    targets = root / "test_targets_bop19.json"
    targets.write_text(
        json.dumps(
            [
                {"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 1},
                {"scene_id": 1, "im_id": 0, "obj_id": 2, "inst_count": 1},
            ]
        ),
        encoding="utf-8",
    )
    detections = root / "detections.json"
    detections.write_text(
        json.dumps(
            [
                {
                    "scene_id": 1,
                    "image_id": 0,
                    "category_id": 1,
                    "score": 0.9,
                    "segmentation": encode_uncompressed(second),
                },
                {
                    "scene_id": 1,
                    "image_id": 0,
                    "category_id": 1,
                    "score": 0.1,
                    "segmentation": encode_uncompressed(first),
                },
            ]
        ),
        encoding="utf-8",
    )
    return targets, detections


def test_coco_rle_decodes_both_representations() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    uncompressed = encode_uncompressed(mask)
    compressed = {
        "size": uncompressed["size"],
        "counts": encode_compressed_counts(uncompressed["counts"]),
    }
    np.testing.assert_array_equal(decode_coco_rle(uncompressed), mask)
    np.testing.assert_array_equal(decode_coco_rle(compressed), mask)


def test_mask_pair_metrics_penalize_shifted_mask() -> None:
    ground_truth = np.zeros((10, 10), dtype=bool)
    ground_truth[2:7, 2:7] = True
    shifted = np.zeros_like(ground_truth)
    shifted[3:8, 3:8] = True
    exact = mask_pair_metrics(ground_truth, ground_truth)
    displaced = mask_pair_metrics(shifted, ground_truth)
    assert exact["iou"] == 1.0
    assert exact["boundary_f1"] == 1.0
    assert 0.0 < displaced["iou"] < 1.0
    assert displaced["center_error_px"] > 0.0


def test_gt_and_detection_sources_share_targets_and_prediction_budget(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    targets, detections = build_bop_fixture(dataset)
    report, rows = evaluate_bop_mask_sources(
        dataset_root=dataset,
        targets_path=targets,
        split="test",
        detection_sources={"cnos": detections},
        opening_kernels={"cnos": 3},
    )
    assert report["target_group_count"] == 2
    assert report["target_instance_count"] == 2
    assert len(report["target_gt_count_mismatch_groups"]) == 1
    gt = report["sources"]["gt"]["overall"]
    cnos = report["sources"]["cnos"]["overall"]
    assert gt["mean_iou_per_target"] == 1.0
    assert gt["at_iou"]["0.50"]["recall"] == 1.0
    assert cnos["selected_prediction_count"] == 1
    assert cnos["discarded_prediction_count"] == 1
    assert cnos["missing_prediction_slot_count"] == 1
    assert cnos["mean_iou_per_target"] == 0.5
    assert cnos["at_iou"]["0.50"]["recall"] == 0.5
    object_one = report["sources"]["cnos"]["per_object"]["1"]
    assert object_one["mean_iou_per_target"] == 1.0
    matched = [
        row
        for row in rows
        if row["source"] == "cnos" and row["status"] == "matched"
    ]
    assert matched[0]["matched_gt_index"] == 1


def test_pose_summary_uses_best_hypothesis_and_common_denominator(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "poses"
    (result_root / "1").mkdir(parents=True)
    records = [
        {
            "scene_id": 1,
            "img_id": 0,
            "obj_id": 1,
            "inst_id": 0,
            "hypothesis_id": 0,
            "score": 0.2,
            "mssd": 20.0,
            "mssd_n": 0.3,
            "mspd": 10.0,
        },
        {
            "scene_id": 1,
            "img_id": 0,
            "obj_id": 1,
            "inst_id": 0,
            "hypothesis_id": 1,
            "score": 0.9,
            "mssd": 5.0,
            "mssd_n": 0.1,
            "mspd": 3.0,
        },
    ]
    (result_root / "1/estimated-poses.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    targets = [
        {"scene_id": 1, "image_id": 0, "object_id": 1, "instance_count": 1},
        {"scene_id": 1, "image_id": 0, "object_id": 2, "instance_count": 1},
    ]
    summary = summarize_foundpose_results(result_root, targets, [1, 2])
    assert summary["target_instances"] == 2
    assert summary["predicted_instances"] == 1
    assert summary["prediction_coverage"] == 0.5
    assert summary["missing_object_ids"] == [2]
    assert summary["per_object"]["1"]["mean_mssd_mm_on_predictions"] == 5.0


def test_gt_mask_inference_config_is_isolated_from_cnos_results() -> None:
    config_root = PROJECT_ROOT / "config/foundpose_bop19"
    cnos = json.loads((config_root / "tless_infer.json").read_text())["infer_opts"]
    gt = json.loads((config_root / "tless_infer_gt.json").read_text())["infer_opts"]
    assert cnos["use_detections"] is True
    assert gt["use_detections"] is False
    assert gt["version"] != cnos["version"]
    assert gt["repre_version"] == cnos["repre_version"]
    assert gt["final_pose_type"] == cnos["final_pose_type"] == "best_coarse"
