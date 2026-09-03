#!/usr/bin/env python3
"""Create an overview and a 6D-pose overlay from FoundPose result JSON."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = PROJECT_ROOT / "third_party/bop_toolkit_eval"
sys.path.insert(0, str(TOOLKIT_ROOT))

from bop_toolkit_lib import inout, misc, pose_error  # noqa: E402
from bop_toolkit_lib.rendering import renderer  # noqa: E402


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(record["R"], dtype=np.float64).reshape(3, 3),
        np.asarray(record["t"], dtype=np.float64).reshape(3, 1),
    )


def _draw_axes(
    image: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: np.ndarray,
    length_mm: float,
) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [length_mm, 0.0, 0.0],
            [0.0, length_mm, 0.0],
            [0.0, 0.0, length_mm],
        ],
        dtype=np.float64,
    )
    camera_points = (rotation @ points.T + translation).T
    pixels = (intrinsics @ camera_points.T).T
    pixels = np.rint(pixels[:, :2] / pixels[:, 2:3]).astype(int)
    origin = tuple(pixels[0])
    cv2.line(image, origin, tuple(pixels[1]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(image, origin, tuple(pixels[2]), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(image, origin, tuple(pixels[3]), (255, 0, 0), 3, cv2.LINE_AA)


def _overview(
    records: list[dict], object_id: int, dataset_name: str, output: Path
) -> None:
    scores = np.asarray([float(item["score"]) for item in records])
    mssd = np.asarray([float(item["mssd"]) for item in records])
    mspd = np.asarray([float(item["mspd"]) for item in records])
    scene_counts = Counter(int(item["scene_id"]) for item in records)
    mssd_limit = float(np.percentile(mssd, 95))
    mspd_limit = float(np.percentile(mspd, 95))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].hist(np.clip(mssd, 0.0, mssd_limit), bins=30, color="#4c78a8")
    axes[0, 0].set_title(f"MSSD distribution (clipped at P95={mssd_limit:.1f} mm)")
    axes[0, 0].set_xlabel("MSSD (mm)")
    axes[0, 0].set_ylabel("Pose count")

    axes[0, 1].hist(np.clip(mspd, 0.0, mspd_limit), bins=30, color="#f58518")
    axes[0, 1].set_title(f"MSPD distribution (clipped at P95={mspd_limit:.1f} px)")
    axes[0, 1].set_xlabel("MSPD (px)")
    axes[0, 1].set_ylabel("Pose count")

    axes[1, 0].scatter(scores, np.clip(mssd, 0.0, mssd_limit), s=18, alpha=0.65)
    axes[1, 0].set_title("FoundPose score vs. MSSD")
    axes[1, 0].set_xlabel("FoundPose score")
    axes[1, 0].set_ylabel("MSSD (mm, clipped at P95)")
    axes[1, 0].grid(alpha=0.25)

    scenes = sorted(scene_counts)
    axes[1, 1].bar([str(scene) for scene in scenes], [scene_counts[x] for x in scenes])
    axes[1, 1].set_title("Valid pose outputs per test scene")
    axes[1, 1].set_xlabel("Scene ID")
    axes[1, 1].set_ylabel("Pose count")

    fig.suptitle(
        f"{dataset_name.upper()} object {object_id} — FoundPose 6D pose inference\n"
        f"N={len(records)} | median MSSD={np.median(mssd):.2f} mm | "
        f"median MSPD={np.median(mspd):.2f} px | mean score={np.mean(scores):.3f}",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _overlay(
    records: list[dict],
    bop_root: Path,
    dataset_name: str,
    split: str,
    model_dir: str,
    object_id: int,
    output: Path,
) -> dict:
    record = min(records, key=lambda item: float(item["mssd"]))
    scene_id = int(record["scene_id"])
    image_id = int(record["img_id"])
    dataset = bop_root / dataset_name
    scene = dataset / split / f"{scene_id:06d}"
    image = cv2.imread(str(scene / "rgb" / f"{image_id:06d}.png"))
    if image is None:
        raise FileNotFoundError(scene / "rgb" / f"{image_id:06d}.png")
    camera = _load_json(scene / "scene_camera.json")[str(image_id)]
    intrinsics = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    all_ground_truth = _load_json(scene / "scene_gt.json")[str(image_id)]
    ground_truth = [item for item in all_ground_truth if int(item["obj_id"]) == object_id]
    model_path = dataset / model_dir / f"obj_{object_id:06d}.ply"
    model = inout.load_ply(str(model_path))["pts"]
    model_info = _load_json(dataset / model_dir / "models_info.json")[str(object_id)]
    symmetries = misc.get_symmetry_transformations(model_info, 0.01)
    rotation_est, translation_est = _pose(record)

    def error_to_gt(item: dict) -> float:
        rotation_gt = np.asarray(item["cam_R_m2c"]).reshape(3, 3)
        translation_gt = np.asarray(item["cam_t_m2c"]).reshape(3, 1)
        return float(
            pose_error.mssd(
                rotation_est,
                translation_est,
                rotation_gt,
                translation_gt,
                model,
                symmetries,
            )
        )

    instance_id = int(record.get("inst_id", -1))
    if 0 <= instance_id < len(all_ground_truth) and int(
        all_ground_truth[instance_id]["obj_id"]
    ) == object_id:
        matched_gt = all_ground_truth[instance_id]
    else:
        matched_gt = min(ground_truth, key=error_to_gt)
    rotation_gt = np.asarray(matched_gt["cam_R_m2c"]).reshape(3, 3)
    translation_gt = np.asarray(matched_gt["cam_t_m2c"]).reshape(3, 1)

    height, width = image.shape[:2]
    render = renderer.create_renderer(width, height, "vispy", mode="depth")
    render.add_object(object_id, str(model_path))
    parameters = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    depth_est = render.render_object(
        object_id, rotation_est, translation_est, *parameters
    )["depth"]
    depth_gt = render.render_object(
        object_id, rotation_gt, translation_gt, *parameters
    )["depth"]
    contours_gt, _ = cv2.findContours(
        (depth_gt > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_est, _ = cv2.findContours(
        (depth_est > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours_gt, -1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.drawContours(image, contours_est, -1, (0, 255, 0), 3, cv2.LINE_AA)
    _draw_axes(image, rotation_est, translation_est, intrinsics, length_mm=25.0)
    cv2.rectangle(image, (10, 10), (710, 82), (20, 20, 20), -1)
    cv2.putText(
        image,
        (
            f"scene={scene_id} image={image_id} inst={record['inst_id']} "
            f"score={float(record['score']):.3f}"
        ),
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        (
            f"MSSD={float(record['mssd']):.2f} mm  "
            f"MSPD={float(record['mspd']):.2f} px  GT=red Pred=green"
        ),
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"failed to write {output}")
    return record


def main() -> None:
    workspace = PROJECT_ROOT.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--bop-root", type=Path, default=workspace / "datasets/bop")
    parser.add_argument("--dataset", default="tless")
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--model-dir", default="models_eval")
    parser.add_argument("--overview-output", type=Path, required=True)
    parser.add_argument("--overlay-output", type=Path, required=True)
    args = parser.parse_args()

    records = _load_json(args.results)
    if not isinstance(records, list) or not records:
        raise SystemExit("result JSON must be a non-empty list")
    _overview(records, args.object_id, args.dataset, args.overview_output)
    example = _overlay(
        records,
        args.bop_root.resolve(),
        args.dataset,
        args.split,
        args.model_dir,
        args.object_id,
        args.overlay_output,
    )
    print(f"overview: {args.overview_output}")
    print(f"best-pose overlay: {args.overlay_output}")
    print(
        "selected: "
        f"scene={example['scene_id']} image={example['img_id']} "
        f"MSSD={float(example['mssd']):.3f} mm "
        f"MSPD={float(example['mspd']):.3f} px"
    )


if __name__ == "__main__":
    main()
