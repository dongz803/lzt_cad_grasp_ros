#!/usr/bin/env python3
"""Evaluate GT and CNOS masks under one BOP19 target protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)

from lzt_perception.mask_evaluation import (  # noqa: E402
    evaluate_bop_mask_sources,
    summarize_foundpose_results,
)


MATCH_COLUMNS = [
    "source",
    "scene_id",
    "image_id",
    "object_id",
    "expected_instances",
    "eligible_gt_instances",
    "source_available_predictions",
    "source_selected_predictions",
    "prediction_rank",
    "prediction_index",
    "prediction_score",
    "matched_gt_index",
    "gt_visibility",
    "status",
    "iou",
    "pixel_precision",
    "pixel_recall",
    "boundary_f1",
    "center_error_px",
    "area_ratio",
]


def _named_paths(values: list[str], argument: str) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{argument} requires NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name or name == "gt":
            raise SystemExit(f"invalid source name for {argument}: {name!r}")
        if name in parsed:
            raise SystemExit(f"duplicate source name for {argument}: {name}")
        parsed[name] = Path(path).expanduser()
    return parsed


def _named_integers(values: list[str], argument: str) -> dict[str, int]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{argument} requires NAME=INTEGER, got: {value}")
        name, raw_number = value.split("=", 1)
        try:
            number = int(raw_number)
        except ValueError as error:
            raise SystemExit(f"invalid integer for {argument}: {value}") from error
        if name in parsed:
            raise SystemExit(f"duplicate source name for {argument}: {name}")
        parsed[name] = number
    return parsed


def main() -> None:
    default_dataset_root = WORKSPACE_ROOT / "datasets/bop/tless"
    parser = argparse.ArgumentParser(
        description=(
            "Compare visible GT masks and BOP-format detection masks using the "
            "same BOP19 targets, prediction budget, and metrics."
        )
    )
    parser.add_argument("--dataset-name", default="tless")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument(
        "--targets",
        type=Path,
        default=default_dataset_root / "test_targets_bop19.json",
    )
    parser.add_argument(
        "--detections",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "BOP/COCO detection JSON; repeat for improved sources. If omitted, "
            "the workspace CNOS-FastSAM T-LESS file is used as source 'cnos'."
        ),
    )
    parser.add_argument(
        "--pose-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional FoundPose output root containing <object>/estimated-poses.json."
        ),
    )
    parser.add_argument(
        "--source-opening",
        action="append",
        default=[],
        metavar="NAME=KERNEL",
        help=(
            "Morphological opening applied before evaluation. CNOS defaults to 3, "
            "matching the current FoundPose input path; use cnos=0 for raw masks."
        ),
    )
    parser.add_argument("--object-ids", type=int, nargs="+")
    parser.add_argument("--minimum-visibility", type=float, default=0.1)
    parser.add_argument("--prediction-factor", type=float, default=1.0)
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75]
    )
    parser.add_argument("--boundary-tolerance-px", type=int, default=2)
    parser.add_argument("--limit-target-groups", type=int)
    parser.add_argument("--image-width", type=int, default=720)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/mask_evaluation/tless_gt_cnos",
    )
    args = parser.parse_args()

    detections = _named_paths(args.detections, "--detections")
    if not detections:
        detections = {
            "cnos": (
                WORKSPACE_ROOT
                / "datasets/bop/detections/cnos-fastsam"
                / "cnos-fastsam_tless-test.json"
            )
        }
    pose_roots = _named_paths(args.pose_root, "--pose-root")
    opening_kernels = _named_integers(
        args.source_opening, "--source-opening"
    )
    if "cnos" in detections and "cnos" not in opening_kernels:
        opening_kernels["cnos"] = 3

    report, match_rows = evaluate_bop_mask_sources(
        dataset_root=args.dataset_root,
        targets_path=args.targets,
        split=args.split,
        detection_sources=detections,
        object_ids=args.object_ids,
        minimum_visibility=args.minimum_visibility,
        prediction_factor=args.prediction_factor,
        opening_kernels=opening_kernels,
        iou_thresholds=args.iou_thresholds,
        boundary_tolerance_px=args.boundary_tolerance_px,
        limit_target_groups=args.limit_target_groups,
        dataset_name=args.dataset_name,
    )
    report["pose_results"] = {}
    for source_name, pose_root in pose_roots.items():
        report["pose_results"][source_name] = summarize_foundpose_results(
            pose_root,
            report["targets"],
            report["selected_object_ids"],
            image_width=args.image_width,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    matches_path = output_dir / "matches.csv"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with matches_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MATCH_COLUMNS)
        writer.writeheader()
        writer.writerows(match_rows)

    concise = {
        name: {
            "mean_iou_per_target": source["overall"]["mean_iou_per_target"],
            "boundary_f1_per_target": source["overall"][
                "mean_boundary_f1_per_target"
            ],
            "recall_at_iou_0.50": source["overall"]["at_iou"]["0.50"][
                "recall"
            ],
            "recall_at_iou_0.75": source["overall"]["at_iou"]["0.75"][
                "recall"
            ],
        }
        for name, source in report["sources"].items()
    }
    print(json.dumps(concise, indent=2))
    print(f"summary: {summary_path}")
    print(f"matches: {matches_path}")


if __name__ == "__main__":
    main()
