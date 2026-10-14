#!/usr/bin/env python3
"""Convert selected real ROBI RealSense scenes into a BOP-compatible split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)

from lzt_benchmark.robi_bop import (  # noqa: E402
    ROBI_OBJECT_IDS,
    convert_robi_realsense_to_bop,
)


def main() -> None:
    workspace = PROJECT_ROOT.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=workspace / "datasets/robi_raw/scenes",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workspace / "datasets/bop/robi",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=tuple(ROBI_OBJECT_IDS),
        default=["Chrome_screw", "Gear"],
    )
    parser.add_argument("--scene-ids", nargs="+", type=int)
    parser.add_argument("--maximum-views-per-scene", type=int)
    parser.add_argument("--pose-units-to-mm", type=float, default=1.0)
    args = parser.parse_args()

    summary = convert_robi_realsense_to_bop(
        source_root=args.source_root,
        output_root=args.output_root,
        object_names=args.objects,
        scene_ids=args.scene_ids,
        maximum_views_per_scene=args.maximum_views_per_scene,
        pose_units_to_mm=args.pose_units_to_mm,
    )
    summary_path = args.output_root / "conversion_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
