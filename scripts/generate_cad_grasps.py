#!/usr/bin/env python3
"""Validate an object config and generate analytic CAD grasp candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.cad_grasp import (  # noqa: E402
    generate_parallel_jaw_candidates,
    load_ascii_ply_geometry,
)
from lzt_cad_grasp_core.object_config import load_object_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_object_config(args.object_config)
    if config.cad_object.mesh_path.suffix.lower() != ".ply":
        raise SystemExit(
            "the dependency-free loader currently accepts ASCII PLY; "
            "convert OBJ/STL to PLY or add a trimesh-backed loader"
        )
    geometry = load_ascii_ply_geometry(
        config.cad_object.mesh_path,
        config.cad_object.units_to_m,
    )
    candidates = generate_parallel_jaw_candidates(config, geometry)
    payload = {
        "schema_version": 1,
        "object_id": config.cad_object.object_id,
        "object_frame": config.object_frame,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "grasp_id": candidate.grasp_id,
                "transform_object_grasp": (
                    candidate.transform_object_grasp.tolist()
                ),
                "jaw_opening_m": candidate.jaw_opening_m,
                "approach_distance_m": candidate.approach_distance_m,
                "antipodal_score": candidate.antipodal_score,
                "analytic_quality": candidate.analytic_quality,
            }
            for candidate in candidates
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"validated {config.cad_object.object_id}; "
        f"wrote {len(candidates)} candidates to {args.output}"
    )


if __name__ == "__main__":
    main()
