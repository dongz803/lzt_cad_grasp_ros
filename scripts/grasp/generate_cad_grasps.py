#!/usr/bin/env python3
"""Validate an object config and generate analytic CAD grasp candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)

from lzt_grasp_planning.cad_grasp import (  # noqa: E402
    generate_parallel_jaw_candidates_with_diagnostics,
    load_ascii_ply_geometry,
)
from lzt_common.object_config import load_object_config  # noqa: E402


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
    candidates, diagnostics = generate_parallel_jaw_candidates_with_diagnostics(
        config, geometry
    )
    payload = {
        "schema_version": 1,
        "object_id": config.cad_object.object_id,
        "object_frame": config.object_frame,
        "candidate_count": len(candidates),
        "generation_diagnostics": {
            "evaluated_frames": diagnostics.evaluated_frames,
            "accepted_before_limit": diagnostics.accepted_before_limit,
            "returned_candidates": diagnostics.returned_candidates,
            "rejection_counts": diagnostics.rejection_counts,
            "center_of_mass_object_m": (
                diagnostics.center_of_mass_object_m.tolist()
            ),
            "mesh_triangle_count": diagnostics.mesh_triangle_count,
            "quality_weights": {
                "antipodal": 0.4,
                "jaw_opening_margin": 0.15,
                "center_of_mass": 0.25,
                "edge_distance": 0.2,
            },
        },
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
                "contact_points_object_m": (
                    None
                    if candidate.contact_points_object_m is None
                    else candidate.contact_points_object_m.tolist()
                ),
                "contact_normals_object": (
                    None
                    if candidate.contact_normals_object is None
                    else candidate.contact_normals_object.tolist()
                ),
                "center_of_mass_offset_m": candidate.center_of_mass_offset_m,
                "minimum_edge_distance_m": candidate.minimum_edge_distance_m,
                "opening_margin_m": candidate.opening_margin_m,
                "finger_collision_free": candidate.finger_collision_free,
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
        f"wrote {len(candidates)} candidates to {args.output}; "
        f"rejections={diagnostics.rejection_counts}"
    )


if __name__ == "__main__":
    main()
