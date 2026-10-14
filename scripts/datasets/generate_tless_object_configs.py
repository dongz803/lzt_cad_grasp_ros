#!/usr/bin/env python3
"""Generate project object YAMLs from authoritative T-LESS model metadata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_MODELS_INFO = (
    WORKSPACE_ROOT / "datasets/bop/tless/models_cad/models_info.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "lzt_description/config/objects"


def _float_list(values: Any, *, expected_length: int, field: str) -> list[float]:
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(f"{field} must contain {expected_length} values")
    return [float(value) for value in values]


def _discrete_transform(values: Any, *, object_id: int) -> list[list[float]]:
    flat = _float_list(
        values,
        expected_length=16,
        field=f"object {object_id} discrete symmetry",
    )
    matrix = [flat[index:index + 4] for index in range(0, 16, 4)]
    for row in range(3):
        matrix[row][3] *= 0.001  # BOP model metadata uses millimetres.
    return matrix


def _symmetry_payload(model_info: dict[str, Any], object_id: int) -> dict[str, Any]:
    continuous = model_info.get("symmetries_continuous", [])
    discrete = model_info.get("symmetries_discrete", [])
    if not isinstance(continuous, list) or not isinstance(discrete, list):
        raise ValueError(f"object {object_id} symmetry metadata must be lists")
    if len(continuous) > 1:
        raise ValueError(
            f"object {object_id} has {len(continuous)} continuous symmetries; "
            "the project schema currently supports one continuous axis"
        )

    transforms = [
        _discrete_transform(values, object_id=object_id) for values in discrete
    ]
    if continuous:
        entry = continuous[0]
        if not isinstance(entry, dict):
            raise ValueError(
                f"object {object_id} continuous symmetry must be a mapping"
            )
        axis = _float_list(
            entry.get("axis"),
            expected_length=3,
            field=f"object {object_id} continuous axis",
        )
        offset_mm = _float_list(
            entry.get("offset", [0.0, 0.0, 0.0]),
            expected_length=3,
            field=f"object {object_id} continuous offset",
        )
        result: dict[str, Any] = {
            "type": "continuous_axis",
            "axis_xyz": axis,
            "offset_m": [value * 0.001 for value in offset_mm],
        }
        if transforms:
            result["transforms"] = transforms
        return result
    if transforms:
        return {"type": "discrete", "transforms": transforms}
    return {"type": "none"}


def _object_payload(model_info: dict[str, Any], object_id: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "object_id": f"tless_{object_id:06d}",
        "display_name": f"T-LESS 工业物料 {object_id}",
        "cad": {
            "mesh": f"${{BOP_PATH}}/tless/models_cad/obj_{object_id:06d}.ply",
            "units_to_m": 0.001,
            "object_frame": "bop_cad_origin",
        },
        "symmetry": _symmetry_payload(model_info, object_id),
        "perception": {
            "roi_source": "instance_mask",
            "expected_modalities": ["rgb"],
        },
        "gripper": {
            "type": "parallel_jaw",
            "minimum_opening_m": 0.005,
            "maximum_opening_m": 0.085,
            "finger_clearance_m": 0.002,
            "finger_thickness_m": 0.008,
            "finger_width_m": 0.022,
            "finger_depth_m": 0.0375,
        },
        "grasp_generation": {
            "approach_distance_m": 0.08,
            "surface_band_fraction": 0.08,
            "minimum_antipodal_score": 0.3,
            "minimum_edge_distance_m": 0.001,
            "maximum_center_of_mass_offset_m": 0.02,
            "lateral_sample_fraction": 0.2,
            "maximum_candidates": 12,
        },
        "task": {
            "placement_position_tolerance_m": 0.005,
            "placement_angle_tolerance_deg": 5.0,
            "functional_surfaces": [],
            "forbidden_grasp_surfaces": [],
        },
    }


def _load_models_info(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("models_info.json must contain an object mapping")
    return payload


def _equivalent(actual: Any, expected: Any) -> bool:
    """Compare parsed YAML payloads while tolerating decimal round-off only."""

    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equivalent(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equivalent(left, right) for left, right in zip(actual, expected)
        )
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15
        )
    return actual == expected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate missing T-LESS object YAMLs from models_info.json while "
            "refusing to overwrite or silently change existing configurations."
        )
    )
    parser.add_argument("--models-info", type=Path, default=DEFAULT_MODELS_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--object-start", type=int, default=1)
    parser.add_argument("--object-end", type=int, default=30)
    parser.add_argument(
        "--write-missing",
        action="store_true",
        help="write missing YAML files; without this flag only validate and report",
    )
    args = parser.parse_args()

    if not 1 <= args.object_start <= args.object_end <= 30:
        raise SystemExit("object range must satisfy 1 <= start <= end <= 30")
    if not args.models_info.is_file():
        raise SystemExit(f"models_info.json is missing: {args.models_info}")

    models_info = _load_models_info(args.models_info)
    written: list[int] = []
    matched: list[int] = []
    missing: list[int] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for object_id in range(args.object_start, args.object_end + 1):
        model_info = models_info.get(str(object_id))
        if not isinstance(model_info, dict):
            raise SystemExit(f"models_info.json has no object {object_id}")
        expected = _object_payload(model_info, object_id)
        output = args.output_dir / f"tless_{object_id:06d}.yaml"
        if output.exists():
            actual = yaml.safe_load(output.read_text(encoding="utf-8"))
            if not _equivalent(actual, expected):
                raise SystemExit(
                    f"existing config differs from authoritative metadata/template: {output}"
                )
            matched.append(object_id)
            continue

        missing.append(object_id)
        if not args.write_missing:
            continue
        output.write_text(
            yaml.safe_dump(
                expected,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        written.append(object_id)

    print(f"matching existing configs: {matched}")
    if args.write_missing:
        print(f"written missing configs: {written}")
    else:
        print(f"missing configs (dry run): {missing}")


if __name__ == "__main__":
    main()
