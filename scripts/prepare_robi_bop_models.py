#!/usr/bin/env python3
"""Convert official ROBI STL CAD files to millimetre BOP PLY models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.robi_bop import ROBI_OBJECT_IDS  # noqa: E402


def _diameter(vertices: np.ndarray, block_size: int = 1024) -> float:
    hull_vertices = vertices
    maximum_squared = 0.0
    for start in range(0, len(hull_vertices), block_size):
        block = hull_vertices[start:start + block_size]
        differences = block[:, None, :] - hull_vertices[None, :, :]
        maximum_squared = max(
            maximum_squared,
            float(np.max(np.sum(differences * differences, axis=2))),
        )
    return float(np.sqrt(maximum_squared))


def _gear_symmetries() -> list[list[float]]:
    transforms = []
    for index in range(1, 12):
        angle = 2.0 * np.pi * index / 12.0
        transform = np.eye(4)
        transform[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        transforms.append(transform.reshape(-1).tolist())
    return transforms


def main() -> None:
    workspace = PROJECT_ROOT.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=workspace / "datasets/robi_raw/models/_CAD_models",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workspace / "datasets/bop/robi/models",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=tuple(ROBI_OBJECT_IDS),
        default=list(ROBI_OBJECT_IDS),
    )
    args = parser.parse_args()

    try:
        import trimesh
    except ImportError as error:
        raise SystemExit("trimesh is required; use the foundpose_gpu Python") from error

    args.output_root.mkdir(parents=True, exist_ok=True)
    models_info_path = args.output_root / "models_info.json"
    models_info = (
        json.loads(models_info_path.read_text(encoding="utf-8"))
        if models_info_path.is_file()
        else {}
    )
    for object_name in args.objects:
        object_id = ROBI_OBJECT_IDS[object_name]
        source = args.model_root / f"{object_name}.stl"
        if not source.is_file():
            raise FileNotFoundError(source)
        mesh = trimesh.load(source, force="mesh", process=False)
        if mesh.is_empty:
            raise ValueError(f"empty ROBI model: {source}")
        destination = args.output_root / f"obj_{object_id:06d}.ply"
        mesh.export(destination, file_type="ply", encoding="ascii")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        hull_vertices = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
        lower = np.min(vertices, axis=0)
        upper = np.max(vertices, axis=0)
        info = {
            "diameter": _diameter(hull_vertices),
            "min_x": float(lower[0]),
            "min_y": float(lower[1]),
            "min_z": float(lower[2]),
            "size_x": float(upper[0] - lower[0]),
            "size_y": float(upper[1] - lower[1]),
            "size_z": float(upper[2] - lower[2]),
        }
        if object_name == "Gear":
            info["symmetries_discrete"] = _gear_symmetries()
        models_info[str(object_id)] = info
        print(
            f"{object_name}: {destination.name}, vertices={len(vertices)}, "
            f"diameter={info['diameter']:.3f} mm"
        )
    models_info_path.write_text(
        json.dumps(models_info, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
