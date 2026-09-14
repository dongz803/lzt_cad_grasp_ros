#!/usr/bin/env python3
"""Export per-object FoundPose JSON files to one BOP19 submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.topk_evaluation import (  # noqa: E402
    aggregate_bop_image_runtimes,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=(
            project_root
            / "outputs/foundpose_bop19/inference/tless_bop19_standard_v1"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-start", type=int, default=1)
    parser.add_argument("--object-end", type=int, default=30)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    missing = []
    for object_id in range(args.object_start, args.object_end + 1):
        path = args.input_root / str(object_id) / "estimated-poses.json"
        if not path.is_file():
            missing.append(object_id)
            continue
        object_records = _load_json(path)
        if not isinstance(object_records, list):
            raise SystemExit(f"invalid result list: {path}")
        records.extend(object_records)
    if missing and not args.allow_missing:
        raise SystemExit(f"missing object results: {missing}")

    image_runtime = aggregate_bop_image_runtimes(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for record in records:
            image_key = (int(record["scene_id"]), int(record["img_id"]))
            rotation = np.asarray(record["R"], dtype=float).reshape(-1)
            translation = np.asarray(record["t"], dtype=float).reshape(-1)
            writer.writerow(
                [
                    record["scene_id"],
                    record["img_id"],
                    record["obj_id"],
                    record["score"],
                    " ".join(map(str, rotation.tolist())),
                    " ".join(map(str, translation.tolist())),
                    image_runtime[image_key],
                ]
            )
    print(
        f"wrote {len(records)} poses from "
        f"{args.object_end - args.object_start + 1 - len(missing)} objects "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
