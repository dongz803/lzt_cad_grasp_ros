#!/usr/bin/env python3
"""Transform refined OpenCV-camera object poses into the robot base frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.pose_gating import foundpose_record_transform  # noqa: E402
from lzt_cad_grasp_core.robot_frames import (  # noqa: E402
    load_robot_camera_calibration,
    transform_camera_object_to_base,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("input pose file must contain a JSON list")
    calibration = load_robot_camera_calibration(args.calibration)
    output = []
    for record in records:
        camera_object = foundpose_record_transform(record)
        # Column-vector convention: T_base_object = T_base_camera @ T_camera_object.
        base_object = transform_camera_object_to_base(camera_object, calibration)
        output.append(
            {
                **record,
                "base_frame": calibration.base_frame,
                "camera_frame": calibration.camera_frame,
                "transform_base_object": base_object.tolist(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calibration": calibration.to_dict(),
                "pose_count": len(output),
                "poses": output,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(output)} robot-base poses to {args.output.resolve()}")


if __name__ == "__main__":
    main()
