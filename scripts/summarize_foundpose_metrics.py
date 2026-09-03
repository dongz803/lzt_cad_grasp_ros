#!/usr/bin/env python3
"""Summarize FoundPose MSSD/MSPD recalls before official VSD evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _average_recall(errors: list[float], denominator: int, thresholds: np.ndarray) -> float:
    if denominator <= 0:
        return float("nan")
    values = np.asarray(errors, dtype=np.float64)
    recalls = [
        float(np.count_nonzero(values < threshold)) / denominator
        for threshold in thresholds
    ]
    return float(np.mean(recalls))


def main() -> None:
    workspace = Path(__file__).resolve().parents[3]
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
    parser.add_argument(
        "--targets",
        type=Path,
        default=workspace / "datasets/bop/tless/test_targets_bop19.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    targets = _load_json(args.targets)
    target_instances = Counter()
    for target in targets:
        target_instances[int(target["obj_id"])] += int(target["inst_count"])

    records_by_object: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    missing_objects = []
    for object_id in range(1, 31):
        path = args.input_root / str(object_id) / "estimated-poses.json"
        if not path.is_file():
            missing_objects.append(object_id)
            continue
        records = _load_json(path)
        if not isinstance(records, list):
            raise SystemExit(f"invalid result list: {path}")
        records_by_object[object_id].extend(records)
    if missing_objects and not args.allow_partial:
        raise SystemExit(f"missing object results: {missing_objects}")

    mssd_thresholds = np.arange(0.05, 0.51, 0.05)
    image_width_relative = 720.0 / 640.0
    mspd_thresholds = np.arange(5.0, 51.0, 5.0) * image_width_relative
    per_object = {}
    for object_id, records in sorted(records_by_object.items()):
        best_by_instance: dict[tuple[int, int, int], dict[str, Any]] = {}
        for record in records:
            if "mssd_n" not in record or "mspd" not in record:
                continue
            key = (
                int(record["scene_id"]),
                int(record["img_id"]),
                int(record.get("inst_id", 0)),
            )
            current = best_by_instance.get(key)
            if current is None or float(record.get("score", 0.0)) > float(
                current.get("score", 0.0)
            ):
                best_by_instance[key] = record
        valid = list(best_by_instance.values())
        denominator = target_instances[object_id]
        per_object[str(object_id)] = {
            "target_instances": denominator,
            "predicted_instances": len(valid),
            "prediction_coverage": len(valid) / denominator if denominator else None,
            "ar_mssd": _average_recall(
                [float(record["mssd_n"]) for record in valid],
                denominator,
                mssd_thresholds,
            ),
            "ar_mspd": _average_recall(
                [float(record["mspd"]) for record in valid],
                denominator,
                mspd_thresholds,
            ),
            "mean_mssd_mm_on_predictions": (
                float(np.mean([float(record["mssd"]) for record in valid]))
                if valid
                else None
            ),
            "mean_mspd_px_on_predictions": (
                float(np.mean([float(record["mspd"]) for record in valid]))
                if valid
                else None
            ),
        }
    object_values = list(per_object.values())
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "tless",
        "protocol": "bop19_preliminary_mssd_mspd",
        "warning": (
            "This is a target-denominator MSSD/MSPD summary. It is not the "
            "official BOP19 composite AR until VSD and official matching pass."
        ),
        "complete_object_set": not missing_objects,
        "missing_object_ids": missing_objects,
        "mssd_normalized_thresholds": mssd_thresholds.tolist(),
        "mspd_pixel_thresholds": mspd_thresholds.tolist(),
        "ar_mssd": (
            float(np.mean([item["ar_mssd"] for item in object_values]))
            if object_values
            else None
        ),
        "ar_mspd": (
            float(np.mean([item["ar_mspd"] for item in object_values]))
            if object_values
            else None
        ),
        "per_object": per_object,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
