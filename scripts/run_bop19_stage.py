#!/usr/bin/env python3
"""Run one FoundPose BOP19 stage per object with resumable profiles."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_CONFIGS = {
    "gen_templates": PROJECT_ROOT / "config/foundpose_bop19/tless_templates.json",
    "gen_repre": PROJECT_ROOT / "config/foundpose_bop19/tless_repre.json",
    "infer": PROJECT_ROOT / "config/foundpose_bop19/tless_infer.json",
}
OPTION_KEYS = {
    "gen_templates": "gen_templates_opts",
    "gen_repre": "gen_repre_opts",
    "infer": "infer_opts",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _object_output(
    output_root: Path, stage: str, opts: dict[str, Any], object_id: int
) -> Path:
    if stage == "gen_templates":
        return (
            output_root
            / "templates"
            / opts["version"]
            / opts["object_dataset"]
            / str(object_id)
        )
    if stage == "gen_repre":
        return (
            output_root
            / "object_repre"
            / opts["object_dataset"]
            / opts["version"]
            / str(object_id)
        )
    return (
        output_root
        / "inference"
        / f"{opts['object_dataset']}_{opts['version']}"
        / str(object_id)
    )


def _is_complete(
    output_root: Path, stage: str, opts: dict[str, Any], object_id: int
) -> bool:
    output = _object_output(output_root, stage, opts, object_id)
    if stage == "gen_templates":
        metadata_path = output / "metadata.json"
        if not metadata_path.is_file():
            return False
        metadata = _load_json(metadata_path)
        minimum_count = (
            int(opts["min_num_viewpoints"])
            * int(opts["num_inplane_rotations"])
            * int(opts["images_per_view"])
        )
        if not isinstance(metadata, list) or len(metadata) < minimum_count:
            return False
        return all(
            len(list((output / kind).glob("template_*.png"))) == len(metadata)
            for kind in ("rgb", "depth", "mask")
        )
    if stage == "gen_repre":
        representation = output / "repre.pth"
        return representation.is_file() and representation.stat().st_size > 0
    result = output / "estimated-poses.json"
    if not result.is_file():
        return False
    return isinstance(_load_json(result), list)


def _write_summary(
    output_root: Path,
    stage: str,
    opts: dict[str, Any],
    requested_ids: list[int],
) -> Path:
    profile_dir = output_root / "profiles" / stage
    profiles = []
    for path in sorted(profile_dir.glob("object_*.json")):
        try:
            profile = _load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        profiles.append(profile)
    completed_ids = [
        object_id
        for object_id in requested_ids
        if _is_complete(output_root, stage, opts, object_id)
    ]
    successful_profiles = [item for item in profiles if item.get("exit_code") == 0]
    peak_values = [
        item["gpu_memory_peak_mib"]
        for item in successful_profiles
        if item.get("gpu_memory_peak_mib") is not None
    ]
    summary = {
        "schema_version": 1,
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_object_ids": requested_ids,
        "completed_object_ids": completed_ids,
        "completed_count": len(completed_ids),
        "requested_count": len(requested_ids),
        "successful_profile_count": len(successful_profiles),
        "total_profiled_duration_s": sum(
            float(item.get("duration_s", 0.0)) for item in successful_profiles
        ),
        "maximum_gpu_memory_peak_mib": max(peak_values) if peak_values else None,
        "gpu_memory_measurement_scope": "full_device",
        "profiles": [str(path) for path in sorted(profile_dir.glob("object_*.json"))],
    }
    path = output_root / "profiles" / f"{stage}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(STAGE_CONFIGS))
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/foundpose_bop19",
    )
    parser.add_argument("--object-start", type=int, default=1)
    parser.add_argument("--object-end", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if not 1 <= args.object_start <= args.object_end <= 30:
        raise SystemExit("T-LESS object range must satisfy 1 <= start <= end <= 30")
    maximum_workers = 3 if args.stage == "gen_templates" else 2
    if not 1 <= args.workers <= maximum_workers:
        raise SystemExit(
            f"--workers must lie in [1, {maximum_workers}] for {args.stage}"
        )
    if args.stage == "gen_repre" and args.workers != 1:
        raise SystemExit("gen_repre must run with --workers=1 on the 6 GiB GPU")
    config_path = (args.config or STAGE_CONFIGS[args.stage]).resolve()
    base_config = _load_json(config_path)
    option_key = OPTION_KEYS[args.stage]
    if set(base_config) != {option_key}:
        raise SystemExit(f"expected one top-level key: {option_key}")
    base_opts = base_config[option_key]
    output_root = args.output_root.resolve()
    object_ids = list(range(args.object_start, args.object_end + 1))
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for object_id in object_ids:
        if not args.force and _is_complete(
            output_root, args.stage, base_opts, object_id
        ):
            print(f"[{object_id:02d}] complete; skipping")
            continue
        object_config = deepcopy(base_config)
        object_config[option_key]["object_lids"] = [object_id]
        run_config = (
            output_root
            / "run_configs"
            / args.stage
            / f"object_{object_id:06d}.json"
        )
        run_config.parent.mkdir(parents=True, exist_ok=True)
        run_config.write_text(
            json.dumps(object_config, indent=2) + "\n",
            encoding="utf-8",
        )
        profile_path = (
            output_root
            / "profiles"
            / args.stage
            / f"object_{object_id:06d}.json"
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/profile_foundpose.py"),
            args.stage,
            "--opts-path",
            str(run_config),
            "--profile-path",
            str(profile_path),
        ]
        print(f"[{object_id:02d}] {' '.join(command)}")
        if args.dry_run:
            continue
        environment = os.environ.copy()
        environment["BOP_OUTPUT_PATH"] = str(output_root)
        tasks.append((object_id, command, environment))

    def execute(task: tuple[int, list[str], dict[str, str]]) -> tuple[int, int]:
        object_id, command, environment = task
        return object_id, subprocess.run(command, env=environment).returncode

    failures = []
    if args.workers == 1:
        completed = (execute(task) for task in tasks)
        for object_id, return_code in completed:
            _write_summary(output_root, args.stage, base_opts, object_ids)
            if return_code != 0:
                failures.append((object_id, return_code))
                break
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(execute, task) for task in tasks]
            for future in as_completed(futures):
                object_id, return_code = future.result()
                _write_summary(output_root, args.stage, base_opts, object_ids)
                if return_code != 0:
                    failures.append((object_id, return_code))

    summary_path = _write_summary(
        output_root,
        args.stage,
        base_opts,
        object_ids,
    )
    print(f"summary: {summary_path}")
    if failures:
        raise SystemExit(f"failed objects: {failures}")


if __name__ == "__main__":
    main()
