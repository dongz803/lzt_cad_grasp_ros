#!/usr/bin/env python3
"""Freeze a verified complete FoundPose BOP19 baseline into one manifest."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "outputs/foundpose_bop19"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument("--official-scores", type=Path, required=True)
    parser.add_argument("--bop-csv", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    required_json = {
        "dataset_validation": output_root / "dataset_validation.json",
        "template_profile": output_root / "profiles/gen_templates_summary.json",
        "representation_profile": output_root / "profiles/gen_repre_summary.json",
        "inference_profile": output_root / "profiles/infer_summary.json",
        "official_scores": args.official_scores.resolve(),
    }
    documents = {}
    for name, path in required_json.items():
        if not path.is_file():
            raise SystemExit(f"cannot freeze; missing {name}: {path}")
        documents[name] = _load_json(path)
    if documents["dataset_validation"].get("complete") is not True:
        raise SystemExit("cannot freeze; dataset validation is incomplete")
    for stage in ("template_profile", "representation_profile", "inference_profile"):
        summary = documents[stage]
        if summary.get("completed_count") != 30 or summary.get("requested_count") != 30:
            raise SystemExit(f"cannot freeze; {stage} is not complete for 30 objects")

    score_keys = {
        "bop19_average_recall_vsd",
        "bop19_average_recall_mssd",
        "bop19_average_recall_mspd",
        "bop19_average_recall",
        "bop19_average_time_per_image",
    }
    if not score_keys.issubset(documents["official_scores"]):
        raise SystemExit("cannot freeze; official BOP19 score keys are incomplete")
    bop_csv = args.bop_csv.resolve()
    if not bop_csv.is_file():
        raise SystemExit(f"cannot freeze; result CSV is missing: {bop_csv}")
    with bop_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_columns = {"scene_id", "im_id", "obj_id", "score", "R", "t", "time"}
    if not rows or set(rows[0]) != expected_columns:
        raise SystemExit("cannot freeze; BOP CSV is empty or has invalid columns")
    csv_object_ids = {int(row["obj_id"]) for row in rows}
    if csv_object_ids != set(range(1, 31)):
        raise SystemExit(
            "cannot freeze; BOP CSV does not contain all T-LESS object IDs"
        )

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "cannot freeze; project Git worktree is dirty; commit the verified "
            "implementation first"
        )

    tracked_inputs = [
        project_root / "config/foundpose_bop19/tless_templates.json",
        project_root / "config/foundpose_bop19/tless_repre.json",
        project_root / "config/foundpose_bop19/tless_infer.json",
        project_root / "third_party/patches/foundpose_topk_coarse.patch",
        project_root / "third_party/patches/foundpose_cpu_kmeans.patch",
        project_root / "third_party/patches/foundpose_repre_version.patch",
        project_root / "scripts/run_bop19_stage.py",
        project_root / "scripts/run_foundpose.py",
        project_root / "scripts/export_foundpose_bop.py",
        project_root / "scripts/evaluate_bop19_official.py",
        project_root / "scripts/freeze_foundpose_baseline.py",
    ]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "FoundPose",
        "dataset_protocol": "T-LESS BOP19 localization",
        "project_git_commit": git_commit,
        "project_git_worktree_clean": True,
        "official_scores": documents["official_scores"],
        "resource_profiles": {
            key: documents[key]
            for key in (
                "template_profile",
                "representation_profile",
                "inference_profile",
            )
        },
        "artifacts": {
            "bop_csv": {"path": str(bop_csv), "sha256": _sha256(bop_csv)},
            "bop_csv_pose_count": len(rows),
            "official_scores": {
                "path": str(required_json["official_scores"]),
                "sha256": _sha256(required_json["official_scores"]),
            },
        },
        "tracked_input_sha256": {
            str(path.relative_to(project_root)): _sha256(path)
            for path in tracked_inputs
        },
    }
    manifest_path = output_root / "FROZEN_BASELINE.json"
    if manifest_path.exists():
        raise SystemExit(
            f"refusing to overwrite immutable baseline manifest: {manifest_path}"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"frozen baseline: {manifest_path}")


if __name__ == "__main__":
    main()
