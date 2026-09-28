#!/usr/bin/env python3
"""Fetch the official BOP19 evaluation subset with a SHA-256 manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from fetch_foundpose_raw import fetch


REPOSITORY = "thodan/bop_toolkit"
DEFAULT_REF = "master"
FILES = (
    "LICENSE",
    "scripts/eval_bop19_pose.py",
    "scripts/eval_calc_errors.py",
    "scripts/eval_calc_scores.py",
    "bop_toolkit_lib/__init__.py",
    "bop_toolkit_lib/call_vsd_worker.py",
    "bop_toolkit_lib/common_utils.py",
    "bop_toolkit_lib/config.py",
    "bop_toolkit_lib/dataset_params.py",
    "bop_toolkit_lib/inout.py",
    "bop_toolkit_lib/misc.py",
    "bop_toolkit_lib/pose_error.py",
    "bop_toolkit_lib/pose_matching.py",
    "bop_toolkit_lib/score.py",
    "bop_toolkit_lib/transform.py",
    "bop_toolkit_lib/visibility.py",
    "bop_toolkit_lib/rendering/renderer.py",
    "bop_toolkit_lib/rendering/renderer_batch.py",
    "bop_toolkit_lib/rendering/renderer_cpp.py",
    "bop_toolkit_lib/rendering/renderer_py.py",
    "bop_toolkit_lib/rendering/renderer_vispy.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--ref", default=DEFAULT_REF)
    args = parser.parse_args()

    destination = args.destination.resolve()
    records = []
    for index, relative in enumerate(FILES, start=1):
        url = (
            "https://raw.githubusercontent.com/"
            f"{REPOSITORY}/{args.ref}/{relative}"
        )
        payload = fetch(url, destination / relative)
        records.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        print(f"[{index:02d}/{len(FILES):02d}] {relative} ({len(payload)} bytes)")

    manifest = {
        "repository": f"https://github.com/{REPOSITORY}",
        "ref": args.ref,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "raw.githubusercontent.com",
        "scope": "BOP19 VSD/MSSD/MSPD localization evaluation subset",
        "files": records,
    }
    manifest_path = destination / "SOURCE_SNAPSHOT.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
