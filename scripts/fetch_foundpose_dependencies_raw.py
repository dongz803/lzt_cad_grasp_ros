#!/usr/bin/env python3
"""Fetch the exact DINOv2/BOP Toolkit subset pinned by FoundPose."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fetch_foundpose_raw import fetch


DEPENDENCIES = {
    "dinov2": {
        "repository": "facebookresearch/dinov2",
        "ref": "e1277af2ba9496fbadf7aec6eba56e8d882d1e35",
        "files": (
            "LICENSE",
            "dinov2/__init__.py",
            "dinov2/hub/__init__.py",
            "dinov2/hub/backbones.py",
            "dinov2/hub/utils.py",
            "dinov2/layers/__init__.py",
            "dinov2/layers/attention.py",
            "dinov2/layers/block.py",
            "dinov2/layers/dino_head.py",
            "dinov2/layers/drop_path.py",
            "dinov2/layers/layer_scale.py",
            "dinov2/layers/mlp.py",
            "dinov2/layers/patch_embed.py",
            "dinov2/layers/swiglu_ffn.py",
            "dinov2/models/__init__.py",
            "dinov2/models/vision_transformer.py",
        ),
    },
    "bop_toolkit": {
        "repository": "thodan/bop_toolkit",
        "ref": "e7ba9f2389d48d721856df8147fb133c0145e99c",
        "files": (
            "LICENSE",
            "bop_toolkit_lib/__init__.py",
            "bop_toolkit_lib/config.py",
            "bop_toolkit_lib/dataset_params.py",
            "bop_toolkit_lib/inout.py",
            "bop_toolkit_lib/misc.py",
            "bop_toolkit_lib/pycoco_utils.py",
            "bop_toolkit_lib/transform.py",
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundpose-root", type=Path, required=True)
    args = parser.parse_args()
    external_root = args.foundpose_root.resolve() / "external"

    for dependency_name, spec in DEPENDENCIES.items():
        destination = external_root / dependency_name
        records = []
        files = spec["files"]
        for index, relative in enumerate(files, start=1):
            url = (
                "https://raw.githubusercontent.com/"
                f"{spec['repository']}/{spec['ref']}/{relative}"
            )
            payload = fetch(url, destination / relative)
            records.append(
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            print(
                f"{dependency_name} [{index:02d}/{len(files):02d}] "
                f"{relative} ({len(payload)} bytes)"
            )

        manifest = {
            "repository": f"https://github.com/{spec['repository']}",
            "ref": spec["ref"],
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "transport": "raw.githubusercontent.com",
            "scope": "runtime subset required by the FoundPose coarse pipeline",
            "files": records,
        }
        (destination / "SOURCE_SNAPSHOT.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
