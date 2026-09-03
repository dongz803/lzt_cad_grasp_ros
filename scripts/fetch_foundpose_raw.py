#!/usr/bin/env python3
"""Fetch the small FoundPose source tree from GitHub's official raw endpoint.

This is a network fallback for machines where git/codeload.github.com is
unreachable but raw.githubusercontent.com works. Third-party source remains
ignored by the parent repository. A SHA-256 manifest is written beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "facebookresearch/foundpose"
DEFAULT_REF = "main"
FILES = (
    ".gitmodules",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "conda_foundpose_gpu.yaml",
    "conda_foundpose_mps.yaml",
    "configs/gen_repre/lmo.json",
    "configs/gen_templates/lmo.json",
    "configs/infer/lmo.json",
    "scripts/gen_repre.py",
    "scripts/gen_templates.py",
    "scripts/infer.py",
    "scripts/prepare_bop_submission.py",
    "utils/__init__.py",
    "utils/cluster_util.py",
    "utils/config_util.py",
    "utils/corresp_util.py",
    "utils/data_util.py",
    "utils/dinov2_utils.py",
    "utils/eval_errors.py",
    "utils/eval_util.py",
    "utils/feature_util.py",
    "utils/geometry.py",
    "utils/html_util.py",
    "utils/infer_pose_util.py",
    "utils/json_util.py",
    "utils/knn_util.py",
    "utils/logging.py",
    "utils/misc.py",
    "utils/pnp_util.py",
    "utils/preprocess_util.py",
    "utils/projector_util.py",
    "utils/render_vis_util.py",
    "utils/renderer.py",
    "utils/renderer_base.py",
    "utils/renderer_builder.py",
    "utils/repre_util.py",
    "utils/structs.py",
    "utils/template_util.py",
    "utils/vis_base_util.py",
    "utils/vis_util.py",
)


def fetch(url: str, destination: Path, attempts: int = 4) -> bytes:
    """Download one file atomically with bounded retries."""
    request = urllib.request.Request(url, headers={"User-Agent": "lzt-foundpose-setup"})
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return payload
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt != attempts:
                time.sleep(attempt * 2)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


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
        digest = hashlib.sha256(payload).hexdigest()
        records.append({"path": relative, "bytes": len(payload), "sha256": digest})
        print(f"[{index:02d}/{len(FILES):02d}] {relative} ({len(payload)} bytes)")

    manifest = {
        "repository": f"https://github.com/{REPOSITORY}",
        "ref": args.ref,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "raw.githubusercontent.com",
        "files": records,
    }
    (destination / "SOURCE_SNAPSHOT.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {destination / 'SOURCE_SNAPSHOT.json'}")

    patch_dir = Path(__file__).resolve().parents[1] / "third_party" / "patches"
    for patch_name in (
        "foundpose_topk_coarse.patch",
        "foundpose_cpu_kmeans.patch",
    ):
        patch_path = patch_dir / patch_name
        subprocess.run(
            ["patch", "-p1", "--forward", "--input", str(patch_path)],
            cwd=destination,
            check=True,
        )
        print(f"applied compatibility patch: {patch_path.name}")


if __name__ == "__main__":
    main()
