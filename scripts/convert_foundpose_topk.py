#!/usr/bin/env python3
"""Convert FoundPose JSON to symmetry-aware project Top-K JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lzt_cad_grasp_core.object_config import load_object_config  # noqa: E402
from lzt_cad_grasp_core.pose_hypotheses import (  # noqa: E402
    hypothesis_sets_to_jsonable,
    load_foundpose_top_k,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--object-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = load_object_config(args.object_config)
    sets = load_foundpose_top_k(args.input, config.cad_object, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(hypothesis_sets_to_jsonable(sets), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(sets)} pose-hypothesis sets to {args.output}")


if __name__ == "__main__":
    main()
