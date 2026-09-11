#!/usr/bin/env python3
"""Run the vendored FoundPose entry points with workspace-local paths."""

from __future__ import annotations

import os
from pathlib import Path
import random
import runpy
import sys


COMMANDS = {
    "gen_templates": "gen_templates.py",
    "gen_repre": "gen_repre.py",
    "infer": "infer.py",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        commands = "|".join(COMMANDS)
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} {{{commands}}} [FoundPose arguments]"
        )

    command = sys.argv[1]
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parents[1]
    foundpose_root = project_root / "third_party" / "foundpose"
    bop_root = Path(
        os.environ.get("BOP_PATH", workspace_root / "datasets" / "bop")
    ).resolve()
    output_root = Path(
        os.environ.get(
            "BOP_OUTPUT_PATH", project_root / "outputs" / "foundpose"
        )
    ).resolve()

    if not foundpose_root.is_dir():
        raise SystemExit(f"FoundPose source is missing: {foundpose_root}")
    if not bop_root.is_dir():
        raise SystemExit(f"BOP dataset root is missing: {bop_root}")

    python_paths = [
        foundpose_root,
        foundpose_root / "external" / "bop_toolkit",
        foundpose_root / "external" / "dinov2",
    ]
    sys.path[:0] = [str(path) for path in python_paths]

    seed = int(os.environ.get("LZT_RANDOM_SEED", "0"))
    random.seed(seed)
    import cv2
    import numpy as np
    import torch

    cv2.setRNGSeed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ["BOP_PATH"] = str(bop_root)
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault(
        "TORCH_HOME", str(workspace_root / ".cache" / "torch")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    from bop_toolkit_lib import config as bop_config

    bop_config.datasets_path = str(bop_root)
    bop_config.output_path = str(output_root)

    entrypoint = foundpose_root / "scripts" / COMMANDS[command]
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
