#!/usr/bin/env python3
"""Run the industrial-CAD workcell with grasp-focused acceptance.
    给工业CAD工作单元运行抓取聚焦的接受模式。

"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> None:
    runner = Path(__file__).with_name("run_industrial_cad_pick_place.py")
    defaults = [
        "--scene-name",
        "industrial_cad_grasp_focus.xml",
        "--slot",
        "place_region_center",
        "--acceptance-mode",
        "grasp_focus",
        "--placement-region-min-xy",
        "-0.38",
        "0.09",
        "--placement-region-max-xy",
        "-0.02",
        "0.47",
    ]
    os.execv(
        sys.executable,
        [sys.executable, str(runner), *defaults, *sys.argv[1:]],
    )


if __name__ == "__main__":
    main()
