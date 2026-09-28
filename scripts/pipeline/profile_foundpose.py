#!/usr/bin/env python3
"""Run a FoundPose stage and record wall time plus peak NVIDIA memory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def gpu_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return max(int(line.strip()) for line in result.stdout.splitlines())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("gen_templates", "gen_repre", "infer"))
    parser.add_argument("--opts-path", type=Path, required=True)
    parser.add_argument("--profile-path", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    runner = project_root / "scripts" / "pipeline" / "run_foundpose.py"
    command = [
        sys.executable,
        str(runner),
        args.stage,
        "--opts-path",
        str(args.opts_path.resolve()),
    ]

    baseline_mib = gpu_memory_mib()
    peak_mib = baseline_mib
    stop = threading.Event()

    def monitor() -> None:
        nonlocal peak_mib
        while not stop.wait(args.poll_seconds):
            current = gpu_memory_mib()
            if current is not None:
                peak_mib = current if peak_mib is None else max(peak_mib, current)

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    process = subprocess.run(command, env=os.environ.copy())
    stop.set()
    thread.join(timeout=max(args.poll_seconds * 2.0, 1.0))
    duration = time.perf_counter() - started

    profile = {
        "stage": args.stage,
        "opts_path": str(args.opts_path.resolve()),
        "started_at_utc": started_at.isoformat(),
        "duration_s": duration,
        "exit_code": process.returncode,
        "gpu_memory_baseline_mib": baseline_mib,
        "gpu_memory_peak_mib": peak_mib,
        "gpu_memory_peak_delta_mib": (
            None
            if baseline_mib is None or peak_mib is None
            else peak_mib - baseline_mib
        ),
        "gpu_memory_measurement_scope": "full_device",
    }
    args.profile_path.parent.mkdir(parents=True, exist_ok=True)
    args.profile_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
