#!/usr/bin/env python3
"""Run the official BOP19 VSD/MSSD/MSPD localization evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from official_eval_guard import (  # noqa: E402
    require_official_eval_dependencies,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    workspace = project_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-csv", type=Path, required=True)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=project_root / "outputs/foundpose_bop19/official_eval",
    )
    parser.add_argument(
        "--bop-root",
        type=Path,
        default=workspace / "datasets/bop",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    result_csv = args.result_csv.resolve()
    if not result_csv.is_file():
        raise SystemExit(f"result CSV is missing: {result_csv}")
    if "_tless-test" not in result_csv.name:
        raise SystemExit(
            "BOP result filename must contain '_tless-test', for example "
            "foundpose-bop19-standard_tless-test.csv"
        )
    toolkit = project_root / "third_party/bop_toolkit_eval"
    evaluator = toolkit / "scripts/eval_bop19_pose.py"
    if not evaluator.is_file():
        raise SystemExit(
            "official evaluator is missing; run scripts/bootstrap/fetch_bop_toolkit_eval_raw.py"
        )

    require_official_eval_dependencies(project_root=project_root)

    eval_root = args.eval_root.resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["BOP_PATH"] = str(args.bop_root.resolve())
    environment["BOP_RESULTS_PATH"] = str(result_csv.parent)
    environment["BOP_EVAL_PATH"] = str(eval_root)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(toolkit),
            environment.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    environment["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).resolve().parent), environment.get("PATH", "")]
    )
    command = [
        sys.executable,
        str(evaluator),
        "--renderer_type=vispy",
        f"--result_filenames={result_csv.name}",
        f"--results_path={result_csv.parent}",
        f"--eval_path={eval_root}",
        f"--num_workers={args.num_workers}",
    ]
    log_path = eval_root / f"{result_csv.stem}.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            command,
            env=environment,
            check=True,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )

    result_name = result_csv.stem
    scores_path = eval_root / result_name / "scores_bop19.json"
    if not scores_path.is_file():
        raise SystemExit(f"official score file was not produced: {scores_path}")
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    print(f"official evaluator log: {log_path}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
