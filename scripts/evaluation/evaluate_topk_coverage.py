#!/usr/bin/env python3
"""Evaluate Top-K pose policies with fixed target coverage and BOP matching."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)

from lzt_benchmark.topk_evaluation import (  # noqa: E402
    common_coverage_comparison,
    failure_waterfall,
    load_detection_counts,
    load_target_counts,
    oracle_topk_summary,
    recall_at_k,
    select_policy_predictions,
    summarize_selected_predictions,
    write_bop_csv,
)

from official_eval_guard import (  # noqa: E402
    require_official_eval_dependencies,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _bop_method_token(name: str) -> str:
    """Return a BOP-safe method token without the filename field separator.

    The BOP toolkit parses the first underscore as the boundary between the
    method name and ``dataset-split``.  Keeping an underscore in a user-facing
    method name such as ``mode_aware`` therefore produces a valid CSV with an
    invalid BOP result filename.
    """

    token = re.sub(r"[^A-Za-z0-9.-]+", "-", name).strip("-.")
    if not token:
        raise ValueError(f"method name has no BOP-safe characters: {name!r}")
    return token


def _named_paths(values: list[str], argument: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{argument} expects NAME=PATH, got: {value}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise SystemExit(f"{argument} expects NAME=PATH, got: {value}")
        result[name].append(Path(raw_path))
    return dict(result)


def _named_values(values: list[str], argument: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{argument} expects NAME=VALUE, got: {value}")
        name, setting = value.split("=", 1)
        result[name] = setting
    return result


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        payload = _load_json(path)
        if not isinstance(payload, list):
            raise SystemExit(f"pose result must be a JSON list: {path}")
        records.extend(payload)
    return records


def _official_subset_scores(
    result_root: Path, object_ids: set[int]
) -> dict[str, Any]:
    per_metric: dict[str, dict[int, list[float]]] = {
        "mssd": defaultdict(list),
        "mspd": defaultdict(list),
        "vsd": defaultdict(list),
    }
    for directory in result_root.glob("error=*"):
        metric = directory.name.split("_", 1)[0].split("=", 1)[-1]
        if metric not in per_metric:
            continue
        for score_path in directory.glob("scores_th=*.json"):
            payload = _load_json(score_path)
            recalls = payload.get("obj_recalls", {})
            for object_id in object_ids:
                if str(object_id) in recalls:
                    per_metric[metric][object_id].append(
                        float(recalls[str(object_id)])
                    )
    per_object: dict[str, dict[str, float | None]] = {}
    for object_id in sorted(object_ids):
        per_object[str(object_id)] = {
            metric: (
                None
                if not values[object_id]
                else float(np.mean(values[object_id]))
            )
            for metric, values in per_metric.items()
        }
        components = [
            value
            for value in per_object[str(object_id)].values()
            if value is not None
        ]
        per_object[str(object_id)]["ar"] = (
            None if not components else float(np.mean(components))
        )
    macro = {}
    for metric in ("mssd", "mspd", "vsd", "ar"):
        values = [
            item[metric]
            for item in per_object.values()
            if item[metric] is not None
        ]
        macro[metric] = None if not values else float(np.mean(values))
    return {
        "protocol": "official_bop19_one_to_one_matching_subset_macro",
        "object_ids": sorted(object_ids),
        "macro_average": macro,
        "per_object": per_object,
    }


def _write_curve_csv(report: dict[str, Any], output: Path) -> None:
    import csv

    with output.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "method",
            "accepted_predictions",
            "coverage",
            "minimum_confidence",
            "selective_risk_mssd",
            "selective_risk_mspd",
            "conditional_ar_mssd",
            "conditional_ar_mspd",
            "fixed_denominator_ar_mssd",
            "fixed_denominator_ar_mspd",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, method in report["methods"].items():
            for point in method["risk_coverage"]:
                writer.writerow({"method": name, **point})


def _polyline(points: list[dict[str, Any]], x_key: str, y_key: str,
              left: float, top: float, width: float, height: float) -> str:
    if not points:
        return ""
    coords = []
    for point in points:
        x = left + float(point[x_key]) * width
        y = top + (1.0 - float(point[y_key])) * height
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def _write_curve_svg(report: dict[str, Any], output: Path) -> None:
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea"]
    panels = [
        ("selective_risk_mssd", "MSSD selective risk"),
        ("selective_risk_mspd", "MSPD selective risk"),
    ]
    width, height = 1000, 460
    panel_width, panel_height = 410, 320
    lefts = [70, 560]
    top = 70
    style = (
        '<style>text{font-family:sans-serif;fill:#111827}'
        '.axis{stroke:#374151;stroke-width:1}'
        '.grid{stroke:#e5e7eb;stroke-width:1}'
        '.line{fill:none;stroke-width:2.5}</style>'
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        style,
        '<text x="500" y="28" text-anchor="middle" font-size="20">'
        'Top-K risk-coverage evaluation</text>',
    ]
    for panel_index, (risk_key, title) in enumerate(panels):
        left = lefts[panel_index]
        elements.append(
            f'<text x="{left + panel_width / 2}" y="52" '
            f'text-anchor="middle" font-size="15">{title}</text>'
        )
        for tick in range(6):
            value = tick / 5
            x = left + value * panel_width
            y = top + (1 - value) * panel_height
            elements.extend(
                [
                    f'<line class="grid" x1="{x}" y1="{top}" '
                    f'x2="{x}" y2="{top + panel_height}"/>',
                    f'<line class="grid" x1="{left}" y1="{y}" '
                    f'x2="{left + panel_width}" y2="{y}"/>',
                    f'<text x="{x}" y="{top + panel_height + 20}" '
                    f'text-anchor="middle" font-size="11">{value:.1f}</text>',
                    f'<text x="{left - 10}" y="{y + 4}" '
                    f'text-anchor="end" font-size="11">{value:.1f}</text>',
                ]
            )
        elements.extend(
            [
                f'<line class="axis" x1="{left}" y1="{top + panel_height}" '
                f'x2="{left + panel_width}" y2="{top + panel_height}"/>',
                f'<line class="axis" x1="{left}" y1="{top}" '
                f'x2="{left}" y2="{top + panel_height}"/>',
                f'<text x="{left + panel_width / 2}" '
                f'y="{top + panel_height + 42}" text-anchor="middle" '
                'font-size="12">Target-instance coverage</text>',
            ]
        )
        for method_index, (name, method) in enumerate(report["methods"].items()):
            color = colors[method_index % len(colors)]
            points = _polyline(
                method["risk_coverage"], "coverage", risk_key,
                left, top, panel_width, panel_height,
            )
            elements.append(
                f'<polyline class="line" stroke="{color}" points="{points}"/>'
            )
            if panel_index == 0:
                legend_x = left + 12 + method_index * 150
                elements.extend(
                    [
                        f'<line x1="{legend_x}" y1="{top + 16}" '
                        f'x2="{legend_x + 24}" y2="{top + 16}" '
                        f'stroke="{color}" stroke-width="3"/>',
                        f'<text x="{legend_x + 30}" y="{top + 20}" '
                        f'font-size="12">{name}</text>',
                    ]
                )
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw and gated Top-K policies using a fixed BOP target "
            "denominator, equal-coverage risk and optional official BOP19 AR."
        )
    )
    parser.add_argument(
        "--method", action="append", required=True,
        help="repeatable NAME=PATH; repeat NAME to combine per-object JSON files",
    )
    parser.add_argument(
        "--score-key", action="append", default=[],
        help="optional NAME=FIELD, default score",
    )
    parser.add_argument("--object-id", type=int, action="append", required=True)
    parser.add_argument(
        "--targets", type=Path,
        default=WORKSPACE_ROOT / "datasets/bop/tless/test_targets_bop19.json",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        help=(
            "CNOS/GT-mask detection JSON used to separate missing detections "
            "from detections whose PnP stage produced no pose"
        ),
    )
    parser.add_argument(
        "--raw-method",
        help=(
            "method name representing pre-gate FoundPose output; defaults to "
            "'raw' when present, otherwise the first --method"
        ),
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        action="append",
        default=[],
        help="repeatable K for Recall@K; defaults to 1, 3 and 5",
    )
    parser.add_argument(
        "--success-field",
        default="mssd_n",
        help="record-local GT error used by Recall@K/failure attribution",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.1,
        help="exclusive correctness threshold for --success-field",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-workers", type=int, default=1)
    args = parser.parse_args()

    if args.run_official:
        require_official_eval_dependencies(project_root=PROJECT_ROOT)

    method_paths = _named_paths(args.method, "--method")
    score_keys = _named_values(args.score_key, "--score-key")
    object_ids = set(args.object_id)
    targets = _load_json(args.targets)
    target_counts = load_target_counts(targets, object_ids)
    denominator = sum(target_counts.values())
    if denominator <= 0:
        raise SystemExit("selected objects have no BOP targets")
    recall_ks = tuple(args.recall_k or (1, 3, 5))
    if any(value <= 0 for value in recall_ks):
        raise SystemExit("--recall-k values must be positive")
    if not np.isfinite(args.success_threshold) or args.success_threshold <= 0.0:
        raise SystemExit("--success-threshold must be finite and positive")

    detection_counts = None
    if args.detections is not None:
        detection_payload = _load_json(args.detections)
        if not isinstance(detection_payload, list):
            raise SystemExit(f"detections must be a JSON list: {args.detections}")
        detection_counts = load_detection_counts(detection_payload, object_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mssd_thresholds = tuple(np.arange(0.05, 0.51, 0.05))
    mspd_thresholds = tuple(np.arange(5.0, 51.0, 5.0) * (720.0 / 640.0))
    report: dict[str, Any] = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "tless",
        "object_ids": sorted(object_ids),
        "protocol": {
            "denominator": "BOP19 target inst_count for selected objects",
            "target_instances": denominator,
            "missing_prediction_policy": "counted as failure and non-coverage",
            "gate_abstention_policy": "counted as failure and non-coverage",
            "top1_policy": "highest method score per detected instance",
            "target_budget": "at most inst_count predictions per scene/image/object",
            "preliminary_errors": (
                "record-local FoundPose MSSD/MSPD; use official_bop19 for "
                "one-to-one matching and VSD"
            ),
            "recall_at_k": (
                f"any of the K highest-scoring hypotheses has "
                f"{args.success_field} < {args.success_threshold}; fixed target "
                "denominator"
            ),
            "failure_waterfall": (
                "mutually exclusive first-failure stages; complete detection/PnP "
                "separation requires --detections"
            ),
        },
        "methods": {},
    }
    method_records: dict[str, tuple[list[dict[str, Any]], str]] = {}
    for name, paths in method_paths.items():
        raw_records = _load_records(paths)
        score_key = score_keys.get(name, "score")
        selected, diagnostics = select_policy_predictions(
            raw_records, target_counts, score_key
        )
        method_summary = summarize_selected_predictions(
            selected,
            target_counts,
            score_key,
            mssd_thresholds,
            mspd_thresholds,
        )
        method_summary["selection"] = diagnostics
        method_summary["input_paths"] = [str(path.resolve()) for path in paths]
        method_summary["score_key"] = score_key
        method_summary["oracle_topk_evaluation_only"] = {
            "mssd": oracle_topk_summary(
                raw_records, target_counts, "mssd_n", mssd_thresholds
            ),
            "mspd": oracle_topk_summary(
                raw_records, target_counts, "mspd", mspd_thresholds
            ),
        }
        method_summary["recall_at_k"] = recall_at_k(
            raw_records,
            target_counts,
            args.success_field,
            args.success_threshold,
            recall_ks,
            score_key,
        )
        report["methods"][name] = method_summary
        method_records[name] = (selected, score_key)
        bop_token = _bop_method_token(name)
        bop_csv = args.output_dir / "bop_results" / f"{bop_token}_tless-test.csv"
        write_bop_csv(selected, bop_csv)
        method_summary["bop_method_token"] = bop_token
        method_summary["bop_result_csv"] = str(bop_csv.resolve())

    raw_method = args.raw_method or (
        "raw" if "raw" in method_paths else next(iter(method_paths))
    )
    if raw_method not in method_paths:
        raise SystemExit(f"--raw-method is not one of the methods: {raw_method}")
    if detection_counts is None:
        report["failure_waterfall"] = {
            "available": False,
            "reason": (
                "--detections was not provided, so missing detections cannot be "
                "separated from detections without a PnP pose"
            ),
            "raw_method": raw_method,
        }
    else:
        raw_input_records = _load_records(method_paths[raw_method])
        raw_score_key = score_keys.get(raw_method, "score")
        report["failure_waterfall"] = {
            "available": True,
            "detections": str(args.detections.resolve()),
            "raw_method": raw_method,
            "methods": {},
        }
        for name, paths in method_paths.items():
            report["failure_waterfall"]["methods"][name] = failure_waterfall(
                raw_input_records,
                _load_records(paths),
                detection_counts,
                target_counts,
                field=args.success_field,
                threshold=args.success_threshold,
                raw_score_key=raw_score_key,
                policy_score_key=score_keys.get(name, "score"),
            )

    report["equal_coverage_comparison"] = common_coverage_comparison(
        method_records,
        denominator,
        mssd_thresholds,
        mspd_thresholds,
    )

    if args.run_official:
        evaluator = PROJECT_ROOT / "scripts/evaluation/evaluate_bop19_official.py"
        for name, method in report["methods"].items():
            result_csv = Path(method["bop_result_csv"])
            eval_root = args.output_dir / "official_eval" / name
            subprocess.run(
                [
                    sys.executable,
                    str(evaluator),
                    "--result-csv",
                    str(result_csv),
                    "--eval-root",
                    str(eval_root),
                    "--num-workers",
                    str(args.official_workers),
                ],
                check=True,
            )
            result_root = eval_root / result_csv.stem
            method["official_bop19"] = _official_subset_scores(
                result_root, object_ids
            )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_curve_csv(report, args.output_dir / "risk_coverage.csv")
    _write_curve_svg(report, args.output_dir / "risk_coverage.svg")
    compact = {
        name: {
            "coverage": method["coverage"],
            "ar_mssd_fixed_denominator": method["mssd"][
                "average_recall_fixed_target_denominator"
            ],
            "ar_mspd_fixed_denominator": method["mspd"][
                "average_recall_fixed_target_denominator"
            ],
            "official_bop19": method.get("official_bop19"),
        }
        for name, method in report["methods"].items()
    }
    print(json.dumps({"summary": str(summary_path.resolve()), **compact}, indent=2))


if __name__ == "__main__":
    main()
