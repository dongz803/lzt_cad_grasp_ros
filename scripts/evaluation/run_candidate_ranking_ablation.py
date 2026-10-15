#!/usr/bin/env python3
"""Run the frozen T-LESS candidate-ranking ablation for Innovation 1.

The experiment reuses the already-rendered FoundPose and hard-gate outputs.  It
does not rerun FoundPose, CAD rendering, refinement, or model fitting.  Every
ranking method sees the same accepted candidates; only its score changes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
EVALUATOR = PROJECT_ROOT / "scripts/evaluation/evaluate_topk_coverage.py"
OFFICIAL_EVALUATOR = PROJECT_ROOT / "scripts/evaluation/evaluate_bop19_official.py"
DEFAULT_RAW_ROOT = (
    PROJECT_ROOT
    / "outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1"
)
DEFAULT_GATED_ROOT = (
    PROJECT_ROOT
    / "outputs/innovation1_v3_full/tless_objects1_30_fixed_10mm_5deg_v1/gating"
)
DEFAULT_DETECTIONS = (
    WORKSPACE_ROOT
    / "datasets/bop/detections/cnos-fastsam/cnos-fastsam_tless-test.json"
)
DEFAULT_TARGETS = WORKSPACE_ROOT / "datasets/bop/tless/test_targets_bop19.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/innovation1_ranking_ablation/tless_objects1_30_v3_frozen_v1"
)

METHODS = {
    "raw": "原始 FoundPose 分数，未经过硬门控",
    "gate_original": "硬门控后按 FoundPose 原始分数排序",
    "gate_reliability_v3": "硬门控后按冻结的 14 维 V3 可靠性概率排序",
    "gate_pnp_score": "硬门控后按 PnP 内点比例排序",
    "gate_pnp_inliers": "硬门控后按 PnP 内点数量排序",
    "gate_iou": "硬门控后按 CAD 投影与观测 Mask IoU 排序",
    "gate_depth": "硬门控后按归一化深度误差从小到大排序",
}

SCORE_KEYS = {
    "raw": "score",
    "gate_original": "original_score",
    "gate_reliability_v3": "calibrated_probability",
    "gate_pnp_score": "ablation_pnp_score",
    "gate_pnp_inliers": "ablation_pnp_inliers",
    "gate_iou": "ablation_silhouette_iou",
    "gate_depth": "ablation_negative_depth_error_over_cad_scale",
}

SUBSETS = {
    "all_1_30": tuple(range(1, 31)),
    "development_2_4": (2, 3, 4),
    "historical_holdout_5_10": tuple(range(5, 11)),
    "cross_object_1_5_30": (1, *range(5, 31)),
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _named_paths(values: Iterable[str], argument: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{argument} expects METHOD=PATH, got: {value}")
        name, raw_path = value.split("=", 1)
        if name not in METHODS or not raw_path:
            raise ValueError(f"{argument} expects a known METHOD=PATH: {value}")
        if name in result:
            raise ValueError(f"duplicate {argument} method: {name}")
        result[name] = Path(raw_path)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def enrich_gated_record(record: dict[str, Any]) -> dict[str, Any]:
    """Copy a gated record and expose frozen single-cue scores at top level."""

    output = dict(record)
    metrics = record.get("pose_gate_metrics", {})
    features = record.get("reliability_features", {})
    output["ablation_pnp_score"] = _finite(metrics.get("pnp_score"), -1.0)
    output["ablation_pnp_inliers"] = _finite(metrics.get("pnp_inliers"), -1.0)
    output["ablation_silhouette_iou"] = _finite(
        metrics.get("silhouette_iou"), -1.0
    )
    normalized_depth_error = _finite(
        features.get("depth_error_over_cad_scale"), 1.0e9
    )
    output["ablation_negative_depth_error_over_cad_scale"] = (
        -normalized_depth_error
    )
    return output


def _record_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["scene_id"]),
        int(record.get("img_id", record.get("im_id"))),
        int(record["obj_id"]),
        int(record.get("inst_id", 0)),
    )


def _label(record: dict[str, Any], threshold: float = 0.1) -> bool | None:
    error = _finite(record.get("mssd_n"), math.inf)
    return None if not math.isfinite(error) else error < threshold


def _roc_auc(labels: Iterable[bool], scores: Iterable[float]) -> float | None:
    pairs = sorted(
        (float(score), bool(label))
        for label, score in zip(labels, scores)
        if math.isfinite(float(score))
    )
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _method_diagnostics(
    records: list[dict[str, Any]], score_key: str
) -> dict[str, Any]:
    labelled = [record for record in records if _label(record) is not None]
    labels = [bool(_label(record)) for record in labelled]
    scores = [_finite(record.get(score_key), -1.0e12) for record in labelled]
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in labelled:
        groups[_record_key(record)].append(record)
    selected = [
        max(
            group,
            key=lambda record: (
                _finite(record.get(score_key), -1.0e12),
                -int(record.get("hypothesis_id", 0)),
            ),
        )
        for group in groups.values()
    ]
    selected_correct = sum(bool(_label(record)) for record in selected)
    result: dict[str, Any] = {
        "score_key": score_key,
        "candidate_count": len(records),
        "labelled_candidate_count": len(labelled),
        "positive_candidate_count": sum(labels),
        "candidate_auc": _roc_auc(labels, scores),
        "predicted_group_count": len(groups),
        "selected_correct_group_count": selected_correct,
        "selected_correct_fraction_on_predicted_groups": (
            selected_correct / len(groups) if groups else 0.0
        ),
    }
    if score_key == "calibrated_probability" and labelled:
        probabilities = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
        targets = np.asarray(labels, dtype=np.float64)
        result["brier"] = float(np.mean((probabilities - targets) ** 2))
    return result


def _paired_top1(
    records: list[dict[str, Any]], first_key: str, second_key: str
) -> dict[str, Any]:
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if _label(record) is not None:
            groups[_record_key(record)].append(record)

    def selected_correct(group: list[dict[str, Any]], score_key: str) -> bool:
        selected = max(
            group,
            key=lambda record: (
                _finite(record.get(score_key), -1.0e12),
                -int(record.get("hypothesis_id", 0)),
            ),
        )
        return bool(_label(selected))

    both_correct = first_only = second_only = both_wrong = 0
    for group in groups.values():
        first = selected_correct(group, first_key)
        second = selected_correct(group, second_key)
        if first and second:
            both_correct += 1
        elif first:
            first_only += 1
        elif second:
            second_only += 1
        else:
            both_wrong += 1
    discordant = first_only + second_only
    p_value = 1.0
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(first_only, second_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "first_score_key": first_key,
        "second_score_key": second_key,
        "group_count": len(groups),
        "both_correct": both_correct,
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "both_wrong": both_wrong,
        "discordant_group_count": discordant,
        "mcnemar_exact_two_sided_p_value": p_value,
    }


def _macro_subset(
    per_object: dict[str, dict[str, float | None]], object_ids: Iterable[int]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("mssd", "mspd", "vsd", "ar"):
        values = [
            per_object[str(object_id)].get(metric)
            for object_id in object_ids
            if str(object_id) in per_object
            and per_object[str(object_id)].get(metric) is not None
        ]
        result[metric] = None if not values else float(np.mean(values))
    return result


def _official_subset_scores(
    result_root: Path, object_ids: set[int]
) -> dict[str, Any]:
    official_scores_path = result_root / "scores_bop19.json"
    if not official_scores_path.is_file():
        raise RuntimeError(f"missing official BOP19 score file: {official_scores_path}")
    official_scores = _load_json(official_scores_path)
    official_overall = {
        "mssd": float(official_scores["bop19_average_recall_mssd"]),
        "mspd": float(official_scores["bop19_average_recall_mspd"]),
        "vsd": float(official_scores["bop19_average_recall_vsd"]),
        "ar": float(official_scores["bop19_average_recall"]),
    }
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
        values: dict[str, float | None] = {}
        for metric in ("mssd", "mspd", "vsd"):
            scores = per_metric[metric].get(object_id, [])
            values[metric] = None if not scores else float(np.mean(scores))
        available = [
            values[metric]
            for metric in ("mssd", "mspd", "vsd")
            if values[metric] is not None
        ]
        values["ar"] = None if not available else float(np.mean(available))
        per_object[str(object_id)] = values
    macro: dict[str, float | None] = {}
    for metric in ("mssd", "mspd", "vsd", "ar"):
        values = [
            item[metric]
            for item in per_object.values()
            if item[metric] is not None
        ]
        macro[metric] = None if not values else float(np.mean(values))
    if any(macro[metric] is None for metric in ("mssd", "mspd", "vsd")):
        raise RuntimeError(f"incomplete official BOP19 output: {result_root}")
    return {
        "protocol": "official_bop19_one_to_one_matching_subset_macro",
        "object_ids": sorted(object_ids),
        "official_overall": official_overall,
        "per_object_macro_average": macro,
        "per_object": per_object,
    }


def _attach_official_results(
    evaluation: dict[str, Any],
    evaluation_dir: Path,
    object_ids: list[int],
    official_workers: int,
    caches: dict[str, Path],
) -> dict[str, Any]:
    csv_paths = {
        name: Path(method["bop_result_csv"])
        for name, method in evaluation["methods"].items()
    }
    hashes = {name: _sha256(path) for name, path in csv_paths.items()}
    groups: dict[str, list[str]] = defaultdict(list)
    for name in METHODS:
        groups[hashes[name]].append(name)
    provenance: dict[str, Any] = {}
    for digest, names in groups.items():
        cached_name = next((name for name in names if name in caches), None)
        if cached_name is not None:
            result_root = caches[cached_name].resolve()
            source_csv = (
                result_root.parents[2]
                / "bop_results"
                / f"{result_root.name}.csv"
            )
            if not source_csv.is_file():
                raise RuntimeError(
                    f"cannot verify cached official result CSV: {source_csv}"
                )
            source_digest = _sha256(source_csv)
            if source_digest != digest:
                raise RuntimeError(
                    "official cache CSV is not hash-identical: "
                    f"{source_csv} ({source_digest}) != {csv_paths[names[0]]} "
                    f"({digest})"
                )
            source = {
                "kind": "verified_hash-identical_cached_official_result",
                "cache_method": cached_name,
                "result_root": str(result_root),
                "source_csv": str(source_csv),
                "source_csv_sha256": source_digest,
            }
        else:
            representative = names[0]
            result_csv = csv_paths[representative]
            eval_root = evaluation_dir / "official_eval_deduplicated" / representative
            command = [
                sys.executable,
                str(OFFICIAL_EVALUATOR),
                "--result-csv",
                str(result_csv),
                "--eval-root",
                str(eval_root),
                "--num-workers",
                str(official_workers),
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            result_root = eval_root / result_csv.stem
            source = {
                "kind": "evaluated_once_for_hash-identical_methods",
                "representative_method": representative,
                "command": command,
                "result_root": str(result_root.resolve()),
            }
        official = _official_subset_scores(result_root, set(object_ids))
        for name in names:
            evaluation["methods"][name]["official_bop19"] = official
            provenance[name] = {
                **source,
                "bop_csv_sha256": digest,
                "hash_identical_methods": names,
            }
    evaluation["official_bop19_provenance"] = provenance
    return evaluation


def _fixed_success_subset(
    per_object: dict[str, dict[str, Any]], object_ids: Iterable[int]
) -> dict[str, float | int]:
    rows = [per_object[str(object_id)] for object_id in object_ids]
    targets = sum(int(row["target_instances"]) for row in rows)
    successes = sum(int(row["successful_target_instances"]) for row in rows)
    return {
        "target_instances": targets,
        "successful_target_instances": successes,
        "fixed_denominator_success_rate": successes / targets if targets else 0.0,
    }


def _build_compact_report(
    evaluation: dict[str, Any], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "innovation1_candidate_ranking_ablation",
        "methods": {},
        "subsets": {name: list(ids) for name, ids in SUBSETS.items()},
        "important_equivalences": diagnostics["important_equivalences"],
        "paired_top1_comparisons": diagnostics["paired_top1_comparisons"],
    }
    waterfall = evaluation["failure_waterfall"]["methods"]
    for name, method in evaluation["methods"].items():
        official = method.get("official_bop19", {})
        per_object = official.get("per_object", {})
        report["methods"][name] = {
            "description": METHODS[name],
            "score_key": method["score_key"],
            "coverage": method["coverage"],
            "prediction_count": method["selection"]["selected_prediction_count"],
            "preliminary_ar_mssd_fixed_denominator": method["mssd"][
                "average_recall_fixed_target_denominator"
            ],
            "official_bop19_all_1_30": official.get("official_overall"),
            "official_bop19_per_object_macro_all_1_30": official.get(
                "per_object_macro_average"
            ),
            "ranking_diagnostics": diagnostics["methods"].get(name),
            "subsets": {
                subset_name: {
                    "official_bop19_macro": _macro_subset(per_object, object_ids),
                    "mssd_n_lt_0.1_top1": _fixed_success_subset(
                        waterfall[name]["per_object"], object_ids
                    ),
                }
                for subset_name, object_ids in SUBSETS.items()
            },
        }
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 创新点一候选排序消融结果",
        "",
        "> 本实验复用冻结的 T-LESS 1–30 FoundPose Top-K 与硬门控输出，只改变候选排序分数；GT 仅用于离线评价。",
        "",
        "| 方法 | 覆盖率 | 官方 BOP19 MSSD | MSPD | VSD | AR | 候选 AUC | 固定分母 Top-1 正确率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in METHODS:
        method = report["methods"][name]
        official = method["official_bop19_all_1_30"] or {}
        diagnostics = method["ranking_diagnostics"] or {}
        top1 = method["subsets"]["all_1_30"]["mssd_n_lt_0.1_top1"]
        values = [official.get(key) for key in ("mssd", "mspd", "vsd", "ar")]
        format_value: Callable[[Any], str] = lambda value: (
            "—" if value is None else f"{float(value):.5f}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{method['coverage']:.5f}",
                    *(format_value(value) for value in values),
                    format_value(diagnostics.get("candidate_auc")),
                    f"{top1['fixed_denominator_success_rate']:.5f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `gate_original` 与 `gate_pnp_score` 若完全相同，是因为当前 FoundPose 原始分数就是 PnP 内点比例，不应把二者宣称为两个独立算法。",
            "- 候选 AUC 和 `MSSD_n < 0.1` Top-1 正确率使用记录内 GT 误差作离线诊断；论文主位姿指标以官方 BOP19 一对一匹配结果为准。",
            "- `development_2_4` 参与了 V3 可靠性拟合；`historical_holdout_5_10` 和 `cross_object_1_5_30` 必须分开报告，不能把全量结果称为完全独立测试。",
            "- 本实验不评价 RGB-D/CAD 精修、对称模式或协方差模块；这些模块需要各自的消融与评价指标。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluation_command(
    raw_path: Path,
    gated_path: Path,
    output_dir: Path,
    detections: Path,
    targets: Path,
    object_ids: Iterable[int],
    run_official: bool,
    official_workers: int,
) -> list[str]:
    command = [sys.executable, str(EVALUATOR)]
    for name in METHODS:
        source = raw_path if name == "raw" else gated_path
        command.extend(["--method", f"{name}={source}"])
        command.extend(["--score-key", f"{name}={SCORE_KEYS[name]}"])
    for object_id in object_ids:
        command.extend(["--object-id", str(object_id)])
    command.extend(
        [
            "--targets",
            str(targets),
            "--detections",
            str(detections),
            "--raw-method",
            "raw",
            "--output-dir",
            str(output_dir),
            "--official-workers",
            str(official_workers),
        ]
    )
    if run_official:
        command.append("--run-official")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen T-LESS 1–30 candidate-ranking ablation."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--gated-root", type=Path, default=DEFAULT_GATED_ROOT)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object-id", type=int, action="append", default=[])
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-workers", type=int, default=4)
    parser.add_argument(
        "--official-cache",
        action="append",
        default=[],
        help=(
            "repeatable METHOD=RESULT_ROOT for a previously completed official "
            "evaluation; reuse is allowed only for hash-identical BOP CSVs"
        ),
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="rebuild compact reports from an already completed output directory",
    )
    args = parser.parse_args()

    try:
        official_caches = _named_paths(args.official_cache, "--official-cache")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    object_ids = sorted(set(args.object_id or range(1, 31)))
    if any(object_id <= 0 for object_id in object_ids):
        raise SystemExit("--object-id values must be positive")
    if args.official_workers <= 0:
        raise SystemExit("--official-workers must be positive")
    output_dir = args.output_dir.resolve()
    if args.finalize_existing:
        evaluation_path = output_dir / "evaluation/summary.json"
        diagnostics_path = output_dir / "ranking_diagnostics.json"
        manifest_path = output_dir / "experiment_manifest.json"
        missing = [
            str(path)
            for path in (evaluation_path, diagnostics_path, manifest_path)
            if not path.is_file()
        ]
        if missing:
            raise SystemExit("missing completed outputs:\n" + "\n".join(missing))
        evaluation = _load_json(evaluation_path)
        provenance = evaluation.get("official_bop19_provenance", {})
        if set(provenance) != set(METHODS):
            raise SystemExit("official BOP19 provenance is incomplete")
        for name, source in provenance.items():
            evaluation["methods"][name]["official_bop19"] = _official_subset_scores(
                Path(source["result_root"]), set(object_ids)
            )
        diagnostics = _load_json(diagnostics_path)
        gated_records = _load_json(
            output_dir / "derived_inputs/gated_candidates_with_ablation_scores.json"
        )
        diagnostics["paired_top1_comparisons"] = {
            "reliability_v3_vs_gate_original": _paired_top1(
                gated_records, "calibrated_probability", "original_score"
            ),
            "reliability_v3_vs_iou": _paired_top1(
                gated_records,
                "calibrated_probability",
                "ablation_silhouette_iou",
            ),
            "reliability_v3_vs_depth": _paired_top1(
                gated_records,
                "calibrated_probability",
                "ablation_negative_depth_error_over_cad_scale",
            ),
        }
        _write_json(diagnostics_path, diagnostics)
        compact = _build_compact_report(evaluation, diagnostics)
        _write_json(evaluation_path, evaluation)
        _write_json(output_dir / "summary.json", compact)
        _write_markdown(output_dir / "RESULT.md", compact)
        manifest = _load_json(manifest_path)
        manifest["reports_finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["report_metric_boundary"] = (
            "official overall BOP19 scores are distinct from per-object subset macro"
        )
        _write_json(manifest_path, manifest)
        print(f"finalized ranking ablation reports: {output_dir}")
        return
    required = [EVALUATOR, OFFICIAL_EVALUATOR, args.detections, args.targets]
    raw_paths = {
        object_id: args.raw_root / str(object_id) / "estimated-poses.json"
        for object_id in object_ids
    }
    gated_paths = {
        object_id: args.gated_root / f"tless_{object_id:06d}" / "gated-poses.json"
        for object_id in object_ids
    }
    required.extend(raw_paths.values())
    required.extend(gated_paths.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required inputs:\n" + "\n".join(missing))
    missing_caches = [
        f"{name}: {path}"
        for name, path in official_caches.items()
        if not path.is_dir()
    ]
    if missing_caches:
        raise SystemExit("missing official cache roots:\n" + "\n".join(missing_caches))

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "derived_inputs"
    raw_output = input_dir / "raw_candidates.json"
    gated_output = input_dir / "gated_candidates_with_ablation_scores.json"

    raw_records: list[dict[str, Any]] = []
    gated_records: list[dict[str, Any]] = []
    for object_id in object_ids:
        raw_records.extend(_load_json(raw_paths[object_id]))
        gated_records.extend(
            enrich_gated_record(record)
            for record in _load_json(gated_paths[object_id])
        )
    _write_json(raw_output, raw_records)
    _write_json(gated_output, gated_records)

    equivalence = all(
        math.isclose(
            _finite(record.get("original_score"), -1.0),
            _finite(record.get("ablation_pnp_score"), -2.0),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for record in gated_records
    )
    diagnostics = {
        "schema_version": 1,
        "correctness_definition": "mssd_n < 0.1 (evaluation only)",
        "methods": {
            "raw": _method_diagnostics(raw_records, SCORE_KEYS["raw"]),
            **{
                name: _method_diagnostics(gated_records, SCORE_KEYS[name])
                for name in METHODS
                if name != "raw"
            },
        },
        "important_equivalences": {
            "gate_original_equals_gate_pnp_score_for_every_candidate": equivalence,
            "explanation": (
                "FoundPose original_score is the stored PnP inlier ratio in this run"
            ),
        },
        "paired_top1_comparisons": {
            "reliability_v3_vs_gate_original": _paired_top1(
                gated_records, "calibrated_probability", "original_score"
            ),
            "reliability_v3_vs_iou": _paired_top1(
                gated_records,
                "calibrated_probability",
                "ablation_silhouette_iou",
            ),
            "reliability_v3_vs_depth": _paired_top1(
                gated_records,
                "calibrated_probability",
                "ablation_negative_depth_error_over_cad_scale",
            ),
        },
    }
    _write_json(output_dir / "ranking_diagnostics.json", diagnostics)

    command = _evaluation_command(
        raw_output,
        gated_output,
        output_dir / "evaluation",
        args.detections.resolve(),
        args.targets.resolve(),
        object_ids,
        False,
        args.official_workers,
    )
    manifest = {
        "schema_version": 1,
        "experiment": "innovation1_candidate_ranking_ablation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "object_ids": object_ids,
        "methods": METHODS,
        "score_keys": SCORE_KEYS,
        "protocol": {
            "candidate_pool": "same hard-gate survivors for every gate_* method",
            "correctness": "mssd_n < 0.1 for evaluation only",
            "target_denominator": "T-LESS BOP19 target inst_count",
            "model_fitting": "none; frozen V3 probabilities are reused",
            "rerun_foundpose_or_rendering": False,
            "ground_truth_used_online": False,
        },
        "inputs": {
            "raw_root": str(args.raw_root.resolve()),
            "gated_root": str(args.gated_root.resolve()),
            "detections": str(args.detections.resolve()),
            "targets": str(args.targets.resolve()),
            "raw_source_sha256": {
                str(object_id): _sha256(raw_paths[object_id])
                for object_id in object_ids
            },
            "gated_source_sha256": {
                str(object_id): _sha256(gated_paths[object_id])
                for object_id in object_ids
            },
        },
        "interpreter": sys.executable,
        "official_bop19_requested": args.run_official,
        "official_cache_roots": {
            name: str(path.resolve()) for name, path in official_caches.items()
        },
        "command": command,
    }
    manifest_path = output_dir / "experiment_manifest.json"
    _write_json(manifest_path, manifest)
    begin = time.perf_counter()
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        evaluation = _load_json(output_dir / "evaluation/summary.json")
        if args.run_official:
            evaluation = _attach_official_results(
                evaluation,
                output_dir / "evaluation",
                object_ids,
                args.official_workers,
                official_caches,
            )
            _write_json(output_dir / "evaluation/summary.json", evaluation)
        compact = _build_compact_report(evaluation, diagnostics)
        _write_json(output_dir / "summary.json", compact)
        _write_markdown(output_dir / "RESULT.md", compact)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["duration_s"] = time.perf_counter() - begin
        _write_json(manifest_path, manifest)
        raise
    manifest["status"] = "completed"
    manifest["duration_s"] = time.perf_counter() - begin
    manifest["outputs"] = {
        "summary": str((output_dir / "summary.json").resolve()),
        "result_markdown": str((output_dir / "RESULT.md").resolve()),
        "evaluation": str((output_dir / "evaluation/summary.json").resolve()),
    }
    _write_json(manifest_path, manifest)
    print(f"completed ranking ablation: {output_dir}")


if __name__ == "__main__":
    main()
