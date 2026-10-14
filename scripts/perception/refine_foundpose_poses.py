#!/usr/bin/env python3
"""Refine FoundPose candidates with CAD silhouette and observed RGB-D feedback."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)

from lzt_grasp_planning.cad_grasp import load_ascii_ply_geometry  # noqa: E402
from lzt_common.contracts import PoseHypothesis, PoseHypothesisSet  # noqa: E402
from lzt_perception.mask_evaluation import decode_coco_rle  # noqa: E402
from lzt_common.object_config import load_object_config  # noqa: E402
from lzt_perception.pose_gating import (  # noqa: E402
    PoseGateConfig,
    PoseGateDecision,
    evaluate_pose_candidate,
    foundpose_record_transform,
    render_model_points,
    sample_model_points,
)
from lzt_perception.pose_refinement import (  # noqa: E402
    PoseRefinementConfig,
    refine_pose_rgbd,
)
from lzt_perception.pose_mode_refinement import (  # noqa: E402
    refine_pose_mixture_rgbd,
)
from lzt_perception.pose_modes import (  # noqa: E402
    PoseModeClusteringConfig,
    cluster_pose_hypotheses,
    pose_mixtures_to_jsonable,
)
from lzt_perception.pose_reliability import (  # noqa: E402
    FEATURE_NAMES as RELIABILITY_FEATURE_NAMES,
    PoseReliabilityModel,
    pose_reliability_features,
)
from lzt_perception.pose_uncertainty import (  # noqa: E402
    PredictivePoseCovarianceModel,
)


STALE_GT_FIELDS = (
    "mssd",
    "mssd_n",
    "mspd",
    "mssd_id",
    "mspd_id",
    "inliers_gt",
    "inliers_gt_err",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _align_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    height, width = shape
    source_height, source_width = mask.shape
    if source_height < height or source_width < width:
        raise ValueError(f"mask {mask.shape} is smaller than image {shape}")
    offset_y = (source_height - height) // 2
    offset_x = (source_width - width) // 2
    aligned = mask[offset_y:offset_y + height, offset_x:offset_x + width]
    if aligned.shape != shape:
        raise ValueError(f"cannot align mask {mask.shape} to {shape}")
    return aligned


def _record_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["scene_id"]),
        int(record["img_id"]),
        int(record["obj_id"]),
        int(record.get("inst_id", 0)),
    )


def _candidate_key(record: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    """Return one key for both FoundPose records and gate-audit rows."""

    return (
        int(record["scene_id"]),
        int(record.get("img_id", record.get("image_id"))),
        int(record.get("obj_id", record.get("object_id"))),
        int(record.get("inst_id", record.get("instance_id", 0))),
        int(record.get("hypothesis_id", 0)),
    )


def _raw_pnp_score(record: Mapping[str, Any]) -> float:
    """Recover the pre-gate PnP score instead of reusing a calibrated score."""

    gate_metrics = record.get("pose_gate_metrics", {})
    if (
        isinstance(gate_metrics, Mapping)
        and gate_metrics.get("pnp_score") is not None
    ):
        return float(gate_metrics["pnp_score"])
    return float(record.get("original_score", record.get("score", 0.0)))


def _records_to_hypothesis_set(
    records: list[dict[str, Any]],
    object_id: str,
) -> PoseHypothesisSet:
    """Convert one record group without collapsing distinct pose modes."""

    if not records:
        raise ValueError("cannot build a hypothesis set from an empty group")
    ordered = sorted(
        records,
        key=lambda item: (
            -float(item.get("score", 0.0)),
            int(item.get("hypothesis_id", 0)),
        ),
    )
    first = ordered[0]
    hypotheses = tuple(
        PoseHypothesis(
            transform_camera_object=foundpose_record_transform(record),
            confidence=float(
                np.clip(float(record.get("score", 0.0)), 0.0, 1.0)
            ),
            symmetry_class=str(record.get("symmetry_class", "identity")),
            source=str(record.get("source", "foundpose_coarse")),
            hypothesis_id=int(record.get("hypothesis_id", index)),
        )
        for index, record in enumerate(ordered)
    )
    return PoseHypothesisSet(
        object_id=object_id,
        scene_id=int(first["scene_id"]),
        image_id=int(first["img_id"]),
        instance_id=int(first.get("inst_id", 0)),
        hypotheses=hypotheses,
    )


def _post_refinement_audit(
    record: Mapping[str, Any],
    decision: PoseGateDecision,
) -> dict[str, Any]:
    key = _candidate_key(record)
    return {
        "scene_id": key[0],
        "image_id": key[1],
        "object_id": key[2],
        "instance_id": key[3],
        "hypothesis_id": key[4],
        "original_score": _raw_pnp_score(record),
        **decision.to_dict(),
        "evaluation_only": {},
    }


def _reliability_contexts(
    gate_candidates: list[dict[str, Any]],
    post_candidates: list[dict[str, Any]],
) -> dict[tuple[int, int, int, int], list[dict[str, Any]]]:
    """Replace accepted gate rows with their post-refinement audit rows.

    Rejected gate rows remain in each group so group-relative features keep the
    same denominator and meaning used when the frozen reliability model was
    trained.
    """

    post_by_key = {_candidate_key(item): item for item in post_candidates}
    contexts: dict[
        tuple[int, int, int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    matched: set[tuple[int, int, int, int, int]] = set()
    for gate_candidate in gate_candidates:
        key = _candidate_key(gate_candidate)
        replacement = post_by_key.get(key)
        contexts[key[:4]].append(
            gate_candidate if replacement is None else replacement
        )
        if replacement is not None:
            matched.add(key)
    missing = sorted(set(post_by_key) - matched)
    if missing:
        raise ValueError(
            "gate audit does not contain refined input candidates: "
            + ", ".join(str(key) for key in missing[:3])
        )
    return dict(contexts)


def _preview(
    rgb_bgr: np.ndarray,
    observed_mask: np.ndarray,
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> np.ndarray:
    image = rgb_bgr.copy()
    layers = (
        (observed_mask, (255, 0, 0)),
        (initial_mask, (0, 0, 255)),
        (refined_mask, (0, 255, 0)),
    )
    for mask, color in layers:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, 2)
    cv2.putText(
        image,
        "observed=blue initial=red refined=green",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--object-config", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE_ROOT / "datasets/bop/tless",
    )
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--render-point-count", type=int, default=5000)
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="print progress every N groups; use 0 to disable",
    )
    parser.add_argument("--reweight-score", action="store_true")
    parser.add_argument(
        "--mode-aware",
        action="store_true",
        help=(
            "cluster symmetry-equivalent Top-K candidates, refine one center "
            "per mode, propagate the correction, then regroup after scoring"
        ),
    )
    parser.add_argument(
        "--mode-translation-threshold-m", type=float, default=0.03
    )
    parser.add_argument(
        "--mode-rotation-threshold-deg", type=float, default=15.0
    )
    parser.add_argument(
        "--mode-continuous-symmetry-samples", type=int, default=360
    )
    parser.add_argument(
        "--reliability-model",
        type=Path,
        help="frozen model used to recalibrate reliability after refinement",
    )
    parser.add_argument(
        "--predictive-covariance-model",
        type=Path,
        help=(
            "frozen GT-calibrated covariance added to within-mode dispersion; "
            "requires --mode-aware"
        ),
    )
    parser.add_argument(
        "--gate-audit",
        type=Path,
        help="gate audit retaining rejected candidates for group-relative features",
    )
    args = parser.parse_args()
    if (args.reliability_model is None) != (args.gate_audit is None):
        raise SystemExit(
            "--reliability-model and --gate-audit must be provided together"
        )
    if args.reweight_score and args.reliability_model is not None:
        raise SystemExit(
            "--reweight-score cannot be combined with reliability recalibration"
        )
    if args.predictive_covariance_model is not None and not args.mode_aware:
        raise SystemExit("--predictive-covariance-model requires --mode-aware")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")

    os.environ.setdefault("BOP_PATH", str(args.dataset_root.resolve().parent))
    object_config = load_object_config(args.object_config)
    object_suffix = object_config.cad_object.object_id.rsplit("_", 1)[-1]
    object_id = int(object_suffix) if object_suffix.isdigit() else None
    if object_id is None:
        raise SystemExit("object config ID must end in a numeric BOP object ID")
    records = _load_json(args.input)
    if not isinstance(records, list):
        raise SystemExit("input pose file must contain a JSON list")
    records = [item for item in records if int(item.get("obj_id", -1)) == object_id]
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_record_key(record)].append(record)
    selected_keys = sorted(groups)
    if args.limit_groups is not None:
        selected_keys = selected_keys[:args.limit_groups]

    detections: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for detection in _load_json(args.detections):
        key = (
            int(detection["scene_id"]),
            int(detection["image_id"]),
            int(detection["category_id"]),
        )
        if key[2] == object_id:
            detections[key].append(detection)
    for group in detections.values():
        group.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    geometry = load_ascii_ply_geometry(
        object_config.cad_object.mesh_path,
        units_to_m=object_config.cad_object.units_to_m,
    )
    model_points = sample_model_points(geometry.vertices_m, args.render_point_count)
    settings = PoseRefinementConfig()
    mode_settings = PoseModeClusteringConfig(
        maximum_translation_distance_m=args.mode_translation_threshold_m,
        maximum_rotation_distance_deg=args.mode_rotation_threshold_deg,
        continuous_symmetry_samples=args.mode_continuous_symmetry_samples,
    )
    reliability_model = (
        None
        if args.reliability_model is None
        else PoseReliabilityModel.load(args.reliability_model)
    )
    predictive_model = (
        None
        if args.predictive_covariance_model is None
        else PredictivePoseCovarianceModel.load(args.predictive_covariance_model)
    )
    try:
        predictive_covariance = (
            None
            if predictive_model is None
            else predictive_model.covariance_for(
                object_config.cad_object.object_id
            )
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    gate_payload = None if args.gate_audit is None else _load_json(args.gate_audit)
    gate_candidates: list[dict[str, Any]] = []
    post_gate_settings = PoseGateConfig()
    post_gate_model_points = model_points
    cad_scale_m = float(
        np.linalg.norm(
            np.max(geometry.vertices_m, axis=0)
            - np.min(geometry.vertices_m, axis=0)
        )
    )
    if reliability_model is not None:
        if not isinstance(gate_payload, dict) or not isinstance(
            gate_payload.get("candidates"), list
        ):
            raise SystemExit("gate audit must contain a candidates list")
        gate_candidates = gate_payload["candidates"]
        gate_config = gate_payload.get("summary", {}).get("config", {})
        if not isinstance(gate_config, dict):
            raise SystemExit("gate audit summary.config must be an object")
        post_gate_settings = PoseGateConfig(**gate_config)
        post_gate_model_points = sample_model_points(
            geometry.vertices_m, post_gate_settings.render_point_count
        )
    split_root = args.dataset_root / args.split

    @lru_cache(maxsize=24)
    def load_observation(scene_id: int, image_id: int):
        root = split_root / f"{scene_id:06d}"
        camera = _load_json(root / "scene_camera.json")[str(image_id)]
        depth_raw = cv2.imread(
            str(root / "depth" / f"{image_id:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        rgb = cv2.imread(str(root / "rgb" / f"{image_id:06d}.png"))
        if depth_raw is None or rgb is None:
            raise FileNotFoundError(f"missing RGB-D image {scene_id}/{image_id}")
        depth_m = (
            depth_raw.astype(np.float64)
            * float(camera.get("depth_scale", 1.0))
            / 1000.0
        )
        intrinsics = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
        return rgb, depth_m, intrinsics

    output_records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    post_candidates: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    preview_count = 0
    initial_mixtures = []
    mode_refinement_audits: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    for group_index, key in enumerate(selected_keys, start=1):
        scene_id, image_id, _, instance_id = key
        detection_group = detections.get((scene_id, image_id, object_id), [])
        candidates = sorted(
            groups[key], key=lambda item: float(item.get("score", 0.0)), reverse=True
        )[:args.top_k]
        if instance_id >= len(detection_group):
            reasons["missing_observation_mask"] += len(candidates)
            continue
        rgb, depth_m, intrinsics = load_observation(scene_id, image_id)
        observed_mask = _align_mask(
            decode_coco_rle(detection_group[instance_id]["segmentation"]),
            depth_m.shape,
        )
        mode_result_by_hypothesis: dict[int, Any] = {}
        mode_id_by_hypothesis: dict[int, int] = {}
        corrected_by_hypothesis: dict[int, np.ndarray] = {}
        if args.mode_aware:
            hypothesis_set = _records_to_hypothesis_set(
                candidates, object_config.cad_object.object_id
            )
            initial_mixture = cluster_pose_hypotheses(
                hypothesis_set,
                object_config.cad_object,
                mode_settings,
                confidence_is_calibrated_probability=(
                    reliability_model is not None
                ),
            )
            mode_refinement = refine_pose_mixture_rgbd(
                hypothesis_set,
                initial_mixture,
                model_points,
                intrinsics,
                observed_mask,
                depth_m,
                settings,
            )
            initial_mixtures.append(initial_mixture)
            mode_refinement_audits.append(
                {
                    "scene_id": scene_id,
                    "image_id": image_id,
                    "object_id": object_id,
                    "instance_id": instance_id,
                    **mode_refinement.to_dict(),
                }
            )
            corrected_by_hypothesis = {
                item.hypothesis_id: item.transform_camera_object
                for item in mode_refinement.hypothesis_set.hypotheses
            }
            for item in mode_refinement.mode_refinements:
                for hypothesis_id in item.member_hypothesis_ids:
                    mode_result_by_hypothesis[hypothesis_id] = item.result
                    mode_id_by_hypothesis[hypothesis_id] = item.mode_id
        for record in candidates:
            initial = foundpose_record_transform(record)
            hypothesis_id = int(record.get("hypothesis_id", 0))
            if args.mode_aware:
                result = mode_result_by_hypothesis[hypothesis_id]
                refined_transform = corrected_by_hypothesis[hypothesis_id]
            else:
                result = refine_pose_rgbd(
                    model_points,
                    initial,
                    intrinsics,
                    observed_mask,
                    depth_m,
                    settings,
                )
                refined_transform = result.refined_transform_camera_object
            output = dict(record)
            output["R"] = refined_transform[:3, :3].tolist()
            output["t"] = (
                1000.0 * refined_transform[:3, 3, None]
            ).tolist()
            input_gt = {field: output.pop(field) for field in STALE_GT_FIELDS if field in output}
            output["source"] = (
                "foundpose_cad_rgbd_refined"
                if result.accepted
                else "foundpose_cad_rgbd_refinement_no_change"
            )
            output["pose_refinement"] = result.to_dict()
            if args.mode_aware:
                output["pose_refinement"].update(
                    {
                        "mode_id": mode_id_by_hypothesis[hypothesis_id],
                        "mode_center_refined": True,
                        "correction_propagated_to_member": True,
                    }
                )
            output["pose_refinement"]["input_evaluation_only"] = input_gt
            post_candidate = None
            if reliability_model is not None:
                scoring_record = dict(output)
                scoring_record["score"] = _raw_pnp_score(record)
                decision = evaluate_pose_candidate(
                    scoring_record,
                    post_gate_model_points,
                    intrinsics,
                    observed_mask,
                    depth_m,
                    config=post_gate_settings,
                )
                post_candidate = _post_refinement_audit(record, decision)
                post_candidates.append(post_candidate)
            if args.reweight_score:
                pre_score = float(record.get("score", 0.0))
                quality = float(np.clip(1.0 - result.refined_metrics.data_loss, 0.0, 1.0))
                output["pre_refinement_score"] = pre_score
                output["score"] = pre_score * quality
            output_records.append(output)
            audit = {
                "scene_id": scene_id,
                "image_id": image_id,
                "object_id": object_id,
                "instance_id": instance_id,
                "hypothesis_id": int(record.get("hypothesis_id", 0)),
                **result.to_dict(),
            }
            if args.mode_aware:
                audit.update(
                    {
                        "mode_id": mode_id_by_hypothesis[hypothesis_id],
                        "mode_center_refined": True,
                        "correction_propagated_to_member": True,
                    }
                )
            if post_candidate is not None:
                audit["post_refinement_pose_audit"] = post_candidate
            audits.append(audit)
            reasons["accepted" if result.accepted else "no_change"] += 1
            if result.accepted and preview_count < args.preview_limit:
                initial_mask, _, _ = render_model_points(
                    model_points,
                    initial,
                    intrinsics,
                    depth_m.shape,
                    splat_radius_px=settings.render_splat_radius_px,
                )
                refined_mask, _, _ = render_model_points(
                    model_points,
                    refined_transform,
                    intrinsics,
                    depth_m.shape,
                    splat_radius_px=settings.render_splat_radius_px,
                )
                preview_dir.mkdir(parents=True, exist_ok=True)
                name = (
                    f"{scene_id}_{image_id}_{object_id}_{instance_id}_"
                    f"{int(record.get('hypothesis_id', 0))}.jpg"
                )
                cv2.imwrite(
                    str(preview_dir / name),
                    _preview(rgb, observed_mask, initial_mask, refined_mask),
                )
                preview_count += 1
        if args.progress_every and (
            group_index % args.progress_every == 0
            or group_index == len(selected_keys)
        ):
            print(
                f"refinement progress: {group_index}/{len(selected_keys)} groups",
                file=sys.stderr,
                flush=True,
            )

    post_rejected_count = 0
    pre_probabilities: list[float] = []
    post_probabilities: list[float] = []
    changed_top1_groups = 0
    if reliability_model is not None:
        try:
            reliability_contexts = _reliability_contexts(
                gate_candidates, post_candidates
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        output_by_key = {_candidate_key(item): item for item in output_records}
        audit_by_key = {_candidate_key(item): item for item in audits}
        calibrated_records: list[dict[str, Any]] = []
        pre_top1: dict[tuple[int, int, int, int], int] = {}
        post_top1: dict[tuple[int, int, int, int], int] = {}
        for group_key, context in reliability_contexts.items():
            refined_group = [
                item for item in context if _candidate_key(item) in output_by_key
            ]
            if refined_group:
                pre_top1[group_key] = int(
                    max(
                        refined_group,
                        key=lambda item: float(
                            output_by_key[_candidate_key(item)].get("score", 0.0)
                        ),
                    )["hypothesis_id"]
                )
            group_calibrated: list[dict[str, Any]] = []
            for post_candidate in refined_group:
                key = _candidate_key(post_candidate)
                output = output_by_key[key]
                audit = audit_by_key[key]
                pre_probability = float(output.get("score", 0.0))
                pre_probabilities.append(pre_probability)
                if not post_candidate["accepted"]:
                    post_rejected_count += 1
                    audit["post_refinement_reliability"] = {
                        "accepted": False,
                        "rejection_reasons": post_candidate["rejection_reasons"],
                    }
                    continue
                features = pose_reliability_features(
                    post_candidate, cad_scale_m, context
                )
                probability = float(reliability_model.predict_proba(features))
                feature_payload = dict(
                    zip(RELIABILITY_FEATURE_NAMES, features.tolist())
                )
                output["pre_refinement_score"] = pre_probability
                output["pre_refinement_calibrated_probability"] = output.get(
                    "calibrated_probability"
                )
                output["pre_refinement_reliability_features"] = output.get(
                    "reliability_features"
                )
                output["post_refinement_pose_gate_metrics"] = post_candidate[
                    "metrics"
                ]
                output["post_refinement_geometric_quality"] = post_candidate[
                    "geometric_quality"
                ]
                output["post_refinement_heuristic_robust_score"] = post_candidate[
                    "robust_score"
                ]
                output["post_refinement_reliability_features"] = feature_payload
                output["reliability_features"] = feature_payload
                output["calibrated_probability"] = probability
                output["score"] = probability
                output["source"] = output["source"] + "_recalibrated"
                audit["post_refinement_reliability"] = {
                    "accepted": True,
                    "pre_refinement_probability": pre_probability,
                    "calibrated_probability": probability,
                    "features": feature_payload,
                }
                post_probabilities.append(probability)
                calibrated_records.append(output)
                group_calibrated.append(output)
            if group_calibrated:
                post_top1[group_key] = int(
                    max(
                        group_calibrated,
                        key=lambda item: float(item["score"]),
                    ).get("hypothesis_id", 0)
                )
        changed_top1_groups = sum(
            pre_top1[key] != post_top1.get(key)
            for key in pre_top1.keys() & post_top1.keys()
        )
        output_records = calibrated_records

    refined_mixtures = []
    if args.mode_aware:
        refined_groups: dict[
            tuple[int, int, int, int], list[dict[str, Any]]
        ] = defaultdict(list)
        for record in output_records:
            refined_groups[_record_key(record)].append(record)
        for group in refined_groups.values():
            refined_set = _records_to_hypothesis_set(
                group, object_config.cad_object.object_id
            )
            refined_mixtures.append(
                cluster_pose_hypotheses(
                    refined_set,
                    object_config.cad_object,
                    mode_settings,
                    predictive_covariance_se3=predictive_covariance,
                    predictive_uncertainty_method=(
                        None if predictive_model is None else predictive_model.method
                    ),
                    confidence_is_calibrated_probability=(
                        reliability_model is not None
                    ),
                )
            )

    improvements = [item["objective_improvement"] for item in audits]
    accepted_mode_count = sum(
        int(item["accepted_mode_count"]) for item in mode_refinement_audits
    )
    initial_mode_count = sum(len(item.modes) for item in initial_mixtures)
    refined_mode_count = sum(len(item.modes) for item in refined_mixtures)
    summary = {
        "schema_version": 1,
        "method": (
            "foundpose_symmetry_modes_rgbd_refinement_rescore"
            if args.mode_aware
            else "foundpose_cad_silhouette_depth_feedback_refinement"
        ),
        "object_id": object_id,
        "input": str(args.input.resolve()),
        "detections": str(args.detections.resolve()),
        "group_count": len(selected_keys),
        "candidate_count": len(output_records),
        "accepted_refinement_count": reasons["accepted"],
        "no_change_count": reasons["no_change"],
        "missing_mask_candidate_count": reasons["missing_observation_mask"],
        "mean_objective_improvement": (
            None if not improvements else float(np.mean(improvements))
        ),
        "median_objective_improvement": (
            None if not improvements else float(np.median(improvements))
        ),
        "config": settings.to_dict(),
        "ground_truth_used_for_refinement": False,
        "stale_input_gt_metrics_removed": True,
        "score_reweighted": args.reweight_score,
        "post_refinement_reliability_recalibrated": reliability_model is not None,
        "post_refinement_rejected_candidate_count": post_rejected_count,
        "pre_refinement_probability_mean": (
            None if not pre_probabilities else float(np.mean(pre_probabilities))
        ),
        "post_refinement_probability_mean": (
            None if not post_probabilities else float(np.mean(post_probabilities))
        ),
        "recalibration_changed_top1_group_count": changed_top1_groups,
        "mode_aware": args.mode_aware,
        "mode_clustering_config": (
            mode_settings.to_dict() if args.mode_aware else None
        ),
        "initial_mode_count": initial_mode_count,
        "accepted_mode_refinement_count": accepted_mode_count,
        "refined_mode_count": refined_mode_count,
        "mode_refinement_evaluation_count": (
            sum(
                mode["refinement"]["evaluations"]
                for group in mode_refinement_audits
                for mode in group["modes"]
            )
            if args.mode_aware
            else 0
        ),
        "reliability_model": (
            None
            if args.reliability_model is None
            else {
                "path": str(args.reliability_model.resolve()),
                "sha256": hashlib.sha256(
                    args.reliability_model.read_bytes()
                ).hexdigest(),
                "ground_truth_used_at_runtime": False,
            }
        ),
        "predictive_covariance_model": (
            None
            if args.predictive_covariance_model is None
            else {
                "path": str(args.predictive_covariance_model.resolve()),
                "sha256": hashlib.sha256(
                    args.predictive_covariance_model.read_bytes()
                ).hexdigest(),
                "conditional_on": predictive_model.conditional_on,
                "object_ids": list(predictive_model.object_ids),
            }
        ),
        "gate_audit": (
            None if args.gate_audit is None else str(args.gate_audit.resolve())
        ),
        "output_score_semantics": (
            "post_refinement_calibrated_probability"
            if reliability_model is not None
            else "heuristic_refinement_weight"
            if args.reweight_score
            else "input_score_preserved"
        ),
    }
    (args.output_dir / "refined-poses.json").write_text(
        json.dumps(output_records, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "candidates": audits,
                "mode_refinements": mode_refinement_audits,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.mode_aware:
        (args.output_dir / "pose-modes-initial.json").write_text(
            json.dumps(
                pose_mixtures_to_jsonable(initial_mixtures), indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "pose-modes-refined.json").write_text(
            json.dumps(
                pose_mixtures_to_jsonable(refined_mixtures), indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
