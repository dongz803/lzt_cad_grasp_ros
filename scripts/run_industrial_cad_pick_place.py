#!/usr/bin/env python3
"""Run a same-source industrial-CAD pick-and-place cycle in MuJoCo.
    在仿真环境里面把工业CAD模型抓起来然后放到指定位置

"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
from _bootstrap import activate_project_packages

activate_project_packages(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-config",
        type=Path,
        default=PROJECT_ROOT / "lzt_description/config/objects/tless_000004.yaml",
    )
    parser.add_argument(
        "--grasp-candidates",
        type=Path,
        default=PROJECT_ROOT / "outputs/cad_grasps/tless_000004.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="explicit new output directory; must be empty or absent",
    )
    parser.add_argument(
        "--run-id",
        help="optional reproducible run label used under the default output root",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scene-name",
        default="industrial_cad_single.xml",
        help="MuJoCo scene under scenes/; defaults to the precise-placement baseline",
    )
    parser.add_argument(
        "--slot",
        choices=[
            *(f"place_slot_{index:02d}" for index in range(1, 7)),
            "place_region_center",
        ],
        default="place_slot_02",
    )
    parser.add_argument(
        "--acceptance-mode",
        choices=("precise_pose", "grasp_focus"),
        default="precise_pose",
        help="precise pose tolerance or grasp-focused rectangular-region acceptance",
    )
    parser.add_argument(
        "--placement-region-min-xy",
        type=float,
        nargs=2,
        metavar=("X_MIN", "Y_MIN"),
        default=(-0.38, 0.09),
    )
    parser.add_argument(
        "--placement-region-max-xy",
        type=float,
        nargs=2,
        metavar=("X_MAX", "Y_MAX"),
        default=(-0.02, 0.47),
    )
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--initial-xy-m",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=(-0.2, -0.12),
    )
    parser.add_argument("--lift-height-m", type=float, default=0.14)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--path-samples", type=int, default=25)
    parser.add_argument(
        "--export-bop-root",
        type=Path,
        help="export the initial MuJoCo observation as a one-image T-LESS BOP split",
    )
    parser.add_argument(
        "--pose-hypotheses",
        type=Path,
        help="project-native Top-K JSON for this exact camera observation",
    )
    parser.add_argument("--pose-set-index", type=int, default=0)
    parser.add_argument(
        "--max-sim-pose-position-error-m", type=float, default=0.03
    )
    parser.add_argument(
        "--max-sim-pose-angle-error-deg", type=float, default=20.0
    )
    parser.add_argument(
        "--grasp-ranking",
        choices=("baseline", "cvar"),
        default="baseline",
        help="fixed expected-score baseline or Top-K CVaR tail-risk ranking",
    )
    parser.add_argument("--cvar-tail-probability", type=float, default=0.25)
    parser.add_argument("--cvar-risk-weight", type=float, default=0.5)
    parser.add_argument("--minimum-grasp-score", type=float, default=0.0)
    parser.add_argument(
        "--confidence-is-calibrated-probability",
        action="store_true",
        help=(
            "preserve absolute Top-K correctness probability and add an "
            "explicit no-valid-pose failure hypothesis"
        ),
    )
    parser.add_argument("--minimum-pose-set-reliability", type=float, default=0.0)
    parser.add_argument("--maximum-cvar-failure-risk", type=float, default=1.0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--close-on-finish", action="store_true")
    return parser.parse_args()


def _safe_run_id(raw: str | None) -> str:
    if raw is None:
        return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    if not raw or re.fullmatch(r"[A-Za-z0-9_.-]+", raw) is None:
        raise ValueError("run-id may contain only letters, numbers, dot, dash and underscore")
    return raw


def _execution_mode(args: argparse.Namespace) -> str:
    source = "gt" if args.pose_hypotheses is None else "predicted"
    phase = "check" if args.check_only else "dynamic"
    profile = "_grasp_focus" if getattr(args, "acceptance_mode", "precise_pose") == "grasp_focus" else ""
    ranking = "_cvar" if getattr(args, "grasp_ranking", "baseline") == "cvar" else ""
    return f"{source}_{phase}{profile}{ranking}"


def _prepare_output_directory(
    args: argparse.Namespace,
    object_id: str,
) -> Path:
    run_id = _safe_run_id(args.run_id)
    output = (
        PROJECT_ROOT
        / "outputs/industrial_cad_pick_place"
        / object_id
        / _execution_mode(args)
        / f"{run_id}_seed{args.seed}"
        if args.output is None
        else args.output
    ).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty run directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot() -> dict[str, Any]:
    def command(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_short": command("status", "--short"),
    }


def _snapshot_run_inputs(
    args: argparse.Namespace,
    output_dir: Path,
    object_id: str,
) -> Path:
    snapshot_dir = output_dir / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "object_config": args.object_config.resolve(),
        "grasp_candidates": args.grasp_candidates.resolve(),
        "scene": (PROJECT_ROOT / "lzt_description/scenes" / args.scene_name).resolve(),
    }
    if args.pose_hypotheses is not None:
        inputs["pose_hypotheses"] = args.pose_hypotheses.resolve()
    input_records = {}
    for label, source in inputs.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix or ".dat"
        destination = snapshot_dir / f"{label}{suffix}"
        shutil.copy2(source, destination)
        input_records[label] = {
            "source": str(source),
            "snapshot": str(destination),
            "sha256": _sha256(source),
        }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "run_id": output_dir.name,
        "object_id": object_id,
        "mode": _execution_mode(args),
        "random_seed": args.seed,
        "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        "output_directory": str(output_dir),
        "inputs": input_records,
        "git": _git_snapshot(),
        "environment": {
            "python": sys.version,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "bop_path": os.environ.get("BOP_PATH"),
        },
    }
    path = output_dir / "run_manifest.json"
    _write_json(path, manifest)
    return path


def _complete_run_manifest(
    path: Path,
    *,
    success: bool,
    result_path: Path,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completed_at"] = datetime.now().astimezone().isoformat()
    payload["success"] = success
    payload["result"] = str(result_path.resolve())
    _write_json(path, payload)


def _candidate_from_dict(item: dict[str, Any]):
    from lzt_common.contracts import GraspCandidate

    return GraspCandidate(
        grasp_id=str(item["grasp_id"]),
        transform_object_grasp=np.asarray(
            item["transform_object_grasp"], dtype=np.float64
        ),
        jaw_opening_m=float(item["jaw_opening_m"]),
        analytic_quality=float(item.get("analytic_quality", 0.0)),
        approach_distance_m=float(item.get("approach_distance_m", 0.08)),
        antipodal_score=float(item.get("antipodal_score", 0.0)),
        contact_points_object_m=(
            None
            if item.get("contact_points_object_m") is None
            else np.asarray(item["contact_points_object_m"], dtype=np.float64)
        ),
        contact_normals_object=(
            None
            if item.get("contact_normals_object") is None
            else np.asarray(item["contact_normals_object"], dtype=np.float64)
        ),
        center_of_mass_offset_m=float(
            item.get("center_of_mass_offset_m", 0.0)
        ),
        minimum_edge_distance_m=float(
            item.get("minimum_edge_distance_m", 0.0)
        ),
        opening_margin_m=float(item.get("opening_margin_m", 0.0)),
        finger_collision_free=bool(item.get("finger_collision_free", True)),
    )


def _load_candidates(path: Path, object_id: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("object_id") != object_id:
        raise ValueError("grasp candidates and object config do not match")
    candidates = [_candidate_from_dict(item) for item in payload["candidates"]]
    if not candidates:
        raise ValueError("grasp candidate file is empty")
    return candidates


def _pose_set_from_dict(raw: dict[str, Any]):
    from lzt_common.contracts import PoseHypothesis, PoseHypothesisSet

    covariance_raw = raw.get("covariance_se3")
    covariance = (
        None
        if covariance_raw is None
        else np.asarray(covariance_raw, dtype=np.float64)
    )
    hypotheses = tuple(
        PoseHypothesis(
            transform_camera_object=np.asarray(
                item["transform_camera_object"], dtype=np.float64
            ),
            confidence=float(item["confidence"]),
            symmetry_class=str(item.get("symmetry_class", "identity")),
            covariance_se3=covariance,
            source=str(item.get("source", "unknown")),
            hypothesis_id=int(item.get("hypothesis_id", index)),
        )
        for index, item in enumerate(raw["hypotheses"])
    )
    return PoseHypothesisSet(
        object_id=str(raw["object_id"]),
        scene_id=int(raw["scene_id"]),
        image_id=int(raw["image_id"]),
        instance_id=int(raw["instance_id"]),
        hypotheses=hypotheses,
        covariance_se3=covariance,
        uncertainty_method=str(raw.get("uncertainty_method", "unavailable")),
        symmetry_discretization_deg=raw.get("symmetry_discretization_deg"),
    )


def _load_pose_set(path: Path, set_index: int, object_id: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("pose-hypotheses JSON must be a non-empty list")
    if not 0 <= set_index < len(payload):
        raise ValueError("pose-set-index is outside the JSON list")
    result = _pose_set_from_dict(payload[set_index])
    if result.object_id != object_id:
        raise ValueError("pose hypotheses and object config do not match")
    return result


def _arm_dof_indices(model, joint_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [model.jnt_dofadr[model.joint(name).id] for name in joint_names],
        dtype=np.int32,
    )


def _arm_qpos_addresses(model, joint_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [model.jnt_qposadr[model.joint(name).id] for name in joint_names],
        dtype=np.int32,
    )


def _external_camera(mujoco):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-0.25, 0.03, 0.48]
    camera.distance = 1.45
    camera.azimuth = 145
    camera.elevation = -24
    return camera


def _depth_preview(depth_m: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth_m) & (depth_m > 0.0)
    result = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not finite.any():
        return result
    near, far = np.percentile(depth_m[finite], [2.0, 98.0])
    span = max(float(far - near), 1e-6)
    intensity = np.zeros(depth_m.shape, dtype=np.uint8)
    normalized = np.clip((depth_m[finite] - near) / span, 0.0, 1.0)
    intensity[finite] = ((1.0 - normalized) * 255.0).astype(np.uint8)
    result[finite] = np.repeat(intensity[..., None], 3, axis=2)[finite]
    return result


def _save_initial_observation(
    mujoco,
    model,
    data,
    output_dir: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    from lzt_mujoco.industrial_workcell import (
        FIXED_CAMERA_NAME,
        camera_intrinsics,
        camera_object_transform,
        target_geom_ids,
    )

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=FIXED_CAMERA_NAME)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=FIXED_CAMERA_NAME)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=FIXED_CAMERA_NAME)
    segmentation = renderer.render().copy()
    renderer.close()
    mask = (
        np.isin(segmentation[..., 0], target_geom_ids(model))
        & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    )
    if not mask.any():
        raise RuntimeError("the injected industrial CAD is not visible to vision_camera")

    overlay = rgb.copy()
    overlay[mask] = (
        0.45 * overlay[mask] + 0.55 * np.array([30, 230, 90])
    ).astype(np.uint8)
    Image.fromarray(rgb).save(output_dir / "initial_rgb.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(
        output_dir / "initial_gt_mask.png"
    )
    Image.fromarray(overlay).save(output_dir / "initial_mask_overlay.png")
    Image.fromarray(_depth_preview(depth)).save(output_dir / "initial_depth.png")
    np.save(output_dir / "initial_depth_m.npy", depth.astype(np.float32))
    transform_camera_object = camera_object_transform(model, data)
    intrinsics = camera_intrinsics(model, FIXED_CAMERA_NAME, width, height)
    return {
        "camera_name": FIXED_CAMERA_NAME,
        "image_size": [height, width],
        "camera_intrinsics": intrinsics.tolist(),
        "transform_camera_object": transform_camera_object.tolist(),
        "target_mask_pixel_count": int(np.count_nonzero(mask)),
        "depth_valid_ratio": float(np.mean(np.isfinite(depth) & (depth > 0.0))),
        "pose_source": "mujoco_ground_truth",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _export_tless_bop_sample(
    export_root: Path,
    source_models_cad: Path,
    output_dir: Path,
    observation: dict[str, Any],
    object_id: str,
) -> Path:
    """Export the captured frame in the exact layout consumed by FoundPose."""

    if not object_id.startswith("tless_"):
        raise ValueError("BOP export currently supports T-LESS object configs")
    object_lid = int(object_id.rsplit("_", 1)[1])
    dataset_root = export_root.resolve() / "tless"
    models_link = dataset_root / "models_cad"
    dataset_root.mkdir(parents=True, exist_ok=True)
    if models_link.is_symlink():
        if models_link.resolve() != source_models_cad.resolve():
            raise ValueError("existing models_cad link points to another dataset")
    elif models_link.exists():
        raise ValueError("BOP export models_cad exists and is not a symlink")
    else:
        models_link.symlink_to(source_models_cad.resolve(), target_is_directory=True)

    split_root = dataset_root / "test_primesense"
    scene_root = split_root / "000001"
    for directory in ("rgb", "depth", "mask", "mask_visib"):
        (scene_root / directory).mkdir(parents=True, exist_ok=True)

    rgb = Image.open(output_dir / "initial_rgb.png").convert("RGB")
    mask = np.asarray(
        Image.open(output_dir / "initial_gt_mask.png").convert("L")
    ) > 0
    depth_m = np.load(output_dir / "initial_depth_m.npy")
    height, width = mask.shape
    if rgb.size != (width, height) or depth_m.shape != mask.shape:
        raise ValueError("captured RGB, depth and mask sizes do not match")
    depth_mm = np.zeros(mask.shape, dtype=np.uint16)
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_mm[valid_depth] = np.clip(
        np.rint(depth_m[valid_depth] * 1000.0), 1, 65535
    ).astype(np.uint16)
    rgb.save(scene_root / "rgb/000000.png")
    Image.fromarray(depth_mm).save(scene_root / "depth/000000.png")
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_image.save(scene_root / "mask/000000_000000.png")
    mask_image.save(scene_root / "mask_visib/000000_000000.png")

    ys, xs = np.nonzero(mask)
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]
    transform = np.asarray(
        observation["transform_camera_object"], dtype=np.float64
    )
    intrinsics = np.asarray(
        observation["camera_intrinsics"], dtype=np.float64
    )
    _write_json(
        scene_root / "scene_camera.json",
        {
            "0": {
                "cam_K": intrinsics.reshape(-1).tolist(),
                "depth_scale": 1.0,
            }
        },
    )
    _write_json(
        scene_root / "scene_gt.json",
        {
            "0": [
                {
                    "cam_R_m2c": transform[:3, :3].reshape(-1).tolist(),
                    "cam_t_m2c": (transform[:3, 3] * 1000.0).tolist(),
                    "obj_id": object_lid,
                }
            ]
        },
    )
    pixel_count = int(np.count_nonzero(mask))
    _write_json(
        scene_root / "scene_gt_info.json",
        {
            "0": [
                {
                    "bbox_obj": bbox,
                    "bbox_visib": bbox,
                    "px_count_all": pixel_count,
                    "px_count_valid": pixel_count,
                    "px_count_visib": pixel_count,
                    "visib_fract": 1.0,
                }
            ]
        },
    )
    _write_json(
        dataset_root / "test_targets_bop19.json",
        [{"im_id": 0, "inst_count": 1, "obj_id": object_lid, "scene_id": 1}],
    )
    # The pose gate consumes the standard BOP detection contract. This record
    # intentionally uses the simulator mask so the first system integration
    # test measures pose estimation and execution rather than segmentation.
    from lzt_perception.mask_refinement import binary_mask_to_rle

    _write_json(
        export_root.resolve() / "gt-mask-detections.json",
        [
            {
                "scene_id": 1,
                "image_id": 0,
                "category_id": object_lid,
                "bbox": bbox,
                "score": 1.0,
                "time": 0.0,
                "segmentation": binary_mask_to_rle(mask),
            }
        ],
    )
    _write_json(
        export_root.resolve() / "manifest.json",
        {
            "dataset": "tless",
            "split": "test_primesense",
            "scene_id": 1,
            "image_id": 0,
            "object_id": object_lid,
            "pose_source": "mujoco_ground_truth_for_evaluation_only",
            "rgb_source": str(output_dir / "initial_rgb.png"),
        },
    )
    return export_root.resolve()


def _solve_execution_plan(
    model,
    data,
    candidate,
    transform_world_pick: np.ndarray,
    transform_world_place: np.ndarray,
    lift_height_m: float,
    path_samples: int,
) -> dict[str, Any] | None:
    from lzt_mujoco.robot_constraints import (
        check_joint_path_collisions,
        solve_site_ik,
    )
    from lzt_mujoco.simulation import UR5E_JOINT_NAMES

    pick_pregrasp = transform_world_pick.copy()
    pick_pregrasp[:3, 3] -= (
        candidate.approach_distance_m * transform_world_pick[:3, 2]
    )
    pick_lift = transform_world_pick.copy()
    pick_lift[2, 3] += lift_height_m
    place_pregrasp = transform_world_place.copy()
    place_pregrasp[2, 3] += lift_height_m
    current_site_id = model.site("ur5e_gripper_pinch").id
    transit_z = max(
        float(data.site_xpos[current_site_id, 2]),
        float(pick_lift[2, 3]),
        float(place_pregrasp[2, 3]),
    )
    place_transit = transform_world_place.copy()
    place_transit[2, 3] = transit_z
    waypoints = (
        ("pick_pregrasp", pick_pregrasp),
        ("pick_grasp", transform_world_pick),
        ("pick_lift", pick_lift),
        ("place_transit", place_transit),
        ("place_pregrasp", place_pregrasp),
        ("place_grasp", transform_world_place),
        ("retreat", place_pregrasp),
    )
    qpos_addresses = _arm_qpos_addresses(model, UR5E_JOINT_NAMES)
    previous = data.qpos[qpos_addresses].copy()
    solved: dict[str, list[float]] = {}
    path_audit = []
    elbow_up_seed = np.array([-2.0, -0.8, 1.0, 1.35, 1.55, -3.13])
    for waypoint_index, (name, transform) in enumerate(waypoints):
        seed_options = [previous]
        if waypoint_index == 0:
            # The workcell has two common UR5e IK branches. The home-seeded
            # branch can fold below the table, so also test a known elbow-up
            # seed and accept only a collision-free path from the real start.
            seed_options.append(elbow_up_seed)
        solution = None
        collision = None
        rejected: list[str] = []
        for seed in seed_options:
            candidate_solution = solve_site_ik(
                model,
                data,
                transform,
                initial_joint_positions_rad=seed,
                max_iterations=1000,
            )
            if not candidate_solution.success:
                rejected.append(candidate_solution.reason)
                continue
            candidate_collision = check_joint_path_collisions(
                model,
                data,
                previous,
                candidate_solution.joint_positions_rad,
                samples=path_samples,
            )
            if not candidate_collision.collision_free:
                rejected.extend(candidate_collision.collision_pairs)
                continue
            solution = candidate_solution
            collision = candidate_collision
            break
        if solution is None or collision is None:
            print(
                f"plan_reject={candidate.grasp_id} stage={name} "
                f"reasons={tuple(sorted(set(rejected)))}",
                flush=True,
            )
            return None
        solved[name] = list(solution.joint_positions_rad)
        path_audit.append(
            {
                "to": name,
                "ik": solution.to_dict(),
                "collision": collision.to_dict(),
            }
        )
        previous = np.asarray(solution.joint_positions_rad)
    return {
        "waypoint_transforms_world": {
            name: transform.tolist() for name, transform in waypoints
        },
        "waypoint_joints_rad": solved,
        "path_audit": path_audit,
    }


def _gripper_control_for_opening(jaw_opening_m: float) -> float:
    # The 2F-85 model uses 0=open, 255=closed. A 5 mm preload gives contact
    # force while reproducing the earlier 18 mm-part calibration (~208).
    commanded_opening = max(0.0, jaw_opening_m - 0.005)
    return float(np.clip(255.0 * (0.085 - commanded_opening) / 0.085, 0.0, 240.0))


def _axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _symmetry_aware_angle_deg(
    first: np.ndarray,
    second: np.ndarray,
    continuous_symmetries,
) -> float:
    if not continuous_symmetries:
        return _axis_angle_deg(first, second)
    # For an axis-symmetric object, rotation around the configured local axis
    # is physically equivalent. Compare only the world direction of the axis.
    return max(
        float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(first @ symmetry.axis_xyz, second @ symmetry.axis_xyz),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        for symmetry in continuous_symmetries
    )


def _transformed_bounds_xy(
    transform_world_object: np.ndarray,
    local_bounds_min_m: np.ndarray,
    local_bounds_max_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the world XY bounds of all eight transformed CAD AABB corners."""

    lower = np.asarray(local_bounds_min_m, dtype=np.float64)
    upper = np.asarray(local_bounds_max_m, dtype=np.float64)
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    world_xy = (transform_world_object @ corners.T).T[:, :2]
    return np.min(world_xy, axis=0), np.max(world_xy, axis=0)


def _object_inside_xy_region(
    transform_world_object: np.ndarray,
    local_bounds_min_m: np.ndarray,
    local_bounds_max_m: np.ndarray,
    region_min_xy: np.ndarray,
    region_max_xy: np.ndarray,
) -> bool:
    """Conservatively require the transformed CAD bounds to fit in the region."""

    object_min_xy, object_max_xy = _transformed_bounds_xy(
        transform_world_object,
        local_bounds_min_m,
        local_bounds_max_m,
    )
    return bool(
        np.all(object_min_xy >= np.asarray(region_min_xy, dtype=np.float64))
        and np.all(object_max_xy <= np.asarray(region_max_xy, dtype=np.float64))
    )


def _risk_rejection_reasons(
    robust_score: float,
    pose_set_reliability: float,
    cvar_failure_risk: float,
    minimum_grasp_score: float,
    minimum_pose_set_reliability: float,
    maximum_cvar_failure_risk: float,
) -> list[str]:
    reasons = []
    if robust_score < minimum_grasp_score:
        reasons.append("low_robust_grasp_score")
    if pose_set_reliability < minimum_pose_set_reliability:
        reasons.append("low_pose_set_reliability")
    if cvar_failure_risk > maximum_cvar_failure_risk:
        reasons.append("high_cvar_failure_risk")
    return reasons


def main() -> None:
    args = parse_args()
    if args.lift_height_m <= 0.08 or args.path_samples < 2:
        raise ValueError("lift height must exceed 0.08 m and path samples >= 2")
    if args.realtime_factor <= 0.0:
        raise ValueError("realtime factor must be positive")
    region_min_xy = np.asarray(args.placement_region_min_xy, dtype=np.float64)
    region_max_xy = np.asarray(args.placement_region_max_xy, dtype=np.float64)
    if np.any(region_min_xy >= region_max_xy):
        raise ValueError("placement region minimum must be below maximum on both axes")
    if not 0.0 < args.cvar_tail_probability <= 1.0:
        raise ValueError("cvar tail probability must lie in (0, 1]")
    if not 0.0 <= args.cvar_risk_weight <= 1.0:
        raise ValueError("cvar risk weight must lie in [0, 1]")
    if not 0.0 <= args.minimum_grasp_score <= 1.0:
        raise ValueError("minimum grasp score must lie in [0, 1]")
    if not 0.0 <= args.minimum_pose_set_reliability <= 1.0:
        raise ValueError("minimum pose-set reliability must lie in [0, 1]")
    if not 0.0 <= args.maximum_cvar_failure_risk <= 1.0:
        raise ValueError("maximum CVaR failure risk must lie in [0, 1]")
    if args.viewer:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("BOP_PATH", str(WORKSPACE_ROOT / "datasets/bop"))

    import mujoco

    from lzt_common.contracts import (
        PickPlaceTask,
        PoseHypothesis,
        PoseHypothesisSet,
    )
    from lzt_mujoco.industrial_workcell import (
        TARGET_BODY_NAME,
        TARGET_JOINT_NAME,
        body_transform,
        build_industrial_cad_workcell,
        camera_object_transform,
        set_freejoint_transform,
    )
    from lzt_common.object_config import load_object_config
    from lzt_mujoco.robot_constraints import (
        MujocoUR5eConstraintEvaluator,
        RobotConstraintConfig,
        current_disallowed_robot_contacts,
    )
    from lzt_common.robot_frames import (
        mujoco_base_camera_calibration,
        mujoco_transform_base_world,
    )
    from lzt_mujoco.simulation import (
        GRIPPER_ACTUATOR_NAME,
        UR5E_ACTUATOR_NAMES,
        UR5E_BASE_BODY,
        UR5E_JOINT_NAMES,
        reset_robot_home,
    )
    from lzt_grasp_planning.task_grasp import rank_task_grasps
    from lzt_grasp_planning.task_grasp import RobustScoringConfig

    config = load_object_config(args.object_config)
    candidates = _load_candidates(args.grasp_candidates, config.cad_object.object_id)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output = _prepare_output_directory(args, config.cad_object.object_id)
    run_manifest_path = _snapshot_run_inputs(
        args, args.output, config.cad_object.object_id
    )
    print(f"run_output={args.output}", flush=True)
    mesh_cache = args.output / "cache" / f"{config.cad_object.object_id}_m.obj"
    model, scene_info = build_industrial_cad_workcell(
        config,
        mesh_cache,
        scene_name=args.scene_name,
        initial_xy_m=tuple(args.initial_xy_m),
        initial_yaw_rad=np.deg2rad(args.initial_yaw_deg),
    )
    data = mujoco.MjData(model)
    reset_robot_home(model, data)
    set_freejoint_transform(
        model,
        data,
        TARGET_JOINT_NAME,
        scene_info.initial_transform_world_object,
    )

    arm_dofs = _arm_dof_indices(model, UR5E_JOINT_NAMES)
    forbidden_contacts: set[str] = set()
    viewer = None
    wall_reference = time.perf_counter()
    sim_reference = float(data.time)

    def controlled_step() -> None:
        nonlocal wall_reference, sim_reference
        data.qfrc_applied[arm_dofs] = data.qfrc_bias[arm_dofs]
        mujoco.mj_step(model, data)
        forbidden_contacts.update(
            current_disallowed_robot_contacts(model, data)
        )
        if viewer is not None:
            target_wall = wall_reference + (
                float(data.time) - sim_reference
            ) / args.realtime_factor
            delay = target_wall - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            viewer.sync()

    for _ in range(int(round(1.0 / model.opt.timestep))):
        controlled_step()
    initial_object = body_transform(data, model.body(TARGET_BODY_NAME).id)
    observation = _save_initial_observation(
        mujoco,
        model,
        data,
        args.output,
        args.image_width,
        args.image_height,
    )
    (args.output / "observation.json").write_text(
        json.dumps(observation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.export_bop_root is not None:
        exported_root = _export_tless_bop_sample(
            args.export_bop_root,
            config.cad_object.mesh_path.parent,
            args.output,
            observation,
            config.cad_object.object_id,
        )
        print(f"exported_bop_root={exported_root}", flush=True)

    slot_position = data.site(args.slot).xpos.copy()
    desired_world_object = scene_info.initial_transform_world_object.copy()
    desired_world_object[:2, 3] = slot_position[:2]
    transform_base_world = mujoco_transform_base_world(model, data)
    transform_world_base = np.linalg.inv(transform_base_world)
    transform_base_goal = transform_base_world @ desired_world_object
    calibration = mujoco_base_camera_calibration(model, data)
    transform_camera_object = camera_object_transform(model, data)
    if args.pose_hypotheses is None:
        covariance = np.zeros((6, 6), dtype=np.float64)
        pose_set = PoseHypothesisSet(
            object_id=config.cad_object.object_id,
            scene_id=0,
            image_id=0,
            instance_id=0,
            hypotheses=(
                PoseHypothesis(
                    transform_camera_object=transform_camera_object,
                    confidence=1.0,
                    covariance_se3=covariance,
                    source="mujoco_ground_truth",
                ),
            ),
            covariance_se3=covariance,
            uncertainty_method="exact_mujoco_ground_truth",
        )
        pose_source = "mujoco_ground_truth"
    else:
        pose_set = _load_pose_set(
            args.pose_hypotheses,
            args.pose_set_index,
            config.cad_object.object_id,
        )
        pose_source = f"top_k_file:{args.pose_hypotheses.resolve()}"
    top_pose = pose_set.hypotheses[0].transform_camera_object
    top_pose_position_error = float(
        np.linalg.norm(top_pose[:3, 3] - transform_camera_object[:3, 3])
    )
    top_pose_angle_error = _symmetry_aware_angle_deg(
        top_pose[:3, :3],
        transform_camera_object[:3, :3],
        config.cad_object.continuous_symmetries,
    )
    task = PickPlaceTask(
        transform_world_object_goal=transform_base_goal,
        position_tolerance_m=config.placement_position_tolerance_m,
        angle_tolerance_deg=config.placement_angle_tolerance_deg,
    )
    constraint_evaluator = MujocoUR5eConstraintEvaluator(
        model,
        data,
        transform_world_base,
        RobotConstraintConfig(path_samples=args.path_samples),
    )
    scoring_config = RobustScoringConfig(
        cvar_tail_probability=args.cvar_tail_probability,
        cvar_risk_weight=(
            args.cvar_risk_weight if args.grasp_ranking == "cvar" else 0.0
        ),
        include_place_uncertainty=args.acceptance_mode == "precise_pose",
        confidence_is_calibrated_probability=(
            args.confidence_is_calibrated_probability
        ),
    )
    ranked = rank_task_grasps(
        pose_set,
        candidates,
        task,
        calibration.transform_base_camera_cv,
        constraint_evaluator,
        config=scoring_config,
    )

    selected = None
    selected_evaluation = None
    execution_plan = None
    # The generic evaluator checks direct joint interpolation. The executor
    # adds high transit waypoints, so its complete path audit is authoritative
    # for dynamic execution and may recover a candidate rejected by the direct
    # interpolation baseline.
    for ranked_item in sorted(
        ranked,
        key=lambda item: (
            -item.robust_score,
            -item.candidate.analytic_quality,
            item.candidate.grasp_id,
        ),
    ):
        for evaluation in ranked_item.hypothesis_evaluations:
            if not evaluation.constraints.feasible:
                print(
                    f"constraint_reject={ranked_item.candidate.grasp_id} "
                    f"hypothesis={evaluation.hypothesis_id} "
                    f"reason={evaluation.constraints.reason} "
                    f"collisions={evaluation.constraints.collision_pairs}",
                    flush=True,
                )
            pick_world = (
                transform_world_base @ evaluation.transform_world_grasp_pick
            )
            place_world = (
                transform_world_base @ evaluation.transform_world_grasp_place
            )
            candidate_plan = _solve_execution_plan(
                model,
                data,
                ranked_item.candidate,
                pick_world,
                place_world,
                args.lift_height_m,
                args.path_samples,
            )
            if candidate_plan is not None:
                selected = ranked_item
                selected_evaluation = evaluation
                execution_plan = candidate_plan
                break
        if execution_plan is not None:
            break
    if (
        selected is None
        or selected_evaluation is None
        or execution_plan is None
    ):
        raise RuntimeError("no CAD grasp candidate has a complete collision-free path")
    selected_pose = next(
        hypothesis
        for hypothesis in pose_set.hypotheses
        if hypothesis.hypothesis_id == selected_evaluation.hypothesis_id
    ).transform_camera_object
    selected_pose_position_error = float(
        np.linalg.norm(
            selected_pose[:3, 3] - transform_camera_object[:3, 3]
        )
    )
    selected_pose_angle_error = _symmetry_aware_angle_deg(
        selected_pose[:3, :3],
        transform_camera_object[:3, :3],
        config.cad_object.continuous_symmetries,
    )
    selected_prediction_safe = bool(
        selected_pose_position_error <= args.max_sim_pose_position_error_m
        and selected_pose_angle_error <= args.max_sim_pose_angle_error_deg
    )
    risk_rejection_reasons = _risk_rejection_reasons(
        selected.robust_score,
        selected.pose_set_reliability,
        selected.cvar_failure_risk,
        args.minimum_grasp_score,
        args.minimum_pose_set_reliability,
        args.maximum_cvar_failure_risk,
    )
    selected_risk_accepted = not risk_rejection_reasons
    ground_truth_base_grasp = (
        calibration.transform_base_camera_cv
        @ transform_camera_object
        @ selected.candidate.transform_object_grasp
    )
    grasp_target_position_error = float(
        np.linalg.norm(
            selected_evaluation.transform_world_grasp_pick[:3, 3]
            - ground_truth_base_grasp[:3, 3]
        )
    )
    grasp_target_angle_error = _axis_angle_deg(
        selected_evaluation.transform_world_grasp_pick[:3, :3],
        ground_truth_base_grasp[:3, :3],
    )

    plan_payload = {
        "schema_version": 1,
        "object_id": config.cad_object.object_id,
        "scene_name": args.scene_name,
        "acceptance_mode": args.acceptance_mode,
        "grasp_ranking": args.grasp_ranking,
        "cvar_tail_probability": scoring_config.cvar_tail_probability,
        "cvar_risk_weight": scoring_config.cvar_risk_weight,
        "confidence_is_calibrated_probability": (
            scoring_config.confidence_is_calibrated_probability
        ),
        "minimum_grasp_score": args.minimum_grasp_score,
        "minimum_pose_set_reliability": args.minimum_pose_set_reliability,
        "maximum_cvar_failure_risk": args.maximum_cvar_failure_risk,
        "pose_source": pose_source,
        "selected_hypothesis_id": selected_evaluation.hypothesis_id,
        "top_pose_sim_position_error_m": top_pose_position_error,
        "top_pose_sim_symmetry_aware_angle_error_deg": top_pose_angle_error,
        "selected_pose_sim_position_error_m": selected_pose_position_error,
        "selected_pose_sim_symmetry_aware_angle_error_deg": (
            selected_pose_angle_error
        ),
        "grasp_target_position_error_m": grasp_target_position_error,
        "grasp_target_angle_error_deg": grasp_target_angle_error,
        "prediction_safe_for_dynamic_execution": selected_prediction_safe,
        "candidate_count": len(candidates),
        "selected_grasp_id": selected.candidate.grasp_id,
        "selected_analytic_quality": selected.candidate.analytic_quality,
        "selected_robust_score": selected.robust_score,
        "selected_pose_set_reliability": selected.pose_set_reliability,
        "selected_expected_task_success": selected.expected_task_success,
        "selected_tail_task_success": selected.tail_task_success,
        "selected_cvar_failure_risk": selected.cvar_failure_risk,
        "selected_risk_accepted": selected_risk_accepted,
        "risk_rejection_reasons": risk_rejection_reasons,
        "jaw_opening_m": selected.candidate.jaw_opening_m,
        "gripper_close_control": _gripper_control_for_opening(
            selected.candidate.jaw_opening_m
        ),
        "goal_transform_world_object": desired_world_object.tolist(),
        "execution_plan": execution_plan,
        "constraint_audit": constraint_evaluator.audit_log,
    }
    (args.output / "plan.json").write_text(
        json.dumps(plan_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"selected {selected.candidate.grasp_id}; "
        f"quality={selected.candidate.analytic_quality:.4f}; "
        f"planned 7 collision-free waypoints",
        flush=True,
    )
    if args.check_only:
        check_report = {
            "success": selected_prediction_safe and selected_risk_accepted,
            "mode": "check_only",
            "execution_attempted": False,
            "scene_name": args.scene_name,
            "acceptance_mode": args.acceptance_mode,
            "grasp_ranking": args.grasp_ranking,
            "random_seed": args.seed,
            "object_id": config.cad_object.object_id,
            "selected_grasp_id": selected.candidate.grasp_id,
            "selected_hypothesis_id": selected_evaluation.hypothesis_id,
            "pose_source": pose_source,
            "top_pose_sim_position_error_m": top_pose_position_error,
            "top_pose_sim_symmetry_aware_angle_error_deg": top_pose_angle_error,
            "selected_pose_sim_position_error_m": selected_pose_position_error,
            "selected_pose_sim_symmetry_aware_angle_error_deg": (
                selected_pose_angle_error
            ),
            "grasp_target_position_error_m": grasp_target_position_error,
            "grasp_target_angle_error_deg": grasp_target_angle_error,
            "prediction_safe_for_dynamic_execution": selected_prediction_safe,
            "selected_robust_score": selected.robust_score,
            "selected_pose_set_reliability": selected.pose_set_reliability,
            "selected_expected_task_success": selected.expected_task_success,
            "selected_tail_task_success": selected.tail_task_success,
            "selected_cvar_failure_risk": selected.cvar_failure_risk,
            "selected_risk_accepted": selected_risk_accepted,
            "risk_rejection_reasons": risk_rejection_reasons,
            "observation": observation,
            "plan_file": str(args.output / "plan.json"),
        }
        (args.output / "result.json").write_text(
            json.dumps(check_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _complete_run_manifest(
            run_manifest_path,
            success=selected_prediction_safe and selected_risk_accepted,
            result_path=args.output / "result.json",
        )
        print(json.dumps(check_report, indent=2, ensure_ascii=False))
        return
    execution_rejection_reasons = list(risk_rejection_reasons)
    if not selected_prediction_safe:
        execution_rejection_reasons.append("predicted_pose_exceeds_safety_gate")
    if execution_rejection_reasons:
        refusal_report = {
            "schema_version": 1,
            "success": False,
            "mode": "dynamic_refused",
            "execution_attempted": False,
            "scene_name": args.scene_name,
            "acceptance_mode": args.acceptance_mode,
            "grasp_ranking": args.grasp_ranking,
            "random_seed": args.seed,
            "object_id": config.cad_object.object_id,
            "selected_grasp_id": selected.candidate.grasp_id,
            "selected_hypothesis_id": selected_evaluation.hypothesis_id,
            "pose_source": pose_source,
            "top_pose_sim_position_error_m": top_pose_position_error,
            "top_pose_sim_symmetry_aware_angle_error_deg": top_pose_angle_error,
            "selected_pose_sim_position_error_m": selected_pose_position_error,
            "selected_pose_sim_symmetry_aware_angle_error_deg": (
                selected_pose_angle_error
            ),
            "prediction_safe_for_dynamic_execution": selected_prediction_safe,
            "selected_robust_score": selected.robust_score,
            "selected_pose_set_reliability": selected.pose_set_reliability,
            "selected_expected_task_success": selected.expected_task_success,
            "selected_tail_task_success": selected.tail_task_success,
            "selected_cvar_failure_risk": selected.cvar_failure_risk,
            "selected_risk_accepted": selected_risk_accepted,
            "risk_rejection_reasons": execution_rejection_reasons,
            "grasp_success": False,
            "runtime_forbidden_contacts": [],
            "observation": observation,
            "plan_file": str(args.output / "plan.json"),
        }
        result_path = args.output / "result.json"
        result_path.write_text(
            json.dumps(refusal_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _complete_run_manifest(
            run_manifest_path,
            success=False,
            result_path=result_path,
        )
        print(json.dumps(refusal_report, indent=2, ensure_ascii=False))
        return

    renderer = mujoco.Renderer(model, height=720, width=960)
    stage_images: list[Path] = []

    def save_stage(index: int, name: str) -> None:
        renderer.update_scene(data, camera=_external_camera(mujoco))
        path = args.output / f"{index:02d}_{name}.png"
        Image.fromarray(renderer.render()).save(path)
        stage_images.append(path)
        print(f"stage={name}", flush=True)

    actuator_ids = np.asarray(
        [model.actuator(name).id for name in UR5E_ACTUATOR_NAMES], dtype=int
    )
    gripper_id = model.actuator(GRIPPER_ACTUATOR_NAME).id

    def move_arm(target: list[float], duration_s: float, settle_s: float = 0.25) -> None:
        start = data.ctrl[actuator_ids].copy()
        destination = np.asarray(target, dtype=np.float64)
        steps = max(1, int(round(duration_s / model.opt.timestep)))
        for step in range(steps):
            phase = (step + 1) / steps
            smooth = phase * phase * (3.0 - 2.0 * phase)
            data.ctrl[actuator_ids] = (1.0 - smooth) * start + smooth * destination
            controlled_step()
        for _ in range(max(0, int(round(settle_s / model.opt.timestep)))):
            controlled_step()

    def ramp_gripper(target: float, duration_s: float, settle_s: float) -> None:
        start = float(data.ctrl[gripper_id])
        steps = max(1, int(round(duration_s / model.opt.timestep)))
        for step in range(steps):
            phase = (step + 1) / steps
            data.ctrl[gripper_id] = (1.0 - phase) * start + phase * target
            controlled_step()
        for _ in range(max(0, int(round(settle_s / model.opt.timestep)))):
            controlled_step()

    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [-0.25, 0.03, 0.48]
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -24
        wall_reference = time.perf_counter()
        sim_reference = float(data.time)

    joints = execution_plan["waypoint_joints_rad"]
    save_stage(0, "home")
    move_arm(joints["pick_pregrasp"], 2.5)
    save_stage(1, "pick_pregrasp")
    move_arm(joints["pick_grasp"], 1.5, 0.15)
    save_stage(2, "pick_grasp")
    ramp_gripper(plan_payload["gripper_close_control"], 1.2, 0.8)
    save_stage(3, "closed")
    before_lift_z = float(data.body(TARGET_BODY_NAME).xpos[2])
    move_arm(joints["pick_lift"], 2.5, 0.5)
    maximum_object_z = float(data.body(TARGET_BODY_NAME).xpos[2])
    save_stage(4, "lifted")
    move_arm(joints["place_transit"], 2.0)
    move_arm(joints["place_pregrasp"], 2.5, 0.35)
    save_stage(5, "place_pregrasp")
    move_arm(joints["place_grasp"], 2.5, 0.5)
    save_stage(6, "place")
    release_object = body_transform(data, model.body(TARGET_BODY_NAME).id)
    ramp_gripper(0.0, 3.0, 2.0)
    save_stage(7, "released")
    move_arm(joints["retreat"], 1.5, 0.5)
    for _ in range(int(round(1.0 / model.opt.timestep))):
        controlled_step()
    save_stage(8, "retreated")
    renderer.close()

    final_object = body_transform(data, model.body(TARGET_BODY_NAME).id)
    target_joint_id = model.joint(TARGET_JOINT_NAME).id
    target_dof = model.jnt_dofadr[target_joint_id]
    target_velocity = data.qvel[target_dof : target_dof + 6].copy()
    xy_error = float(
        np.linalg.norm(final_object[:2, 3] - desired_world_object[:2, 3])
    )
    position_error = float(
        np.linalg.norm(final_object[:3, 3] - desired_world_object[:3, 3])
    )
    angle_error = _symmetry_aware_angle_deg(
        final_object[:3, :3],
        desired_world_object[:3, :3],
        config.cad_object.continuous_symmetries,
    )
    stable = bool(
        np.linalg.norm(target_velocity[:3]) < 0.02
        and np.linalg.norm(target_velocity[3:]) < 0.10
    )
    lifted = maximum_object_z - before_lift_z > 0.08
    precise_placement_ok = bool(
        position_error <= config.placement_position_tolerance_m
        and angle_error <= config.placement_angle_tolerance_deg
    )
    final_object_min_xy, final_object_max_xy = _transformed_bounds_xy(
        final_object,
        scene_info.local_bounds_min_m,
        scene_info.local_bounds_max_m,
    )
    placement_region_ok = _object_inside_xy_region(
        final_object,
        scene_info.local_bounds_min_m,
        scene_info.local_bounds_max_m,
        region_min_xy,
        region_max_xy,
    )
    placement_ok = (
        placement_region_ok
        if args.acceptance_mode == "grasp_focus"
        else precise_placement_ok
    )
    grasp_success = bool(lifted)
    success = bool(
        grasp_success and placement_ok and stable and not forbidden_contacts
    )
    report = {
        "schema_version": 1,
        "success": success,
        "execution_attempted": True,
        "random_seed": args.seed,
        "scene_name": args.scene_name,
        "acceptance_mode": args.acceptance_mode,
        "grasp_ranking": args.grasp_ranking,
        "mode": (
            "dynamic_ground_truth_pose"
            if args.pose_hypotheses is None
            else "dynamic_predicted_pose"
        ),
        "pose_source": pose_source,
        "object_id": config.cad_object.object_id,
        "selected_grasp_id": selected.candidate.grasp_id,
        "selected_hypothesis_id": selected_evaluation.hypothesis_id,
        "top_pose_sim_position_error_m": top_pose_position_error,
        "top_pose_sim_symmetry_aware_angle_error_deg": top_pose_angle_error,
        "selected_pose_sim_position_error_m": selected_pose_position_error,
        "selected_pose_sim_symmetry_aware_angle_error_deg": (
            selected_pose_angle_error
        ),
        "selected_robust_score": selected.robust_score,
        "selected_pose_set_reliability": selected.pose_set_reliability,
        "selected_expected_task_success": selected.expected_task_success,
        "selected_tail_task_success": selected.tail_task_success,
        "selected_cvar_failure_risk": selected.cvar_failure_risk,
        "selected_risk_accepted": selected_risk_accepted,
        "risk_rejection_reasons": risk_rejection_reasons,
        "grasp_target_position_error_m": grasp_target_position_error,
        "grasp_target_angle_error_deg": grasp_target_angle_error,
        "initial_transform_world_object": initial_object.tolist(),
        "release_transform_world_object": release_object.tolist(),
        "goal_transform_world_object": desired_world_object.tolist(),
        "final_transform_world_object": final_object.tolist(),
        "lifted": lifted,
        "lift_distance_m": maximum_object_z - before_lift_z,
        "grasp_success": grasp_success,
        "placement_ok": placement_ok,
        "precise_placement_ok_diagnostic": precise_placement_ok,
        "placement_pose_error_used_for_acceptance": (
            args.acceptance_mode == "precise_pose"
        ),
        "placement_region_ok": placement_region_ok,
        "placement_region_min_xy_m": region_min_xy.tolist(),
        "placement_region_max_xy_m": region_max_xy.tolist(),
        "final_object_bounds_min_xy_m": final_object_min_xy.tolist(),
        "final_object_bounds_max_xy_m": final_object_max_xy.tolist(),
        "final_xy_error_m": xy_error,
        "final_position_error_m": position_error,
        "final_angle_error_deg": angle_error,
        "position_tolerance_m": config.placement_position_tolerance_m,
        "angle_tolerance_deg": config.placement_angle_tolerance_deg,
        "stable": stable,
        "final_linear_speed_m_s": float(np.linalg.norm(target_velocity[:3])),
        "final_angular_speed_rad_s": float(np.linalg.norm(target_velocity[3:])),
        "runtime_forbidden_contacts": sorted(forbidden_contacts),
        "observation": observation,
        "plan_file": str(args.output / "plan.json"),
        "stage_images": [str(path) for path in stage_images],
    }
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _complete_run_manifest(
        run_manifest_path,
        success=success,
        result_path=args.output / "result.json",
    )
    frames = [Image.open(path).convert("RGB") for path in stage_images]
    frames[0].save(
        args.output / "pick_place.gif",
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0,
    )
    for frame in frames:
        frame.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if viewer is not None:
        viewer.sync()
        if not args.close_on_finish:
            print("上下料完成；关闭 MuJoCo 窗口即可退出。", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 60.0)
        viewer.close()
    if not success:
        raise RuntimeError("industrial CAD pick-and-place did not meet acceptance criteria")


if __name__ == "__main__":
    main()
