"""ROS-independent contracts for CAD-driven industrial pick and place."""

from .contracts import (
    CADObject,
    CameraObservation,
    ContinuousSymmetry,
    GraspCandidate,
    PickPlaceTask,
    PoseEstimator,
    PoseHypothesis,
    PoseHypothesisSet,
    TaskGraspPlanner,
)
from .object_config import ObjectConfig, load_object_config
from .pose_hypotheses import foundpose_records_to_top_k, load_foundpose_top_k
from .cad_grasp import (
    GraspGenerationDiagnostics,
    MeshGeometry,
    generate_parallel_jaw_candidates,
    generate_parallel_jaw_candidates_with_diagnostics,
)
from .task_grasp import (
    MotionConstraintResult,
    RankedTaskGrasp,
    RobustScoringConfig,
    rank_task_grasps,
)
from .topk_evaluation import (
    common_coverage_comparison,
    fixed_denominator_metric,
    load_target_counts,
    oracle_topk_summary,
    risk_coverage_curve,
    select_policy_predictions,
    summarize_selected_predictions,
    write_bop_csv,
)
from .robi_bop import ROBI_OBJECT_IDS, convert_robi_realsense_to_bop
from .mask_evaluation import (
    boundary_f1,
    decode_coco_rle,
    evaluate_bop_mask_sources,
    mask_pair_metrics,
    summarize_foundpose_results,
)
from .mask_refinement import (
    RGBDMaskRefinementConfig,
    RGBDMaskRefinementResult,
    binary_mask_to_rle,
    mask_bbox_xywh,
    refine_rgbd_mask,
)
from .pose_gating import (
    PoseGateConfig,
    PoseGateDecision,
    contour_reprojection_error_px,
    evaluate_pose_candidate,
    foundpose_record_transform,
    render_model_points,
    reweighted_foundpose_record,
    sample_model_points,
)
from .pose_refinement import (
    PoseAlignmentMetrics,
    PoseRefinementConfig,
    PoseRefinementResult,
    evaluate_pose_alignment,
    refine_pose_rgbd,
)
from .robot_frames import (
    RobotCameraCalibration,
    load_robot_camera_calibration,
    mujoco_base_camera_calibration,
    mujoco_transform_base_world,
    transform_camera_object_to_base,
    validate_transform,
)


_MUJOCO_EXPORTS = {
    "CollisionCheckResult",
    "IKSolution",
    "MujocoUR5eConstraintEvaluator",
    "RobotConstraintConfig",
    "check_joint_path_collisions",
    "solve_site_ik",
}


def __getattr__(name):
    """Load robot constraints lazily for perception-only Python environments."""
    if name in _MUJOCO_EXPORTS:
        from . import robot_constraints

        return getattr(robot_constraints, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_workcell_model(*args, **kwargs):
    """Import MuJoCo only for callers that actually build the workcell."""
    from .simulation import build_workcell_model as implementation

    return implementation(*args, **kwargs)


def build_workcell_spec(*args, **kwargs):
    """Import MuJoCo only for callers that actually build the workcell spec."""
    from .simulation import build_workcell_spec as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "CADObject",
    "CameraObservation",
    "ContinuousSymmetry",
    "GraspCandidate",
    "PickPlaceTask",
    "PoseEstimator",
    "PoseHypothesis",
    "PoseHypothesisSet",
    "TaskGraspPlanner",
    "ObjectConfig",
    "load_object_config",
    "foundpose_records_to_top_k",
    "load_foundpose_top_k",
    "MeshGeometry",
    "GraspGenerationDiagnostics",
    "generate_parallel_jaw_candidates",
    "generate_parallel_jaw_candidates_with_diagnostics",
    "MotionConstraintResult",
    "RankedTaskGrasp",
    "RobustScoringConfig",
    "rank_task_grasps",
    "common_coverage_comparison",
    "fixed_denominator_metric",
    "load_target_counts",
    "oracle_topk_summary",
    "risk_coverage_curve",
    "select_policy_predictions",
    "summarize_selected_predictions",
    "write_bop_csv",
    "ROBI_OBJECT_IDS",
    "convert_robi_realsense_to_bop",
    "boundary_f1",
    "decode_coco_rle",
    "evaluate_bop_mask_sources",
    "mask_pair_metrics",
    "summarize_foundpose_results",
    "RGBDMaskRefinementConfig",
    "RGBDMaskRefinementResult",
    "binary_mask_to_rle",
    "mask_bbox_xywh",
    "refine_rgbd_mask",
    "PoseGateConfig",
    "PoseGateDecision",
    "contour_reprojection_error_px",
    "evaluate_pose_candidate",
    "foundpose_record_transform",
    "render_model_points",
    "reweighted_foundpose_record",
    "sample_model_points",
    "PoseAlignmentMetrics",
    "PoseRefinementConfig",
    "PoseRefinementResult",
    "evaluate_pose_alignment",
    "refine_pose_rgbd",
    "CollisionCheckResult",
    "IKSolution",
    "MujocoUR5eConstraintEvaluator",
    "RobotConstraintConfig",
    "check_joint_path_collisions",
    "solve_site_ik",
    "RobotCameraCalibration",
    "load_robot_camera_calibration",
    "mujoco_base_camera_calibration",
    "mujoco_transform_base_world",
    "transform_camera_object_to_base",
    "validate_transform",
    "build_workcell_model",
    "build_workcell_spec",
]
