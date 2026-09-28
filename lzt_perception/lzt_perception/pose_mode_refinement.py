"""Mode-level RGB-D refinement for symmetry-aware pose mixtures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from lzt_common.contracts import PoseHypothesis, PoseHypothesisSet

from .pose_modes import PoseMixture
from .pose_refinement import (
    PoseRefinementConfig,
    PoseRefinementResult,
    refine_pose_rgbd,
)


@dataclass(frozen=True)
class PoseModeRefinement:
    """Refinement result for one mode center and its propagated correction."""

    mode_id: int
    representative_hypothesis_id: int
    member_hypothesis_ids: tuple[int, ...]
    result: PoseRefinementResult
    correction_camera: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "representative_hypothesis_id": self.representative_hypothesis_id,
            "member_hypothesis_ids": list(self.member_hypothesis_ids),
            "correction_camera": self.correction_camera.tolist(),
            "refinement": self.result.to_dict(),
        }


@dataclass(frozen=True)
class PoseMixtureRefinementResult:
    """Corrected hypotheses after refining one center per non-equivalent mode."""

    hypothesis_set: PoseHypothesisSet
    mode_refinements: tuple[PoseModeRefinement, ...]

    @property
    def accepted_mode_count(self) -> int:
        return sum(item.result.accepted for item in self.mode_refinements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_count": len(self.mode_refinements),
            "accepted_mode_count": self.accepted_mode_count,
            "modes": [item.to_dict() for item in self.mode_refinements],
        }


def refine_pose_mixture_rgbd(
    hypothesis_set: PoseHypothesisSet,
    mixture: PoseMixture,
    model_points_m: np.ndarray,
    camera_intrinsics: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    config: PoseRefinementConfig | None = None,
) -> PoseMixtureRefinementResult:
    """Refine each mode center once and propagate its SE(3) correction.

    The left-multiplied camera-frame correction preserves relative offsets and
    uncertainty within a mode.  Non-equivalent modes remain separate until the
    corrected candidates are rescored and clustered again by the caller.
    """

    identity = (
        hypothesis_set.object_id,
        hypothesis_set.scene_id,
        hypothesis_set.image_id,
        hypothesis_set.instance_id,
    )
    mixture_identity = (
        mixture.object_id,
        mixture.scene_id,
        mixture.image_id,
        mixture.instance_id,
    )
    if identity != mixture_identity:
        raise ValueError("hypothesis set and pose mixture identity must match")

    hypotheses = {
        item.hypothesis_id: item for item in hypothesis_set.hypotheses
    }
    expected_ids = set(hypotheses)
    mode_ids = [
        hypothesis_id
        for mode in mixture.modes
        for hypothesis_id in mode.member_hypothesis_ids
    ]
    if len(mode_ids) != len(set(mode_ids)) or set(mode_ids) != expected_ids:
        raise ValueError("pose modes must partition all source hypotheses exactly once")

    corrected: dict[int, PoseHypothesis] = {}
    refinements: list[PoseModeRefinement] = []
    for mode in mixture.modes:
        result = refine_pose_rgbd(
            model_points_m,
            mode.transform_camera_object,
            camera_intrinsics,
            observed_mask,
            observed_depth_m,
            config,
        )
        correction = (
            result.refined_transform_camera_object
            @ np.linalg.inv(mode.transform_camera_object)
        )
        for hypothesis_id in mode.member_hypothesis_ids:
            hypothesis = hypotheses[hypothesis_id]
            corrected[hypothesis_id] = replace(
                hypothesis,
                transform_camera_object=(
                    correction @ hypothesis.transform_camera_object
                ),
                covariance_se3=None,
                source=(
                    f"{hypothesis.source}_mode_rgbd_refined"
                    if result.accepted
                    else f"{hypothesis.source}_mode_rgbd_no_change"
                ),
            )
        refinements.append(
            PoseModeRefinement(
                mode_id=mode.mode_id,
                representative_hypothesis_id=(
                    mode.representative_hypothesis_id
                ),
                member_hypothesis_ids=mode.member_hypothesis_ids,
                result=result,
                correction_camera=correction,
            )
        )

    ordered = tuple(
        sorted(
            corrected.values(),
            key=lambda item: (-item.confidence, item.hypothesis_id),
        )
    )
    return PoseMixtureRefinementResult(
        hypothesis_set=PoseHypothesisSet(
            object_id=hypothesis_set.object_id,
            scene_id=hypothesis_set.scene_id,
            image_id=hypothesis_set.image_id,
            instance_id=hypothesis_set.instance_id,
            hypotheses=ordered,
            uncertainty_method="mode_level_rgbd_refinement_pending_rescore",
        ),
        mode_refinements=tuple(refinements),
    )
