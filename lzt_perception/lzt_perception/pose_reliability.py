"""Calibrated reliability for physically valid Top-K 6D pose hypotheses.

Ground-truth pose errors are used only to create offline training labels.  The
runtime feature vector contains FoundPose/PnP, CAD silhouette and RGB-D depth
consistency measurements that are already available from ``pose_gating``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "pnp_score",
    "log1p_pnp_inliers",
    "silhouette_iou",
    "silhouette_boundary_f1",
    "absolute_log_area_ratio",
    "center_error_over_mask_scale",
    "reprojection_error_over_mask_scale",
    "depth_overlap_fraction",
    "depth_error_over_cad_scale",
    "depth_inlier_fraction",
    "relative_pnp_score",
    "relative_geometric_quality",
    "accepted_group_fraction",
    "group_score_entropy",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _unit(value: Any) -> float:
    return float(np.clip(_finite(value), 0.0, 1.0))


def _group_entropy(records: Sequence[Mapping[str, Any]]) -> float:
    scores = np.asarray(
        [max(_finite(item.get("robust_score")), 0.0) for item in records],
        dtype=np.float64,
    )
    if scores.size <= 1:
        return 0.0
    if np.sum(scores) <= 1e-12:
        return 1.0
    probabilities = scores / np.sum(scores)
    entropy = -float(
        np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
    )
    return entropy / float(np.log(scores.size))


def pose_reliability_features(
    record: Mapping[str, Any],
    cad_scale_m: float,
    group_records: Sequence[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    """Build a dimensionless runtime feature vector from one gate audit row."""

    if not np.isfinite(cad_scale_m) or cad_scale_m <= 0.0:
        raise ValueError("cad_scale_m must be finite and positive")
    metrics = record.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    group = tuple(group_records or (record,))
    observed_pixels = max(_finite(metrics.get("observed_mask_pixels")), 1.0)
    rendered_pixels = max(_finite(metrics.get("rendered_silhouette_pixels")), 0.0)
    mask_scale = max(float(np.sqrt(observed_pixels)), 1.0)
    pnp_score = _unit(metrics.get("pnp_score", record.get("original_score")))
    geometric_quality = _unit(record.get("geometric_quality"))
    group_pnp_max = max(
        (
            _unit(item.get("metrics", {}).get("pnp_score", item.get("original_score")))
            for item in group
        ),
        default=pnp_score,
    )
    group_geometry_max = max(
        (_unit(item.get("geometric_quality")) for item in group),
        default=geometric_quality,
    )
    # New audits reserve ``accepted`` for the minimal physical hard gate.  The
    # explicit soft-pass field preserves the meaning used by models trained on
    # legacy audits, where accepted also implied passing all RGB-D thresholds.
    accepted_fraction = float(
        np.mean(
            [
                bool(item.get("soft_pass", item.get("accepted", False)))
                for item in group
            ]
        )
    )
    depth_error = _finite(metrics.get("median_depth_error_m"), cad_scale_m * 10.0)
    center_error = _finite(metrics.get("silhouette_center_error_px"), mask_scale * 10.0)
    reprojection_error = _finite(metrics.get("reprojection_error_px"), mask_scale * 10.0)
    features = np.asarray(
        [
            pnp_score,
            np.log1p(max(_finite(metrics.get("pnp_inliers")), 0.0)),
            _unit(metrics.get("silhouette_iou")),
            _unit(metrics.get("silhouette_boundary_f1")),
            min(abs(float(np.log((rendered_pixels + 1.0) / (observed_pixels + 1.0)))), 10.0),
            min(max(center_error, 0.0) / mask_scale, 10.0),
            min(max(reprojection_error, 0.0) / mask_scale, 10.0),
            _unit(_finite(metrics.get("depth_overlap_pixels")) / observed_pixels),
            min(max(depth_error, 0.0) / cad_scale_m, 10.0),
            _unit(metrics.get("depth_inlier_fraction")),
            pnp_score / max(group_pnp_max, 1e-12),
            geometric_quality / max(group_geometry_max, 1e-12),
            accepted_fraction,
            _group_entropy(group),
        ],
        dtype=np.float64,
    )
    if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
        raise ValueError("pose reliability features must be finite")
    return features


@dataclass(frozen=True)
class PoseReliabilityModel:
    """Standardized L2 logistic model with a portable JSON representation."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    label_mssd_normalized_threshold: float = 0.1
    method: str = "l2_logistic_regression"

    def __post_init__(self) -> None:
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        expected = (len(FEATURE_NAMES),)
        if mean.shape != expected or scale.shape != expected or coefficients.shape != expected:
            raise ValueError(f"model vectors must have shape {expected}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("model mean and coefficients must be finite")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("model feature scales must be finite and positive")
        if not np.isfinite(self.intercept):
            raise ValueError("model intercept must be finite")
        if self.label_mssd_normalized_threshold <= 0.0:
            raise ValueError("label threshold must be positive")
        object.__setattr__(self, "feature_mean", mean.copy())
        object.__setattr__(self, "feature_scale", scale.copy())
        object.__setattr__(self, "coefficients", coefficients.copy())

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(values)):
            raise ValueError("features have an invalid shape or contain non-finite values")
        logits = ((values - self.feature_mean) / self.feature_scale) @ self.coefficients
        logits = np.clip(logits + self.intercept, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities[0] if single else probabilities

    def with_logit_calibration(
        self,
        logit_scale: float,
        logit_bias: float,
    ) -> "PoseReliabilityModel":
        """Fold an affine probability calibrator into the portable model.

        The transformation is ``sigmoid(scale * raw_logit + bias)``.  Folding
        it into the coefficients keeps the runtime path independent of the
        offline calibration implementation.
        """

        if not np.isfinite(logit_scale) or logit_scale < 0.0:
            raise ValueError("logit calibration scale must be finite and non-negative")
        if not np.isfinite(logit_bias):
            raise ValueError("logit calibration bias must be finite")
        return PoseReliabilityModel(
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            coefficients=self.coefficients * logit_scale,
            intercept=float(self.intercept * logit_scale + logit_bias),
            label_mssd_normalized_threshold=self.label_mssd_normalized_threshold,
            method=f"{self.method}+affine_logit_calibration",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": self.method,
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "label": {
                "name": "mssd_normalized_within_threshold",
                "threshold": self.label_mssd_normalized_threshold,
                "ground_truth_used_at_runtime": False,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PoseReliabilityModel":
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("reliability model feature schema does not match this code")
        label = payload.get("label", {})
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            label_mssd_normalized_threshold=float(label.get("threshold", 0.1)),
            method=str(payload.get("method", "l2_logistic_regression")),
        )

    @classmethod
    def load(cls, path: Path | str) -> "PoseReliabilityModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination


def fit_pose_reliability_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    label_mssd_normalized_threshold: float = 0.1,
    l2_regularization: float = 1.0,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
    balance_classes: bool = False,
) -> PoseReliabilityModel:
    """Fit logistic regression using deterministic Newton updates.

    Class balancing is optional because it can improve minority ranking while
    destroying the probability interpretation unless an additional prior
    correction is applied.  The default maximum-likelihood fit is therefore
    the probability-calibration baseline.
    """

    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError("features must be an N x feature-count matrix")
    if len(values) != len(targets) or len(values) < 2:
        raise ValueError("features and labels must contain matching samples")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(targets, [0.0, 1.0])):
        raise ValueError("training data must be finite with binary labels")
    positives = float(np.sum(targets == 1.0))
    negatives = float(np.sum(targets == 0.0))
    if positives == 0.0 or negatives == 0.0:
        raise ValueError("training labels must contain both classes")
    if l2_regularization < 0.0 or maximum_iterations < 1 or tolerance <= 0.0:
        raise ValueError("invalid optimizer configuration")

    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    constant_features = [
        name for name, value in zip(FEATURE_NAMES, scale) if value < 1e-6
    ]
    if constant_features:
        raise ValueError(
            "training features contain constant dimensions: "
            + ", ".join(constant_features)
        )
    standardized = (values - mean) / scale
    design = np.column_stack((np.ones(len(values)), standardized))
    sample_weight = (
        np.where(
            targets == 1.0,
            0.5 * len(targets) / positives,
            0.5 * len(targets) / negatives,
        )
        if balance_classes
        else np.ones(len(targets), dtype=np.float64)
    )
    beta = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    for _ in range(maximum_iterations):
        logits = np.clip(design @ beta, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = (
            design.T @ (sample_weight * (probability - targets)) / len(targets)
            + l2_regularization * (regularizer @ beta) / len(targets)
        )
        curvature = sample_weight * np.maximum(probability * (1.0 - probability), 1e-8)
        hessian = (
            (design.T * curvature) @ design / len(targets)
            + l2_regularization * regularizer / len(targets)
        )
        step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-10, gradient)
        beta -= step
        if float(np.linalg.norm(step)) < tolerance:
            break
    return PoseReliabilityModel(
        feature_mean=mean,
        feature_scale=scale,
        coefficients=beta[1:],
        intercept=float(beta[0]),
        label_mssd_normalized_threshold=label_mssd_normalized_threshold,
        method=(
            "balanced_l2_logistic_regression"
            if balance_classes
            else "l2_logistic_regression"
        ),
    )


def fit_affine_logit_calibrator(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    maximum_iterations: int = 100,
    tolerance: float = 1e-10,
    l2_regularization: float = 1e-6,
) -> tuple[float, float]:
    """Fit a two-parameter Platt-style calibrator on an independent split.

    The slope is required to be non-negative so calibration cannot silently
    reverse the candidate ranking learned on the training split.  If the
    unconstrained Newton optimum has a negative slope, the boundary solution
    uses a constant probability equal to the calibration prevalence.
    """

    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(targets) != len(scores) or len(targets) < 2:
        raise ValueError("calibration labels and probabilities must be aligned")
    if not np.all(np.isin(targets, [0.0, 1.0])):
        raise ValueError("calibration labels must be binary")
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration probabilities must be finite")
    positives = int(np.sum(targets == 1.0))
    negatives = int(np.sum(targets == 0.0))
    if positives == 0 or negatives == 0:
        raise ValueError("calibration labels must contain both classes")
    if maximum_iterations < 1 or tolerance <= 0.0 or l2_regularization < 0.0:
        raise ValueError("invalid calibrator optimizer configuration")

    clipped = np.clip(scores, 1e-9, 1.0 - 1e-9)
    raw_logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(len(raw_logits)), raw_logits))
    beta = np.asarray([0.0, 1.0], dtype=np.float64)
    regularizer = np.diag([0.0, l2_regularization])
    for _ in range(maximum_iterations):
        logits = np.clip(design @ beta, -40.0, 40.0)
        calibrated = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (calibrated - targets) / len(targets)
        gradient += regularizer @ beta / len(targets)
        curvature = np.maximum(calibrated * (1.0 - calibrated), 1e-8)
        hessian = (design.T * curvature) @ design / len(targets)
        hessian += regularizer / len(targets)
        step = np.linalg.solve(hessian + np.eye(2) * 1e-10, gradient)
        beta -= step
        if float(np.linalg.norm(step)) < tolerance:
            break

    bias = float(beta[0])
    scale = float(beta[1])
    if scale < 0.0:
        prevalence = float(np.mean(targets))
        bias = float(np.log(prevalence / (1.0 - prevalence)))
        scale = 0.0
    return scale, bias


def apply_affine_logit_calibration(
    probabilities: np.ndarray,
    logit_scale: float,
    logit_bias: float,
) -> np.ndarray:
    """Apply the affine-logit calibration used by ``with_logit_calibration``."""

    scores = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("probabilities must be finite")
    if not np.isfinite(logit_scale) or logit_scale < 0.0:
        raise ValueError("logit calibration scale must be finite and non-negative")
    if not np.isfinite(logit_bias):
        raise ValueError("logit calibration bias must be finite")
    clipped = np.clip(scores, 1e-9, 1.0 - 1e-9)
    raw_logits = np.log(clipped / (1.0 - clipped))
    logits = np.clip(logit_scale * raw_logits + logit_bias, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    calibration_bins: int = 10,
) -> dict[str, float]:
    """Return threshold-independent discrimination and calibration metrics."""

    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(targets) != len(scores) or len(targets) == 0:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    if calibration_bins < 2 or not np.all(np.isin(targets, [0.0, 1.0])):
        raise ValueError("invalid labels or calibration bin count")
    scores = np.clip(scores, 1e-9, 1.0 - 1e-9)
    brier = float(np.mean((scores - targets) ** 2))
    log_loss = -float(
        np.mean(targets * np.log(scores) + (1.0 - targets) * np.log(1.0 - scores))
    )
    ece = 0.0
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    for index in range(calibration_bins):
        if index + 1 == calibration_bins:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(scores[mask])) - float(np.mean(targets[mask]))
            )

    positives = int(np.sum(targets == 1.0))
    negatives = int(np.sum(targets == 0.0))
    auc = float("nan")
    if positives and negatives:
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and scores[order[end]] == scores[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        auc = float(
            (
                np.sum(ranks[targets == 1.0])
                - positives * (positives + 1) * 0.5
            )
            / (positives * negatives)
        )
    return {
        "sample_count": float(len(targets)),
        "positive_fraction": float(np.mean(targets)),
        "brier_score": brier,
        "log_loss": log_loss,
        "expected_calibration_error": ece,
        "roc_auc": auc,
    }
