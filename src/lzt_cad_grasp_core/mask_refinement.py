"""RGB-D guided refinement for externally predicted instance masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RGBDMaskRefinementConfig:
    """Thresholds for conservative, label-free RGB-D mask correction."""

    seed_erosion_px: int = 2
    search_dilation_px: int = 1
    minimum_seed_pixels: int = 24
    minimum_output_pixels: int = 20
    depth_quantile_low: float = 0.05
    depth_quantile_high: float = 0.95
    depth_margin_m: float = 0.008
    depth_mad_scale: float = 3.0
    color_mad_scale: float = 3.0
    minimum_lab_tolerance: tuple[float, float, float] = (18.0, 12.0, 12.0)
    opening_kernel: int = 1
    closing_kernel: int = 1
    preserve_invalid_depth: bool = True
    minimum_retained_fraction: float = 0.15
    minimum_correction_fraction: float = 0.10
    maximum_correction_fraction: float = 0.25
    maximum_growth_fraction: float = 0.08

    def __post_init__(self) -> None:
        integer_fields = (
            self.seed_erosion_px,
            self.search_dilation_px,
            self.minimum_seed_pixels,
            self.minimum_output_pixels,
            self.opening_kernel,
            self.closing_kernel,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("pixel and kernel parameters must be non-negative")
        if self.minimum_seed_pixels <= 0 or self.minimum_output_pixels <= 0:
            raise ValueError("minimum pixel counts must be positive")
        if not 0.0 <= self.depth_quantile_low < self.depth_quantile_high <= 1.0:
            raise ValueError("depth quantiles must be ordered in [0, 1]")
        if self.depth_margin_m < 0.0 or self.depth_mad_scale < 0.0:
            raise ValueError("depth tolerances must be non-negative")
        if self.color_mad_scale < 0.0:
            raise ValueError("color_mad_scale must be non-negative")
        if len(self.minimum_lab_tolerance) != 3 or any(
            value <= 0.0 for value in self.minimum_lab_tolerance
        ):
            raise ValueError("minimum_lab_tolerance must contain three positives")
        if not 0.0 <= self.minimum_retained_fraction <= 1.0:
            raise ValueError("minimum_retained_fraction must lie in [0, 1]")
        fractions = (
            self.minimum_correction_fraction,
            self.maximum_correction_fraction,
            self.maximum_growth_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("correction fractions must lie in [0, 1]")
        if self.minimum_correction_fraction > self.maximum_correction_fraction:
            raise ValueError("minimum correction fraction exceeds maximum")
        for field_name in ("opening_kernel", "closing_kernel"):
            value = int(getattr(self, field_name))
            if value not in (0, 1) and value % 2 == 0:
                raise ValueError(f"{field_name} must be zero, one, or odd")


@dataclass(frozen=True)
class RGBDMaskRefinementResult:
    """Refined binary mask and auditable statistics."""

    mask: np.ndarray
    status: str
    original_area: int
    refined_area: int
    added_pixels: int
    removed_pixels: int
    seed_pixels: int
    valid_depth_fraction: float
    depth_median_m: float | None
    depth_lower_m: float | None
    depth_upper_m: float | None
    proposed_added_pixels: int
    proposed_removed_pixels: int
    proposed_change_fraction: float
    applied: bool

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "original_area": self.original_area,
            "refined_area": self.refined_area,
            "added_pixels": self.added_pixels,
            "removed_pixels": self.removed_pixels,
            "seed_pixels": self.seed_pixels,
            "valid_depth_fraction": self.valid_depth_fraction,
            "depth_median_m": self.depth_median_m,
            "depth_lower_m": self.depth_lower_m,
            "depth_upper_m": self.depth_upper_m,
            "proposed_added_pixels": self.proposed_added_pixels,
            "proposed_removed_pixels": self.proposed_removed_pixels,
            "proposed_change_fraction": self.proposed_change_fraction,
            "applied": self.applied,
        }


def _ellipse(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _morphology(mask: np.ndarray, operation: int, kernel_size: int) -> np.ndarray:
    if kernel_size in (0, 1):
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel).astype(bool)


def _robust_depth_interval(
    values: np.ndarray,
    config: RGBDMaskRefinementConfig,
) -> tuple[float, float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    adaptive_margin = max(config.depth_margin_m, config.depth_mad_scale * robust_sigma)
    low_quantile, high_quantile = np.quantile(
        values,
        [config.depth_quantile_low, config.depth_quantile_high],
    )
    lower = float(low_quantile - adaptive_margin)
    upper = float(high_quantile + adaptive_margin)
    return median, lower, upper


def _seed_component(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask
    best_label = 0
    best_key = (-1, -1)
    for label in range(1, count):
        component = labels == label
        overlap = int(np.count_nonzero(component & seed))
        area = int(statistics[label, cv2.CC_STAT_AREA])
        key = (overlap, area)
        if key > best_key:
            best_label = label
            best_key = key
    return labels == best_label


def refine_rgbd_mask(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    initial_mask: np.ndarray,
    config: RGBDMaskRefinementConfig | None = None,
) -> RGBDMaskRefinementResult:
    """Refine one instance mask using robust depth support and RGB edge growth.

    Existing mask pixels are trimmed only when they have a valid, incompatible
    depth. RGB consistency is used for adding nearby pixels, so reflective or
    textured regions already inside a detector mask are not removed by color.
    Invalid-depth pixels can be preserved to avoid erasing metal surfaces.
    """

    settings = config or RGBDMaskRefinementConfig()
    image = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float64)
    original = np.asarray(initial_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be uint8 with shape (height, width, 3)")
    if depth.shape != image.shape[:2] or original.shape != depth.shape:
        raise ValueError("depth and mask must match the RGB image size")
    original_area = int(np.count_nonzero(original))
    if original_area == 0:
        return RGBDMaskRefinementResult(
            mask=original.copy(),
            status="empty_input",
            original_area=0,
            refined_area=0,
            added_pixels=0,
            removed_pixels=0,
            seed_pixels=0,
            valid_depth_fraction=0.0,
            depth_median_m=None,
            depth_lower_m=None,
            depth_upper_m=None,
            proposed_added_pixels=0,
            proposed_removed_pixels=0,
            proposed_change_fraction=0.0,
            applied=False,
        )

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    valid_original = original & valid_depth
    valid_fraction = float(np.count_nonzero(valid_original) / original_area)
    if settings.seed_erosion_px > 0:
        seed = cv2.erode(
            original.astype(np.uint8),
            _ellipse(settings.seed_erosion_px),
        ).astype(bool)
    else:
        seed = original.copy()
    seed &= valid_depth
    if np.count_nonzero(seed) < settings.minimum_seed_pixels:
        seed = valid_original
    seed_count = int(np.count_nonzero(seed))
    if seed_count < settings.minimum_seed_pixels:
        return RGBDMaskRefinementResult(
            mask=original.copy(),
            status="insufficient_valid_depth",
            original_area=original_area,
            refined_area=original_area,
            added_pixels=0,
            removed_pixels=0,
            seed_pixels=seed_count,
            valid_depth_fraction=valid_fraction,
            depth_median_m=None,
            depth_lower_m=None,
            depth_upper_m=None,
            proposed_added_pixels=0,
            proposed_removed_pixels=0,
            proposed_change_fraction=0.0,
            applied=False,
        )

    median, lower, upper = _robust_depth_interval(depth[seed], settings)
    depth_consistent = valid_depth & (depth >= lower) & (depth <= upper)

    if settings.search_dilation_px > 0:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)
        seed_colors = lab[seed]
        color_median = np.median(seed_colors, axis=0)
        color_mad = np.median(np.abs(seed_colors - color_median), axis=0)
        color_tolerance = np.maximum(
            np.asarray(settings.minimum_lab_tolerance, dtype=np.float64),
            settings.color_mad_scale * 1.4826 * color_mad,
        )
        color_consistent = np.all(
            np.abs(lab - color_median) <= color_tolerance,
            axis=2,
        )
        search = cv2.dilate(
            original.astype(np.uint8),
            _ellipse(settings.search_dilation_px),
        ).astype(bool)
        growth = search & depth_consistent & color_consistent
    else:
        growth = np.zeros_like(original)
    retained = original & depth_consistent
    if settings.preserve_invalid_depth:
        retained |= original & ~valid_depth
    combined = retained | growth
    combined = _morphology(combined, cv2.MORPH_CLOSE, settings.closing_kernel)
    combined = _morphology(combined, cv2.MORPH_OPEN, settings.opening_kernel)
    combined = _seed_component(combined, seed)

    proposed_area = int(np.count_nonzero(combined))
    proposed_added = int(np.count_nonzero(combined & ~original))
    proposed_removed = int(np.count_nonzero(original & ~combined))
    proposed_change_fraction = (proposed_added + proposed_removed) / max(
        original_area, 1
    )
    growth_fraction = proposed_added / max(original_area, 1)
    retained_fraction = proposed_area / max(original_area, 1)
    if (
        proposed_area < settings.minimum_output_pixels
        or retained_fraction < settings.minimum_retained_fraction
    ):
        combined = original.copy()
        refined_area = original_area
        status = "fallback_small_output"
        applied = False
    elif (
        proposed_change_fraction < settings.minimum_correction_fraction
        or proposed_change_fraction > settings.maximum_correction_fraction
        or growth_fraction > settings.maximum_growth_fraction
    ):
        combined = original.copy()
        refined_area = original_area
        status = "unchanged_conservative_guard"
        applied = False
    else:
        refined_area = proposed_area
        status = "refined"
        applied = True
    return RGBDMaskRefinementResult(
        mask=combined,
        status=status,
        original_area=original_area,
        refined_area=refined_area,
        added_pixels=int(np.count_nonzero(combined & ~original)),
        removed_pixels=int(np.count_nonzero(original & ~combined)),
        seed_pixels=seed_count,
        valid_depth_fraction=valid_fraction,
        depth_median_m=median,
        depth_lower_m=lower,
        depth_upper_m=upper,
        proposed_added_pixels=proposed_added,
        proposed_removed_pixels=proposed_removed,
        proposed_change_fraction=proposed_change_fraction,
        applied=applied,
    )


def binary_mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as an uncompressed COCO-compatible RLE."""

    flattened = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    run = 0
    for value in flattened:
        integer = int(value)
        if integer == current:
            run += 1
        else:
            counts.append(run)
            current = integer
            run = 1
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def mask_bbox_xywh(mask: np.ndarray) -> list[int]:
    """Return an integer COCO bbox for a non-empty binary mask."""

    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if len(rows) == 0:
        return [0, 0, 0, 0]
    left = int(np.min(columns))
    right = int(np.max(columns))
    top = int(np.min(rows))
    bottom = int(np.max(rows))
    return [left, top, right - left + 1, bottom - top + 1]
