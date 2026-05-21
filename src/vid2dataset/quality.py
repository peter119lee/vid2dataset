"""Quality filters: Laplacian blur and luma sanity checks.

Pure functions, no I/O, easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityResult:
    """Outcome of running a frame through the quality gates."""

    passed: bool
    blur_score: float  # Laplacian variance — higher is sharper
    mean_luma: float  # 0..255
    luma_std: float
    reason: str = ""


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard cheap blur metric.

    Higher values mean more high-frequency detail, i.e. sharper. Typical
    thresholds: <50 = obviously blurry, 50-150 = borderline, >150 = sharp.
    MMD dance footage with motion blur often sits in 30-120, so a threshold
    around 100 is a sensible default to drop the worst frames.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def luma_stats(gray: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of luma in 0..255."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def evaluate_frame(
    frame_bgr: np.ndarray,
    *,
    blur_threshold: float,
    min_brightness: float,
    max_brightness: float,
    min_contrast: float,
) -> QualityResult:
    """Run all quality gates against one BGR frame.

    The first failing gate sets the rejection reason; we still report all
    metrics so callers can log/debug.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = laplacian_variance(gray)
    mean, std = luma_stats(gray)

    reason = ""
    if blur < blur_threshold:
        reason = f"blur {blur:.1f} < {blur_threshold:.1f}"
    elif mean < min_brightness:
        reason = f"too dark ({mean:.1f})"
    elif mean > max_brightness:
        reason = f"too bright ({mean:.1f})"
    elif std < min_contrast:
        reason = f"low contrast ({std:.1f})"

    return QualityResult(
        passed=not reason,
        blur_score=blur,
        mean_luma=mean,
        luma_std=std,
        reason=reason,
    )
