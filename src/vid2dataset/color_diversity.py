"""Color/lighting diversity filter.

Prevents oversampling frames with the same lighting condition. Computes
a compact HSV histogram fingerprint per frame and rejects new frames
whose chi-squared distance to recent accepted frames is below a threshold.

This catches the common MMD case where 80% of the video has the same
warm sunset lighting and you end up with a dataset biased toward one
color palette.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_color_fingerprint(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a normalised HSV histogram as a compact color fingerprint.

    Uses 16 hue bins, 8 saturation bins, 4 value bins = 512-dim vector.
    Normalised to sum=1 for chi-squared comparison.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [16, 8, 4], [0, 180, 0, 256, 0, 256]
    )
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def chi_squared_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Chi-squared distance between two normalised histograms.

    Returns 0 for identical, higher for more different.
    Typical range: 0..2 for normalised histograms.
    """
    denom = h1 + h2
    mask = denom > 0
    if not mask.any():
        return 0.0
    diff = (h1[mask] - h2[mask]) ** 2 / denom[mask]
    return float(diff.sum())


class ColorDiversityFilter:
    """Reject frames with color distribution too similar to recent accepts."""

    def __init__(self, *, min_distance: float = 0.15, max_compare: int = 15):
        self.min_distance = min_distance
        self.max_compare = max_compare
        self._fingerprints: list[np.ndarray] = []

    def is_diverse(self, frame_bgr: np.ndarray) -> bool:
        """Return True if frame's color palette is sufficiently different."""
        fp = compute_color_fingerprint(frame_bgr)
        compare_set = self._fingerprints[-self.max_compare:]
        for prev_fp in compare_set:
            if chi_squared_distance(fp, prev_fp) < self.min_distance:
                return False
        return True

    def accept(self, frame_bgr: np.ndarray) -> None:
        fp = compute_color_fingerprint(frame_bgr)
        self._fingerprints.append(fp)
        if len(self._fingerprints) > self.max_compare:
            self._fingerprints = self._fingerprints[-self.max_compare:]

    def reset(self) -> None:
        self._fingerprints.clear()
