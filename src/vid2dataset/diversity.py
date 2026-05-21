"""SSIM-based diversity filter.

After a frame passes the quality gate, we compare it against all
previously accepted frames *within the same scene*. If it's too similar
to any of them (SSIM > threshold), we skip it.

This catches the case where pHash says "different" but the actual visual
content is nearly identical (same pose, slightly different arm angle).
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray, *, resize_to: int = 128) -> float:
    """Compute structural similarity between two BGR images.

    Both images are resized to a small square for speed — we only need a
    coarse similarity signal, not pixel-perfect SSIM.
    """
    a = cv2.resize(img_a, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    b = cv2.resize(img_b, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)

    mu_a = a_gray.mean()
    mu_b = b_gray.mean()
    sigma_a = a_gray.std()
    sigma_b = b_gray.std()
    sigma_ab = ((a_gray - mu_a) * (b_gray - mu_b)).mean()

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ssim = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (sigma_a**2 + sigma_b**2 + c2)
    )
    return float(ssim)


class DiversityFilter:
    """Track accepted frames and reject new ones that are too similar."""

    def __init__(self, *, ssim_threshold: float = 0.85, max_compare: int = 20):
        self.threshold = ssim_threshold
        self.max_compare = max_compare
        self._accepted: list[np.ndarray] = []

    def is_diverse(self, frame_bgr: np.ndarray) -> bool:
        """Return True if frame is sufficiently different from all accepted."""
        compare_set = self._accepted[-self.max_compare:]
        return all(compute_ssim(frame_bgr, prev) <= self.threshold for prev in compare_set)

    def accept(self, frame_bgr: np.ndarray) -> None:
        """Register a frame as accepted (store a small thumbnail for future comparisons)."""
        thumb = cv2.resize(frame_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        self._accepted.append(thumb)

    def reset(self) -> None:
        """Clear for a new scene."""
        self._accepted.clear()
