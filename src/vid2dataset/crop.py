"""Letterbox / pillarbox detection and cropping.

MMD videos are frequently a 9:16 character render embedded in a 16:9 frame
(or vice versa), padded with solid black. We detect those padding bars and
return the content rect.

Pure-numpy implementation, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropRect:
    """Inclusive content rectangle in pixel coords."""

    x: int
    y: int
    w: int
    h: int

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]

    @classmethod
    def full(cls, image: np.ndarray) -> CropRect:
        h, w = image.shape[:2]
        return cls(0, 0, w, h)


def _is_black_axis(
    arr2d: np.ndarray,
    *,
    threshold: int,
    min_ratio: float,
    axis: int,
) -> np.ndarray:
    """Return a 1-D bool mask of which rows (axis=1) or cols (axis=0) are black.

    A line is "black" when at least ``min_ratio`` of its pixels are below
    ``threshold``.
    """
    below = arr2d < threshold
    ratio = below.mean(axis=axis)
    return ratio >= min_ratio


def detect_letterbox(
    frame_bgr: np.ndarray,
    *,
    threshold: int = 16,
    min_ratio: float = 0.98,
) -> CropRect:
    """Find the inner content rect of ``frame_bgr`` after stripping black bars.

    Args:
        frame_bgr: HxWx3 BGR image.
        threshold: Pixel value below which a pixel counts as "black".
        min_ratio: Fraction of a row/column that must be black for it to be
            counted as letterbox.

    Returns:
        A ``CropRect`` covering the non-letterbox content. Falls back to the
        full frame if detection found no bars or stripped everything.
    """
    if frame_bgr.size == 0:
        return CropRect.full(frame_bgr)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    rows_black = _is_black_axis(gray, threshold=threshold, min_ratio=min_ratio, axis=1)
    cols_black = _is_black_axis(gray, threshold=threshold, min_ratio=min_ratio, axis=0)

    # Top: count contiguous black rows from the top
    top = int(np.argmax(~rows_black)) if (~rows_black).any() else h
    bottom = int(np.argmax(~rows_black[::-1])) if (~rows_black).any() else h
    left = int(np.argmax(~cols_black)) if (~cols_black).any() else w
    right = int(np.argmax(~cols_black[::-1])) if (~cols_black).any() else w

    new_x = left
    new_y = top
    new_w = w - left - right
    new_h = h - top - bottom

    # Sanity checks: don't crop everything, don't return tiny slivers, and
    # require at least a 4-pixel bar to bother cropping (avoids JPEG noise
    # masquerading as letterbox).
    if new_w < 16 or new_h < 16:
        return CropRect.full(frame_bgr)
    if (top + bottom + left + right) < 4:
        return CropRect.full(frame_bgr)

    return CropRect(new_x, new_y, new_w, new_h)
