"""Body/subject completeness filter.

Rejects frames where the main subject is likely cut off at the frame
edges. Uses a simple heuristic: if there's significant edge activity in
the border strips (top/bottom/left/right 10%), the subject is probably
extending beyond the frame.

Also checks that the "center of visual mass" isn't too close to an edge,
which indicates the subject is partially out of frame.

No ML required — pure OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_completeness_score(frame_bgr: np.ndarray, *, border_ratio: float = 0.10) -> float:
    """Return a 0..1 score where 1 = subject fully contained, 0 = heavily cut off.

    The score is based on:
    1. Edge density in border strips vs center (lower border edges = better)
    2. Center-of-mass position (closer to center = better)
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    bh = max(1, int(h * border_ratio))
    bw = max(1, int(w * border_ratio))

    # Edge density in border strips
    border_mask = np.zeros_like(edges)
    border_mask[:bh, :] = 1  # top
    border_mask[-bh:, :] = 1  # bottom
    border_mask[:, :bw] = 1  # left
    border_mask[:, -bw:] = 1  # right

    center_mask = 1 - border_mask

    border_pixels = border_mask.sum()
    center_pixels = center_mask.sum()

    if border_pixels == 0 or center_pixels == 0:
        return 1.0

    border_edge_density = (edges * border_mask).sum() / (border_pixels * 255)
    center_edge_density = (edges * center_mask).sum() / (center_pixels * 255)

    # If border has more edge activity relative to center, subject is cut off
    if center_edge_density < 0.001:
        edge_score = 1.0
    else:
        ratio = border_edge_density / center_edge_density
        edge_score = max(0.0, 1.0 - ratio)

    # Center-of-mass check: where is the "visual weight"?
    # Use thresholded foreground (non-background pixels)
    _, fg = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(fg)
    if moments["m00"] < 1:
        com_score = 1.0
    else:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        # Normalise to 0..1 where 0.5 = center
        nx = abs(cx / w - 0.5) * 2  # 0 = center, 1 = edge
        ny = abs(cy / h - 0.5) * 2
        com_score = max(0.0, 1.0 - max(nx, ny))

    return (edge_score * 0.6 + com_score * 0.4)


def is_subject_complete(
    frame_bgr: np.ndarray,
    *,
    min_score: float = 0.35,
    border_ratio: float = 0.10,
) -> bool:
    """Return True if the subject appears to be fully within the frame."""
    return compute_completeness_score(frame_bgr, border_ratio=border_ratio) >= min_score


def compute_subject_ratio(frame_bgr: np.ndarray) -> float:
    """Estimate what fraction of the frame is occupied by the foreground subject.

    Uses Otsu thresholding on grayscale to separate foreground from background.
    Returns 0.0..1.0 where 1.0 = entire frame is foreground.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Pick whichever side (black or white) is smaller as "foreground"
    white_ratio = mask.sum() / (mask.size * 255)
    return min(white_ratio, 1.0 - white_ratio) * 2  # scale so 50/50 = 1.0


def is_subject_large_enough(
    frame_bgr: np.ndarray,
    *,
    min_ratio: float = 0.15,
) -> bool:
    """Return True if the subject occupies at least min_ratio of the frame.

    Safe: if detection fails or gives nonsensical results, returns True
    (never blocks a frame due to detection failure).
    """
    try:
        ratio = compute_subject_ratio(frame_bgr)
        return ratio >= min_ratio
    except Exception:
        return True  # fail-open
