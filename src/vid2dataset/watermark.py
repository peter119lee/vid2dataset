"""Watermark detection for MMD / video content.

Detects static overlays (text, logos, URLs, banners) that appear at the
SAME screen position across many frames. Common cases:
- "@xinhai1999" or similar artist tags burned into MMD output
- "afdian.net/xxx" patreon-style URLs
- Streaming software HUDs / subscription banners
- Recording software watermarks (OBS, Bandicam)

Why this matters: a LoRA trained on watermarked frames will generate
images with the watermark embedded as a "feature" - sometimes producing
fake URLs, garbled text, or visual artifacts in random places.

Algorithm:
    1. Sample N evenly-spaced frames.
    2. Compute per-pixel std-dev across frames.
    3. Static mask = pixels with very low std (constant overlay).
    4. Dilate static mask to merge clusters into rectangles.
    5. For each rectangle, check that mean frame has text-like edge
       density inside it (not a uniform static region).
    6. Filter by size + position to drop false positives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatermarkRegion:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    location: str

    def as_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "confidence": round(self.confidence, 3),
            "location": self.location,
        }

    @property
    def area(self) -> int:
        return self.w * self.h


def _classify_position(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> str:
    cx = x + w / 2
    cy = y + h / 2
    horiz = "left" if cx < frame_w * 0.35 else ("right" if cx > frame_w * 0.65 else "center")
    vert = "top" if cy < frame_h * 0.35 else ("bottom" if cy > frame_h * 0.65 else "middle")
    return f"{vert}-{horiz}"


def _sample_frames(video_path: Path, count: int = 15) -> list[np.ndarray]:
    """Open video, sample ``count`` evenly-spaced grayscale frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap = cv2.VideoCapture(rf"\\?\{Path(video_path).resolve()}")
    if not cap.isOpened():
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total < 5:
            return []
        step = max(1, total // count)
        frames: list[np.ndarray] = []
        for i in range(count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ok, fr = cap.read()
            if ok and fr is not None:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
        return frames
    finally:
        cap.release()


def detect_watermarks(
    video_path: Path,
    *,
    sample_count: int = 8,
    min_confidence: float = 0.6,
) -> list[WatermarkRegion]:
    """Return suspected watermark rectangles, or [] if none."""
    frames = _sample_frames(video_path, count=sample_count)
    if len(frames) < 5:
        return []

    stack = np.stack(frames).astype(np.float32)
    h, w = stack.shape[1], stack.shape[2]
    pixel_std = stack.std(axis=0)
    mean_frame = stack.mean(axis=0).astype(np.uint8)

    # Static mask: very low pixel-wise std across frames
    static_mask = (pixel_std < 8.0).astype(np.uint8) * 255

    # Dilate to cluster nearby static pixels (text characters group together)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    static_blobs = cv2.dilate(static_mask, kernel, iterations=2)

    # Edges of mean frame for text-likeness check
    edges = cv2.Canny(mean_frame, 60, 150)

    contours, _ = cv2.findContours(static_blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[WatermarkRegion] = []
    frame_area = h * w
    # Watermarks live near edges. Define strict edge zones (within 12% of border).
    edge_x = w * 0.12
    edge_y = h * 0.12
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 800:
            continue
        # Watermarks are SMALL: cap at 4%% of frame area.
        if area > frame_area * 0.04:
            continue
        if cw < 30 or ch < 10:
            continue
        # Position filter: rectangle must touch (or be within) one of the four edge zones.
        cx = x + cw / 2
        cy = y + ch / 2
        in_top    = cy < edge_y
        in_bottom = cy > h - edge_y
        in_left   = cx < edge_x
        in_right  = cx > w - edge_x
        if not (in_top or in_bottom or in_left or in_right):
            continue
        roi_edges = edges[y : y + ch, x : x + cw]
        roi_static = static_mask[y : y + ch, x : x + cw]
        roi_pixels = mean_frame[y : y + ch, x : x + cw]
        edge_density = float((roi_edges > 0).sum()) / max(1, ch * cw)
        static_density = float((roi_static > 0).sum()) / max(1, ch * cw)
        if edge_density < 0.02 or edge_density > 0.40:
            continue
        if static_density < 0.01:
            continue
        # Bimodality test: text/logo has bimodal histogram (dark text on light
        # bg, or vice versa). Smooth scene fabric has unimodal/normal histogram.
        hist = np.histogram(roi_pixels, bins=8, range=(0, 256))[0]
        if hist.sum() == 0:
            continue
        hist_norm = hist / hist.sum()
        # Bimodality score: how concentrated in top-2 bins vs spread out
        sorted_h = np.sort(hist_norm)[::-1]
        bimodal_score = float(sorted_h[0] + sorted_h[-1])  # peak + tail
        if bimodal_score < 0.45:
            continue
        aspect_score = min(1.0, cw / max(1, ch * 1.5))
        confidence = min(1.0, edge_density * 6.0) * aspect_score * min(1.0, bimodal_score * 1.5)
        if confidence < min_confidence:
            continue
        loc = _classify_position(x, y, cw, ch, w, h)
        candidates.append(WatermarkRegion(
            x=int(x), y=int(y), w=int(cw), h=int(ch),
            confidence=float(confidence),
            location=loc,
        ))

    candidates.sort(key=lambda r: -r.confidence)
    return candidates[:3]


def expand_crop_for_watermarks(
    base_x: int, base_y: int, base_w: int, base_h: int,
    watermarks: list[WatermarkRegion],
    *,
    only_peripheral: bool = True,
) -> tuple[int, int, int, int]:
    """Given a base crop and watermark regions, return an expanded crop
    that excludes the peripheral ones. Center watermarks are NEVER cropped
    (would slice the subject)."""
    new_x, new_y = base_x, base_y
    new_x2 = base_x + base_w
    new_y2 = base_y + base_h
    full_h = base_h
    full_w = base_w
    edge_margin_x = full_w * 0.20
    edge_margin_y = full_h * 0.20
    for wm in watermarks:
        if only_peripheral:
            wm_cx = wm.x + wm.w / 2
            wm_cy = wm.y + wm.h / 2
            on_left = wm_cx < edge_margin_x
            on_right = wm_cx > full_w - edge_margin_x
            on_top = wm_cy < edge_margin_y
            on_bottom = wm_cy > full_h - edge_margin_y
            if not (on_left or on_right or on_top or on_bottom):
                continue
        if wm.y < full_h * 0.5:
            new_y = max(new_y, wm.y + wm.h + 2)
        else:
            new_y2 = min(new_y2, wm.y - 2)
        if wm.x < full_w * 0.5:
            new_x = max(new_x, wm.x + wm.w + 2)
        else:
            new_x2 = min(new_x2, wm.x - 2)
    new_w = max(1, new_x2 - new_x)
    new_h = max(1, new_y2 - new_y)
    return new_x, new_y, new_w, new_h
