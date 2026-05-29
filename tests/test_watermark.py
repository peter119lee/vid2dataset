"""Unit tests for vid2dataset.watermark."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vid2dataset import watermark as wm_mod
from vid2dataset.watermark import (
    WatermarkRegion,
    detect_watermarks,
    expand_crop_for_watermarks,
)


def _make_frames(*, count: int = 10, with_watermark: bool = True) -> list[np.ndarray]:
    """Build N grayscale frames simulating real video content + optional watermark.

    Each frame is a smoothly-varying scene (mid-grey with slow per-frame drift)
    plus, optionally, a static high-contrast text overlay.
    """
    H, W = 720, 1280
    rng = np.random.default_rng(0)
    frames: list[np.ndarray] = []
    for i in range(count):
        frame = np.full((H, W), 100 + (i * 8) % 50, dtype=np.uint8)
        cy = 400 + (i * 5) % 100
        cv2.circle(frame, (640, cy), 150, 180, -1)
        cv2.rectangle(frame, (200, 500), (500, 700), 60, -1)
        frame = cv2.add(frame, rng.integers(0, 25, (H, W), dtype=np.uint8))
        if with_watermark:
            cv2.putText(
                frame, "@xinhai1999", (W - 240, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2,
            )
        frames.append(frame)
    return frames


def test_detect_watermark_finds_synthetic_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        wm_mod, "_sample_frames", lambda *a, **k: _make_frames(with_watermark=True)
    )
    regions = detect_watermarks(tmp_path / "fake.mp4", min_confidence=0.4)
    assert len(regions) >= 1
    r = regions[0]
    assert r.location.startswith("top")


def test_detect_no_false_positive_on_clean_scene(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        wm_mod, "_sample_frames", lambda *a, **k: _make_frames(with_watermark=False)
    )
    regions = detect_watermarks(tmp_path / "fake.mp4", min_confidence=0.4)
    assert regions == []


def test_detect_returns_empty_for_missing_video(tmp_path: Path) -> None:
    fake = tmp_path / "doesnotexist.mp4"
    assert detect_watermarks(fake) == []


def test_watermark_region_serialises() -> None:
    r = WatermarkRegion(x=10, y=20, w=100, h=30, confidence=0.85, location="top-right")
    d = r.as_dict()
    assert d["x"] == 10
    assert d["confidence"] == 0.85
    assert d["location"] == "top-right"


def test_expand_crop_excludes_corner_watermark() -> None:
    wm = WatermarkRegion(x=850, y=10, w=140, h=40, confidence=0.9, location="top-right")
    x, y, w, h = expand_crop_for_watermarks(0, 0, 1000, 600, [wm])
    assert x + w <= wm.x + 5


def test_expand_crop_keeps_center_watermark() -> None:
    wm = WatermarkRegion(x=400, y=300, w=200, h=50, confidence=0.9, location="middle-center")
    x, y, w, h = expand_crop_for_watermarks(0, 0, 1000, 600, [wm])
    assert (x, y, w, h) == (0, 0, 1000, 600)