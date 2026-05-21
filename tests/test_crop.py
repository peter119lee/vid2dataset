"""Tests for crop.py — letterbox detection."""

from __future__ import annotations

import numpy as np

from vid2dataset.crop import detect_letterbox


def _content_with_bars(
    *,
    content_w: int = 100,
    content_h: int = 100,
    bar_top: int = 20,
    bar_bottom: int = 30,
    bar_left: int = 0,
    bar_right: int = 0,
    bar_value: int = 0,
    content_value: int = 200,
) -> np.ndarray:
    """Make a synthetic frame with the given letterbox bars."""
    full_h = bar_top + content_h + bar_bottom
    full_w = bar_left + content_w + bar_right
    frame = np.full((full_h, full_w, 3), bar_value, dtype=np.uint8)
    frame[bar_top : bar_top + content_h, bar_left : bar_left + content_w, :] = content_value
    return frame


def test_no_letterbox_returns_full() -> None:
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    rect = detect_letterbox(frame)
    assert (rect.x, rect.y, rect.w, rect.h) == (0, 0, 100, 100)


def test_horizontal_bars() -> None:
    frame = _content_with_bars(bar_top=20, bar_bottom=30)
    rect = detect_letterbox(frame)
    assert rect.x == 0
    assert rect.y == 20
    assert rect.h == 100
    assert rect.w == 100


def test_pillarbox() -> None:
    frame = _content_with_bars(bar_top=0, bar_bottom=0, bar_left=15, bar_right=25)
    rect = detect_letterbox(frame)
    assert rect.y == 0
    assert rect.x == 15
    assert rect.w == 100
    assert rect.h == 100


def test_combined_letter_and_pillar() -> None:
    frame = _content_with_bars(bar_top=10, bar_bottom=10, bar_left=10, bar_right=10)
    rect = detect_letterbox(frame)
    assert rect.x == 10 and rect.y == 10
    assert rect.w == 100 and rect.h == 100


def test_threshold_respects_dark_grey() -> None:
    # bars at value 50 — above default threshold of 16, should NOT crop
    frame = _content_with_bars(bar_top=20, bar_bottom=20, bar_value=50)
    rect = detect_letterbox(frame, threshold=16)
    # Full frame (no crop)
    assert rect.y == 0
    # …but with a higher threshold we should crop them
    rect2 = detect_letterbox(frame, threshold=80)
    assert rect2.y == 20


def test_tiny_noise_does_not_trigger_crop() -> None:
    # Pure content with two pixels of border noise — should not crop.
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    frame[0:1, :, :] = 0  # one row of black at top
    rect = detect_letterbox(frame)
    # Single row of black is below the 4-px sanity floor, so no crop.
    assert rect.y == 0 and rect.h == 100
