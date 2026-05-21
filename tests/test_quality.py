"""Tests for quality.py — blur and luma metrics."""

from __future__ import annotations

import cv2
import numpy as np

from vid2dataset.quality import evaluate_frame, laplacian_variance, luma_stats


def _checkerboard(size: int = 64, square: int = 8) -> np.ndarray:
    """High-frequency BGR pattern with sharp edges = high Laplacian variance."""
    g = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            g[y, x] = 255 if ((x // square) + (y // square)) % 2 == 0 else 0
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _flat(value: int = 128, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_laplacian_high_for_checkerboard() -> None:
    img = _checkerboard()
    var = laplacian_variance(img)
    assert var > 1000, f"checkerboard should be very sharp, got {var}"


def test_laplacian_low_for_flat() -> None:
    var = laplacian_variance(_flat(128))
    assert var < 1.0, f"flat image should have ~0 variance, got {var}"


def test_blurred_checkerboard_drops_score() -> None:
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (15, 15), sigmaX=5)
    assert laplacian_variance(blurred) < laplacian_variance(sharp)


def test_luma_stats_flat() -> None:
    mean, std = luma_stats(_flat(128))
    assert mean == 128
    assert std == 0


def test_evaluate_passes_clean_frame() -> None:
    res = evaluate_frame(
        _checkerboard(),
        blur_threshold=100,
        min_brightness=10,
        max_brightness=245,
        min_contrast=5,
    )
    assert res.passed, res.reason


def test_evaluate_rejects_blurry() -> None:
    blurred = cv2.GaussianBlur(_checkerboard(), (21, 21), sigmaX=10)
    res = evaluate_frame(
        blurred,
        blur_threshold=10000,  # absurdly high
        min_brightness=0,
        max_brightness=255,
        min_contrast=0,
    )
    assert not res.passed
    assert "blur" in res.reason


def test_evaluate_rejects_pure_black() -> None:
    res = evaluate_frame(
        _flat(0),
        blur_threshold=0,
        min_brightness=10,
        max_brightness=245,
        min_contrast=0,
    )
    assert not res.passed
    assert "dark" in res.reason


def test_evaluate_rejects_pure_white() -> None:
    res = evaluate_frame(
        _flat(255),
        blur_threshold=0,
        min_brightness=0,
        max_brightness=245,
        min_contrast=0,
    )
    assert not res.passed
    assert "bright" in res.reason
