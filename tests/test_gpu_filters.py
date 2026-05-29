"""Unit tests for vid2dataset.gpu_filters."""

from __future__ import annotations

import numpy as np

from vid2dataset.gpu_filters import (
    BatchColorFilter,
    BatchSSIMFilter,
    best_device,
    device_summary,
    is_torch_available,
)


def _img(seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def test_module_imports_without_torch() -> None:
    # Even if torch is missing, the module must still import and report "no GPU"
    is_torch_available()  # boolean, no crash
    summary = device_summary()
    assert isinstance(summary, str)
    assert summary  # non-empty


def test_best_device_returns_known_value() -> None:
    dev = best_device()
    assert dev in {"cpu", "cuda", "mps"}


def test_ssim_filter_first_frame_always_diverse() -> None:
    f = BatchSSIMFilter()
    # Empty filter: first frame should be considered diverse
    assert f.is_diverse(_img(0)) is True


def test_ssim_filter_same_frame_not_diverse() -> None:
    f = BatchSSIMFilter(ssim_threshold=0.85)
    img = _img(0)
    f.accept(img)
    # The same image again should be flagged as not diverse
    assert f.is_diverse(img) is False


def test_ssim_filter_different_frame_is_diverse() -> None:
    f = BatchSSIMFilter()
    f.accept(_img(0))
    # Random different frame should be visually distinct
    assert f.is_diverse(_img(123)) is True


def test_ssim_filter_max_compare_bounded() -> None:
    f = BatchSSIMFilter(max_compare=5)
    for s in range(20):
        f.accept(_img(s))
    # Internal store should not exceed max_compare
    if hasattr(f, "_thumbs") and f._thumbs is not None:
        # Either tensor or list, must have <= 5
        try:
            n = f._thumbs.shape[0]
        except AttributeError:
            n = len(f._thumbs)
        assert n <= 5


def test_color_filter_first_frame_diverse() -> None:
    f = BatchColorFilter()
    assert f.is_diverse(_img(0)) is True


def test_color_filter_same_frame_not_diverse() -> None:
    f = BatchColorFilter()
    img = _img(0)
    f.accept(img)
    assert f.is_diverse(img) is False


def test_color_filter_reset_clears_state() -> None:
    f = BatchColorFilter()
    f.accept(_img(0))
    f.reset()
    # After reset, any frame should be diverse again
    assert f.is_diverse(_img(0)) is True
