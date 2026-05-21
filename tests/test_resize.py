"""Tests for resize.py — bucket selection and Anima-aligned resize."""

from __future__ import annotations

import numpy as np
import pytest

from vid2dataset.resize import (
    Bucket,
    contain_resize_and_pad,
    cover_resize_and_crop,
    generate_buckets,
    longest_edge_resize,
    select_bucket,
)


def test_generate_buckets_anima_defaults() -> None:
    buckets = generate_buckets(resolution=1024, min_bucket=512, max_bucket=2048, step=64)
    assert all(b.width % 64 == 0 and b.height % 64 == 0 for b in buckets)
    assert all(b.width >= 512 and b.height >= 512 for b in buckets)
    # Constant-token cap: w*h <= 1024*1024
    assert all(b.width * b.height <= 1024 * 1024 for b in buckets)
    # The square 1024x1024 must be present.
    assert Bucket(1024, 1024) in buckets


def test_generate_buckets_rejects_misaligned_resolution() -> None:
    with pytest.raises(ValueError):
        generate_buckets(resolution=1000, min_bucket=512, max_bucket=2048, step=64)


def test_select_bucket_for_square() -> None:
    buckets = generate_buckets(resolution=1024, min_bucket=512, max_bucket=2048, step=64)
    chosen = select_bucket(1920, 1920, buckets)
    assert chosen.width == chosen.height
    # Should pick the largest square <= 1024^2 token cap, i.e. 1024x1024.
    assert chosen == Bucket(1024, 1024)


def test_select_bucket_for_landscape() -> None:
    buckets = generate_buckets(resolution=1024, min_bucket=512, max_bucket=2048, step=64)
    chosen = select_bucket(1920, 1080, buckets)
    assert chosen.width > chosen.height
    # Closeness to 16:9 = 1.777, should be e.g. 1344x768 or 1408x768.
    ar = chosen.width / chosen.height
    assert 1.6 < ar < 1.95


def test_select_bucket_for_portrait() -> None:
    buckets = generate_buckets(resolution=1024, min_bucket=512, max_bucket=2048, step=64)
    chosen = select_bucket(1080, 1920, buckets)
    assert chosen.height > chosen.width


def test_cover_resize_outputs_exact_bucket() -> None:
    src = np.random.randint(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    bucket = Bucket(1024, 768)
    out = cover_resize_and_crop(src, bucket)
    assert out.shape == (768, 1024, 3)


def test_cover_resize_with_portrait_source() -> None:
    src = np.random.randint(0, 255, size=(1920, 1080, 3), dtype=np.uint8)
    bucket = Bucket(704, 1024)
    out = cover_resize_and_crop(src, bucket)
    assert out.shape == (1024, 704, 3)


def test_contain_resize_pads() -> None:
    src = np.full((100, 200, 3), 200, dtype=np.uint8)
    bucket = Bucket(640, 640)
    out = contain_resize_and_pad(src, bucket, pad_color=(0, 0, 0))
    assert out.shape == (640, 640, 3)
    # Top and bottom rows should be padded (black).
    assert (out[0, 0] == 0).all()


def test_longest_edge_resize_preserves_aspect() -> None:
    src = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = longest_edge_resize(src, 1024)
    h, w = out.shape[:2]
    assert max(w, h) == 1024
    # Aspect preserved within rounding.
    assert abs((w / h) - (1280 / 720)) < 0.01
