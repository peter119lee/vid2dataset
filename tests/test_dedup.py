"""Tests for dedup.py — pHash distance and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vid2dataset.dedup import DedupIndex, hash_image


def _img(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def test_hash_image_identical() -> None:
    img = _img(0)
    h1 = hash_image(img)
    h2 = hash_image(img)
    assert h1 == h2
    assert (h1 - h2) == 0


def test_hash_image_different() -> None:
    h1 = hash_image(_img(0))
    h2 = hash_image(_img(123))
    assert (h1 - h2) > 5


def test_index_detects_duplicate() -> None:
    idx = DedupIndex(distance=5)
    h1 = hash_image(_img(0))
    idx.add(h1, "a.png")
    h2 = hash_image(_img(0))
    assert idx.is_duplicate(h2) == "a.png"


def test_index_passes_distinct() -> None:
    idx = DedupIndex(distance=5)
    idx.add(hash_image(_img(0)), "a.png")
    h2 = hash_image(_img(999))
    assert idx.is_duplicate(h2) is None


def test_index_blur_is_near_duplicate() -> None:
    """Slight blur shouldn't prevent dedup detection."""
    img = _img(0)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    idx = DedupIndex(distance=5)
    idx.add(hash_image(img), "a.png")
    # Blur shifts pHash slightly, so distance > 0 but should be small.
    h_blurred = hash_image(blurred)
    assert idx.is_duplicate(h_blurred) == "a.png" or (
        hash_image(img) - h_blurred
    ) <= 5


def test_persistence_roundtrip(tmp_path: Path) -> None:
    idx = DedupIndex(hash_size=8, distance=4)
    for i in range(5):
        idx.add(hash_image(_img(i)), f"frame_{i}.png")

    path = tmp_path / "dedup.json"
    idx.save(path)

    loaded = DedupIndex.load(path)
    assert len(loaded) == 5
    assert loaded.hash_size == 8
    assert loaded.distance == 4
    # Hashes round-trip cleanly.
    for h_a, h_b in zip(idx.hashes, loaded.hashes, strict=True):
        assert h_a == h_b


def test_load_or_new_handles_missing(tmp_path: Path) -> None:
    path = tmp_path / "no.json"
    idx = DedupIndex.load_or_new(path, hash_size=8, distance=5)
    assert len(idx) == 0
    assert idx.distance == 5


def test_load_or_new_overrides_distance(tmp_path: Path) -> None:
    """Reloading should honour the *current* distance, not the stored one."""
    saved = DedupIndex(hash_size=8, distance=3)
    saved.add(hash_image(_img(0)), "x.png")
    path = tmp_path / "d.json"
    saved.save(path)

    reloaded = DedupIndex.load_or_new(path, hash_size=8, distance=10)
    assert reloaded.distance == 10


def test_save_writes_valid_json(tmp_path: Path) -> None:
    idx = DedupIndex()
    idx.add(hash_image(_img(7)), "z.png")
    path = tmp_path / "d.json"
    idx.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["entries"][0]["source"] == "z.png"
