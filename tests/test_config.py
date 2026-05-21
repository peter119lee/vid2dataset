"""Smoke test for the config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from vid2dataset.config import ExtractConfig


def test_defaults_match_anima(tmp_path: Path) -> None:
    """Anima official defaults: resolution 1024, min_pixels 500_000, step 64."""
    cfg = ExtractConfig(input=tmp_path)
    assert cfg.resolution == 1024
    assert cfg.min_pixels == 500_000
    assert cfg.bucket_step == 64
    assert cfg.min_bucket == 512
    assert cfg.max_bucket == 2048


def test_resolution_must_be_aligned(tmp_path: Path) -> None:
    # The config validator enforces multiple-of-8; the full multiple-of-step
    # check happens in resize.generate_buckets.
    with pytest.raises(ValueError):
        ExtractConfig(input=tmp_path, resolution=1023)


def test_to_toml_dict_strips_none_and_paths(tmp_path: Path) -> None:
    cfg = ExtractConfig(input=tmp_path, output=tmp_path / "out")
    d = cfg.to_toml_dict()
    assert isinstance(d["input"], str)
    assert isinstance(d["output"], str)
    # dedup_index is None → should be dropped
    assert "dedup_index" not in d


def test_load_preset_overrides_via_toml(tmp_path: Path) -> None:
    f = tmp_path / "p.toml"
    f.write_text(
        'resolution = 768\nblur_threshold = 50\n', encoding="utf-8"
    )
    cfg = ExtractConfig.from_toml(f, overrides={"input": tmp_path})
    assert cfg.resolution == 768
    assert cfg.blur_threshold == 50
    # Untouched fields keep defaults
    assert cfg.min_pixels == 500_000
