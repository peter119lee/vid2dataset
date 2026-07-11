"""Tests for Advanced mode core: segment filtering + manual single-frame capture."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vid2dataset.cli import _parse_segments
from vid2dataset.config import ExtractConfig
from vid2dataset.extractor import _in_segments, _segments_for, process_single_frame


def test_parse_segments_cli() -> None:
    segs = _parse_segments(["dance.mp4:30-95.5", "dance.mp4:120-130", "b.mkv:0-10"])
    assert segs == {
        "dance.mp4": [(30.0, 95.5), (120.0, 130.0)],
        "b.mkv": [(0.0, 10.0)],
    }


def test_parse_segments_rejects_bad_specs() -> None:
    for bad in ["dance.mp4", "dance.mp4:10", "dance.mp4:abc-5", "dance.mp4:10-5"]:
        with pytest.raises(ValueError):
            _parse_segments([bad])


def test_segments_for_matches_filename_and_normalizes(tmp_path) -> None:
    cfg = ExtractConfig(
        input=tmp_path,
        output=tmp_path,
        segments={"v.mp4": [(9.0, 3.0), (20.0, 30.0)]},
    )
    segs = _segments_for(cfg, tmp_path / "sub" / "v.mp4")
    assert segs == [(3.0, 9.0), (20.0, 30.0)]  # reversed pair fixed, sorted
    assert _segments_for(cfg, tmp_path / "other.mp4") is None


def test_in_segments_boundaries() -> None:
    segs = [(3.0, 9.0), (20.0, 30.0)]
    assert _in_segments(3.0, segs)
    assert _in_segments(9.0, segs)
    assert _in_segments(25.0, segs)
    assert not _in_segments(2.9, segs)
    assert not _in_segments(15.0, segs)


def test_process_single_frame_writes_bucketed_image(tmp_path) -> None:
    cfg = ExtractConfig(
        input=tmp_path,
        output=tmp_path / "out",
        resolution=512,
        min_bucket=256,
        max_bucket=768,
    )
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    out = process_single_frame(cfg, frame, tmp_path / "clip.mp4", seq=1)
    assert out is not None and out.exists()
    assert out.name == "clip_manual_00001.png"
    w, h = Image.open(out).size
    assert w % 64 == 0 and h % 64 == 0  # landed on the bucket grid


def test_process_single_frame_respects_kohya_folder(tmp_path) -> None:
    cfg = ExtractConfig(
        input=tmp_path,
        output=tmp_path / "out",
        flatten_output=True,
        kohya_repeats=5,
        trigger_word="tw",
    )
    frame = np.full((512, 512, 3), 200, dtype=np.uint8)
    out = process_single_frame(cfg, frame, tmp_path / "clip.mp4", seq=3)
    assert out is not None
    assert out.parent.name == "5_tw"
    assert out.name == "clip_manual_00003.png"


def test_process_single_frame_rejects_empty(tmp_path) -> None:
    cfg = ExtractConfig(input=tmp_path, output=tmp_path / "out")
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert process_single_frame(cfg, empty, tmp_path / "clip.mp4", seq=1) is None
