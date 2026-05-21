"""Tests for io_utils.py — sanitize, video discovery."""

from __future__ import annotations

from pathlib import Path

from vid2dataset.io_utils import discover_videos, sanitize_stem


def test_sanitize_strips_path_separators() -> None:
    assert "/" not in sanitize_stem("a/b/c")
    assert "\\" not in sanitize_stem(r"a\b\c")


def test_sanitize_collapses_whitespace() -> None:
    assert sanitize_stem("a b   c") == "a_b_c"


def test_sanitize_handles_chinese() -> None:
    out = sanitize_stem("心海大喜")
    assert out == "心海大喜"


def test_sanitize_truncates() -> None:
    s = "a" * 200
    assert len(sanitize_stem(s, max_len=80)) == 80


def test_sanitize_falls_back_to_default() -> None:
    assert sanitize_stem("///") == "video"
    assert sanitize_stem("") == "video"


def test_discover_videos_single_file(tmp_path: Path) -> None:
    f = tmp_path / "x.mp4"
    f.write_bytes(b"")
    assert discover_videos(f) == [f]


def test_discover_videos_dir_recurses(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.mkv").write_bytes(b"")
    (tmp_path / "sub" / "ignore.txt").write_bytes(b"")

    out = discover_videos(tmp_path)
    names = sorted(p.name for p in out)
    assert names == ["a.mp4", "b.mkv"]


def test_discover_videos_filters_non_video(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "a.txt").write_bytes(b"")
    out = discover_videos(tmp_path)
    assert len(out) == 1
    assert out[0].suffix == ".mp4"
