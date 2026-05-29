"""Unit tests for vid2dataset.report."""

from __future__ import annotations

from pathlib import Path

from vid2dataset.report import generate_report


def _summary(num_videos: int = 2) -> dict:
    return {
        "total_written": 10,
        "total_candidates": 50,
        "elapsed_s": 12.3,
        "videos": [
            {
                "video": f"D:/in/video{i}.mp4",
                "written": 5,
                "candidates": 25,
                "rejected_blur": 3,
                "rejected_ssim": 1,
                "rejected_color": 2,
                "rejected_dup": 0,
                "elapsed_s": 6.0,
                "watermarks": (
                    [{"x": 10, "y": 5, "w": 100, "h": 30,
                      "confidence": 0.85, "location": "top-right"}]
                    if i == 0 else []
                ),
                "records": [
                    {"blur": 50.0 + j * 10, "bucket": [1024, 576]}
                    for j in range(5)
                ],
            }
            for i in range(num_videos)
        ],
    }


def test_generates_html_file(tmp_path: Path) -> None:
    out = tmp_path / "_report.html"
    result = generate_report(_summary(), out)
    assert result == out
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "</html>" in text


def test_report_includes_basic_stats(tmp_path: Path) -> None:
    out = tmp_path / "_report.html"
    generate_report(_summary(), out)
    text = out.read_text(encoding="utf-8")
    # Total written value should appear
    assert ">10<" in text
    # Mentions watermark check section
    assert "Watermark check" in text


def test_report_flags_videos_with_watermarks(tmp_path: Path) -> None:
    out = tmp_path / "_report.html"
    generate_report(_summary(), out)
    text = out.read_text(encoding="utf-8")
    assert "Suspected static overlays" in text
    assert "top-right" in text
    assert "0.85" in text  # confidence


def test_report_clean_when_no_watermarks(tmp_path: Path) -> None:
    s = _summary()
    for v in s["videos"]:
        v["watermarks"] = []
    out = tmp_path / "_report.html"
    generate_report(s, out)
    text = out.read_text(encoding="utf-8")
    assert "No suspected watermarks detected" in text


def test_report_handles_empty_videos(tmp_path: Path) -> None:
    s = {"total_written": 0, "total_candidates": 0, "elapsed_s": 0.0, "videos": []}
    out = tmp_path / "_report.html"
    generate_report(s, out)
    assert out.exists()


def test_report_escapes_filenames(tmp_path: Path) -> None:
    s = _summary(1)
    s["videos"][0]["video"] = "D:/scary<name&here>.mp4"
    out = tmp_path / "_report.html"
    generate_report(s, out)
    text = out.read_text(encoding="utf-8")
    # Raw script tag must NOT appear (escaped)
    # Special chars must be escaped
    assert "&lt;name" in text
    assert "<name" not in text
