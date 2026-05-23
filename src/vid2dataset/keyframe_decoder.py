"""Fast keyframe extraction via ffmpeg.

Uses imageio-ffmpeg's bundled ffmpeg binary to decode only I-frames
(keyframes) from a video. This is 5-20x faster than OpenCV seek-based
extraction for keyframe-only sampling.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _ffmpeg_exe() -> str | None:
    """Return path to ffmpeg binary, or None if not available."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    return None


def has_ffmpeg() -> bool:
    return _ffmpeg_exe() is not None


def probe_resolution(video_path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream. (0,0) on failure."""
    exe = _ffmpeg_exe()
    if not exe:
        return (0, 0)
    try:
        result = subprocess.run(
            [exe, "-i", str(video_path), "-hide_banner"],
            capture_output=True, text=True, timeout=10, errors="ignore",
        )
        for line in result.stderr.splitlines():
            if "Video:" in line:
                m = re.search(r"(\d{2,5})x(\d{2,5})", line)
                if m:
                    return (int(m.group(1)), int(m.group(2)))
    except Exception as e:
        log.debug("ffprobe failed for %s: %s", video_path, e)
    return (0, 0)


def extract_keyframes(
    video_path: Path,
    *,
    max_count: int | None = None,
    downscale_long_edge: int | None = None,
    timeout: float = 600.0,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_seconds, frame_bgr) for I-frames via ffmpeg pipe."""
    exe = _ffmpeg_exe()
    if not exe:
        raise ImportError("ffmpeg not available. Install imageio-ffmpeg.")

    width, height = probe_resolution(video_path)
    if width <= 0:
        raise OSError(f"Could not probe resolution of {video_path}")

    out_w, out_h = width, height
    if downscale_long_edge and max(width, height) > downscale_long_edge:
        scale = downscale_long_edge / max(width, height)
        out_w = max(2, int(round(width * scale)) // 2 * 2)
        out_h = max(2, int(round(height * scale)) // 2 * 2)

    vf_parts = ["select='eq(pict_type,I)'"]
    if (out_w, out_h) != (width, height):
        vf_parts.append(f"scale={out_w}:{out_h}")
    vf = ",".join(vf_parts)

    cmd = [
        exe, "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", vf, "-vsync", "vfr", "-an",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

    frame_size = out_w * out_h * 3
    yielded = 0
    approx_interval = 2.0
    import contextlib
    try:
        while True:
            if max_count is not None and yielded >= max_count:
                break
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((out_h, out_w, 3))
            yield yielded * approx_interval, frame
            yielded += 1
    finally:
        with contextlib.suppress(Exception):
            proc.stdout.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.returncode not in (0, None) and yielded == 0:
            err = b""
            with contextlib.suppress(Exception):
                err = proc.stderr.read() or b''
            raise OSError(
                f"ffmpeg exited {proc.returncode}: "
                f"{err.decode('utf-8', errors='ignore')[:500]}"
            )


def extract_at_timestamps(
    video_path: Path,
    timestamps: list[float],
    *,
    downscale_long_edge: int | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, frame_bgr) by ffmpeg -ss seek per timestamp."""
    exe = _ffmpeg_exe()
    if not exe:
        raise ImportError("ffmpeg not available")

    width, height = probe_resolution(video_path)
    if width <= 0:
        raise OSError(f"Could not probe {video_path}")

    out_w, out_h = width, height
    if downscale_long_edge and max(width, height) > downscale_long_edge:
        scale = downscale_long_edge / max(width, height)
        out_w = max(2, int(round(width * scale)) // 2 * 2)
        out_h = max(2, int(round(height * scale)) // 2 * 2)

    for ts in timestamps:
        cmd = [
            exe, "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-an",
        ]
        if (out_w, out_h) != (width, height):
            cmd.extend(["-vf", f"scale={out_w}:{out_h}"])
        cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            need = out_w * out_h * 3
            if len(result.stdout) >= need:
                frame = np.frombuffer(result.stdout[:need], dtype=np.uint8).reshape((out_h, out_w, 3))
                yield ts, frame
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg timeout at ts=%.2f for %s", ts, video_path)
