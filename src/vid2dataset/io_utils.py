"""I/O helpers: video discovery, frame-accurate seeking, image writing,
caption sidecar generation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv", ".ts"}


# ── Video discovery ────────────────────────────────────────────────────


def discover_videos(input_path: Path) -> list[Path]:
    """Return a sorted list of video files under ``input_path``.

    If ``input_path`` is a single file, return ``[input_path]``.
    If a directory, recurse into subfolders (Anima's nested
    ``image_dataset`` layout is mirrored so this is the natural choice).
    """
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    return sorted(
        p
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


# ── Filename hygiene ───────────────────────────────────────────────────

_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_stem(s: str, *, max_len: int = 80) -> str:
    """Make ``s`` safe + globally unique as a filename stem.

    - NFC-normalise unicode
    - Strip path separators and reserved characters
    - Collapse whitespace to underscore
    - If the result was truncated or fell back to ``video``, append
      a short hash of the original to avoid collisions.
    """
    import hashlib
    original = s
    s = unicodedata.normalize("NFC", s)
    s = _INVALID_FS_CHARS.sub("", s)
    s = re.sub(r"\s+", "_", s).strip("_.")
    needs_hash = (not s) or (len(s) > max_len)
    if not s:
        s = "video"
    if needs_hash:
        suffix = hashlib.md5(original.encode("utf-8", errors="replace")).hexdigest()[:8]
        max_base = max_len - 9
        s = s[:max_base] + "_" + suffix
    return s[:max_len]


# ── Video iteration ────────────────────────────────────────────────────


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


@contextmanager
def open_capture(path: Path) -> Iterator[cv2.VideoCapture]:
    """Open an OpenCV VideoCapture, ensure it's released afterward.

    Uses a workaround for non-ASCII paths on Windows: if the normal open
    fails, retry with the short 8.3 path.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        # Workaround: try reading via numpy buffer for non-ASCII paths
        # For VideoCapture there's no clean workaround, but we can try
        # the extended-length path prefix which sometimes helps.
        alt = f"\\\\?\\{path.resolve()}"
        cap = cv2.VideoCapture(alt)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    try:
        yield cap
    finally:
        cap.release()


def probe_video(path: Path) -> VideoMeta:
    """Read basic metadata without decoding any frames."""
    with open_capture(path) as cap:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return VideoMeta(
        path=path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def read_frames_at(
    path: Path,
    indices: Iterable[int],
    *,
    seek_accurate: bool = True,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame_bgr)`` pairs for the requested indices.

    ``seek_accurate=True`` uses ``CAP_PROP_POS_FRAMES`` which on most
    backends decodes from the previous keyframe — slower but exact.
    Setting it to False simply seeks and reads the next frame, which can
    drift to the keyframe boundary on some codecs.
    """
    indices = sorted({int(i) for i in indices})
    if not indices:
        return
    with open_capture(path) as cap:
        for idx in indices:
            if seek_accurate:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            yield idx, frame


# ── Output writing ─────────────────────────────────────────────────────


def write_image(
    image_bgr: np.ndarray,
    out_path: Path,
    *,
    fmt: str = "png",
    jpg_quality: int = 95,
    webp_quality: int = 95,
) -> None:
    """Write ``image_bgr`` to ``out_path`` in the requested format.

    Uses cv2.imencode + Path.write_bytes to handle non-ASCII paths on Windows.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    ext = f".{fmt}" if fmt != "jpg" else ".jpg"
    params: list[int] = []
    if fmt == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
    elif fmt == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(webp_quality)]
    elif fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

    ok, buf = cv2.imencode(ext, image_bgr, params)
    if not ok:
        raise OSError(f"Failed to encode {out_path}")
    out_path.write_bytes(buf.tobytes())
