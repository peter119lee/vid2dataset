"""Async PNG/JPG/WebP writer pool.

Encoding + writing a 1024x1024 PNG takes 50-100ms. With one image per
extracted frame and ~1000 frames per dataset, that's a minute of pure
encode time blocking the main loop.

This module submits encode+write jobs to a small thread pool so the
extractor can keep filtering while previous frames are still being
written. Errors are logged but don't propagate — write failures are
non-critical (the source frame is gone but the run continues).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _encode_and_write(
    image_bgr: np.ndarray,
    out_path: Path,
    fmt: str,
    jpg_quality: int,
    webp_quality: int,
) -> None:
    """Encode + write. Used by the worker thread."""
    fmt = fmt.lower()
    ext = f".{fmt}" if fmt != "jpg" else ".jpg"
    params: list[int] = []
    if fmt == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
    elif fmt == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(webp_quality)]
    elif fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ext, image_bgr, params)
    if not ok:
        log.warning("Failed to encode %s", out_path)
        return
    try:
        out_path.write_bytes(buf.tobytes())
    except OSError as e:
        log.warning("Failed to write %s: %s", out_path, e)


class AsyncWriter:
    """Thread-pool backed async image writer.

    Usage::

        with AsyncWriter(workers=2) as w:
            w.submit(img, path, fmt="png")
            ...
        # all writes complete by exit
    """

    def __init__(self, workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="vid2dataset-writer"
        )
        self._futures: list[Future] = []
        self._lock = threading.Lock()

    def submit(
        self,
        image_bgr: np.ndarray,
        out_path: Path,
        *,
        fmt: str = "png",
        jpg_quality: int = 95,
        webp_quality: int = 95,
    ) -> None:
        # Take a copy of the buffer because the caller may reuse/free the source.
        img = image_bgr.copy()
        f = self._pool.submit(
            _encode_and_write, img, out_path, fmt, jpg_quality, webp_quality
        )
        with self._lock:
            self._futures.append(f)

    def flush(self) -> None:
        """Wait for all submitted writes to complete."""
        with self._lock:
            futures = self._futures[:]
            self._futures.clear()
        for f in futures:
            try:
                f.result()
            except Exception as e:
                log.warning("Writer task failed: %s", e)

    def close(self) -> None:
        self.flush()
        self._pool.shutdown(wait=True)

    def __enter__(self) -> AsyncWriter:
        return self

    def __exit__(self, *args) -> None:
        self.close()
