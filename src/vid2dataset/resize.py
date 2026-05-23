"""Bucket-aware resize matching Anima's preprocessing algorithm.

This mirrors the resize logic in
``I:\\Lora trainer\\anima_lora\\preprocess\\resize_images.py``:

1. Pick a target bucket (W, H) on the bucket grid (multiples of
   ``bucket_step``) such that the target is closest in aspect ratio to
   the input *and* the longer side is bounded by ``resolution``.
2. Resize the input preserving aspect ratio so that it *covers* the
   bucket.
3. Center-crop the excess to land exactly on the bucket size.

This way the output is guaranteed to be a multiple of ``bucket_step``
on both axes, which is what Anima's bucketing dataloader expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Bucket:
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def __iter__(self):
        yield self.width
        yield self.height


def generate_buckets(
    *,
    resolution: int,
    min_bucket: int,
    max_bucket: int,
    step: int,
) -> list[Bucket]:
    """Generate the bucket grid Anima trains on.

    A bucket is any (W, H) where:
    - W and H are multiples of ``step``
    - min_bucket <= min(W, H)
    - max(W, H) <= max_bucket
    - W * H <= resolution**2 (the constant-token cap; matches Anima's
      ``constant_token_buckets=True``)
    """
    if resolution % step != 0:
        raise ValueError(f"resolution {resolution} not a multiple of step {step}")
    if min_bucket % step != 0:
        raise ValueError(f"min_bucket {min_bucket} not a multiple of step {step}")
    if max_bucket % step != 0:
        raise ValueError(f"max_bucket {max_bucket} not a multiple of step {step}")
    if min_bucket > max_bucket:
        raise ValueError("min_bucket > max_bucket")

    max_pixels = resolution * resolution
    buckets: list[Bucket] = []
    w = min_bucket
    while w <= max_bucket:
        h = min_bucket
        while h <= max_bucket:
            if w * h <= max_pixels:
                buckets.append(Bucket(w, h))
            h += step
        w += step
    return buckets


def select_bucket(
    src_w: int,
    src_h: int,
    buckets: list[Bucket],
) -> Bucket:
    """Pick the bucket whose aspect ratio is closest to the source.

    Tie-breaker: the bucket whose pixel count is largest (use as much
    detail as the constant-token cap allows).
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError("invalid source dimensions")
    if not buckets:
        raise ValueError("empty bucket list")

    src_ar = src_w / src_h

    def score(b: Bucket) -> tuple[float, int]:
        # Lower aspect-ratio error is better, larger pixels is better.
        return (abs(b.aspect - src_ar), -b.pixels)

    return min(buckets, key=score)


def cover_resize_and_crop(
    image_bgr: np.ndarray,
    bucket: Bucket,
) -> np.ndarray:
    """Scale-and-center-crop ``image_bgr`` to exactly ``bucket`` size.

    Algorithm matches Anima's ``preprocess/resize_images.py``:

        ar_img = w / h
        ar_bucket = bw / bh
        if ar_img > ar_bucket:
            new_h = bh
            new_w = round(bh * ar_img)
        else:
            new_w = bw
            new_h = round(bw / ar_img)
        # then center-crop to (bw, bh)
    """
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid image dimensions: {w}x{h}")
    bw, bh = bucket.width, bucket.height
    ar_img = w / h
    ar_bucket = bw / bh

    if ar_img > ar_bucket:
        new_h = bh
        new_w = round(bh * ar_img)
    else:
        new_w = bw
        new_h = round(bw / ar_img)

    interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LANCZOS4
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)

    left = (new_w - bw) // 2
    top = (new_h - bh) // 2
    return resized[top : top + bh, left : left + bw]


def contain_resize_and_pad(
    image_bgr: np.ndarray,
    bucket: Bucket,
    *,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Fit the image fully inside the bucket, padding with ``pad_color``.

    Useful when you want to preserve every pixel (no center crop) at the
    cost of black bars in the output. Not Anima's default.
    """
    h, w = image_bgr.shape[:2]
    bw, bh = bucket.width, bucket.height
    scale = min(bw / w, bh / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)

    canvas = np.full((bh, bw, 3), pad_color, dtype=image_bgr.dtype)
    off_x = (bw - new_w) // 2
    off_y = (bh - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return canvas


def longest_edge_resize(image_bgr: np.ndarray, target: int) -> np.ndarray:
    """Scale so the longer edge equals ``target``, preserve aspect.

    Output is *not* on the bucket grid — the trainer will bucket it later.
    Use only for casual extraction; for Anima training prefer ``cover``.
    """
    h, w = image_bgr.shape[:2]
    if max(w, h) == target:
        return image_bgr
    scale = target / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)
