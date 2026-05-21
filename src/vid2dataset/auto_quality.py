"""Auto-quality threshold tuning.

Samples N random frames from a video, computes their blur scores, and
picks a threshold at a given percentile. This solves the "what blur
threshold should I use for THIS specific video?" problem.

Different videos have wildly different baseline sharpness:
- 4K rendered MMD: Laplacian variance 200-800
- 1080p phone-recorded dance: 30-150
- Compressed gameplay: 50-200

Auto-tuning picks a threshold that keeps the top X% sharpest frames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from vid2dataset.io_utils import open_capture
from vid2dataset.quality import laplacian_variance

log = logging.getLogger(__name__)


def auto_detect_blur_threshold(
    video_path: Path,
    *,
    sample_count: int = 50,
    keep_percentile: float = 60.0,
) -> float:
    """Sample random frames and return a blur threshold at the given percentile.

    Args:
        video_path: Path to video file.
        sample_count: How many random frames to sample for calibration.
        keep_percentile: What percentile of frames to keep. 60 means
            "keep the top 60% sharpest frames" → threshold at the 40th
            percentile of blur scores.

    Returns:
        Suggested blur_threshold value.
    """
    with open_capture(video_path) as cap:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames < 10:
            return 50.0

        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(total_frames, size=min(sample_count, total_frames), replace=False))

        scores: list[float] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scores.append(laplacian_variance(gray))

    if not scores:
        return 50.0

    # We want to keep the top `keep_percentile`% of frames.
    # So the threshold is at (100 - keep_percentile) percentile.
    threshold_percentile = 100.0 - keep_percentile
    threshold = float(np.percentile(scores, threshold_percentile))

    log.info(
        "Auto-quality: sampled %d frames, blur range [%.1f, %.1f], "
        "keeping top %.0f%% → threshold = %.1f",
        len(scores), min(scores), max(scores), keep_percentile, threshold,
    )
    return threshold
