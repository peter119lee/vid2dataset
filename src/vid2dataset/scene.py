"""Scene detection wrapper around PySceneDetect.

We only use the lightweight ``ContentDetector`` (HSV-based content delta).
For pure dance footage where the camera angle rarely cuts, this still
catches choreography phase changes; combined with frame oversampling per
scene we get good coverage without decoding every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video


@dataclass(frozen=True)
class Scene:
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float

    @property
    def midpoint_frame(self) -> int:
        return (self.start_frame + self.end_frame) // 2

    @property
    def length_frames(self) -> int:
        return max(1, self.end_frame - self.start_frame)


def detect_scenes(
    video_path: Path | str,
    *,
    threshold: float = 27.0,
    frame_skip: int = 4,
) -> list[Scene]:
    """Return a list of detected scenes.

    `frame_skip`: process every (frame_skip + 1)-th frame. 4 means we sample
    every 5th frame \u2014 4-5x faster on 60fps videos, no accuracy loss for
    typical scene changes (>=0.5s). 0 disables.

    Falls back to a single "scene" spanning the whole video if nothing found.
    """
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    # Use frame_skip to sample every Nth frame; ~4x speedup on 60fps videos
    # with no accuracy loss for typical scene changes (>= 0.5s long).
    sm.detect_scenes(video=video, show_progress=False, frame_skip=max(0, frame_skip))
    raw = sm.get_scene_list()

    if not raw:
        # Single-scene fallback: take the whole video.
        duration = video.duration
        fps = video.frame_rate or 30.0
        end_frame = max(1, int(duration.get_seconds() * fps)) if duration else 1
        return [
            Scene(
                start_frame=0,
                end_frame=end_frame,
                start_time=0.0,
                end_time=end_frame / fps,
            )
        ]

    scenes: list[Scene] = []
    for start, end in raw:
        scenes.append(
            Scene(
                start_frame=start.get_frames(),
                end_frame=end.get_frames(),
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
            )
        )
    return scenes


def sample_indices_for_scene(scene: Scene, *, count: int) -> list[int]:
    """Return ``count`` evenly-spaced frame indices inside ``scene``.

    We bias the samples slightly inward to dodge transition frames at the
    very start/end of a cut, which are often half-blended and useless for
    training.
    """
    n = scene.length_frames
    if n <= 1:
        return [scene.start_frame]
    if count <= 1:
        return [scene.midpoint_frame]

    margin = max(1, n // 10)
    span_start = scene.start_frame + margin
    span_end = scene.end_frame - margin
    if span_end <= span_start:
        span_start, span_end = scene.start_frame, scene.end_frame

    step = (span_end - span_start) / (count - 1) if count > 1 else 0
    return [int(round(span_start + i * step)) for i in range(count)]
