"""Pipeline orchestrator.

Wires all modules together:

    discover videos
        (optional) auto-quality calibration per video
        for each video:
            scene detection
                for each scene:
                    sample N candidate frame indices
                    decode (accurate or keyframe-snap)
                    letterbox-crop
                    quality gate (blur + luma)
                    completeness filter (optional)
                    bucket-resize
                    SSIM diversity check
                    color diversity check
                    pHash global dedup
                    write image
            per-video stats.json
        contact sheet + HTML gallery
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vid2dataset.auto_quality import auto_detect_blur_threshold
from vid2dataset.color_diversity import ColorDiversityFilter
from vid2dataset.completeness import is_subject_complete, is_subject_large_enough
from vid2dataset.config import ExtractConfig
from vid2dataset.crop import detect_letterbox
from vid2dataset.dedup import DedupIndex, hash_image
from vid2dataset.diversity import DiversityFilter
from vid2dataset.gallery import generate_contact_sheet, generate_html_gallery
from vid2dataset.io_utils import (
    VideoMeta,
    discover_videos,
    probe_video,
    read_frames_at,
    sanitize_stem,
    write_image,
)
from vid2dataset.quality import evaluate_frame
from vid2dataset.resize import (
    Bucket,
    contain_resize_and_pad,
    cover_resize_and_crop,
    generate_buckets,
    longest_edge_resize,
    select_bucket,
)
from vid2dataset.scene import detect_scenes, sample_indices_for_scene

log = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────


@dataclass
class FrameRecord:
    video: str
    frame_index: int
    out_path: str
    blur: float
    bucket: tuple[int, int]
    pixels: int


@dataclass
class VideoStats:
    video: str
    duration_s: float
    fps: float
    width: int
    height: int
    scenes: int
    candidates: int
    written: int
    rejected_blur: int = 0
    rejected_luma: int = 0
    rejected_too_small: int = 0
    rejected_dup: int = 0
    rejected_ssim: int = 0
    rejected_color: int = 0
    rejected_completeness: int = 0
    auto_blur_threshold: float | None = None
    elapsed_s: float = 0.0
    records: list[FrameRecord] = field(default_factory=list)


@dataclass
class PipelineResult:
    config: ExtractConfig
    videos: list[VideoStats]
    total_written: int
    total_candidates: int
    elapsed_s: float
    contact_sheet_path: str | None = None
    html_gallery_path: str | None = None

    def to_summary_dict(self) -> dict:
        return {
            "total_written": self.total_written,
            "total_candidates": self.total_candidates,
            "elapsed_s": round(self.elapsed_s, 2),
            "contact_sheet": self.contact_sheet_path,
            "html_gallery": self.html_gallery_path,
            "videos": [
                {
                    "video": v.video,
                    "written": v.written,
                    "candidates": v.candidates,
                    "rejected_blur": v.rejected_blur,
                    "rejected_luma": v.rejected_luma,
                    "rejected_too_small": v.rejected_too_small,
                    "rejected_dup": v.rejected_dup,
                    "rejected_ssim": v.rejected_ssim,
                    "rejected_color": v.rejected_color,
                    "rejected_completeness": v.rejected_completeness,
                    "auto_blur_threshold": v.auto_blur_threshold,
                    "elapsed_s": round(v.elapsed_s, 2),
                }
                for v in self.videos
            ],
        }


ProgressCallback = Callable[[str, int, int], None]


# ── Internals ─────────────────────────────────────────────────────────


def _resize_to_bucket(
    image_bgr: np.ndarray,
    cfg: ExtractConfig,
    buckets: list[Bucket],
) -> tuple[np.ndarray, Bucket] | None:
    h, w = image_bgr.shape[:2]
    if cfg.resize_mode == "longest":
        out = longest_edge_resize(image_bgr, cfg.resolution)
        oh, ow = out.shape[:2]
        return out, Bucket(width=ow, height=oh)

    bucket = select_bucket(w, h, buckets)
    if cfg.resize_mode == "cover":
        out = cover_resize_and_crop(image_bgr, bucket)
    elif cfg.resize_mode == "contain":
        out = contain_resize_and_pad(image_bgr, bucket)
    else:
        raise ValueError(f"Unknown resize_mode: {cfg.resize_mode}")
    return out, bucket


def _output_dir_for(cfg: ExtractConfig, video: Path) -> Path:
    if cfg.flatten_output:
        return cfg.output
    return cfg.output / sanitize_stem(video.stem)


def _stats_path(cfg: ExtractConfig, video: Path) -> Path:
    return _output_dir_for(cfg, video) / "_stats.json"


def _process_video(
    video: Path,
    *,
    cfg: ExtractConfig,
    buckets: list[Bucket],
    dedup_index: DedupIndex | None,
    seq_offset: int,
    progress: ProgressCallback | None,
) -> VideoStats:
    """Extract frames from a single video."""
    t0 = time.perf_counter()
    meta: VideoMeta = probe_video(video)
    out_dir = _output_dir_for(cfg, video)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Processing %s (%dx%d, %.1fs, %d frames, mode=%s)",
        video.name, meta.width, meta.height, meta.duration_s,
        meta.frame_count, cfg.decode_mode,
    )

    # ── Auto-quality calibration ─────────────────────────────────
    effective_blur_threshold = cfg.blur_threshold
    auto_threshold = None
    if cfg.auto_quality:
        auto_threshold = auto_detect_blur_threshold(
            video,
            sample_count=50,
            keep_percentile=cfg.auto_quality_percentile,
        )
        effective_blur_threshold = auto_threshold
        log.info("Auto-quality threshold for %s: %.1f", video.name, auto_threshold)

    # ── Pick candidate indices ───────────────────────────────────
    scenes = detect_scenes(video, threshold=cfg.scene_threshold)

    indices: list[int] = []
    if cfg.sampling in ("scene", "hybrid"):
        for sc in scenes:
            sc_indices = sample_indices_for_scene(sc, count=cfg.frames_per_scene)
            if cfg.sampling == "hybrid" and meta.fps > 0:
                max_per_scene = max(
                    cfg.frames_per_scene,
                    int((sc.end_time - sc.start_time) / cfg.interval_seconds),
                )
                if len(sc_indices) < max_per_scene:
                    sc_indices = sample_indices_for_scene(sc, count=max_per_scene)
            indices.extend(sc_indices)
    else:  # interval
        if meta.fps > 0:
            step = max(1, int(cfg.interval_seconds * meta.fps))
            indices = list(range(0, meta.frame_count, step))

    indices = sorted(set(indices))

    # In keyframe mode, thin out indices to approximate keyframe positions.
    # Most codecs use GOP of 60-300 frames. We keep one index per GOP-sized
    # chunk, which naturally snaps to keyframes on seek.
    if cfg.decode_mode == "keyframe" and meta.fps > 0:
        gop_estimate = max(30, int(meta.fps * 2))  # ~2 seconds
        thinned: list[int] = []
        last = -gop_estimate
        for idx in indices:
            if idx - last >= gop_estimate:
                thinned.append(idx)
                last = idx
        indices = thinned

    stats = VideoStats(
        video=str(video),
        duration_s=meta.duration_s,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        scenes=len(scenes),
        candidates=len(indices),
        written=0,
        auto_blur_threshold=auto_threshold,
    )

    # ── Diversity filters (per-video state) ──────────────────────
    ssim_filter = DiversityFilter(ssim_threshold=cfg.ssim_threshold) if cfg.ssim_filter else None
    color_filter = (
        ColorDiversityFilter(min_distance=cfg.color_distance) if cfg.color_diversity else None
    )

    # ── Decode + filter + write ─────────────────────────────────
    seq = seq_offset
    written_for_this_video = 0
    seek_mode = cfg.decode_mode != "keyframe"

    # ALL frames that pass quality+resize go here as fallback for min guarantee.
    all_quality_passed: list[tuple[int, np.ndarray, float, Bucket]] = []

    effective_max = cfg.max_per_video
    effective_min = cfg.min_per_video
    if effective_max and effective_min > effective_max:
        effective_min = effective_max

    for n, (idx, frame_bgr) in enumerate(
        read_frames_at(video, indices, seek_accurate=seek_mode)
    ):
        if progress and (n % 4 == 0):
            progress("decode", n, len(indices))

        # Letterbox crop first.
        if cfg.detect_letterbox:
            rect = detect_letterbox(
                frame_bgr,
                threshold=cfg.letterbox_threshold,
                min_ratio=cfg.letterbox_min_ratio,
            )
            frame_bgr = rect.apply(frame_bgr)

        # Quality gate.
        q = evaluate_frame(
            frame_bgr,
            blur_threshold=effective_blur_threshold,
            min_brightness=cfg.min_brightness,
            max_brightness=cfg.max_brightness,
            min_contrast=cfg.min_contrast,
        )
        if not q.passed:
            if "blur" in q.reason:
                stats.rejected_blur += 1
            else:
                stats.rejected_luma += 1
            continue

        # Completeness filter (soft: rejected frames still go to backup).
        if cfg.completeness_filter and not is_subject_complete(
            frame_bgr, min_score=cfg.completeness_threshold
        ):
            stats.rejected_completeness += 1
            continue

        # Subject size filter (soft: skip but don't lose frame).
        if cfg.subject_size_filter and not is_subject_large_enough(
            frame_bgr, min_ratio=cfg.subject_min_ratio
        ):
            stats.rejected_completeness += 1
            continue

        # Resize to bucket.
        rr = _resize_to_bucket(frame_bgr, cfg, buckets)
        if rr is None:
            stats.rejected_too_small += 1
            continue
        out_img, bucket = rr
        if bucket.pixels < cfg.min_pixels:
            stats.rejected_too_small += 1
            continue

        # Frame passed quality + resize — save as backup for min guarantee.
        all_quality_passed.append((idx, out_img, q.blur_score, bucket))

        # SSIM diversity.
        if ssim_filter is not None and not ssim_filter.is_diverse(out_img):
            stats.rejected_ssim += 1
            continue

        # Color diversity — auto-relax if too strict.
        if color_filter is not None and not color_filter.is_diverse(out_img):
            stats.rejected_color += 1
            continue

        # pHash dedup.
        if cfg.dedup and dedup_index is not None:
            h = hash_image(out_img, hash_size=cfg.phash_size)
            dup = dedup_index.is_duplicate(h)
            if dup is not None:
                stats.rejected_dup += 1
                continue

        # ── Accept frame ─────────────────────────────────────────
        seq += 1
        stem = f"{sanitize_stem(video.stem)}_{seq:05d}"
        out_path = out_dir / f"{stem}.{cfg.output_format}"
        write_image(
            out_img, out_path, fmt=cfg.output_format,
            jpg_quality=cfg.jpg_quality, webp_quality=cfg.webp_quality,
        )
        if ssim_filter is not None:
            ssim_filter.accept(out_img)
        if color_filter is not None:
            color_filter.accept(out_img)
        if cfg.dedup and dedup_index is not None:
            dedup_index.add(h, str(out_path))

        stats.records.append(
            FrameRecord(
                video=str(video), frame_index=idx, out_path=str(out_path),
                blur=q.blur_score, bucket=(bucket.width, bucket.height),
                pixels=bucket.pixels,
            )
        )
        stats.written += 1
        written_for_this_video += 1

        if effective_max and written_for_this_video >= effective_max:
            log.info("Reached max_per_video=%d for %s", effective_max, video.name)
            break

    # ── Min per-video guarantee ──────────────────────────────────
    # If we didn't meet the minimum, pull from ALL quality-passed frames
    # (sorted by sharpness). This is the REAL guarantee — it uses frames
    # that passed quality+resize regardless of diversity/dedup rejection.
    if effective_min > 0 and written_for_this_video < effective_min:
        needed = effective_min - written_for_this_video
        if effective_max:
            needed = min(needed, effective_max - written_for_this_video)
        # Exclude already-written frame indices
        written_indices = {r.frame_index for r in stats.records}
        available = [c for c in all_quality_passed if c[0] not in written_indices]
        available.sort(key=lambda x: x[2], reverse=True)  # sharpest first
        for b_idx, b_img, b_blur, b_bucket in available[:needed]:
            seq += 1
            stem = f"{sanitize_stem(video.stem)}_{seq:05d}"
            out_path = out_dir / f"{stem}.{cfg.output_format}"
            write_image(
                b_img, out_path, fmt=cfg.output_format,
                jpg_quality=cfg.jpg_quality, webp_quality=cfg.webp_quality,
            )
            if cfg.dedup and dedup_index is not None:
                bh = hash_image(b_img, hash_size=cfg.phash_size)
                dedup_index.add(bh, str(out_path))
            stats.records.append(
                FrameRecord(
                    video=str(video), frame_index=b_idx, out_path=str(out_path),
                    blur=b_blur, bucket=(b_bucket.width, b_bucket.height),
                    pixels=b_bucket.pixels,
                )
            )
            stats.written += 1
            written_for_this_video += 1
        if needed > 0 and available:
            log.info(
                "Min guarantee: added %d backup frames for %s",
                min(needed, len(available)), video.name,
            )

    stats.elapsed_s = time.perf_counter() - t0

    stats_path = _stats_path(cfg, video)
    stats_path.write_text(
        json.dumps(asdict(stats), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


# ── Public entry point ────────────────────────────────────────────────


def run_pipeline(
    cfg: ExtractConfig,
    *,
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    """Extract a training set from one video or a directory of videos."""
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    videos = discover_videos(cfg.input)
    if not videos:
        raise RuntimeError(f"No videos found under {cfg.input}")

    cfg.output.mkdir(parents=True, exist_ok=True)
    buckets = generate_buckets(
        resolution=cfg.resolution,
        min_bucket=cfg.min_bucket,
        max_bucket=cfg.max_bucket,
        step=cfg.bucket_step,
    )
    if not buckets:
        raise RuntimeError("No valid buckets — check resolution/min_bucket/max_bucket/step.")

    dedup_index = (
        DedupIndex.load_or_new(
            cfg.dedup_index, hash_size=cfg.phash_size, distance=cfg.phash_distance
        )
        if cfg.dedup
        else None
    )

    t_start = time.perf_counter()
    all_stats: list[VideoStats] = []
    seq_offset = 0

    for vi, video in enumerate(videos):
        if progress:
            progress("video", vi, len(videos))

        if cfg.skip_existing and _stats_path(cfg, video).exists():
            log.info("Skip (already done): %s", video.name)
            continue

        try:
            vs = _process_video(
                video,
                cfg=cfg,
                buckets=buckets,
                dedup_index=dedup_index,
                seq_offset=seq_offset,
                progress=progress,
            )
        except (cv2.error, OSError, ValueError) as e:
            log.error("Failed to process %s: %s", video.name, e)
            continue

        all_stats.append(vs)
        seq_offset += vs.written
        if dedup_index and cfg.dedup_index:
            dedup_index.save(cfg.dedup_index)

    elapsed = time.perf_counter() - t_start

    # ── Gallery generation ───────────────────────────────────────
    all_image_paths = [
        Path(r.out_path) for vs in all_stats for r in vs.records
    ]
    cs_path = None
    html_path = None

    if cfg.contact_sheet and all_image_paths:
        cs = generate_contact_sheet(
            all_image_paths, cfg.output / "_contact_sheet.png"
        )
        cs_path = str(cs) if cs else None

    if cfg.html_gallery and all_image_paths:
        hg = generate_html_gallery(
            all_image_paths, cfg.output / "_gallery.html"
        )
        html_path = str(hg) if hg else None

    result = PipelineResult(
        config=cfg,
        videos=all_stats,
        total_written=sum(v.written for v in all_stats),
        total_candidates=sum(v.candidates for v in all_stats),
        elapsed_s=elapsed,
        contact_sheet_path=cs_path,
        html_gallery_path=html_path,
    )

    summary_path = cfg.output / "_run_summary.json"
    summary_path.write_text(
        json.dumps(result.to_summary_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(
        "Done: %d images written across %d videos in %.1fs",
        result.total_written, len(all_stats), elapsed,
    )
    return result
