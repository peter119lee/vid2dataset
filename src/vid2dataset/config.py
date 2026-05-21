"""Central configuration for vid2dataset.

All extraction parameters are defined here as a single Pydantic model.
Defaults are chosen to match the Anima LoRA trainer's preprocessing pipeline:

- ``resolution = 1024``: matches Anima's ``resolution`` in ``configs/base.toml``.
- ``min_pixels = 500_000`` (0.5 MP): images below this are *auto-dropped* by
  Anima's ``preprocess/resize_images.py``, so we refuse to emit anything
  smaller.
- ``bucket_step = 64``: Anima's ``bucket_reso_steps``, required for the
  constant-token bucket grid.
- ``min_bucket = 512`` / ``max_bucket = 2048``: Anima's bucket bounds.

References:
- ``I:\\Lora trainer\\anima_lora\\TRAINING.md``
- ``I:\\Lora trainer\\anima_lora\\configs\\base.toml``
- ``I:\\Lora trainer\\anima_lora\\preprocess\\resize_images.py``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


ResizeMode = Literal["cover", "contain", "longest"]
SamplingMode = Literal["scene", "interval", "hybrid"]
ImageFormat = Literal["png", "jpg", "webp"]
DecodeMode = Literal["accurate", "keyframe"]


class ExtractConfig(BaseModel):
    """All knobs for the extraction pipeline.

    Every field has a sensible Anima-aligned default. Override per-run via
    CLI flags, the GUI form, or a TOML preset.
    """

    # ── I/O ────────────────────────────────────────────────────────────
    input: Path = Field(
        ...,
        description="Input video file or directory of videos.",
    )
    output: Path = Field(
        Path("output"),
        description="Output directory. Sub-folders created per video.",
    )
    flatten_output: bool = Field(
        False,
        description="If True, write all images directly into output/ "
        "(filenames stay globally unique). If False, one sub-folder per video.",
    )
    output_format: ImageFormat = Field(
        "png",
        description="Output image format. PNG is lossless; recommended for training.",
    )
    jpg_quality: int = Field(95, ge=50, le=100)
    webp_quality: int = Field(95, ge=50, le=100)

    # ── Sampling strategy ─────────────────────────────────────────────
    sampling: SamplingMode = Field(
        "hybrid",
        description="scene = one frame per detected scene; interval = every N "
        "seconds; hybrid = scene-based with an upper-bound interval cap.",
    )
    scene_threshold: float = Field(
        27.0,
        ge=1.0,
        le=100.0,
        description="PySceneDetect ContentDetector threshold. Lower = more scenes.",
    )
    frames_per_scene: int = Field(
        5,
        ge=1,
        le=30,
        description="Candidate frames sampled per scene before quality + dedup.",
    )
    interval_seconds: float = Field(
        2.0,
        gt=0,
        description="Used in interval/hybrid mode as the maximum gap between "
        "candidate frames within a single scene.",
    )

    # ── Quality filters ───────────────────────────────────────────────
    blur_threshold: float = Field(
        100.0,
        ge=0,
        description="Min Laplacian variance. Below = blurry, dropped. "
        "MMD dance footage often needs 100-200.",
    )
    min_brightness: float = Field(
        15.0,
        ge=0,
        le=255,
        description="Drop frames with mean luma below this (near-black flashes).",
    )
    max_brightness: float = Field(
        245.0,
        ge=0,
        le=255,
        description="Drop frames with mean luma above this (near-white flashes).",
    )
    min_contrast: float = Field(
        10.0,
        ge=0,
        description="Drop frames with luma std-dev below this (flat colour fields).",
    )

    # ── Letterbox / black bar crop ───────────────────────────────────
    detect_letterbox: bool = Field(
        True,
        description="Auto-detect and crop top/bottom/left/right black bars.",
    )
    letterbox_threshold: int = Field(
        16,
        ge=0,
        le=64,
        description="Pixel value below which a row/column is considered 'black'.",
    )
    letterbox_min_ratio: float = Field(
        0.98,
        ge=0.5,
        le=1.0,
        description="Fraction of pixels in a row/column that must be below "
        "letterbox_threshold for it to count as black.",
    )

    # ── Resize / bucketing (Anima-aligned) ───────────────────────────
    resolution: int = Field(
        1024,
        ge=256,
        le=4096,
        description="Target long-edge resolution (Anima default: 1024).",
    )
    min_bucket: int = Field(
        512,
        ge=64,
        description="Min short-edge after bucketing (Anima default: 512).",
    )
    max_bucket: int = Field(
        2048,
        ge=64,
        description="Max long-edge after bucketing (Anima default: 2048).",
    )
    bucket_step: int = Field(
        64,
        ge=8,
        description="Bucket grid step. Output dims will be multiples of this. "
        "Anima default: 64.",
    )
    min_pixels: int = Field(
        500_000,
        ge=0,
        description="Reject final crops below this pixel count. Anima auto-drops "
        "below 500_000 (0.5 MP); matching that prevents wasted I/O.",
    )
    resize_mode: ResizeMode = Field(
        "cover",
        description="cover = fill bucket, center-crop excess (Anima style); "
        "contain = pad to bucket; longest = scale long edge only, no crop.",
    )

    # ── Dedup ─────────────────────────────────────────────────────────
    dedup: bool = Field(True, description="Enable perceptual-hash deduplication.")
    phash_size: int = Field(
        8, ge=4, le=16, description="pHash hash size (8 = 64-bit, default)."
    )
    phash_distance: int = Field(
        5,
        ge=0,
        le=64,
        description="Max Hamming distance to count as duplicate. 0 = identical, "
        "5 ≈ very similar, 10+ = loose.",
    )
    dedup_index: Path | None = Field(
        None,
        description="Optional path to persist the global pHash index across runs.",
    )

    # ── Diversity filters ────────────────────────────────────────────
    ssim_filter: bool = Field(
        True,
        description="Enable SSIM-based diversity filter. Ensures accepted frames "
        "within a scene are visually distinct (different poses).",
    )
    ssim_threshold: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Max SSIM between two frames to consider them 'different'. "
        "Lower = stricter diversity. 0.85 works well for MMD dance.",
    )
    color_diversity: bool = Field(
        True,
        description="Enable color/lighting diversity filter. Prevents oversampling "
        "frames with identical lighting conditions.",
    )
    color_distance: float = Field(
        0.08,
        ge=0.0,
        description="Min chi-squared distance between HSV histograms. "
        "Higher = more color variety required. 0.08 works for single-scene MMD.",
    )
    completeness_filter: bool = Field(
        False,
        description="Reject frames where the subject is cut off at edges. "
        "Useful for character LoRA, less important for style.",
    )
    completeness_threshold: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Min completeness score (0=cut off, 1=fully contained).",
    )
    subject_size_filter: bool = Field(
        False,
        description="Reject frames where the subject is too small (far away / "
        "overhead shots). Safe: auto-disables per video if all frames rejected.",
    )
    subject_min_ratio: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description="Min foreground-to-frame ratio. 0.15 = subject must occupy "
        "at least 15%% of the frame.",
    )
    min_per_video: int = Field(
        3,
        ge=0,
        description="Minimum images guaranteed per video. If filters are too "
        "strict, progressively relaxes them to meet this floor. "
        "Capped by max_per_video if set. 0 = no guarantee.",
    )

    # ── Auto-quality ─────────────────────────────────────────────────
    auto_quality: bool = Field(
        False,
        description="Auto-detect blur threshold per video by sampling random "
        "frames. Overrides blur_threshold with a data-driven value.",
    )
    auto_quality_percentile: float = Field(
        60.0,
        ge=10.0,
        le=95.0,
        description="Keep the top N%% sharpest frames when auto_quality is on.",
    )

    # ── Decode mode ──────────────────────────────────────────────────
    decode_mode: Literal["accurate", "keyframe"] = Field(
        "accurate",
        description="'accurate' = frame-exact seeking (slow but precise); "
        "'keyframe' = snap to nearest I-frame (10-20x faster for 60fps video).",
    )

    # ── Gallery output ───────────────────────────────────────────────
    contact_sheet: bool = Field(
        True,
        description="Generate a contact-sheet PNG overview after extraction.",
    )
    html_gallery: bool = Field(
        True,
        description="Generate an HTML gallery for visual QA.",
    )

    # ── Performance / housekeeping ──────────────────────────────────
    workers: int = Field(
        2,
        ge=1,
        le=32,
        description="Parallel worker processes. OpenCV decoding is GIL-bound, "
        "so 2-4 is usually enough.",
    )
    seek_accurate: bool = Field(
        True,
        description="Use frame-accurate seeking. Slower but avoids keyframe drift.",
    )
    skip_existing: bool = Field(
        True,
        description="Skip videos whose stats.json already exists (resumable runs).",
    )
    max_per_video: int | None = Field(
        None,
        ge=1,
        description="Hard cap on output images per video, after all filters.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")

    # ── Validators ───────────────────────────────────────────────────
    @field_validator("resolution", "min_bucket", "max_bucket")
    @classmethod
    def _multiple_of_step(cls, v: int) -> int:
        # We can't access bucket_step here in v1 of pydantic-v2 cross-field
        # validation, so just check 8-alignment as a soft floor; full check
        # happens in resize.py.
        if v % 8 != 0:
            raise ValueError(f"{v} must be a multiple of 8")
        return v

    @field_validator("output", "input", "dedup_index", mode="before")
    @classmethod
    def _expand_path(cls, v):  # type: ignore[no-untyped-def]
        if v is None or isinstance(v, Path):
            return v
        return Path(str(v)).expanduser()

    # ── Loading helpers ──────────────────────────────────────────────
    @classmethod
    def from_toml(cls, path: Path | str, *, overrides: dict | None = None) -> ExtractConfig:
        """Load a config from TOML, optionally merging CLI/GUI overrides on top."""
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_toml_dict(self) -> dict:
        """Serialise to a TOML-friendly dict (Path → str)."""
        d = self.model_dump()
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif v is None:
                d.pop(k)
        return d
