"""vid2dataset Gradio GUI — polished, publication-ready interface.

Launch with: vid2dataset gui
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from vid2dataset import __version__
from vid2dataset.config import ExtractConfig
from vid2dataset.extractor import run_pipeline
from vid2dataset.presets import list_presets, load_preset

# ── Constants ─────────────────────────────────────────────────────────

PRESET_CHOICES = ["(none)"] + [n for n, _ in list_presets()]
PRESET_DESCRIPTIONS = dict(list_presets())

CSS = """
.main-header { text-align: center; margin-bottom: 8px; }
.main-header h1 { font-size: 1.8rem; font-weight: 700; margin: 0; }
.main-header p { color: #888; margin: 4px 0 0 0; font-size: 0.9rem; }
.preset-info { padding: 8px 12px; border-radius: 6px; background: #1e293b;
               border-left: 3px solid #3b82f6; font-size: 0.85rem; color: #94a3b8; }
.run-btn { font-size: 1.1rem !important; padding: 12px 32px !important; }
.stats-box { font-family: monospace; font-size: 0.8rem; }
footer { display: none !important; }
"""


# ── Log capture ───────────────────────────────────────────────────────


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.buffer: list[str] = []
        self.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(self.format(record))

    def get_text(self) -> str:
        return "\n".join(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


# ── Core logic ────────────────────────────────────────────────────────


def _get_preset_info(name: str) -> str:
    if not name or name == "(none)":
        return ""
    desc = PRESET_DESCRIPTIONS.get(name, "")
    return f"**{name}** — {desc}" if desc else ""


def _apply_preset(name: str) -> list:
    """Return updated values for all parameter fields when preset changes."""
    defaults = {
        "sampling": "hybrid",
        "scene_threshold": 27.0,
        "frames_per_scene": 5,
        "blur_threshold": 100.0,
        "resolution": 1024,
        "min_bucket": 512,
        "max_bucket": 2048,
        "bucket_step": 64,
        "min_pixels": 500000,
        "resize_mode": "cover",
        "phash_distance": 5,
        "ssim_threshold": 0.85,
        "color_distance": 0.08,
        "auto_quality": False,
        "decode_mode": "accurate",
        "completeness_filter": False,
    }
    if name and name != "(none)":
        cfg = load_preset(name)
        defaults.update(cfg)

    return [
        gr.update(value=defaults.get("sampling", "hybrid")),
        gr.update(value=defaults.get("scene_threshold", 27.0)),
        gr.update(value=defaults.get("frames_per_scene", 5)),
        gr.update(value=defaults.get("blur_threshold", 100.0)),
        gr.update(value=defaults.get("resolution", 1024)),
        gr.update(value=defaults.get("min_bucket", 512)),
        gr.update(value=defaults.get("max_bucket", 2048)),
        gr.update(value=defaults.get("bucket_step", 64)),
        gr.update(value=defaults.get("min_pixels", 500000)),
        gr.update(value=defaults.get("resize_mode", "cover")),
        gr.update(value=defaults.get("phash_distance", 5)),
        gr.update(value=defaults.get("ssim_threshold", 0.85)),
        gr.update(value=defaults.get("color_distance", 0.08)),
        gr.update(value=defaults.get("auto_quality", False)),
        gr.update(value=defaults.get("decode_mode", "accurate")),
        gr.update(value=defaults.get("completeness_filter", False)),
        _get_preset_info(name),
    ]


def _run_extraction(
    input_path: str,
    output_path: str,
    preset_name: str,
    sampling: str,
    scene_threshold: float,
    frames_per_scene: int,
    blur_threshold: float,
    resolution: int,
    min_bucket: int,
    max_bucket: int,
    bucket_step: int,
    min_pixels: int,
    resize_mode: str,
    phash_distance: int,
    ssim_threshold: float,
    color_distance: float,
    auto_quality: bool,
    decode_mode: str,
    completeness_filter: bool,
    output_format: str,
    max_per_video: int,
    flatten: bool,
):
    """Generator that yields (log_text, status, gallery_images) tuples."""
    if not input_path.strip():
        yield "Error: input path is required.", "Error: no input path", []
        return
    if not output_path.strip():
        yield "Error: output path is required.", "Error: no output path", []
        return

    # Build config
    base: dict = {}
    if preset_name and preset_name != "(none)":
        base = load_preset(preset_name)

    base.update({
        "input": Path(input_path),
        "output": Path(output_path),
        "flatten_output": flatten,
        "sampling": sampling,
        "scene_threshold": scene_threshold,
        "frames_per_scene": int(frames_per_scene),
        "blur_threshold": blur_threshold,
        "resolution": int(resolution),
        "min_bucket": int(min_bucket),
        "max_bucket": int(max_bucket),
        "bucket_step": int(bucket_step),
        "min_pixels": int(min_pixels),
        "resize_mode": resize_mode,
        "phash_distance": int(phash_distance),
        "ssim_threshold": ssim_threshold,
        "color_distance": color_distance,
        "auto_quality": auto_quality,
        "decode_mode": decode_mode,
        "completeness_filter": completeness_filter,
        "output_format": output_format,
        "max_per_video": int(max_per_video) if max_per_video > 0 else None,
    })

    log_capture = LogCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(log_capture)
    root_logger.setLevel(logging.INFO)

    try:
        cfg = ExtractConfig(**base)
        log_capture.buffer.append(f"Config resolved. Input: {cfg.input}")
        log_capture.buffer.append(f"Output: {cfg.output}")
        log_capture.buffer.append(f"Decode mode: {cfg.decode_mode}")
        yield log_capture.get_text(), "Starting...", []

        result = run_pipeline(cfg)

        # Collect output images for gallery
        images = []
        for vs in result.videos:
            for rec in vs.records:
                p = Path(rec.out_path)
                if p.exists():
                    images.append(str(p))

        summary_lines = [
            "",
            "--- COMPLETE ---",
            f"Total images: {result.total_written}",
            f"Total candidates inspected: {result.total_candidates}",
            f"Time: {result.elapsed_s:.1f}s",
            "",
        ]
        for vs in result.videos:
            summary_lines.append(
                f"  {Path(vs.video).name}: {vs.written} written "
                f"(blur:{vs.rejected_blur} ssim:{vs.rejected_ssim} "
                f"color:{vs.rejected_color} dup:{vs.rejected_dup})"
            )
            if vs.auto_blur_threshold is not None:
                summary_lines.append(f"    auto blur threshold: {vs.auto_blur_threshold:.1f}")

        log_capture.buffer.extend(summary_lines)
        status = f"Done — {result.total_written} images in {result.elapsed_s:.1f}s"
        yield log_capture.get_text(), status, images

    except (ValueError, OSError, RuntimeError) as e:
        log_capture.buffer.append(f"\nERROR: {e}")
        yield log_capture.get_text(), f"Error: {e}", []
    finally:
        root_logger.removeHandler(log_capture)


# ── UI Builder ────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title=f"vid2dataset {__version__}",
    ) as app:
        # Header
        gr.HTML(
            f'<div class="main-header">'
            f"<h1>vid2dataset</h1>"
            f"<p>Smart video-to-image extractor for LoRA training sets &mdash; v{__version__}</p>"
            f"</div>"
        )

        # ── Top: I/O + Preset ────────────────────────────────────────
        with gr.Group():
            with gr.Row():
                input_path = gr.Textbox(
                    label="Input (video file or folder)",
                    placeholder=r"D:\videos\my_mmd_set",
                    scale=3,
                )
                output_path = gr.Textbox(
                    label="Output folder",
                    placeholder=r"D:\datasets\output",
                    value="output",
                    scale=2,
                )
            with gr.Row():
                preset = gr.Dropdown(
                    choices=PRESET_CHOICES,
                    value="anima-style",
                    label="Preset",
                    scale=1,
                )
                preset_info = gr.Markdown(
                    value=_get_preset_info("anima-style"),
                    elem_classes=["preset-info"],
                )

        # ── Parameters (tabs for organization) ───────────────────────
        with gr.Tabs():
            with gr.Tab("Sampling"):
                with gr.Row():
                    sampling = gr.Radio(
                        ["scene", "interval", "hybrid"],
                        value="hybrid",
                        label="Sampling strategy",
                    )
                    decode_mode = gr.Radio(
                        ["accurate", "keyframe"],
                        value="keyframe",
                        label="Decode mode",
                        info="keyframe = 10-20x faster for 60fps video",
                    )
                with gr.Row():
                    scene_threshold = gr.Slider(
                        5, 80, value=27, step=1,
                        label="Scene detection threshold",
                        info="Lower = more scene cuts detected",
                    )
                    frames_per_scene = gr.Slider(
                        1, 20, value=6, step=1,
                        label="Frames per scene",
                        info="Candidates sampled before filtering",
                    )

            with gr.Tab("Quality"):
                with gr.Row():
                    blur_threshold = gr.Slider(
                        0, 300, value=50, step=5,
                        label="Blur threshold (Laplacian variance)",
                        info="Higher = stricter. MMD dance: 50-100",
                    )
                    auto_quality = gr.Checkbox(
                        value=True,
                        label="Auto-detect threshold",
                        info="Samples 50 random frames to calibrate",
                    )
                with gr.Row():
                    completeness_filter = gr.Checkbox(
                        value=False,
                        label="Subject completeness filter",
                        info="Reject frames where character is cut off",
                    )

            with gr.Tab("Resize / Bucketing"):
                gr.Markdown("*Anima/SDXL-aligned defaults. Change only if you know what you're doing.*")
                with gr.Row():
                    resolution = gr.Slider(
                        256, 2048, value=1024, step=64,
                        label="Resolution (long edge)",
                    )
                    bucket_step = gr.Slider(
                        8, 128, value=64, step=8,
                        label="Bucket step",
                    )
                with gr.Row():
                    min_bucket = gr.Slider(
                        64, 1024, value=512, step=64,
                        label="Min bucket (short edge)",
                    )
                    max_bucket = gr.Slider(
                        512, 4096, value=2048, step=64,
                        label="Max bucket (long edge)",
                    )
                with gr.Row():
                    min_pixels = gr.Number(
                        value=500000,
                        label="Min pixels",
                        info="Anima drops below 500k",
                    )
                    resize_mode = gr.Radio(
                        ["cover", "contain", "longest"],
                        value="cover",
                        label="Resize mode",
                        info="cover = Anima default (crop excess)",
                    )

            with gr.Tab("Diversity / Dedup"):
                with gr.Row():
                    ssim_threshold = gr.Slider(
                        0.5, 1.0, value=0.85, step=0.01,
                        label="SSIM threshold",
                        info="Lower = more diverse poses required",
                    )
                    color_distance = gr.Slider(
                        0.0, 0.5, value=0.08, step=0.01,
                        label="Color distance",
                        info="Higher = more lighting variety",
                    )
                with gr.Row():
                    phash_distance = gr.Slider(
                        0, 20, value=5, step=1,
                        label="pHash distance",
                        info="0=identical, 5=similar, 10+=loose",
                    )

            with gr.Tab("Output"):
                with gr.Row():
                    output_format = gr.Radio(
                        ["png", "jpg", "webp"],
                        value="png",
                        label="Image format",
                        info="PNG = lossless (recommended for training)",
                    )
                    flatten = gr.Checkbox(
                        value=False,
                        label="Flatten output",
                        info="All images in one folder (no per-video subdirs)",
                    )
                with gr.Row():
                    max_per_video = gr.Number(
                        value=0,
                        label="Max images per video",
                        info="0 = no limit",
                        precision=0,
                    )

        # ── Run button + status ──────────────────────────────────────
        with gr.Row():
            run_btn = gr.Button(
                "Extract dataset",
                variant="primary",
                size="lg",
                elem_classes=["run-btn"],
                scale=2,
            )
            status = gr.Textbox(
                label="Status",
                interactive=False,
                scale=3,
            )

        # ── Output area ──────────────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("Log"):
                log_box = gr.Textbox(
                    label="Extraction log",
                    lines=15,
                    interactive=False,
                    elem_classes=["stats-box"],
                )
            with gr.Tab("Preview"):
                gallery = gr.Gallery(
                    label="Extracted images",
                    columns=6,
                    height=500,
                    object_fit="cover",
                    preview=True,
                )

        # ── Event wiring ─────────────────────────────────────────────
        param_outputs = [
            sampling, scene_threshold, frames_per_scene, blur_threshold,
            resolution, min_bucket, max_bucket, bucket_step, min_pixels,
            resize_mode, phash_distance, ssim_threshold, color_distance,
            auto_quality, decode_mode, completeness_filter, preset_info,
        ]

        preset.change(
            _apply_preset,
            inputs=[preset],
            outputs=param_outputs,
        )

        run_btn.click(
            _run_extraction,
            inputs=[
                input_path, output_path, preset,
                sampling, scene_threshold, frames_per_scene, blur_threshold,
                resolution, min_bucket, max_bucket, bucket_step, min_pixels,
                resize_mode, phash_distance, ssim_threshold, color_distance,
                auto_quality, decode_mode, completeness_filter,
                output_format, max_per_video, flatten,
            ],
            outputs=[log_box, status, gallery],
        )

    return app


def launch(*, host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    ui = build_ui()
    ui.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=True,
        css=CSS,
    )


if __name__ == "__main__":
    launch()
