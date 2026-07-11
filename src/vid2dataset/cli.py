"""Typer-based CLI with rich progress + summary tables."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from vid2dataset import __version__
from vid2dataset.config import ExtractConfig
from vid2dataset.extractor import PipelineResult, run_pipeline
from vid2dataset.presets import list_presets, load_preset

app = typer.Typer(
    name="vid2dataset",
    add_completion=False,
    no_args_is_help=True,
    help="Smart video-to-image extractor for LoRA training sets.",
    rich_markup_mode="rich",
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"vid2dataset {__version__}")
        raise typer.Exit(0)


# ── Global options ────────────────────────────────────────────────────


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """vid2dataset — smart video-to-image extractor for LoRA training sets."""


# ── extract ───────────────────────────────────────────────────────────


@app.command()
def extract(  # noqa: PLR0913
    input: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            help="Input video file or directory of videos (recursive).",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory."),
    ] = Path("output"),
    preset: Annotated[
        str | None,
        typer.Option("--preset", "-p", help="Built-in preset name (see `presets`)."),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Override defaults from a TOML file."),
    ] = None,
    # ── Sampling ────────────────────────────────────────────────
    sampling: Annotated[
        str | None,
        typer.Option("--sampling", help="scene | interval | hybrid"),
    ] = None,
    scene_threshold: Annotated[
        float | None,
        typer.Option("--scene-threshold", help="PySceneDetect threshold (default 27)."),
    ] = None,
    frames_per_scene: Annotated[
        int | None,
        typer.Option("--frames-per-scene", help="Candidate frames per scene."),
    ] = None,
    interval_seconds: Annotated[
        float | None,
        typer.Option("--interval-seconds", help="Cap between candidates inside a scene."),
    ] = None,
    # ── Quality ─────────────────────────────────────────────────
    blur_threshold: Annotated[
        float | None,
        typer.Option("--blur-threshold", help="Min Laplacian variance (default 100)."),
    ] = None,
    # ── Resize ──────────────────────────────────────────────────
    resolution: Annotated[
        int | None,
        typer.Option("--resolution", "-r", help="Long-edge resolution (Anima default 1024)."),
    ] = None,
    min_bucket: Annotated[int | None, typer.Option("--min-bucket")] = None,
    max_bucket: Annotated[int | None, typer.Option("--max-bucket")] = None,
    bucket_step: Annotated[int | None, typer.Option("--bucket-step")] = None,
    min_pixels: Annotated[
        int | None,
        typer.Option("--min-pixels", help="Reject crops below this pixel count."),
    ] = None,
    resize_mode: Annotated[
        str | None,
        typer.Option("--resize-mode", help="cover | contain | longest"),
    ] = None,
    # ── Dedup ───────────────────────────────────────────────────
    no_dedup: Annotated[bool, typer.Option("--no-dedup", help="Disable dedup.")] = False,
    phash_distance: Annotated[int | None, typer.Option("--phash-distance")] = None,
    dedup_index: Annotated[
        Path | None,
        typer.Option("--dedup-index", help="Persist global pHash index to this file."),
    ] = None,
    # ── Diversity ───────────────────────────────────────────────
    no_ssim: Annotated[
        bool, typer.Option("--no-ssim", help="Disable SSIM diversity filter.")
    ] = False,
    ssim_threshold: Annotated[
        float | None,
        typer.Option("--ssim-threshold", help="Max SSIM to accept (default 0.85)."),
    ] = None,
    no_color_diversity: Annotated[
        bool, typer.Option("--no-color-diversity", help="Disable color diversity filter.")
    ] = False,
    color_distance: Annotated[
        float | None,
        typer.Option("--color-distance", help="Min HSV histogram distance (default 0.15)."),
    ] = None,
    completeness: Annotated[
        bool, typer.Option("--completeness", help="Enable subject completeness filter.")
    ] = False,
    # ── Auto-quality ────────────────────────────────────────────
    auto_quality: Annotated[
        bool, typer.Option("--auto-quality", help="Auto-detect blur threshold per video.")
    ] = False,
    # ── Decode mode ─────────────────────────────────────────────
    keyframe: Annotated[
        bool, typer.Option("--keyframe", help="Keyframe-only decode (10-20x faster).")
    ] = False,
    # ── Gallery ─────────────────────────────────────────────────
    no_gallery: Annotated[
        bool, typer.Option("--no-gallery", help="Skip contact sheet + HTML gallery.")
    ] = False,
    # ── Output / housekeeping ──────────────────────────────────
    flatten: Annotated[
        bool, typer.Option("--flatten", help="Write all images into a flat output dir.")
    ] = False,
    output_format: Annotated[
        str | None,
        typer.Option("--format", "-f", help="png | jpg | webp"),
    ] = None,
    max_per_video: Annotated[int | None, typer.Option("--max-per-video")] = None,
    no_skip_existing: Annotated[
        bool,
        typer.Option("--no-skip-existing", help="Re-process videos even if stats.json exists."),
    ] = False,
    log_level: Annotated[
        str, typer.Option("--log-level", help="DEBUG | INFO | WARNING | ERROR")
    ] = "INFO",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print resolved config and exit without extracting."),
    ] = False,
) -> None:
    """Extract a LoRA training set from one or more videos."""

    # ── Build config ─────────────────────────────────────────────
    base: dict = {}
    if preset:
        base = load_preset(preset)
    elif config_file:
        base = ExtractConfig.from_toml(config_file).model_dump()

    overrides: dict = {
        "input": input,
        "output": output,
        "sampling": sampling,
        "scene_threshold": scene_threshold,
        "frames_per_scene": frames_per_scene,
        "interval_seconds": interval_seconds,
        "blur_threshold": blur_threshold,
        "resolution": resolution,
        "min_bucket": min_bucket,
        "max_bucket": max_bucket,
        "bucket_step": bucket_step,
        "min_pixels": min_pixels,
        "resize_mode": resize_mode,
        "phash_distance": phash_distance,
        "dedup_index": dedup_index,
        "ssim_threshold": ssim_threshold,
        "color_distance": color_distance,
        "output_format": output_format,
        "max_per_video": max_per_video,
        "log_level": log_level,
    }
    # Boolean flags: only override preset when user explicitly passed them.
    if flatten:
        overrides["flatten_output"] = True
    if no_dedup:
        overrides["dedup"] = False
    if no_ssim:
        overrides["ssim_filter"] = False
    if no_color_diversity:
        overrides["color_diversity"] = False
    if completeness:
        overrides["completeness_filter"] = True
    if auto_quality:
        overrides["auto_quality"] = True
    if keyframe:
        overrides["decode_mode"] = "keyframe"
    if no_gallery:
        overrides["contact_sheet"] = False
        overrides["html_gallery"] = False
    if no_skip_existing:
        overrides["skip_existing"] = False
    base.update({k: v for k, v in overrides.items() if v is not None})
    cfg = ExtractConfig(**base)

    if dry_run:
        _print_config(cfg)
        raise typer.Exit(0)

    _print_config(cfg)

    # ── Run with rich progress ──────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[stage]}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as bar:
        task_id = bar.add_task("video", total=1, stage="starting")

        def cb(stage: str, current: int, total: int) -> None:
            bar.update(task_id, completed=current, total=total, stage=stage)

        result = run_pipeline(cfg, progress=cb)

    _print_summary(result)


# ── presets ───────────────────────────────────────────────────────────


@app.command("presets")
def list_presets_cmd() -> None:
    """List built-in presets."""
    presets = list_presets()
    if not presets:
        console.print("[yellow]No presets bundled.[/]")
        return
    table = Table(title="Built-in presets", show_lines=False)
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    for name, desc in presets:
        table.add_row(name, desc)
    console.print(table)


# ── gui ───────────────────────────────────────────────────────────────


@app.command("gui")
def gui_cmd(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 7860,
    share: Annotated[bool, typer.Option("--share", help="Enable public link.")] = False,
) -> None:
    """Launch the Gradio web GUI (requires ``vid2dataset[gui]``)."""
    try:
        from vid2dataset.gui import launch
    except ImportError as e:  # pragma: no cover
        console.print(
            "[red]Gradio GUI not available.[/] Install with: [bold]pip install vid2dataset[gui][/]"
        )
        console.print(f"[dim]{e}[/]")
        sys.exit(1)
    launch(host=host, port=port, share=share)


@app.command("app")
def app_cmd() -> None:
    """Launch the native desktop application (no browser needed)."""
    from vid2dataset.app import main

    main()


# ── helpers ───────────────────────────────────────────────────────────


def _print_config(cfg: ExtractConfig) -> None:
    table = Table(title="Resolved config", show_lines=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    for k, v in cfg.model_dump().items():
        if v is None:
            continue
        table.add_row(k, str(v))
    console.print(table)


def _print_summary(result: PipelineResult) -> None:
    table = Table(title="Extraction summary", show_lines=False)
    table.add_column("Video", style="cyan", overflow="fold")
    table.add_column("Written", justify="right", style="green")
    table.add_column("Candidates", justify="right")
    table.add_column("Blur", justify="right", style="yellow")
    table.add_column("Luma", justify="right", style="yellow")
    table.add_column("SSIM", justify="right", style="yellow")
    table.add_column("Color", justify="right", style="yellow")
    table.add_column("Dup", justify="right", style="magenta")
    table.add_column("Time", justify="right")
    for v in result.videos:
        table.add_row(
            Path(v.video).name,
            str(v.written),
            str(v.candidates),
            str(v.rejected_blur),
            str(v.rejected_luma),
            str(v.rejected_ssim),
            str(v.rejected_color),
            str(v.rejected_dup),
            f"{v.elapsed_s:.1f}s",
        )
    console.print(table)
    console.print(
        f"[bold green]Total: {result.total_written} images[/]  "
        f"in [bold]{result.elapsed_s:.1f}s[/]  "
        f"({result.total_candidates} candidates)"
    )
    if result.contact_sheet_path:
        console.print(f"[dim]Contact sheet:[/] {result.contact_sheet_path}")
    if result.html_gallery_path:
        console.print(f"[dim]HTML gallery:[/] {result.html_gallery_path}")


@app.command("tag")
def tag(
    folder: Annotated[Path, typer.Argument(help="Folder of images to caption.")],
    trigger: Annotated[
        str, typer.Option("--trigger", "-t", help="LoRA trigger word (first caption token).")
    ] = "",
    model: Annotated[str, typer.Option("--model", "-m", help="WD tagger model name.")] = (
        "wd-eva02-large-tagger-v3"
    ),
    threshold: Annotated[
        float, typer.Option("--threshold", help="General tag confidence threshold.")
    ] = 0.35,
    character_threshold: Annotated[
        float, typer.Option("--character-threshold", help="Character tag threshold.")
    ] = 0.85,
    blacklist: Annotated[
        str, typer.Option("--blacklist", help="Comma-separated tags never written to captions.")
    ] = "",
    always: Annotated[
        str, typer.Option("--always", help="Comma-separated tags added after the trigger word.")
    ] = "",
    prune_threshold: Annotated[
        float,
        typer.Option(
            "--prune-threshold",
            help="Remove tags present in at least this fraction of images (0 = off).",
        ),
    ] = 0.0,
    require: Annotated[
        str,
        typer.Option("--require", help="Images missing any of these tags move to _rejected/."),
    ] = "",
    exclude: Annotated[
        str,
        typer.Option("--exclude", help="Images having any of these tags move to _rejected/."),
    ] = "",
    cpu: Annotated[bool, typer.Option("--cpu", help="Force CPU inference.")] = False,
) -> None:
    """Write WD-tagger caption .txt sidecars for an existing image folder.

    Works on any folder — not just vid2dataset output. Downloads the model
    (and onnxruntime if needed) to a per-user cache on first use.
    """
    from vid2dataset.tagger import TAGGER_MODELS, collect_images, tag_folder

    if not folder.exists():
        console.print(f"[red]Path not found:[/] {folder}")
        raise typer.Exit(1)
    if model not in TAGGER_MODELS:
        console.print(f"[red]Unknown model:[/] {model}. Available: {', '.join(TAGGER_MODELS)}")
        raise typer.Exit(1)
    n_images = len(collect_images(folder))
    if n_images == 0:
        console.print(f"[yellow]No images found under[/] {folder}")
        raise typer.Exit(0)

    console.print(f"Tagging [bold]{n_images}[/] images with [bold]{model}[/]...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task_id = prog.add_task("preparing", total=n_images)

        def cb(stage: str, done: int, total: int) -> None:
            if stage == "tagging":
                prog.update(task_id, description="tagging", completed=done, total=total)
            elif total > 0:  # download progress (bytes)
                prog.update(
                    task_id,
                    description=f"downloading {stage}",
                    completed=done // 1048576,
                    total=total // 1048576,
                )

        summary = tag_folder(
            folder,
            model_name=model,
            trigger_word=trigger,
            general_threshold=threshold,
            character_threshold=character_threshold,
            blacklist=blacklist,
            always=always,
            trait_prune_threshold=prune_threshold,
            require=require,
            exclude=exclude,
            use_gpu=not cpu,
            progress_cb=cb,
        )

    console.print(
        f"[green]Done:[/] {summary.tagged} tagged, {summary.failed} failed (of {summary.total})."
    )
    if summary.rejected:
        console.print(
            f"[yellow]{len(summary.rejected)} image(s) moved to _rejected/[/] "
            f"by --require/--exclude rules."
        )
    if summary.pruned_tags:
        console.print(f"[dim]Pruned constant traits:[/] {', '.join(summary.pruned_tags)}")
    if summary.tag_counts:
        top = ", ".join(f"{t} ({c})" for t, c in summary.tag_counts.most_common(10))
        console.print(f"[dim]Top tags:[/] {top}")


@app.command("gpu-test")
def gpu_test() -> None:
    """Diagnose GPU runtime activation. Prints what loads / what fails.

    Used to debug 'cannot load GPU runtime' errors in the .exe build.
    """
    import traceback

    from vid2dataset.gpu_runtime import (
        RUNTIME_DIR,
        activate_runtime,
        detect_gpu,
        runtime_status,
    )

    print(f"[INFO] Runtime dir: {RUNTIME_DIR}")
    print(f"[INFO] Cache exists: {RUNTIME_DIR.exists()}")
    if RUNTIME_DIR.exists():
        files = sorted(p.name for p in RUNTIME_DIR.iterdir())
        print(f"[INFO] Cache contents ({len(files)} entries): {files[:15]}")
    s = runtime_status()
    print(f"[INFO] cached={s.cached} size={s.size_mb:.1f}MB available={s.available}")
    print(f"[INFO] GPU: {detect_gpu()}")
    print(f"[INFO] sys.path before activate: {sys.path[:6]}")

    print("\n[STEP] Calling activate_runtime()...")
    ok, err = activate_runtime()
    print(f"[RESULT] ok={ok}, err={err!r}")
    if not ok:
        print("[FAIL] Activation failed.")
        raise typer.Exit(1)

    # Now actually exercise things
    try:
        print("\n[STEP] import torch from cache...")
        import torch

        print(f"  torch={torch.__version__} from {torch.__file__}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device: {torch.cuda.get_device_name(0)}")
            x = torch.randn(64, 64, device="cuda")
            y = (x @ x.T).sum().item()
            print(f"  matmul OK: sum={y:.4f}")
    except Exception as e:
        print(f"[FAIL] torch test failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise typer.Exit(1) from e

    try:
        print("\n[STEP] import gpu_filters...")
        from vid2dataset.gpu_filters import (
            BatchSSIMFilter,
            device_summary,
            is_gpu_pipeline_available,
        )

        print(f"  device_summary: {device_summary()}")
        print(f"  pipeline available: {is_gpu_pipeline_available()}")
        # Construct filter (this is what the real pipeline does)
        flt = BatchSSIMFilter(ssim_threshold=0.85)
        print(f"  BatchSSIMFilter constructed OK: {flt}")
    except Exception as e:
        print(f"[FAIL] gpu_filters test failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise typer.Exit(1) from e

    print("\n[OK] === GPU runtime fully working ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
