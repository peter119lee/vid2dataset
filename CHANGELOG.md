# Changelog

All notable changes to vid2dataset are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning is [SemVer](https://semver.org/).

## [1.1.0] - 2026-07-11

### Added — caption quality controls

- **Tag blacklist** (`tag_blacklist`, GUI "Blacklist"): tags never written
  into captions (image kept). Accepts `long_hair` or `long hair` forms.
- **Always-tags** (`tag_always`, config/CLI): tags written right after the
  trigger word in every caption (e.g. `anime screencap`).
- **Trait pruning** (`trait_prune_threshold`, GUI "Prune ≥"): tags present in
  ≥ N% of tagged images are removed from ALL captions, so the trigger word
  absorbs the character's constant traits — the standard character-LoRA
  practice, now automatic. Pruned tags are listed in `_report.html`.
- **Tag-based image filtering** (`tag_require` / `tag_exclude`, GUI
  "Require" / "Reject if"): images missing a required tag (e.g. `1girl`) or
  having an excluded tag (e.g. `multiple girls`) are MOVED to
  `output/_rejected/` — the tagger's understanding of each frame now curates
  the dataset, killing multi-character pollution from gameplay/anime sources.
  Stale sidecars move along with the image; gallery/contact sheet drop the
  rejected entries; counts appear in report + CLI.
- All five controls available on `vid2dataset tag` too (`--blacklist`,
  `--always`, `--prune-threshold`, `--require`, `--exclude`).

## [1.0.0] - 2026-07-11

### Added
- **Auto-tagging (WD tagger)**: tick `Auto-tag images`, set a trigger word,
  and every output image gets a kohya-ready `.txt` caption sidecar
  (`trigger, character tags, general tags`). Models: wd-eva02-large-tagger-v3
  (default, ~1.2 GB) and wd-swinv2-tagger-v3 (~450 MB), downloaded on first
  enable to `%LOCALAPPDATA%/vid2dataset/tagger_models/` (huggingface.co with
  hf-mirror.com fallback). Inference via onnxruntime-DirectML — GPU-accelerated
  on any Windows GPU vendor, automatic CPU fallback. The .exe does not grow;
  onnxruntime (~60 MB) downloads on demand like the GPU runtime.
- **`vid2dataset tag FOLDER`** CLI: caption any existing image folder
  (`--trigger`, `--model`, `--threshold`, `--character-threshold`, `--cpu`).
- **kohya repeats folder**: set `kohya_repeats = 10` (config) with
  `flatten_output` to write images into a `10_<trigger>/` subfolder — the
  kohya-ss dreambooth folder convention.
- **Report + gallery**: `_report.html` gains an Auto-tagging section with
  tagged/failed counts and a top-30 tag frequency table (spot dataset bias
  at a glance); `_gallery.html` hover info now shows each image's tags.
- Caption formatting follows kohya conventions and the sd-image-sorter tagger
  audit: single LF-terminated line, underscores to spaces with a kaomoji
  preserve list (`^_^` stays `^_^`), case-insensitive dedup, rating tags never
  enter captions.

### Notes
- Tagging is OFF by default and runs as a strictly separate post-pipeline
  pass — extraction behavior is byte-identical when the box is unticked.
- Scope guard: vid2dataset prepares datasets. Training, upscaling, image
  editing, and prompt tools are permanently out of scope.

## [0.9.0] - 2026-07-11

### Fixed
- **RTX 50-series (Blackwell) GPU runtime was broken**: Blackwell was mapped
  to cu124 + torch 2.5.1, whose wheels carry no sm_100/sm_120 kernels — users
  downloaded 2.4 GB that could never run. Blackwell now gets **cu128**
  (sm 7.5–12.0). Everything else gets **cu126** (sm 5.0–9.0, Maxwell through
  Hopper — wider legacy coverage than the old cu121, same 12.x driver family).
- **Stale-cache overlay**: upgrading the runtime used to extract the new torch
  on top of the old one, leaving orphaned modules/DLLs behind. The downloader
  now wipes files from a previous runtime version first (wheels being
  re-downloaded are kept for retry).
- **Interrupted downloads**: wheels now download to a `.part` file and rename
  on completion, so a partial file is never mistaken for a finished wheel.
  Cached wheels are also keyed by their real filename (version + CUDA tag),
  so a leftover from another version/tag is never reused.

### Changed
- **GPU runtime: torch 2.5.1 → 2.11.0** (newest torch published for both
  cu126 and cu128 Windows wheels; cu121/cu124 are no longer built by PyTorch).
  `RUNTIME_VERSION` bump invalidates old caches; the app will offer to
  re-download (~2.5 GB cu126 / ~2.7 GB cu128).
- Runtime dependency pins synced to the build venv: numpy 2.4.6 (ABI must
  match the .exe bundle), sympy 1.14.0 (torch ≥ 2.6 needs ≥ 1.13.3),
  **setuptools 80.9.0 added** (hard torch dep on Python 3.12), plus
  typing_extensions / filelock / fsspec / networkx / jinja2 / MarkupSafe.

### Added
- **GPU-swap detection**: the manifest now records which CUDA tag the cache
  was built for. If the detected GPU needs a different tag (e.g. you upgraded
  from an RTX 40 card to an RTX 50 card), the app offers to download the
  matching build instead of silently activating an incompatible one.
- Unit tests for `gpu_runtime` (GPU classification, CUDA tag selection,
  pin-consistency guards, wheel cache naming, stale-cache wipe).

## [0.8.0] - 2026-05-29

### Changed
- **GPU runtime now downloads on-demand**: the standalone 2.4 GB `vid2dataset-gpu.exe`
  is gone. The single `vid2dataset.exe` (151 MB) shows a GPU checkbox; ticking it on
  the first time prompts to download PyTorch + CUDA 12.1 (~2.4 GB) to`
  `%LOCALAPPDATA%/vid2dataset/gpu_runtime/`. Subsequent runs use the cache
  and skip the download.
- **GPU acceleration without bloating the .exe**: 99%% of users (CPU-only / no
  NVIDIA GPU / don't want GPU) save 2.3 GB on the initial download. Users who
  do want GPU pay the cost once.

### Added
- `gpu_runtime.py` module with `runtime_status()`, `download_runtime()`,
  `activate_runtime()`, `remove_runtime()`.
- Background-thread downloader with per-package progress (`Downloading torch: 47%%`).
- Self-healing: corrupt cache is detected by missing manifest or torch import failure;
  user can remove `%LOCALAPPDATA%/vid2dataset/gpu_runtime` and re-download.

### Removed
- `vid2dataset-gpu.exe` and `build_exe_gpu.py` (the 2.4 GB bundled-CUDA variant).

## [0.7.0] - 2026-05-29

### Added
- **Watermark cropping (opt-in)**: tick the new `Crop watermarks` checkbox
  to actually expand the bucket crop and remove peripheral watermarks
  detected by v0.6's scanner. Default OFF; warn-only behaviour from
  v0.6 is preserved when unticked.
- **Gallery hover info**: every image in `_gallery.html` now shows
  blur score, bucket size, source frame, and SSIM-rejection siblings
  on mouse hover.
- **Output folder toggle**: GUI checkbox `Flatten output` to switch
  between per-video subfolders (default) and a single flat folder.
- **Cross-platform Open Output Folder**: Linux uses `xdg-open`,
  macOS uses `open`, Windows uses `os.startfile` (was Windows-only).
- **README support matrix**: clearly marks which OS / GPU combinations
  are verified vs untested.
- **Unit tests** for `gpu_filters`, `watermark`, `report` modules.
- **GitHub Actions CI**: runs `pytest` + `ruff check` on every push.
- **CHANGELOG.md** (this file).

### Fixed
- Several small lint cleanups in newly-added modules.

## [0.6.1] - 2026-05-29

### Fixed
- GUI checkbox for `Detect watermarks` was missing in v0.6.0; now
  exposed alongside other quality toggles. Default ON.

## [0.6.0] - 2026-05-29

### Added
- **Watermark detection (warn-only)**: scans every video for static
  text/logo overlays (URLs, artist tags, recording HUDs). Detection
  uses pixel std-dev across 8 sampled frames + edge density + bimodal
  histogram + edge-only position constraint. Logs a WARNING and
  saves region info to per-video stats. Pipeline does **not** modify
  output bytes. Validated: 0 false positives on real 34-video MMD
  dataset; 1 hit on synthetic test (intentional).
- **Pre-flight HTML report (`_report.html`)**: bucket distribution,
  blur histogram, per-video rejection breakdown, watermark warnings.

### Changed
- Total runtime on 34 4K MMD videos: 8.21 min → 9.31 min
  (+1.1 min for the watermark scan + report; turn off via GUI
  checkbox or `detect_watermark = False`).

### Notes
- For the same input, output **PNG bytes are identical** to v0.5.0.
  Only `_report.html` and a `watermarks` field in stats are new.

## [0.5.0] - 2026-05-29

### Changed
- **Skip PySceneDetect in keyframe mode**: when `decode_mode='keyframe'`
  AND ffmpeg is available, the frame source streams I-frames directly,
  so PySceneDetect's output was being ignored on this code path. Skip
  it entirely. Saves ~25% of total per-video time.
- **Async PNG writer pool**: encoding + disk write moved to a 2-worker
  thread pool. Negligible gain on SSD, helps on HDD or with many small
  videos.

### Performance
- Total runtime on 34 4K MMD videos: 14.13 min → 8.21 min (1.72x)
- Single 218 s 4K video: 22.4 s → 8.5 s (2.6x)

## [0.4.0] - 2026-05-29

### Added
- **GPU SSIM diversity filter** via PyTorch (`gpu_filters.BatchSSIMFilter`).
  Auto-detects CUDA / MPS / CPU. Self-validates on first frame against
  CPU output; auto-disables on mismatch. 4.6x faster than CPU SSIM in
  microbenchmark.
- **`gpu_accel`** config field + GUI checkbox.
- **`vid2dataset-gpu.exe`** variant: bundles PyTorch + CUDA 12.1
  (~2.4 GB, split into 7z parts to fit GitHub's 2 GB asset limit).

### Changed
- Total runtime: 16.0 min → 14.1 min (1.13x). The pipeline is filter-
  bound, so GPU SSIM only saves what SSIM was costing (~30% of time).

### Notes
- `BatchColorFilter` was implemented but disabled because GPU was
  0.7x slower than CPU due to per-frame transfer overhead. Code kept
  as reference.

## [0.3.1] - 2026-05-23

### Fixed
- **Console window flood on Windows**: every ffmpeg subprocess call
  popped up a transient console window that stole keyboard focus.
  Fixed by adding `CREATE_NO_WINDOW` to all subprocess calls.

### Added
- **Opt-in ffmpeg hwaccel** (`-hwaccel cuda/qsv/d3d11va/...`) with
  output validation against CPU.

### Performance
- Same dataset: 16 min → still ~16 min. NVDEC decode is fast but
  pipeline is filter-bound, so the win is small.

## [0.3.0] - 2026-05-22

### Added
- **Parallel video processing** via `ThreadPoolExecutor` (workers
  auto-detected from CPU cores + RAM).
- **Cancel button** to abort mid-extraction (cooperative
  `threading.Event`).
- **Completion sound** via `winsound`.
- **Drag-and-drop folder** (Windows, via `windnd`).
- **Hover tooltips** on parameter fields (custom widget).
- **Hardware-aware worker auto-detection** (`hardware.py`): scales
  with CPU cores and available RAM, never crashes weak machines.
- **ffmpeg keyframe decoder** module (`keyframe_decoder.py`).
- **PySceneDetect frame_skip** for ~4x faster scene detection on
  60 fps content.
- **`sanitize_stem`** with MD5 hash suffix on collision.

### Performance
- Total runtime: 50 min → 16 min (3.1x).

## [0.2.0] - 2026-05-23

### Fixed
- **`min_per_video` guarantee was fake**: backup pool only contained
  diversity-rejected frames. Now uses a bounded heap of all
  quality-passed frames.
- **Chinese / non-ASCII path support**: cv2.VideoCapture and imwrite
  silently failed on non-ANSI Windows paths. Now uses `\\?\` prefix
  for input + `cv2.imencode` + `Path.write_bytes` for output.
- **Preset loading on PyInstaller .exe**: added `sys._MEIPASS`
  fallback for `importlib.resources`.
- **Memory leaks**: `DiversityFilter._accepted` and
  `ColorDiversityFilter._fingerprints` were unbounded.
- **Integer overflow**: `completeness` filter used uint8 sums on
  >16 MP frames.
- **Wrong widget**: error dialogs used `CTkInputDialog` (a text input)
  instead of a real message box.

### Added
- **One-click update checker** that queries GitHub Releases and stages
  an `.exe` swap on next launch.

## [0.1.0] - 2026-05-22

Initial release.

### Added
- Scene-aware sampling (PySceneDetect)
- Blur + brightness quality gate
- Letterbox auto-crop
- Bucket-aware resize (Anima/SDXL 64-step grid)
- SSIM diversity + Color diversity + pHash dedup
- Auto blur threshold detection (per video)
- Per-video minimum guarantee
- Subject size filter
- Contact sheet + HTML gallery
- ETA estimation
- 中文 / English UI toggle
- Remember last-used settings
- 3 built-in presets: `anima-style`, `anima-character`, `fast-preview`